import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from app.services.model_trainer import ModelTrainer
from app.core.config import settings
from app.core.logging import loggers
from app.services.realtor_api import RealtorAPIService
from app.services.data_collector import DataCollector
import asyncio
from typing import List, Dict, Any
import json
from datetime import datetime

logger = loggers['ml']

def safe_read_csv(path):
    try:
        if os.path.getsize(path) == 0:
            loggers['ml'].warning(f"File {path} is empty. Skipping.")
            return None
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            loggers['ml'].warning(f"File {path} has no columns or data. Skipping.")
            return None
        return df
    except Exception as e:
        loggers['ml'].warning(f"Could not read {path}: {e}. Skipping.")
        return None

async def collect_property_data(locations: List[str], max_properties_per_location: int = 50) -> List[Dict[str, Any]]:
    """
    Collect individual property data from multiple locations for FAISS training.
    
    Args:
        locations: List of location strings (city, state combinations)
        max_properties_per_location: Maximum number of properties to collect per location
        
    Returns:
        List of property dictionaries with complete feature data
    """
    try:
        logger.info(f"Starting property data collection from {len(locations)} locations")
        
        # Initialize services
        realtor_api = RealtorAPIService()
        data_collector = DataCollector()
        
        all_properties = []
        
        for location in locations:
            try:
                logger.info(f"Collecting properties from {location}")
                
                # Parse location (assuming format "City, ST")
                if ',' in location:
                    city, state_code = [part.strip() for part in location.split(',')]
                else:
                    logger.warning(f"Invalid location format: {location}. Expected 'City, ST'")
                    continue
                
                # Search for properties in this location
                properties = await realtor_api.search_properties(
                    city=city,
                    state_code=state_code,
                    limit=max_properties_per_location,
                    property_type="single_family"
                )
                
                if not properties:
                    logger.warning(f"No properties found for {location}")
                    continue
                
                logger.info(f"Found {len(properties)} properties in {location}")
                
                # Validate and clean property data
                valid_properties = []
                for prop in properties:
                    if validate_property_data(prop):
                        valid_properties.append(prop)
                    else:
                        logger.debug(f"Skipping invalid property: {prop.get('property_id', 'unknown')}")
                
                logger.info(f"Validated {len(valid_properties)} properties from {location}")
                all_properties.extend(valid_properties)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting properties from {location}: {str(e)}")
                continue
        
        logger.info(f"Total properties collected: {len(all_properties)}")
        return all_properties
        
    except Exception as e:
        logger.error(f"Error in property data collection: {str(e)}")
        raise

def validate_property_data(property_data: Dict[str, Any]) -> bool:
    """
    Validate that property data contains required fields for training.
    
    Args:
        property_data: Property dictionary
        
    Returns:
        True if property data is valid, False otherwise
    """
    try:
        # Required fields for autoencoder training
        required_fields = [
            'property_id', 'price', 'beds', 'baths', 'area',
            'latitude', 'longitude'
        ]
        
        # Check if all required fields exist and are not None/empty
        for field in required_fields:
            value = property_data.get(field)
            if value is None or value == '' or value == 0:
                return False
                
            # Check if numeric fields are actually numeric
            if field in ['price', 'beds', 'baths', 'area', 'latitude', 'longitude']:
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False
        
        # Additional validation
        price = float(property_data.get('price', 0))
        area = float(property_data.get('area', 0))
        
        # Basic sanity checks
        if price < 10000 or price > 50000000:  # Reasonable price range
            return False
        if area < 100 or area > 20000:  # Reasonable area range
            return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Error validating property data: {str(e)}")
        return False

def prepare_training_data():
    """Prepare training data from CSV files."""
    try:
        # Load Zillow data
        zillow_metro = safe_read_csv(settings.DATA_DIR / 'zillow_metro.csv')
        zillow_county = safe_read_csv(settings.DATA_DIR / 'zillow_county.csv')
        zillow_zip = safe_read_csv(settings.DATA_DIR / 'zillow_zip.csv')
        
        # Load Realtor data
        realtor_metro = safe_read_csv(settings.DATA_DIR / 'realtor_metro.csv')
        realtor_county = safe_read_csv(settings.DATA_DIR / 'realtor_county.csv')
        realtor_zip = safe_read_csv(settings.DATA_DIR / 'realtor_zip.csv')
        
        # Combine data
        dfs = [df for df in [realtor_metro, realtor_county, realtor_zip, zillow_metro, zillow_county, zillow_zip] if df is not None]
        if not dfs:
            raise ValueError("No valid data files found for training.")
        data = pd.concat(dfs, ignore_index=True)
        # Map column names to expected features
        column_mapping = {
            'median_listing_price': 'median_list_price',
            'median_listing_price_per_square_foot': 'median_list_price_per_sqft',
            'median_days_on_market': 'median_dom',
            'active_listing_count': 'inventory',
            'median_square_feet': 'sqft',
            'average_listing_price': 'avg_list_price'
        }
        # Rename columns that exist in the data
        existing_columns = {k: v for k, v in column_mapping.items() if k in data.columns}
        data = data.rename(columns=existing_columns)
        # Remove duplicate columns by keeping only the first occurrence (after renaming)
        data = data.loc[:, ~data.columns.duplicated(keep='first')]
        
        # Ensure all required features exist
        for feature in settings.REQUIRED_FEATURES:
            if feature not in data.columns:
                data[feature] = float('nan')
        
        # Calculate additional features
        if 'median_list_price' in data.columns:
            print('DEBUG: type of data["median_list_price"]:', type(data['median_list_price']))
            print('DEBUG: head of data["median_list_price"]:', data['median_list_price'].head())
            data['median_list_price'] = pd.to_numeric(data['median_list_price'], errors='coerce')
            median_price = data['median_list_price'].median(skipna=True)
            mean_price = data['median_list_price'].mean(skipna=True)
            data['price_to_median_ratio'] = data['median_list_price'] / median_price
            data['price_to_avg_ratio'] = data['median_list_price'] / mean_price
        
        # Create target variables
        if 'median_list_price' in data.columns:
            data['price'] = pd.to_numeric(data['median_list_price'], errors='coerce')
        else:
            data['price'] = float('nan')
        # For ROI, we'll use the year-over-year price change if available, otherwise use a default value
        if 'median_listing_price_yy' in data.columns:
            data['roi'] = pd.to_numeric(data['median_listing_price_yy'], errors='coerce') / 100  # Convert percentage to decimal
        else:
            data['roi'] = 0.05  # Default to 5% ROI if no historical data available
        
        # Handle missing values
        for feature in settings.REQUIRED_FEATURES:
            if feature in data.columns:
                data[feature] = data[feature].fillna(data[feature].median())
        
        return data
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise

async def main():
    """Main function to train models."""
    try:
        logger.info("Starting comprehensive model training process")
        
        # Create model directory if it doesn't exist
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        
        # Define locations to collect property data from
        target_locations = [
            "Dallas, TX",
            "Austin, TX", 
            "Houston, TX",
            "San Antonio, TX",
            "Phoenix, AZ",
            "Atlanta, GA",
            "Miami, FL",
            "Orlando, FL",
            "Denver, CO",
            "Seattle, WA"
        ]
        
        logger.info("Phase 1: Collecting individual property data for FAISS training")
        # Collect individual property data for FAISS similarity search
        try:
            property_data = await collect_property_data(target_locations, max_properties_per_location=100)
            
            if len(property_data) < 50:  # Minimum threshold for meaningful training
                logger.warning(f"Only {len(property_data)} properties collected. This may not be sufficient for good similarity search.")
                logger.info("Proceeding with available data...")
            
            # Train autoencoder and build FAISS index
            if property_data:
                logger.info("Phase 2: Training autoencoder and building FAISS index")
                trainer = ModelTrainer()
                await trainer.train_autoencoder_and_faiss(property_data)
            else:
                logger.error("No property data available for FAISS training")
        
        except Exception as e:
            logger.error(f"Error in FAISS training phase: {str(e)}")
            logger.info("Continuing with traditional model training...")
        
        logger.info("Phase 3: Training traditional prediction models")
        # Traditional model training for property value and investment prediction
        try:
            # Prepare aggregated training data
            training_data = prepare_training_data()
            logger.info(f"Prepared training data with {len(training_data)} samples")
            
            # Initialize model trainer
            trainer = ModelTrainer()
            
            # Train traditional models
            results = trainer.train_models(training_data)
            
            logger.info("Traditional model training completed successfully")
            logger.info(f"Training results: {results}")
        
        except Exception as e:
            logger.error(f"Error in traditional model training: {str(e)}")
            raise
        
        logger.info("All model training phases completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 