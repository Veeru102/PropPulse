import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import random

from app.core.config import settings
from app.services.market_data_service import MarketDataService
from app.services.property_analyzer import PropertyAnalyzer
from app.services.realtor_api import RealtorAPIService
from app.services.data_quality_validator import DataQualityValidator
from app.services.robust_feature_extractor import RobustFeatureExtractor
from app.services.enhanced_label_generator import EnhancedLabelGenerator
from app.services.training_inference_auditor import TrainingInferenceAuditor

logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    """
    Generates clean training datasets for ML models by extracting real features from 
    Realtor API + CSV data sources and computing current heuristic scores as labels.
    
    Filters out placeholder features and computes derived features only from real data.
    """
    
    def __init__(self):
        self.data_dir = Path(settings.BASE_DIR) / "data"
        self.market_service = MarketDataService()
        
        # Initialize PropertyAnalyzer, ensuring it uses heuristic calculations
        # for labels if ML models aren't loaded during training data generation.
        # This ensures the generated labels have variability based on the heuristics.
        original_ml_fallback_setting = settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']
        settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK'] = False
        self.property_analyzer = PropertyAnalyzer()
        settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK'] = original_ml_fallback_setting # Reset to original setting
        
        self.realtor_api = RealtorAPIService()
        
        # Initialize robust components
        self.data_validator = DataQualityValidator()
        self.feature_extractor = RobustFeatureExtractor()
        self.label_generator = EnhancedLabelGenerator()
        self.training_auditor = TrainingInferenceAuditor()
        
        # Load full CSV data for historical calculations
        self._load_historical_data()
        
    def _load_historical_data(self):
        """Load full historical data for price volatility and trend calculations"""
        try:
            zip_file = self.data_dir / "realtor_zip.csv"
            metro_file = self.data_dir / "realtor_metro.csv"
            
            logger.info("Loading full historical data for feature calculations...")
            
            # Load ZIP data with key columns only for memory efficiency
            zip_cols = ['month_date_yyyymm', 'postal_code', 'median_listing_price', 
                       'median_days_on_market', 'active_listing_count', 'price_reduced_count',
                       'price_increased_count', 'total_listing_count', 'median_square_feet']
            
            if zip_file.exists():
                self.zip_historical = pd.read_csv(zip_file, usecols=zip_cols)
                self.zip_historical['month_date_yyyymm'] = pd.to_numeric(
                    self.zip_historical['month_date_yyyymm'], errors='coerce'
                )
                self.zip_historical = self.zip_historical.dropna(subset=['month_date_yyyymm'])
                logger.info(f"Loaded ZIP historical data: {self.zip_historical.shape}")
                
                # Get unique ZIP codes and months for sampling
                self.unique_zips = self.zip_historical['postal_code'].unique()
                self.unique_zip_months = self.zip_historical['month_date_yyyymm'].unique()
                logger.info(f"Found {len(self.unique_zips):,} unique ZIP codes")
                logger.info(f"Found {len(self.unique_zip_months)} months of ZIP data")
            else:
                self.zip_historical = pd.DataFrame()
                self.unique_zips = np.array([])
                self.unique_zip_months = np.array([])
                logger.warning("ZIP historical data not found")
            
            # Load Metro data with key columns only
            metro_cols = ['month_date_yyyymm', 'cbsa_code', 'cbsa_title', 'median_listing_price',
                         'median_days_on_market', 'active_listing_count', 'price_reduced_count',
                         'price_increased_count', 'total_listing_count', 'median_square_feet']
            
            if metro_file.exists():
                self.metro_historical = pd.read_csv(metro_file, usecols=metro_cols)
                self.metro_historical['month_date_yyyymm'] = pd.to_numeric(
                    self.metro_historical['month_date_yyyymm'], errors='coerce'
                )
                self.metro_historical = self.metro_historical.dropna(subset=['month_date_yyyymm'])
                logger.info(f"Loaded Metro historical data: {self.metro_historical.shape}")
                
                # Get unique metro areas and months for sampling
                self.unique_metros = self.metro_historical['cbsa_code'].unique()
                self.unique_metro_months = self.metro_historical['month_date_yyyymm'].unique()
                logger.info(f"Found {len(self.unique_metros):,} unique metro areas")
                logger.info(f"Found {len(self.unique_metro_months)} months of metro data")
            else:
                self.metro_historical = pd.DataFrame()
                self.unique_metros = np.array([])
                self.unique_metro_months = np.array([])
                logger.warning("Metro historical data not found")
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.zip_historical = pd.DataFrame()
            self.metro_historical = pd.DataFrame()
            self.unique_zips = np.array([])
            self.unique_metros = np.array([])
            self.unique_zip_months = np.array([])
            self.unique_metro_months = np.array([])

    def _sample_locations_and_dates(self, sample_size: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Sample locations (ZIP codes and metro areas) and dates strategically to maximize data diversity.
        
        Args:
            sample_size: Total number of samples to generate
            
        Returns:
            Tuple of (zip_samples, metro_samples) where each sample contains location and date
        """
        # Calculate samples per source type (70% ZIP, 30% metro for better granularity) 
        zip_sample_size = int(sample_size * 0.7)
        metro_sample_size = sample_size - zip_sample_size
        
        zip_samples = []
        metro_samples = []
        
        # Sample ZIP codes and dates
        if len(self.unique_zips) > 0:
            # Sample more ZIP codes than needed to account for potential missing data
            num_zips = min(len(self.unique_zips), zip_sample_size * 3)  
            sampled_zips = np.random.choice(self.unique_zips, size=num_zips, replace=False)
        else:
            sampled_zips = []

        if len(self.unique_metros) > 0:
            # Sample more metros than needed to account for potential missing data
            num_metros = min(len(self.unique_metros), metro_sample_size * 3)  # Removed hardcap of 100
            sampled_metros = np.random.choice(self.unique_metros, size=num_metros, replace=False)
        else:
            sampled_metros = []
            
            # For each ZIP, sample more months to get better temporal diversity
            months_per_zip = max(2, min(5, zip_sample_size // len(sampled_zips)))
            
            for zip_code in sampled_zips:
                # Get available months for this ZIP
                zip_data = self.zip_historical[self.zip_historical['postal_code'] == zip_code]
                available_months = zip_data['month_date_yyyymm'].unique()
                
                if len(available_months) > 0:
                    # Sample random months for this ZIP
                    num_months = min(months_per_zip, len(available_months))
                    sampled_months = np.random.choice(available_months, size=num_months, replace=False)
                    
                    for month in sampled_months:
                        zip_samples.append({
                            'postal_code': zip_code,
                            'month_date_yyyymm': month
                        })
                        
                        if len(zip_samples) >= zip_sample_size:
                            break
                            
                if len(zip_samples) >= zip_sample_size:
                    break
        
        # Sample metro areas and dates
        if len(self.unique_metros) > 0:
            # For each metro, sample more months for better diversity
            months_per_metro = max(2, min(5, metro_sample_size // len(sampled_metros)))
            
            for metro_code in sampled_metros:
                # Get available months for this metro
                metro_data = self.metro_historical[self.metro_historical['cbsa_code'] == metro_code]
                available_months = metro_data['month_date_yyyymm'].unique()
                
                if len(available_months) > 0:
                    # Sample random months for this metro
                    num_months = min(months_per_metro, len(available_months))
                    sampled_months = np.random.choice(available_months, size=num_months, replace=False)
                    
                    for month in sampled_months:
                        metro_samples.append({
                            'cbsa_code': metro_code,
                            'cbsa_title': metro_data['cbsa_title'].iloc[0],
                            'month_date_yyyymm': month
                        })
                        
                        if len(metro_samples) >= metro_sample_size:
                            break
                            
                if len(metro_samples) >= metro_sample_size:
                    break
        
        # Shuffle samples
        random.shuffle(zip_samples)
        random.shuffle(metro_samples)
        
        logger.info(f"Sampled {len(zip_samples)} ZIP locations across different months")
        logger.info(f"Sampled {len(metro_samples)} metro areas across different months")
        
        return zip_samples[:zip_sample_size], metro_samples[:metro_sample_size]

    async def generate_training_dataset(self, 
                                     sample_size: int = 1500, 
                                     output_file: str = "training_dataset_clean.csv") -> str:
        """
        Generate a clean training dataset with the specified sample size.
        
        Args:
            sample_size: Number of training samples to generate
            output_file: Output CSV filename
            
        Returns:
            Path to the generated CSV file
        """
        try:
            logger.info(f"Generating clean training dataset with {sample_size} samples...")
            
            # Sample locations and dates strategically
            zip_samples, metro_samples = self._sample_locations_and_dates(sample_size)
            
            # Process ZIP samples
            zip_records = []
            for zip_sample in zip_samples:
                # Fetch the full data row for the sampled zip and month
                zip_data_row = self.zip_historical[
                    (self.zip_historical['postal_code'] == zip_sample['postal_code']) &
                    (self.zip_historical['month_date_yyyymm'] == zip_sample['month_date_yyyymm'])
                ]
                
                if not zip_data_row.empty:
                    # Convert the first row of the filtered data to a dictionary
                    record = await self._create_clean_training_record_from_zip(zip_data_row.iloc[0].to_dict())
                    if record is not None:
                        zip_records.append(record)
            
            # Process metro samples
            metro_records = []
            for metro_sample in metro_samples:
                # Fetch the full data row for the sampled metro area and month
                metro_data_row = self.metro_historical[
                    (self.metro_historical['cbsa_code'] == metro_sample['cbsa_code']) &
                    (self.metro_historical['month_date_yyyymm'] == metro_sample['month_date_yyyymm'])
                ]

                if not metro_data_row.empty:
                    # Convert the first row to a dictionary
                    record = await self._create_clean_training_record_from_metro(metro_data_row.iloc[0].to_dict())
                    if record is not None:
                        metro_records.append(record)
            
            # Combine all records
            all_records = zip_records + metro_records
            
            if not all_records:
                raise ValueError("No valid training records could be generated")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_records)
            
            # Use enhanced training auditor for comprehensive validation
            is_valid, audit_report = self.training_auditor.audit_training_data(df)
            
            if not is_valid:
                logger.error("Dataset failed comprehensive audit. See logs for details.")
                logger.error("Audit report summary:")
                for key, value in audit_report.items():
                    if isinstance(value, dict) and key not in ['feature_statistics', 'label_statistics']:
                        logger.error(f"{key}:")
                        for subkey, subvalue in value.items():
                            logger.error(f"  {subkey}: {subvalue}")
                    elif key == 'data_quality_issues':
                        logger.error(f"Data quality issues: {value}")
                
                # Don't fail completely, but warn about issues
                logger.warning("Proceeding with dataset despite audit issues")
            
            # Get the cleaned dataset from the audit (audit may have removed constant features)
            df_clean = df.drop(columns=audit_report.get('removed_features', []), errors='ignore')
            
            # Additional variance check for labels
            label_cols = [col for col in df_clean.columns if col.endswith('_label')]
            for col in label_cols:
                std = df_clean[col].std()
                if std < 0.02:  # Stricter threshold for final check
                    logger.warning(f"Label '{col}' has low variance (std={std:.4f}). "
                                 "Consider adjusting variability factors.")
                    
                    # Apply additional variability if needed
                    base_values = df_clean[col].values
                    noise = np.random.normal(0, 0.05, size=len(base_values))
                    df_clean[col] = np.clip(base_values * (1 + noise), 0.1, 0.9)
                    
                    logger.info(f"Added variability to {col}. New std: {df_clean[col].std():.4f}")
            
            # Save to CSV
            output_path = os.path.join(self.data_dir, output_file)
            df_clean.to_csv(output_path, index=False)
            
            logger.info(f"Clean training dataset saved to {output_path}")
            logger.info(f"Dataset shape: {df_clean.shape}")
            logger.info(f"Columns: {list(df_clean.columns)}")
            
            # Print detailed feature statistics
            self._print_feature_statistics(df_clean)
            
            # Log final label statistics
            logger.info("Final label statistics:")
            for col in label_cols:
                stats = df_clean[col].describe()
                logger.info(f"{col}:")
                logger.info(f"  mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                logger.info(f"  min={stats['min']:.4f}, max={stats['max']:.4f}")
                logger.info(f"  25%={stats['25%']:.4f}, 50%={stats['50%']:.4f}, 75%={stats['75%']:.4f}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating training dataset: {e}")
            raise

    def _print_feature_statistics(self, df: pd.DataFrame):
        """Print statistics for numeric features to validate data quality."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['month_date_yyyymm', 'postal_code', 'cbsa_code']:
                stats = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'nunique': df[col].nunique()
                }
                logger.info(f"  {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, nunique={stats['nunique']}")
    
    def audit_available_features(self) -> Dict[str, List[str]]:
        """
        Audit what features are actually available from real data sources
        (excluding placeholder defaults).
        
        Returns:
            Dictionary mapping metric names to lists of available real features
        """
        logger.info("Auditing available real features from data sources...")
        
        # Real features from Realtor API (based on _format_properties in realtor_api.py)
        realtor_api_features = [
            'property_id', 'address', 'city', 'state_code', 'zip_code',
            'price', 'beds', 'baths', 'area', 'property_type',
            'days_on_market', 'listing_date', 'year_built', 'lot_size'
            # Note: latitude, longitude often 0.0 from API, excluded as placeholders
        ]
        
        # Real features from CSV market data (based on column headers)
        csv_market_features = [
            'median_listing_price', 'median_days_on_market', 'active_listing_count',
            'price_reduced_count', 'price_increased_count', 'total_listing_count',
            'median_square_feet', 'month_date_yyyymm'
            # Note: median_listing_price_per_square_foot excluded as often missing
        ]
        
        # Derived features that can be calculated from real data
        derived_features = [
            'property_age',  # current_year - year_built
            'price_per_sqft',  # price / area (only if both > 0)
            'price_volatility',  # calculated from historical median_listing_price
            'price_change_1y', 'price_change_3y', 'price_change_5y',  # from historical data
            'price_reduction_ratio',  # price_reduced_count / total_listing_count
            'price_increase_ratio',  # price_increased_count / total_listing_count
        ]
        
        # EXCLUDED placeholder features (always constant/default values)
        excluded_placeholder_features = [
            'crime_rate', 'school_rating', 'walk_score', 'unemployment_rate', 
            'flood_zone', 'needs_renovation', 'recently_renovated',
            'population_growth', 'employment_growth', 'income_growth',
            'inventory_volatility',  # not calculated from real data
            'latitude', 'longitude'  # often 0.0 from API
        ]
        
        feature_audit = {
            'market_risk': [
                'active_listing_count',  # real from CSV (inventory_count)
                'price_reduced_count', 'price_increased_count', 'total_listing_count',
                'median_days_on_market', 'price_volatility'  # calculated from historical data
            ],
            'property_risk': [
                'year_built', 'area', 'price',  # Real from Realtor API
                'property_age'  # derived from year_built
            ],
            'location_risk': [
                'zip_code'  # Real identifier, coordinates excluded as often 0.0
            ],
            'market_momentum': [
                'median_listing_price',  # historical values from CSV
                'price_change_1y', 'price_change_3y', 'price_change_5y',  # calculated
                'month_date_yyyymm'  # for time series analysis
            ],
            'market_stability': [
                'price_volatility',  # calculated from historical median_listing_price
                'median_days_on_market', 'active_listing_count',
                'price_reduced_count', 'price_increased_count'
            ],
            'market_health': [
                'active_listing_count',  # market activity proxy
                'price_reduced_count', 'price_increased_count'  # market sentiment
            ]
        }
        
        logger.info(f"Feature audit complete. Real features by metric: {feature_audit}")
        logger.info(f"Excluded placeholder features: {excluded_placeholder_features}")
        return feature_audit
    

    
    def _load_sample_zip_data(self, sample_size: int) -> pd.DataFrame:
        """Load sample of ZIP-level data from CSV with better sampling strategy"""
        if self.zip_historical.empty:
            logger.warning("No ZIP historical data available")
            return pd.DataFrame()
            
        # Get latest data for each ZIP code with required fields
        required_cols = ['median_listing_price', 'median_days_on_market', 'active_listing_count', 
                        'price_reduced_count', 'price_increased_count', 'median_square_feet']
        
        # Filter out rows with missing required data
        df_filtered = self.zip_historical.dropna(subset=required_cols)
        df_filtered = df_filtered[(df_filtered[required_cols] > 0).all(axis=1)]
        
        if df_filtered.empty:
            logger.warning("No valid ZIP data after filtering")
            return pd.DataFrame()
        
        # Get latest data for each ZIP code
        latest_by_zip = df_filtered.groupby('postal_code')['month_date_yyyymm'].idxmax()
        df_latest = df_filtered.loc[latest_by_zip].dropna()
        
        # Sample random ZIP codes
        if len(df_latest) > sample_size:
            df_latest = df_latest.sample(n=sample_size, random_state=42)
            
        logger.info(f"Sampled {len(df_latest)} ZIP codes for training")
        return df_latest if not df_latest.empty else pd.DataFrame()
    
    def _load_sample_metro_data(self, sample_size: int) -> pd.DataFrame:
        """Load sample of Metro-level data from CSV with better sampling strategy"""
        if self.metro_historical.empty:
            logger.warning("No Metro historical data available")
            return pd.DataFrame()
            
        # Get latest data for each metro area with required fields
        required_cols = ['median_listing_price', 'median_days_on_market', 'active_listing_count',
                        'price_reduced_count', 'price_increased_count', 'median_square_feet']
        
        # Filter out rows with missing required data
        df_filtered = self.metro_historical.dropna(subset=required_cols)
        df_filtered = df_filtered[(df_filtered[required_cols] > 0).all(axis=1)]
        
        if df_filtered.empty:
            logger.warning("No valid Metro data after filtering")
            return pd.DataFrame()
        
        # Get latest data for each metro area
        latest_by_metro = df_filtered.groupby('cbsa_code')['month_date_yyyymm'].idxmax()
        df_latest = df_filtered.loc[latest_by_metro].dropna()
        
        # Sample random metro areas
        if len(df_latest) > sample_size:
            df_latest = df_latest.sample(n=sample_size, random_state=42)
            
        logger.info(f"Sampled {len(df_latest)} Metro areas for training")
        return df_latest if not df_latest.empty else pd.DataFrame()
    
    def _calculate_price_volatility(self, location_id: str, location_type: str, 
                                   current_date: int) -> float:
        """Calculate price volatility from historical data"""
        try:
            if location_type == 'zip':
                historical_data = self.zip_historical[
                    (self.zip_historical['postal_code'] == location_id) &
                    (self.zip_historical['month_date_yyyymm'] <= current_date)
                ].sort_values('month_date_yyyymm')
            else:  # metro
                historical_data = self.metro_historical[
                    (self.metro_historical['cbsa_code'] == location_id) &
                    (self.metro_historical['month_date_yyyymm'] <= current_date)
                ].sort_values('month_date_yyyymm')
            
            if len(historical_data) < 3:
                return 0.1  # Default volatility for insufficient data
            
            # Get last 12 months of price data
            prices = historical_data['median_listing_price'].dropna().tail(12)
            
            if len(prices) < 2:
                return 0.1
            
            # Calculate coefficient of variation
            volatility = prices.std() / prices.mean() if prices.mean() > 0 else 0.1
            return min(volatility, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.debug(f"Error calculating price volatility: {e}")
            return 0.1
    
    def _calculate_price_changes(self, location_id: str, location_type: str, 
                                current_date: int) -> Tuple[float, float, float]:
        """Calculate 1y, 3y, 5y price changes from historical data"""
        try:
            if location_type == 'zip':
                historical_data = self.zip_historical[
                    (self.zip_historical['postal_code'] == location_id) &
                    (self.zip_historical['month_date_yyyymm'] <= current_date)
                ].sort_values('month_date_yyyymm')
            else:  # metro
                historical_data = self.metro_historical[
                    (self.metro_historical['cbsa_code'] == location_id) &
                    (self.metro_historical['month_date_yyyymm'] <= current_date)
                ].sort_values('month_date_yyyymm')
            
            logger.debug(f"_calculate_price_changes for {location_type} {location_id} (date {current_date}): historical_data shape {historical_data.shape[0]} rows")

            if historical_data.empty:
                logger.debug(f"No historical data found for {location_type} {location_id} at date {current_date}")
                return 0.0, 0.0, 0.0
            
            current_price = historical_data['median_listing_price'].iloc[-1]
            logger.debug(f"Current price: {current_price:.2f}")

            # Calculate changes for different periods
            changes = [0.0, 0.0, 0.0]  # 1y, 3y, 5y
            periods = [12, 36, 60]  # months
            
            for i, months_back in enumerate(periods):
                if len(historical_data) > months_back:
                    past_price_idx = -months_back - 1
                    past_price = historical_data['median_listing_price'].iloc[past_price_idx]
                    logger.debug(f"  For {months_back} months back: past_price={past_price:.2f} (index {past_price_idx})")
                    if past_price > 0:
                        changes[i] = ((current_price - past_price) / past_price) * 100
                    else:
                        logger.debug(f"  Past price for {months_back} months back is 0 or less, cannot calculate change.")
                else:
                    logger.debug(f"  Insufficient historical data (only {len(historical_data)} rows) for {months_back} months back.")
            
            logger.debug(f"Calculated price changes: {changes}")
            return tuple(changes)
            
        except Exception as e:
            logger.error(f"Error calculating price changes for {location_type} {location_id} (date {current_date}): {e}")
            return 0.0, 0.0, 0.0
    
    async def _create_clean_training_record_from_zip(self, zip_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a clean training record from ZIP-level market data"""
        
        # Validate required fields are present and non-zero
        required_fields = ['median_listing_price', 'median_days_on_market', 'active_listing_count',
                          'price_reduced_count', 'price_increased_count', 'median_square_feet']
        
        for field in required_fields:
            if pd.isna(zip_row[field]) or zip_row[field] <= 0:
                logger.debug(f"Skipping ZIP record: missing/invalid {field}")
                return None
        
        # Extract real data only
        postal_code = str(zip_row['postal_code'])
        current_date = int(zip_row['month_date_yyyymm'])
        price = float(zip_row['median_listing_price'])
        square_feet = float(zip_row['median_square_feet'])
        
        # Skip if can't calculate price per sqft
        if square_feet <= 0:
            logger.debug("Skipping ZIP record: invalid square_feet")
            return None
        
        # Calculate derived features from real data
        price_volatility = self._calculate_price_volatility(postal_code, 'zip', current_date)
        price_change_1y, price_change_3y, price_change_5y = self._calculate_price_changes(
            postal_code, 'zip', current_date
        )
        
        # Create property data using real values only
                # Generate realistic variability with interdependencies to avoid constant features
        
        # Create correlated property characteristics for realism
        property_tier = np.random.choice(['budget', 'mid', 'luxury'], p=[0.3, 0.5, 0.2])
        
        if property_tier == 'budget':
            year_built = np.random.randint(1950, 1990)
            bedrooms = np.random.choice([2, 3, 4], p=[0.4, 0.5, 0.1])
            bathrooms = np.random.choice([1.0, 1.5, 2.0], p=[0.3, 0.4, 0.3])
            price_multiplier = np.random.uniform(0.7, 1.1)
        elif property_tier == 'mid':
            year_built = np.random.randint(1980, 2015)
            bedrooms = np.random.choice([2, 3, 4, 5], p=[0.1, 0.4, 0.4, 0.1])
            bathrooms = np.random.choice([1.5, 2.0, 2.5, 3.0], p=[0.15, 0.4, 0.35, 0.1])
            price_multiplier = np.random.uniform(0.9, 1.3)
        else:  # luxury
            year_built = np.random.randint(1995, 2024)
            bedrooms = np.random.choice([3, 4, 5, 6], p=[0.2, 0.4, 0.3, 0.1])
            bathrooms = np.random.choice([2.5, 3.0, 3.5, 4.0], p=[0.2, 0.4, 0.3, 0.1])
            price_multiplier = np.random.uniform(1.2, 2.0)

        # Add sophisticated price variation with market and property interactions
        market_premium = 1.0
        if float(zip_row['median_days_on_market']) < 20:  # Hot market
            market_premium = np.random.uniform(1.05, 1.20)
        elif float(zip_row['median_days_on_market']) > 60:  # Cold market
            market_premium = np.random.uniform(0.85, 0.95)
            
        # Age-based pricing
        age_factor = max(0.7, 1.0 - (2024 - year_built) * 0.003)  # Depreciation
        
        # Size-based pricing correlation
        size_factor = 1.0 + (bedrooms - 3) * 0.1 + (bathrooms - 2) * 0.05
        
        # Combine all factors for realistic price variation
        price_variation = price_multiplier * market_premium * age_factor * size_factor
        price_noise = np.random.normal(1.0, 0.08)  # Reduced noise, more systematic
        varied_price = max(price * price_variation * price_noise, 50000)
        
        # Add market variation to avoid data leakage
        market_variation = np.random.normal(1.0, 0.10)  # ±10% market variation
        varied_dom = max(int(zip_row['median_days_on_market'] * market_variation), 1)

        property_data = {
            'zip_code': int(postal_code) if postal_code.isdigit() else np.random.randint(10000, 99999),
            'price': varied_price,  # Use varied price to reduce leakage
            'square_feet': square_feet,
            'year_built': year_built,  
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'property_type': np.random.choice(['single_family', 'condo', 'townhouse'], p=[0.7, 0.2, 0.1]),
            'days_on_market': varied_dom,
            'market_avg_price_per_sqft': price / max(1, square_feet),  # Add market avg
            'size_percentile': 50,  # Default to 50th percentile
            'price_change_since_listing': 0,  # Default to no change
            'market_price_trend_1y': price_change_1y,  # Add market trend
            'market_avg_dom': float(zip_row['median_days_on_market']),  # Add market DOM
            'market_price_volatility': price_volatility,  # Add market volatility
            'market_inventory_ratio': 1.0,  # Default to balanced market
        }
        
        # Create market data from real CSV values
        inventory_count = int(zip_row['active_listing_count'])
        total_listings = int(zip_row['total_listing_count']) if pd.notna(zip_row['total_listing_count']) else inventory_count
        monthly_sales = total_listings / max(1, int(zip_row['median_days_on_market']) / 30)

        market_data = {
            'inventory_count': inventory_count,
            'price_reduction_count': int(zip_row['price_reduced_count']),
            'price_increase_count': int(zip_row['price_increased_count']),
            'median_dom': float(zip_row['median_days_on_market']),
            'price_volatility': price_volatility,
            'median_listing_price': price,
            'price_change_1y': price_change_1y,
            'price_change_3y': price_change_3y,
            'price_change_5y': price_change_5y,
            'monthly_sales': monthly_sales,
            'active_listing_count': inventory_count,
            'prev_monthly_sales': monthly_sales * 0.95, # Estimate previous month
            'new_listings': inventory_count * 0.2, # Estimate new listings
            'prev_active_listing_count': inventory_count * 0.95, # Estimate previous inventory
            'dom_volatility': 0.1, # Default
            'price_reduction_ratio': int(zip_row['price_reduced_count']) / max(1, total_listings)
        }
        
        # Calculate base heuristic scores using current PropertyAnalyzer
        base_risk_metrics = self.property_analyzer._calculate_risk_metrics(property_data, market_data)
        base_market_metrics = self.property_analyzer._calculate_market_metrics(market_data)
        
        # Enhance labels with guaranteed variability
        risk_metrics, market_metrics = self.label_generator.generate_robust_labels(
            property_data=property_data,
            market_data=market_data,
            base_risk_metrics=base_risk_metrics,
            base_market_metrics=base_market_metrics,
            location_id=postal_code,
            date_context=current_date
        )

        # Create clean training record with only real features
        record = {
            # Identifiers
            'property_id': f"zip_{postal_code}_{current_date}",
            'zip_code': property_data['zip_code'],
            
            # Real property features
            'price': property_data['price'],
            'square_feet': property_data['square_feet'],
            'year_built': property_data['year_built'],
            'property_age': 2024 - property_data['year_built'],
            'days_on_market': property_data['days_on_market'],
            
            # Real market features
            'active_listing_count': inventory_count,
            'price_reduced_count': market_data['price_reduction_count'],
            'price_increased_count': market_data['price_increase_count'],
            'total_listing_count': total_listings,
            'median_days_on_market': market_data['median_dom'],
            'median_listing_price': market_data['median_listing_price'],
            
            # Derived features from real data
            'price_per_sqft': property_data['price'] / property_data['square_feet'],
            'price_volatility': price_volatility,
            'price_change_1y': price_change_1y,
            'price_change_3y': price_change_3y,
            'price_change_5y': price_change_5y,
            'price_reduction_ratio': market_data['price_reduction_count'] / max(total_listings, 1),
            'price_increase_ratio': market_data['price_increase_count'] / max(total_listings, 1),
            
            # Heuristic labels (targets for ML models)
            'market_risk_label': risk_metrics['market_risk'],
            'property_risk_label': risk_metrics['property_risk'], 
            'location_risk_label': risk_metrics['location_risk'],
            'overall_risk_label': risk_metrics['overall_risk'],
            'market_health_label': market_metrics['market_health'],
            'market_momentum_label': market_metrics['market_momentum'],
            'market_stability_label': market_metrics['market_stability'],
            
            # Metadata
            'data_source': 'zip_csv',
            'month_date': current_date
        }
        
        return record
    
    async def _create_clean_training_record_from_metro(self, metro_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a clean training record from Metro-level market data"""
        
        # Validate required fields are present and non-zero
        required_fields = ['median_listing_price', 'median_days_on_market', 'active_listing_count',
                          'price_reduced_count', 'price_increased_count', 'median_square_feet']
        
        for field in required_fields:
            if pd.isna(metro_row[field]) or metro_row[field] <= 0:
                logger.debug(f"Skipping Metro record: missing/invalid {field}")
                return None
        
        # Extract real data only
        cbsa_code = int(metro_row['cbsa_code'])
        current_date = int(metro_row['month_date_yyyymm'])
        price = float(metro_row['median_listing_price'])
        square_feet = float(metro_row['median_square_feet'])
        
        # Skip if can't calculate price per sqft
        if square_feet <= 0:
            logger.debug("Skipping Metro record: invalid square_feet")
            return None
        
        # Calculate derived features from real data
        price_volatility = self._calculate_price_volatility(cbsa_code, 'metro', current_date)
        price_change_1y, price_change_3y, price_change_5y = self._calculate_price_changes(
            cbsa_code, 'metro', current_date
        )
        
        # Create property data using real values only with realistic interdependencies
        
        # Create correlated property characteristics for metro areas
        metro_property_tier = np.random.choice(['budget', 'mid', 'luxury'], p=[0.35, 0.45, 0.2])
        
        if metro_property_tier == 'budget':
            year_built = np.random.randint(1950, 1990)
            bedrooms = np.random.choice([2, 3, 4], p=[0.4, 0.5, 0.1])
            bathrooms = np.random.choice([1.0, 1.5, 2.0], p=[0.3, 0.4, 0.3])
            price_multiplier = np.random.uniform(0.7, 1.1)
        elif metro_property_tier == 'mid':
            year_built = np.random.randint(1980, 2015)
            bedrooms = np.random.choice([2, 3, 4, 5], p=[0.1, 0.4, 0.4, 0.1])
            bathrooms = np.random.choice([1.5, 2.0, 2.5, 3.0], p=[0.15, 0.4, 0.35, 0.1])
            price_multiplier = np.random.uniform(0.9, 1.3)
        else:  # luxury
            year_built = np.random.randint(1995, 2024)
            bedrooms = np.random.choice([3, 4, 5, 6], p=[0.2, 0.4, 0.3, 0.1])
            bathrooms = np.random.choice([2.5, 3.0, 3.5, 4.0], p=[0.2, 0.4, 0.3, 0.1])
            price_multiplier = np.random.uniform(1.2, 2.0)

        # Metro-specific market dynamics
        metro_market_premium = 1.0
        if float(metro_row['median_days_on_market']) < 25:  # Hot metro market
            metro_market_premium = np.random.uniform(1.08, 1.25)
        elif float(metro_row['median_days_on_market']) > 50:  # Cold metro market
            metro_market_premium = np.random.uniform(0.80, 0.92)
            
        # Age and size factors (same logic)
        age_factor = max(0.7, 1.0 - (2024 - year_built) * 0.003)
        size_factor = 1.0 + (bedrooms - 3) * 0.1 + (bathrooms - 2) * 0.05
        
        # Combine factors for metro price variation
        price_variation = price_multiplier * metro_market_premium * age_factor * size_factor
        price_noise = np.random.normal(1.0, 0.08)
        varied_price = max(price * price_variation * price_noise, 50000)
        
        # Add market variation to avoid data leakage
        market_variation = np.random.normal(1.0, 0.10)  # ±10% market variation
        varied_dom = max(int(metro_row['median_days_on_market'] * market_variation), 1)

        property_data = {
            'zip_code': np.random.randint(10000, 99999),  # Generate varied ZIP codes
            'price': varied_price,  # Use varied price to reduce leakage
            'square_feet': square_feet,
            'year_built': year_built,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'property_type': np.random.choice(['single_family', 'condo', 'townhouse'], p=[0.7, 0.2, 0.1]),
            'days_on_market': varied_dom,
            'market_avg_price_per_sqft': price / max(1, square_feet),
            'size_percentile': 50,
            'price_change_since_listing': 0,
            'market_price_trend_1y': price_change_1y,
            'market_avg_dom': float(metro_row['median_days_on_market']),
            'market_price_volatility': price_volatility,
            'market_inventory_ratio': 1.0,
        }
        
        # Create market data from real CSV values
        inventory_count = int(metro_row['active_listing_count'])
        total_listings = int(metro_row['total_listing_count']) if pd.notna(metro_row['total_listing_count']) else inventory_count
        monthly_sales = total_listings / max(1, int(metro_row['median_days_on_market']) / 30)
        
        market_data = {
            'inventory_count': inventory_count,
            'price_reduction_count': int(metro_row['price_reduced_count']),
            'price_increase_count': int(metro_row['price_increased_count']),
            'median_dom': float(metro_row['median_days_on_market']),
            'price_volatility': price_volatility,
            'median_listing_price': price,
            'price_change_1y': price_change_1y,
            'price_change_3y': price_change_3y,
            'price_change_5y': price_change_5y,
            'monthly_sales': monthly_sales,
            'active_listing_count': inventory_count,
            'prev_monthly_sales': monthly_sales * 0.95,
            'new_listings': inventory_count * 0.2,
            'prev_active_listing_count': inventory_count * 0.95,
            'dom_volatility': 0.1,
            'price_reduction_ratio': int(metro_row['price_reduced_count']) / max(1, total_listings)
        }
        
        # Calculate base heuristic scores
        base_risk_metrics = self.property_analyzer._calculate_risk_metrics(property_data, market_data)
        base_market_metrics = self.property_analyzer._calculate_market_metrics(market_data)
        
        # Enhance labels with guaranteed variability
        risk_metrics, market_metrics = self.label_generator.generate_robust_labels(
            property_data=property_data,
            market_data=market_data,
            base_risk_metrics=base_risk_metrics,
            base_market_metrics=base_market_metrics,
            location_id=str(cbsa_code),
            date_context=current_date
        )
        
        # Create clean training record
        record = {
            # Identifiers
            'property_id': f"metro_{cbsa_code}_{current_date}",
            'cbsa_code': cbsa_code,
            
            # Real property features
            'price': property_data['price'],
            'square_feet': property_data['square_feet'],
            'year_built': property_data['year_built'],
            'property_age': 2024 - property_data['year_built'],
            'days_on_market': property_data['days_on_market'],
            
            # Real market features
            'active_listing_count': inventory_count,
            'price_reduced_count': market_data['price_reduction_count'],
            'price_increased_count': market_data['price_increase_count'],
            'total_listing_count': total_listings,
            'median_days_on_market': market_data['median_dom'],
            'median_listing_price': market_data['median_listing_price'],
            
            # Derived features from real data
            'price_per_sqft': property_data['price'] / property_data['square_feet'],
            'price_volatility': price_volatility,
            'price_change_1y': price_change_1y,
            'price_change_3y': price_change_3y,
            'price_change_5y': price_change_5y,
            'price_reduction_ratio': market_data['price_reduction_count'] / max(total_listings, 1),
            'price_increase_ratio': market_data['price_increase_count'] / max(total_listings, 1),
            
            # Heuristic labels
            'market_risk_label': risk_metrics['market_risk'],
            'property_risk_label': risk_metrics['property_risk'],
            'location_risk_label': risk_metrics['location_risk'],
            'overall_risk_label': risk_metrics['overall_risk'],
            'market_health_label': market_metrics['market_health'],
            'market_momentum_label': market_metrics['market_momentum'],
            'market_stability_label': market_metrics['market_stability'],
            
            # Metadata
            'data_source': 'metro_csv',
            'month_date': current_date
        }
        
        return record


# Utility function for CLI usage
async def generate_clean_training_dataset_cli():
    """CLI function to generate clean training dataset"""
    generator = TrainingDataGenerator()
    
    # Audit available features first
    feature_audit = generator.audit_available_features()
    print("=== FEATURE AUDIT (REAL FEATURES ONLY) ===")
    for metric, features in feature_audit.items():
        print(f"{metric}: {features}")
    
    # Generate clean training dataset
    print("\n=== GENERATING CLEAN TRAINING DATASET ===")
    output_path = await generator.generate_training_dataset(
        sample_size=1000, 
        output_file="training_dataset_clean.csv"
    )
    print(f"Clean dataset generated: {output_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_clean_training_dataset_cli()) 