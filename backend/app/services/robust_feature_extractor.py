"""
Robust feature extraction that handles missing and incomplete data gracefully.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RobustFeatureExtractor:
    """Extracts features robustly, handling missing data and edge cases."""
    
    def __init__(self):
        self.default_property_values = {
            'price': 250000.0,  # Reasonable default price
            'square_feet': 1500.0,  # Reasonable default size
            'year_built': 2010,  # Reasonable default age
            'bedrooms': 3,
            'bathrooms': 2.0,
            'days_on_market': 30.0,
            'property_type': 'single_family'
        }
        
        self.default_market_values = {
            'inventory_count': 100.0,
            'price_reduction_count': 10.0,
            'price_increase_count': 5.0,
            'median_dom': 30.0,
            'price_volatility': 0.1,
            'median_listing_price': 250000.0,
            'price_change_1y': 0.0,
            'price_change_3y': 0.0,
            'price_change_5y': 0.0,
            'monthly_sales': 50.0,
            'active_listing_count': 100.0
        }
    
    def extract_features_safely(self, property_data: Dict[str, Any], 
                               market_data: Dict[str, Any],
                               property_id: str = "unknown") -> Optional[Dict[str, float]]:
        """
        Extract ML features with robust handling of missing data.
        
        Args:
            property_data: Property-specific data (may be empty or incomplete)
            market_data: Market-level data (may be incomplete)
            property_id: Identifier for logging
            
        Returns:
            Dictionary of features or None if extraction fails completely
        """
        try:
            # Start with empty features dict
            features = {}
            
            # Handle empty or missing property_data
            if not property_data or all(v in [None, 0, 0.0, ''] for v in property_data.values()):
                logger.debug(f"Empty property_data for {property_id}, using market-based defaults")
                property_data = self._create_property_from_market(market_data)
            
            # Ensure market_data has required fields
            market_data = self._ensure_market_data_complete(market_data)
            
            # Extract basic property features
            features.update(self._extract_property_features(property_data, property_id))
            
            # Extract market features
            features.update(self._extract_market_features(market_data, property_id))
            
            # Extract derived features
            features.update(self._extract_derived_features(property_data, market_data, property_id))
            
            # Add missing data flags
            features.update(self._add_missing_data_flags(property_data, market_data))
            
            # Validate extracted features
            features = self._validate_and_clean_features(features, property_id)
            
            logger.debug(f"Successfully extracted {len(features)} features for {property_id}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {property_id}: {e}")
            return self._create_fallback_features(property_id)
    
    def _create_property_from_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create reasonable property data from market data when property_data is empty."""
        property_data = self.default_property_values.copy()
        
        # Use market data to improve defaults where possible
        if 'median_listing_price' in market_data and market_data['median_listing_price'] > 0:
            property_data['price'] = float(market_data['median_listing_price'])
        
        if 'median_dom' in market_data and market_data['median_dom'] > 0:
            property_data['days_on_market'] = float(market_data['median_dom'])
        elif 'median_days_on_market' in market_data and market_data['median_days_on_market'] > 0:
            property_data['days_on_market'] = float(market_data['median_days_on_market'])
        
        # Calculate reasonable square feet from price if available
        if property_data['price'] > 0:
            # Assume $150-200 per sqft as reasonable range
            property_data['square_feet'] = property_data['price'] / 175.0
        
        return property_data
    
    def _ensure_market_data_complete(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure market data has all required fields with reasonable defaults."""
        complete_market_data = self.default_market_values.copy()
        
        # Update with actual market data where available
        for key, value in market_data.items():
            if value is not None and value != '' and not (isinstance(value, (int, float)) and np.isnan(value)):
                complete_market_data[key] = float(value) if isinstance(value, (int, float)) else value
        
        # Ensure consistency in market data
        if complete_market_data['active_listing_count'] <= 0:
            complete_market_data['active_listing_count'] = complete_market_data['inventory_count']
        
        if complete_market_data['inventory_count'] <= 0:
            complete_market_data['inventory_count'] = complete_market_data['active_listing_count']
        
        # Calculate total_listing_count if missing
        if 'total_listing_count' not in complete_market_data or complete_market_data.get('total_listing_count', 0) <= 0:
            complete_market_data['total_listing_count'] = max(
                complete_market_data['active_listing_count'],
                complete_market_data['price_reduction_count'] + complete_market_data['price_increase_count']
            )
        
        return complete_market_data
    
    def _extract_property_features(self, property_data: Dict[str, Any], property_id: str) -> Dict[str, float]:
        """Extract property-specific features."""
        features = {}
        
        # Basic property features
        features['price'] = float(property_data.get('price', self.default_property_values['price']))
        features['square_feet'] = float(property_data.get('square_feet', self.default_property_values['square_feet']))
        features['days_on_market'] = float(property_data.get('days_on_market', self.default_property_values['days_on_market']))
        
        # Ensure positive values
        features['price'] = max(features['price'], 1000.0)  # Minimum price
        features['square_feet'] = max(features['square_feet'], 100.0)  # Minimum size
        features['days_on_market'] = max(features['days_on_market'], 1.0)  # Minimum DOM
        
        # Calculate property age
        year_built = property_data.get('year_built', self.default_property_values['year_built'])
        current_year = 2024
        features['property_age'] = max(0, current_year - int(year_built))
        
        # Calculate price per square foot
        features['price_per_sqft'] = features['price'] / features['square_feet']
        
        logger.debug(f"Extracted property features for {property_id}: price=${features['price']:.0f}, "
                    f"sqft={features['square_feet']:.0f}, dom={features['days_on_market']:.0f}")
        
        return features
    
    def _extract_market_features(self, market_data: Dict[str, Any], property_id: str) -> Dict[str, float]:
        """Extract market-level features."""
        features = {}
        
        # Direct market features
        features['active_listing_count'] = float(market_data['active_listing_count'])
        features['price_reduced_count'] = float(market_data['price_reduction_count'])
        features['price_increased_count'] = float(market_data['price_increase_count'])
        features['total_listing_count'] = float(market_data.get('total_listing_count', features['active_listing_count']))
        features['median_days_on_market'] = float(market_data.get('median_dom', market_data.get('median_days_on_market', 30.0)))
        features['median_listing_price'] = float(market_data.get('median_listing_price', 250000.0))
        
        # Market volatility and trends
        features['price_volatility'] = float(market_data.get('price_volatility', 0.1))
        features['price_change_1y'] = float(market_data.get('price_change_1y', 0.0))
        features['price_change_3y'] = float(market_data.get('price_change_3y', 0.0))
        features['price_change_5y'] = float(market_data.get('price_change_5y', 0.0))
        
        # Ensure positive counts
        features['active_listing_count'] = max(features['active_listing_count'], 1.0)
        features['total_listing_count'] = max(features['total_listing_count'], features['active_listing_count'])
        
        logger.debug(f"Extracted market features for {property_id}: active_listings={features['active_listing_count']:.0f}, "
                    f"median_price=${features['median_listing_price']:.0f}")
        
        return features
    
    def _extract_derived_features(self, property_data: Dict[str, Any], 
                                 market_data: Dict[str, Any], property_id: str) -> Dict[str, float]:
        """Extract derived/calculated features."""
        features = {}
        
        # Calculate ratios safely
        total_listings = max(float(market_data.get('total_listing_count', 1)), 1.0)
        features['price_reduction_ratio'] = float(market_data['price_reduction_count']) / total_listings
        features['price_increase_ratio'] = float(market_data['price_increase_count']) / total_listings
        
        # Market activity ratios
        active_count = max(float(market_data['active_listing_count']), 1.0)
        monthly_sales = float(market_data.get('monthly_sales', active_count * 0.5))
        features['sales_to_inventory_ratio'] = monthly_sales / active_count
        
        # Price relative to market
        property_price = float(property_data.get('price', market_data.get('median_listing_price', 250000)))
        market_price = float(market_data.get('median_listing_price', 250000))
        features['price_vs_market_ratio'] = property_price / max(market_price, 1000.0)
        
        # DOM relative to market
        property_dom = float(property_data.get('days_on_market', 30))
        market_dom = float(market_data.get('median_dom', market_data.get('median_days_on_market', 30)))
        features['dom_vs_market_ratio'] = property_dom / max(market_dom, 1.0)
        
        logger.debug(f"Extracted derived features for {property_id}: price_reduction_ratio={features['price_reduction_ratio']:.3f}")
        
        return features
    
    def _add_missing_data_flags(self, property_data: Dict[str, Any], 
                               market_data: Dict[str, Any]) -> Dict[str, float]:
        """Add flags indicating which data was missing/imputed."""
        flags = {}
        
        # Property data flags
        flags['property_data_missing'] = 1.0 if not property_data or len(property_data) == 0 else 0.0
        flags['price_imputed'] = 1.0 if property_data.get('price', 0) <= 0 else 0.0
        flags['sqft_imputed'] = 1.0 if property_data.get('square_feet', 0) <= 0 else 0.0
        flags['dom_imputed'] = 1.0 if property_data.get('days_on_market', 0) <= 0 else 0.0
        
        # Market data flags
        flags['market_price_missing'] = 1.0 if market_data.get('median_listing_price', 0) <= 0 else 0.0
        flags['market_activity_low'] = 1.0 if market_data.get('active_listing_count', 0) < 10 else 0.0
        
        return flags
    
    def _validate_and_clean_features(self, features: Dict[str, float], property_id: str) -> Dict[str, float]:
        """Validate and clean extracted features."""
        cleaned_features = {}
        
        for key, value in features.items():
            # Handle None, NaN, and infinite values
            if value is None or (isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value))):
                logger.warning(f"Invalid value for feature '{key}' in {property_id}, using default")
                cleaned_features[key] = 0.0
            else:
                # Convert to float and apply reasonable bounds
                try:
                    float_val = float(value)
                    
                    # Apply reasonable bounds based on feature type
                    if 'ratio' in key.lower():
                        float_val = np.clip(float_val, 0.0, 1.0)
                    elif 'price' in key.lower() and 'ratio' not in key.lower():
                        float_val = max(float_val, 0.0)
                    elif 'count' in key.lower():
                        float_val = max(float_val, 0.0)
                    elif 'age' in key.lower():
                        float_val = np.clip(float_val, 0.0, 200.0)  # Reasonable age bounds
                    
                    cleaned_features[key] = float_val
                    
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert feature '{key}' value '{value}' to float for {property_id}")
                    cleaned_features[key] = 0.0
        
        return cleaned_features
    
    def _create_fallback_features(self, property_id: str) -> Dict[str, float]:
        """Create minimal fallback features when extraction completely fails."""
        logger.warning(f"Using fallback features for {property_id}")
        
        return {
            'price': 250000.0,
            'square_feet': 1500.0,
            'days_on_market': 30.0,
            'active_listing_count': 100.0,
            'price_reduced_count': 10.0,
            'price_increased_count': 5.0,
            'total_listing_count': 100.0,
            'median_days_on_market': 30.0,
            'median_listing_price': 250000.0,
            'price_per_sqft': 167.0,
            'price_volatility': 0.1,
            'price_change_1y': 0.0,
            'price_change_3y': 0.0,
            'price_change_5y': 0.0,
            'price_reduction_ratio': 0.1,
            'price_increase_ratio': 0.05,
            'property_data_missing': 1.0,
            'has_missing_data': 1.0
        }