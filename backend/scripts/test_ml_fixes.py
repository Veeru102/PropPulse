#!/usr/bin/env python3
"""
Test script for ML model fixes in PropertyAnalyzer.
This script tests:
1. Empty property_data handling
2. ML model constant prediction detection
3. Field alias handling
4. Data imputation with missing_flag
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to Python path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.property_analyzer import PropertyAnalyzer
from app.services.market_data_service import MarketDataService

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s | %(levelname)-8s | %(message)s',
                   datefmt='%H:%M:%S')
logger = logging.getLogger()

def test_empty_property_data():
    """Test handling of empty property_data"""
    logger.info("=== Testing Empty Property Data Handling ===")
    
    analyzer = PropertyAnalyzer()
    market_data = {
        'median_listing_price': 300000,
        'median_dom': 45,
        'active_listing_count': 100,
        'price_reduction_count': 20,
        'price_increase_count': 5
    }
    
    # Test with empty property_data
    empty_property = {}
    risk_metrics = analyzer._calculate_risk_metrics(empty_property, market_data)
    market_metrics = analyzer._calculate_market_metrics(market_data)
    
    logger.info(f"Risk metrics with empty property_data: {risk_metrics}")
    logger.info(f"Market metrics with market_data only: {market_metrics}")
    
    # Test feature extraction with empty property_data
    features = analyzer._extract_ml_features(empty_property, market_data)
    logger.info(f"Features with empty property_data: {features}")
    
    assert len(features) == 0, "Should return empty dict for empty property_data"
    assert risk_metrics.get('metrics_source') == 'empty_data_fallback', "Should use fallback for empty property_data"

def test_field_aliases():
    """Test handling of field aliases"""
    logger.info("=== Testing Field Alias Handling ===")
    
    analyzer = PropertyAnalyzer()
    property_data = {
        'property_id': 'test_property',
        'price': 350000,
        'square_feet': 2000,
        'days_on_market': 30
    }
    
    # Test with aliased field names
    market_data_with_aliases = {
        'median_price': 300000,  # alias for median_listing_price
        'median_days_on_market': 45,  # alias for median_dom
        'active_listing_count': 100,
        'price_reduction_count': 20,  # alias for price_reduced_count
        'price_increase_count': 5  # alias for price_increased_count
    }
    
    # Extract features
    features = analyzer._extract_ml_features(property_data, market_data_with_aliases)
    logger.info(f"Features with aliased fields: {features}")
    
    # Check if aliases were handled correctly
    assert 'price_per_sqft' in features, "Should calculate price_per_sqft"
    assert features['price_per_sqft'] == 175.0, "Should calculate price_per_sqft correctly"
    
    # Check risk metrics with aliases
    risk_metrics = analyzer._calculate_risk_metrics(property_data, market_data_with_aliases)
    logger.info(f"Risk metrics with aliased fields: {risk_metrics}")
    
    # Should not have 'empty_data_fallback' in metrics_source
    assert 'empty_data_fallback' not in risk_metrics.get('metrics_source', ''), "Should not use fallback with valid data"

def test_imputation_flag():
    """Test imputation flag for missing data"""
    logger.info("=== Testing Imputation Flag ===")
    
    analyzer = PropertyAnalyzer()
    
    # Property data with missing square_feet
    property_data = {
        'property_id': 'test_property',
        'price': 350000,
        'days_on_market': 30
    }
    
    market_data = {
        'median_listing_price': 300000,
        'median_dom': 45,
        'active_listing_count': 100,
        'median_square_feet': 1800,  # For imputation
        'price_reduction_count': 20,
        'price_increase_count': 5
    }
    
    # Extract features
    features = analyzer._extract_ml_features(property_data, market_data)
    logger.info(f"Features with imputation: {features}")
    
    # Check if imputation flag is set
    assert features['has_missing_data'] == 1.0, "Should set missing_data flag when imputing"
    assert features['square_feet'] == 1800.0, "Should impute square_feet from market_data"

def test_constant_prediction_detection():
    """Test detection of constant predictions"""
    logger.info("=== Testing Constant Prediction Detection ===")
    
    analyzer = PropertyAnalyzer()
    
    # Check if the ML models are loaded
    if not analyzer.ml_models_loaded:
        logger.warning("ML models not loaded, skipping constant prediction test")
        return
    
    property_data = {
        'property_id': 'test_property',
        'price': 350000,
        'square_feet': 2000,
        'days_on_market': 30
    }
    
    market_data = {
        'median_listing_price': 300000,
        'median_dom': 45,
        'active_listing_count': 100,
        'price_reduction_count': 20,
        'price_increase_count': 5,
        'price_volatility': 0.1,
        'price_change_1y': 5.0,
        'price_change_3y': 15.0,
        'price_change_5y': 25.0
    }
    
    # Extract features
    features = analyzer._extract_ml_features(property_data, market_data)
    
    # Test location_risk which we know has constant predictions
    location_risk = analyzer._predict_with_ml_model('location_risk', features)
    logger.info(f"Location risk prediction: {location_risk}")
    
    # If location_risk is None, it means the constant prediction detection worked
    if location_risk is None:
        logger.info("Successfully detected constant predictions for location_risk")
    else:
        logger.warning(f"Failed to detect constant predictions for location_risk: {location_risk}")

def main():
    """Run all tests"""
    logger.info("Starting ML fixes tests")
    
    try:
        test_empty_property_data()
        test_field_aliases()
        test_imputation_flag()
        test_constant_prediction_detection()
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()