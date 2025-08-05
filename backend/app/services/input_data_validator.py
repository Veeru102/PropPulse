"""
Input data validator that ensures property_data is never empty at inference.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class InputDataValidator:
    """Validates and enriches input data to prevent empty property_data at inference."""
    
    def __init__(self, fallback_stats_path: str = None):
        self.fallback_stats_path = fallback_stats_path or "data/fallback_property_stats.json"
        self.fallback_stats = self._load_fallback_stats()
    
    def validate_and_enrich_property_data(self, 
                                        property_data: Dict[str, Any], 
                                        market_data: Dict[str, Any],
                                        location_context: str = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate and enrich property data to ensure it's never empty.
        
        Args:
            property_data: Raw property data (may be empty or incomplete)
            market_data: Market data for imputation
            location_context: Location identifier for context-aware imputation
            
        Returns:
            (enriched_property_data, imputation_log)
        """
        imputation_log = []
        
        # Start with a copy to avoid modifying original
        enriched_data = property_data.copy() if property_data else {}
        
        # Check if property_data is effectively empty
        if self._is_property_data_empty(property_data):
            logger.warning(f"Property data is empty or invalid for location {location_context}. Applying comprehensive imputation.")
            enriched_data = self._create_property_from_market_and_fallbacks(market_data, location_context)
            imputation_log.append("Created complete property data from market data and fallbacks")
        else:
            # Validate and impute missing critical fields
            enriched_data, field_imputations = self._impute_missing_critical_fields(enriched_data, market_data, location_context)
            imputation_log.extend(field_imputations)
        
        # Validate final data quality
        validation_issues = self._validate_final_property_data(enriched_data)
        if validation_issues:
            logger.error(f"Property data validation failed even after imputation: {validation_issues}")
            # Apply emergency fallbacks
            enriched_data, emergency_fixes = self._apply_emergency_fallbacks(enriched_data, validation_issues)
            imputation_log.extend(emergency_fixes)
        
        # Log the imputation summary
        if imputation_log:
            logger.info(f"Property data imputation summary for {location_context}: {len(imputation_log)} fixes applied")
            for log_entry in imputation_log:
                logger.debug(f"  - {log_entry}")
        
        return enriched_data, imputation_log
    
    def validate_market_data(self, market_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate and enrich market data to ensure completeness.
        
        Returns:
            (validated_market_data, validation_log)
        """
        validation_log = []
        validated_data = market_data.copy() if market_data else {}
        
        # Required market fields with fallback values
        required_fields = {
            'active_listing_count': 100.0,
            'inventory_count': 100.0,
            'price_reduction_count': 10.0,
            'price_increase_count': 5.0,
            'median_listing_price': 250000.0,
            'median_dom': 30.0,
            'price_volatility': 0.1,
            'price_change_1y': 0.0,
            'price_change_3y': 0.0,
            'price_change_5y': 0.0,
            'monthly_sales': 50.0,
            'total_listing_count': 100.0
        }
        
        for field, fallback_value in required_fields.items():
            if field not in validated_data or validated_data[field] in [None, '', np.nan]:
                validated_data[field] = fallback_value
                validation_log.append(f"Added missing market field '{field}' with fallback value {fallback_value}")
            elif isinstance(validated_data[field], (int, float)) and validated_data[field] <= 0:
                # Handle zero/negative values for count fields
                if 'count' in field or field in ['monthly_sales', 'active_listing_count', 'inventory_count']:
                    validated_data[field] = fallback_value
                    validation_log.append(f"Replaced zero/negative '{field}' with fallback value {fallback_value}")
        
        # Ensure consistency between related fields
        if validated_data['inventory_count'] != validated_data['active_listing_count']:
            # Use the larger value for consistency
            max_count = max(validated_data['inventory_count'], validated_data['active_listing_count'])
            validated_data['inventory_count'] = max_count
            validated_data['active_listing_count'] = max_count
            validation_log.append(f"Synchronized inventory_count and active_listing_count to {max_count}")
        
        # Calculate total_listing_count if missing or inconsistent
        expected_total = max(
            validated_data['active_listing_count'],
            validated_data['price_reduction_count'] + validated_data['price_increase_count']
        )
        
        if validated_data['total_listing_count'] < expected_total:
            validated_data['total_listing_count'] = expected_total
            validation_log.append(f"Adjusted total_listing_count to {expected_total}")
        
        # Calculate ratios
        total_count = max(validated_data['total_listing_count'], 1.0)
        validated_data['price_reduction_ratio'] = validated_data['price_reduction_count'] / total_count
        validated_data['price_increase_ratio'] = validated_data['price_increase_count'] / total_count
        
        return validated_data, validation_log
    
    def _is_property_data_empty(self, property_data: Dict[str, Any]) -> bool:
        """Check if property data is effectively empty."""
        if not property_data:
            return True
        
        # Check if all values are None, empty, or zero
        non_empty_values = 0
        for key, value in property_data.items():
            if key in ['property_id', 'zip_code', 'cbsa_code']:  # Skip ID fields
                continue
            if value not in [None, '', 0, 0.0, np.nan]:
                non_empty_values += 1
        
        return non_empty_values == 0
    
    def _create_property_from_market_and_fallbacks(self, 
                                                  market_data: Dict[str, Any], 
                                                  location_context: str = None) -> Dict[str, Any]:
        """Create complete property data from market data and fallback statistics."""
        
        # Start with market-derived values
        property_data = {}
        
        # Price from market median with some variation
        market_price = float(market_data.get('median_listing_price', 250000))
        if location_context:
            # Add location-based variation (±20%)
            location_hash = hash(str(location_context)) % 100
            price_variation = (location_hash - 50) / 250  # -0.2 to +0.2
            property_data['price'] = market_price * (1 + price_variation)
        else:
            property_data['price'] = market_price
        
        # Square feet based on price and regional averages
        price_per_sqft = self.fallback_stats.get('price_per_sqft_median', 175.0)
        property_data['square_feet'] = property_data['price'] / price_per_sqft
        
        # Days on market from market data
        property_data['days_on_market'] = float(market_data.get('median_dom', 30))
        
        # Property characteristics from fallback stats
        property_data['year_built'] = self.fallback_stats.get('year_built_median', 2010)
        property_data['bedrooms'] = self.fallback_stats.get('bedrooms_median', 3)
        property_data['bathrooms'] = self.fallback_stats.get('bathrooms_median', 2.0)
        property_data['property_type'] = self.fallback_stats.get('property_type_mode', 'single_family')
        
        # Derived fields
        current_year = 2024
        property_data['property_age'] = current_year - property_data['year_built']
        property_data['price_per_sqft'] = property_data['price'] / property_data['square_feet']
        
        # Market context fields
        property_data['market_avg_price_per_sqft'] = price_per_sqft
        property_data['market_avg_dom'] = property_data['days_on_market']
        property_data['market_price_volatility'] = float(market_data.get('price_volatility', 0.1))
        property_data['market_inventory_ratio'] = 1.0  # Neutral ratio
        
        # Pricing context
        property_data['price_change_since_listing'] = 0  # Assume no change
        property_data['market_price_trend_1y'] = float(market_data.get('price_change_1y', 0)) / 100
        property_data['size_percentile'] = 50  # Assume median size
        
        # Add property ID if missing
        if location_context:
            property_data['property_id'] = f"imputed_{location_context}_{int(property_data['price'])}"
        else:
            property_data['property_id'] = f"imputed_unknown_{int(property_data['price'])}"
        
        logger.info(f"Created complete property data: price=${property_data['price']:.0f}, "
                   f"sqft={property_data['square_feet']:.0f}, dom={property_data['days_on_market']:.0f}")
        
        return property_data
    
    def _impute_missing_critical_fields(self, 
                                      property_data: Dict[str, Any], 
                                      market_data: Dict[str, Any],
                                      location_context: str = None) -> Tuple[Dict[str, Any], List[str]]:
        """Impute missing critical fields in existing property data."""
        imputation_log = []
        enriched_data = property_data.copy()
        
        # Critical fields that must be present
        critical_fields = {
            'price': {
                'fallback_source': 'market',
                'market_key': 'median_listing_price',
                'default_value': 250000.0
            },
            'square_feet': {
                'fallback_source': 'derived',
                'calculation': lambda data: data.get('price', 250000) / 175.0,  # Assume $175/sqft
                'default_value': 1500.0
            },
            'days_on_market': {
                'fallback_source': 'market',
                'market_key': 'median_dom',
                'default_value': 30.0
            },
            'year_built': {
                'fallback_source': 'fallback_stats',
                'stats_key': 'year_built_median',
                'default_value': 2010
            },
            'bedrooms': {
                'fallback_source': 'fallback_stats',
                'stats_key': 'bedrooms_median',
                'default_value': 3
            },
            'bathrooms': {
                'fallback_source': 'fallback_stats',
                'stats_key': 'bathrooms_median',
                'default_value': 2.0
            }
        }
        
        for field, config in critical_fields.items():
            if field not in enriched_data or enriched_data[field] in [None, '', 0, 0.0, np.nan]:
                
                if config['fallback_source'] == 'market':
                    market_key = config['market_key']
                    value = market_data.get(market_key, config['default_value'])
                    imputation_log.append(f"Imputed '{field}' from market data '{market_key}': {value}")
                
                elif config['fallback_source'] == 'derived':
                    try:
                        value = config['calculation'](enriched_data)
                        imputation_log.append(f"Calculated '{field}' from other fields: {value}")
                    except:
                        value = config['default_value']
                        imputation_log.append(f"Used default for '{field}' (calculation failed): {value}")
                
                elif config['fallback_source'] == 'fallback_stats':
                    stats_key = config['stats_key']
                    value = self.fallback_stats.get(stats_key, config['default_value'])
                    imputation_log.append(f"Imputed '{field}' from fallback stats: {value}")
                
                else:
                    value = config['default_value']
                    imputation_log.append(f"Used default value for '{field}': {value}")
                
                enriched_data[field] = float(value) if isinstance(value, (int, float)) else value
        
        # Calculate derived fields
        if 'price_per_sqft' not in enriched_data or enriched_data['price_per_sqft'] <= 0:
            if enriched_data.get('square_feet', 0) > 0:
                enriched_data['price_per_sqft'] = enriched_data['price'] / enriched_data['square_feet']
                imputation_log.append(f"Calculated price_per_sqft: {enriched_data['price_per_sqft']:.2f}")
        
        if 'property_age' not in enriched_data:
            current_year = 2024
            enriched_data['property_age'] = current_year - enriched_data.get('year_built', 2010)
            imputation_log.append(f"Calculated property_age: {enriched_data['property_age']}")
        
        return enriched_data, imputation_log
    
    def _validate_final_property_data(self, property_data: Dict[str, Any]) -> List[str]:
        """Validate final property data and return list of issues."""
        issues = []
        
        # Check critical numeric fields
        critical_numeric = {
            'price': (1000, 10000000),  # $1K to $10M
            'square_feet': (100, 20000),  # 100 to 20K sqft
            'days_on_market': (0, 1000),  # 0 to 1000 days
        }
        
        for field, (min_val, max_val) in critical_numeric.items():
            value = property_data.get(field)
            if value is None:
                issues.append(f"Missing critical field: {field}")
            elif not isinstance(value, (int, float)):
                issues.append(f"Non-numeric value for {field}: {value}")
            elif value < min_val or value > max_val:
                issues.append(f"Value out of reasonable range for {field}: {value} (expected {min_val}-{max_val})")
        
        # Check for reasonable relationships
        if property_data.get('price') and property_data.get('square_feet'):
            price_per_sqft = property_data['price'] / property_data['square_feet']
            if price_per_sqft < 10 or price_per_sqft > 2000:
                issues.append(f"Unreasonable price per sqft: ${price_per_sqft:.2f}")
        
        return issues
    
    def _apply_emergency_fallbacks(self, 
                                 property_data: Dict[str, Any], 
                                 issues: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply emergency fallbacks for validation failures."""
        emergency_fixes = []
        fixed_data = property_data.copy()
        
        # Emergency fallback values
        emergency_values = {
            'price': 250000.0,
            'square_feet': 1500.0,
            'days_on_market': 30.0,
            'year_built': 2010,
            'bedrooms': 3,
            'bathrooms': 2.0,
            'property_type': 'single_family'
        }
        
        for issue in issues:
            if "Missing critical field:" in issue:
                field = issue.split(": ")[1]
                if field in emergency_values:
                    fixed_data[field] = emergency_values[field]
                    emergency_fixes.append(f"Applied emergency fallback for {field}: {emergency_values[field]}")
            
            elif "Value out of reasonable range" in issue:
                field = issue.split(" for ")[1].split(":")[0]
                if field in emergency_values:
                    fixed_data[field] = emergency_values[field]
                    emergency_fixes.append(f"Applied emergency fallback for out-of-range {field}: {emergency_values[field]}")
        
        # Recalculate derived fields
        if 'price' in fixed_data and 'square_feet' in fixed_data:
            fixed_data['price_per_sqft'] = fixed_data['price'] / fixed_data['square_feet']
            emergency_fixes.append(f"Recalculated price_per_sqft: {fixed_data['price_per_sqft']:.2f}")
        
        return fixed_data, emergency_fixes
    
    def _load_fallback_stats(self) -> Dict[str, Any]:
        """Load fallback statistics from file or use defaults."""
        stats_path = Path(self.fallback_stats_path)
        
        # Default fallback statistics
        defaults = {
            'price_per_sqft_median': 175.0,
            'year_built_median': 2010,
            'bedrooms_median': 3,
            'bathrooms_median': 2.0,
            'property_type_mode': 'single_family',
            'days_on_market_median': 30.0,
            'square_feet_median': 1500.0
        }
        
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    loaded_stats = json.load(f)
                    # Merge with defaults
                    defaults.update(loaded_stats)
                    logger.info(f"Loaded fallback statistics from {stats_path}")
            except Exception as e:
                logger.warning(f"Failed to load fallback statistics: {e}, using defaults")
        else:
            logger.info("No fallback statistics file found, using defaults")
        
        return defaults
    
    def save_fallback_stats(self, training_data: pd.DataFrame):
        """Generate and save fallback statistics from training data."""
        stats = {}
        
        if 'price_per_sqft' in training_data.columns:
            stats['price_per_sqft_median'] = float(training_data['price_per_sqft'].median())
        
        if 'year_built' in training_data.columns:
            stats['year_built_median'] = int(training_data['year_built'].median())
        
        if 'bedrooms' in training_data.columns:
            stats['bedrooms_median'] = int(training_data['bedrooms'].median())
        
        if 'bathrooms' in training_data.columns:
            stats['bathrooms_median'] = float(training_data['bathrooms'].median())
        
        if 'property_type' in training_data.columns:
            stats['property_type_mode'] = training_data['property_type'].mode().iloc[0]
        
        if 'days_on_market' in training_data.columns:
            stats['days_on_market_median'] = float(training_data['days_on_market'].median())
        
        if 'square_feet' in training_data.columns:
            stats['square_feet_median'] = float(training_data['square_feet'].median())
        
        # Save to file
        stats_path = Path(self.fallback_stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved fallback statistics to {stats_path}")
        self.fallback_stats = stats