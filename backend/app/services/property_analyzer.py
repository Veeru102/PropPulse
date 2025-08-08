import pandas as pd
import numpy as np
import random
from typing import Dict, Any, List, Optional
import xgboost as xgb
from app.core.logging import loggers
from app.core.config import settings
from pathlib import Path
import json
from datetime import datetime
from app.services.market_data_service import MarketDataService

logger = loggers['ml']

# Import here to avoid circular imports
def _get_robust_feature_extractor():
    from app.services.robust_feature_extractor import RobustFeatureExtractor
    return RobustFeatureExtractor()

def _get_input_validator():
    from app.services.input_data_validator import InputDataValidator
    return InputDataValidator()

def _get_training_inference_auditor():
    from app.services.training_inference_auditor import TrainingInferenceAuditor
    return TrainingInferenceAuditor()

class PropertyAnalyzer:
    """Analyzes properties for investment potential using various metrics."""
    
    def __init__(
        self,
        data_collector: Optional[Any] = None,
        market_data_service: Optional[Any] = None
    ):
        """
        Initialize the PropertyAnalyzer.
        
        Args:
            data_collector: Service for collecting property data
            market_data_service: Service for collecting market data
        """
        # Use provided services if given; otherwise fall back to singletons so that
        # methods like get_comparable_properties don't fail with NoneType errors.
        if data_collector is None:
            from app.services.service_manager import ServiceManager
            self.data_collector = ServiceManager.get_data_collector()
        else:
            self.data_collector = data_collector

        if market_data_service is None:
            from app.services.service_manager import ServiceManager
            self.market_data_service = ServiceManager.get_market_data_service()
        else:
            self.market_data_service = market_data_service
        self.base_model_dir = Path(settings.MODEL_DIR) # Store base model directory
        self.value_model = None
        self.investment_model = None
        self.risk_metrics = {}
        
        # ML models for risk assessment
        self.ml_models = {}
        self.ml_models_loaded = False
        
        self._set_latest_model_dir() # New method to set latest model directory
        self._load_models()
        self._load_ml_models()
        logger.info("PropertyAnalyzer initialized with MarketDataService and ML models")
        
    def _set_latest_model_dir(self):
        """Set self.model_dir to the latest versioned subdirectory."""
        try:
            # Find latest model version subdirectory
            model_versions = sorted([d for d in self.base_model_dir.iterdir() if d.is_dir()])
            if model_versions:
                self.model_dir = model_versions[-1]
                logger.info(f"Set model directory to latest version: {self.model_dir.name}")
            else:
                self.model_dir = self.base_model_dir
                logger.warning(f"No versioned model directories found in {self.base_model_dir}, using base directory.")
        except Exception as e:
            logger.error(f"Error setting latest model directory: {e}. Falling back to base model directory.")
            self.model_dir = self.base_model_dir
    
    def _get_default_feature_names(self):
        """Get default feature names matching EXACT legacy model training features."""
        return [
            'price', 'square_feet', 'days_on_market', 'active_listing_count',
            'price_reduced_count', 'price_increased_count', 'total_listing_count',
            'median_days_on_market', 'median_listing_price', 'price_per_sqft',
            'price_volatility', 'price_change_1y', 'price_change_3y', 'price_change_5y',
            'price_reduction_ratio', 'price_increase_ratio'
        ]
    
    def _load_models(self):
        """Load trained models and risk metrics."""
        try:
            # Ensure model_dir is set before attempting to load
            if not hasattr(self, 'model_dir') or not self.model_dir.is_dir():
                logger.warning("Model directory not properly set or does not exist. Skipping XGBoost model loading.")
                return

            models_loaded = 0
            
            # Load value model directly from model_dir
            value_model_path = self.model_dir / "property_value_model.json"
            if value_model_path.exists():
                self.value_model = xgb.Booster()
                self.value_model.load_model(str(value_model_path))
                logger.info("Loaded XGBoost value model")
                models_loaded += 1
            else:
                logger.debug(f"Value model not found at {value_model_path}")
            
            # Load investment model directly from model_dir
            investment_model_path = self.model_dir / "investment_model.json"
            if investment_model_path.exists():
                self.investment_model = xgb.Booster()
                self.investment_model.load_model(str(investment_model_path))
                logger.info("Loaded XGBoost investment model")
                models_loaded += 1
            else:
                logger.debug(f"Investment model not found at {investment_model_path}")
            
            # Load risk metrics directly from model_dir
            risk_metrics_path = self.model_dir / "risk_metrics.json"
            if risk_metrics_path.exists():
                with open(risk_metrics_path, 'r') as f:
                    self.risk_metrics = json.load(f)
                logger.info("Loaded risk metrics")
                models_loaded += 1
            else:
                logger.debug(f"Risk metrics not found at {risk_metrics_path}")
            
            if models_loaded > 0:
                logger.info(f"Successfully loaded {models_loaded} XGBoost models/metrics from {self.model_dir.name}")
            else:
                logger.debug(f"No XGBoost models (property_value_model.json, investment_model.json) found in {self.model_dir.name} directory")
            
        except Exception as e:
            logger.warning(f"Error loading XGBoost models from {self.model_dir}: {str(e)}, continuing with ML models only")
    
    def _load_ml_models(self):
        """Load Random Forest models for risk assessment."""
        try:
            from app.ml_models.model_utils import ModelUtils
            
            ml_model_files = {
                'market_risk': 'market_risk_rf.joblib',
                'property_risk': 'property_risk_rf.joblib',
                'location_risk': 'location_risk_rf.joblib',
                'overall_risk': 'overall_risk_rf.joblib',
                'market_health': 'market_health_rf.joblib',
                'market_momentum': 'market_momentum_rf.joblib',
                'market_stability': 'market_stability_rf.joblib'
            }
            
            models_loaded = 0
            for metric_name, filename in ml_model_files.items():
                try:
                    model_path = self.model_dir / filename
                    if model_path.exists():
                        model_data = ModelUtils.load_model(str(model_path))
                        # Handle both old format (direct model) and new format (dict with metadata)
                        if isinstance(model_data, dict):
                            self.ml_models[metric_name] = {
                                'model': model_data['model'],
                                'feature_names': model_data.get('feature_names', []),
                                'target_name': model_data.get('target_name', metric_name)
                            }
                        else:
                            # Legacy format - model saved directly
                            logger.info(f"Loading legacy model format for {metric_name} (will be updated on next training)")
                            self.ml_models[metric_name] = {
                                'model': model_data,
                                'feature_names': self._get_default_feature_names(),
                                'target_name': metric_name
                            }
                        models_loaded += 1
                        logger.info(f"Loaded ML model for {metric_name}")
                    else:
                        # If not found in versioned directory, try base directory
                        base_model_path = self.base_model_dir / filename
                        if base_model_path.exists():
                            model_data = ModelUtils.load_model(str(base_model_path))
                            # Handle both old format (direct model) and new format (dict with metadata)
                            if isinstance(model_data, dict):
                                # Extract model and metadata
                                self.ml_models[metric_name] = {
                                    'model': model_data['model'],
                                    'feature_names': model_data.get('feature_names', []),
                                    'target_name': model_data.get('target_name', metric_name),
                                    'metadata': model_data.get('metadata', {})
                                }
                                
                                # Log model metadata for debugging
                                metadata = model_data.get('metadata', {})
                                if metadata:
                                    logger.info(f"Loaded {metric_name} model metadata: "
                                              f"type={metadata.get('model_type')}, "
                                              f"shape={metadata.get('training_shape')}, "
                                              f"timestamp={metadata.get('training_timestamp')}")
                            else:
                                # Legacy format - model saved directly
                                logger.info(f"Loading legacy model format for {metric_name} from base directory (will be updated on next training)")
                                self.ml_models[metric_name] = {
                                    'model': model_data,
                                    'feature_names': self._get_default_feature_names(),
                                    'target_name': metric_name
                                }
                            models_loaded += 1
                            logger.info(f"Loaded ML model for {metric_name} from base directory")
                        else:
                            logger.warning(f"ML model file not found in versioned ({model_path}) or base ({base_model_path}) directories: {filename}")
                except Exception as e:
                    logger.error(f"Error loading ML model for {metric_name}: {e}")
                    logger.debug(f"Model loading traceback:", exc_info=True)
            
            if models_loaded > 0:
                self.ml_models_loaded = True
                logger.info(f"Successfully loaded {models_loaded}/{len(ml_model_files)} ML models")
            else:
                logger.warning("No ML models could be loaded, will use heuristic fallback")
                
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self.ml_models_loaded = False
    
    def _extract_ml_features(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for ML model prediction using robust feature extraction."""
        try:
            # Note: Debug logging removed to reduce console flooding
            # logger.debug(f"_extract_ml_features input - property_data: {property_data}")
            # logger.debug(f"_extract_ml_features input - market_data: {market_data}")

            # Use robust feature extractor to handle empty/incomplete data
            extractor = _get_robust_feature_extractor()
            property_id = property_data.get('property_id', 'unknown')
            
            # Extract base features
            features = extractor.extract_features_safely(
                property_data=property_data,
                market_data=market_data,
                property_id=property_id
            )
            
            if features is None:
                return {}
                
            # Check if we have any models with metadata to guide feature extraction
            model_with_metadata = next(
                (model_info for model_info in self.ml_models.values() 
                 if model_info.get('metadata', {}).get('feature_dtypes')),
                None
            )
            
            if model_with_metadata:
                # Use metadata to ensure correct feature types
                metadata = model_with_metadata['metadata']
                feature_dtypes = metadata.get('feature_dtypes', {})
                
                for feature_name, dtype in feature_dtypes.items():
                    if feature_name in features:
                        try:
                            # Convert to the expected type
                            if 'float' in dtype:
                                features[feature_name] = float(features[feature_name])
                            elif 'int' in dtype:
                                features[feature_name] = int(float(features[feature_name]))
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not convert {feature_name} to {dtype}: {e}")
                            features[feature_name] = 0  # Safe fallback
            
            if features is None:
                logger.warning(f"Feature extraction failed completely for property_id: {property_id}")
                return {}
                
            # Convert to the expected format for ML models
            ml_features = {}
            
            # Map robust features to ML model expected format
            expected_features = [
                'price', 'square_feet', 'days_on_market', 'active_listing_count',
                'price_reduced_count', 'price_increased_count', 'total_listing_count',
                'median_days_on_market', 'median_listing_price', 'price_per_sqft',
                'price_volatility', 'price_change_1y', 'price_change_3y', 'price_change_5y',
                'price_reduction_ratio', 'price_increase_ratio'
            ]
            
            # Create proper feature
            feature_mapping = {
                # Direct mappings - NO zip_code, year_built, property_age (not in legacy models)
                'price': features.get('price', property_data.get('price', 250000.0)),
                'square_feet': features.get('square_feet', property_data.get('square_feet', 1800.0)),
                'days_on_market': features.get('days_on_market', property_data.get('days_on_market', 30.0)),
                
                # Market data mappings  
                'active_listing_count': market_data.get('active_listing_count', market_data.get('inventory_count', 100.0)),
                'price_reduced_count': market_data.get('price_reduced_count', market_data.get('price_reduction_count', 10.0)),
                'price_increased_count': market_data.get('price_increased_count', market_data.get('price_increase_count', 5.0)),
                'total_listing_count': market_data.get('total_listing_count', market_data.get('active_listing_count', 100.0)),
                'median_days_on_market': market_data.get('median_dom', market_data.get('median_days_on_market', 30.0)),
                'median_listing_price': market_data.get('median_listing_price', 250000.0),
                'price_volatility': market_data.get('price_volatility', 0.1),
                'price_change_1y': market_data.get('price_change_1y', 0.0),
                'price_change_3y': market_data.get('price_change_3y', 0.0),
                'price_change_5y': market_data.get('price_change_5y', 0.0),
                
                # Calculated features
                'price_per_sqft': features.get('price_per_sqft', 
                    property_data.get('price', 250000.0) / max(property_data.get('square_feet', 1800.0), 1.0)),
                'price_reduction_ratio': market_data.get('price_reduction_count', 10.0) / max(market_data.get('total_listing_count', 100.0), 1.0),
                'price_increase_ratio': market_data.get('price_increase_count', 5.0) / max(market_data.get('total_listing_count', 100.0), 1.0),
            }
            
            # Apply the mapped features
            for feature_name in expected_features:
                if feature_name in feature_mapping:
                    ml_features[feature_name] = float(feature_mapping[feature_name])
                else:
                    # Fallback defaults for any missing features
                    if 'price' in feature_name and 'ratio' not in feature_name:
                        ml_features[feature_name] = 250000.0
                    elif 'count' in feature_name:
                        ml_features[feature_name] = 100.0
                    elif 'ratio' in feature_name:
                        ml_features[feature_name] = 0.1
                    elif 'volatility' in feature_name:
                        ml_features[feature_name] = 0.1
                    else:
                        ml_features[feature_name] = 0.0
            
            # Add missing data flags
            ml_features['has_missing_data'] = features.get('property_data_missing', 0.0)
            
            # Note: Debug logging removed to reduce console flooding
            # logger.debug(f"_extract_ml_features output - ml_features: {ml_features}")
            return ml_features
            
        except Exception as e:
            logger.error(f"Error extracting ML features: {e}")
            return {}
    
    def _predict_with_ml_model(self, metric_name: str, features: Dict[str, float]) -> Optional[float]:
        """Make prediction using ML model with normalization."""
        try:
            if not self.ml_models_loaded or metric_name not in self.ml_models:
                return None
                
            # Skip prediction if features is empty (validation failed)
            if not features:
                logger.warning(f"Empty features provided for {metric_name} prediction. Skipping.")
                return None
            
            model_data = self.ml_models[metric_name]
            model = model_data['model']
            feature_names = model_data['feature_names']
            
            # Create feature vector as a pandas DataFrame with correct column names
            feature_dict = {}
            for feature_name in feature_names:
                # Use .get() with a default of 0.0 for missing features.
                # This handles cases where input data might not contain all features
                # the model was trained on, preventing KeyErrors and ensuring consistent input shape.
                feature_dict[feature_name] = features.get(feature_name, 0.0)
            
            # Validate feature completeness before prediction
            expected_features = set(feature_names)
            provided_features = set(feature_dict.keys())
            
            if expected_features != provided_features:
                missing = expected_features - provided_features
                extra = provided_features - expected_features
                if missing:
                    logger.warning(f"Missing features for {metric_name}: {missing}")
                if extra:
                    logger.debug(f"Extra features for {metric_name}: {extra}")
            
            # Create DataFrame with a single row
            feature_df = pd.DataFrame([feature_dict])
            
            # Validate data types and ranges
            for col in feature_df.columns:
                if feature_df[col].dtype == 'object':
                    logger.warning(f"Non-numeric feature {col} in {metric_name} prediction")
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0.0)
            
            # Check if model is likely to be returning constant predictions
            # We do this by making a small batch of predictions with slightly varied inputs
            test_batch_size = 5
            test_batch = []
            
            # Create test batch with small variations to detect constant predictions
            for i in range(test_batch_size):
                test_dict = feature_dict.copy()
                # Add small random variations to numeric features
                for k, v in test_dict.items():
                    if isinstance(v, (int, float)) and k != 'has_missing_data':
                        test_dict[k] = v * (1 + (random.random() - 0.5) * 0.1)  # ±5% variation
                test_batch.append(test_dict)
                
            test_df = pd.DataFrame(test_batch)
            batch_predictions = model.predict(test_df)
            
            # Check if predictions are suspiciously similar (potential constant model)
            pred_std = np.std(batch_predictions)
            if pred_std < 0.02:
                logger.warning(f"ML model for {metric_name} may be returning constant predictions "
                           f"(stdev: {pred_std:.6f}). Consider retraining with more varied data.")
                # Log the predictions for debugging
                logger.error(f"Test batch predictions: {batch_predictions}")
                
                # If predictions are essentially constant, fall back to heuristic
                if pred_std < 0.001:
                    logger.error(f"Bypassing ML model for {metric_name} due to constant predictions.")
                    return None
            
            # Make prediction using DataFrame with proper feature names
            try:
                raw_prediction = model.predict(feature_df)[0]
                
                # Validate prediction is not NaN or infinite
                if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                    logger.warning(f"Invalid prediction from {metric_name} model: {raw_prediction}")
                    return None
                
                # Ensure the result is a float in [0, 1]
                # The predictions should naturally be within a reasonable range for regression models.
                # We clip them to [0, 1] as the final output range for risk/health scores.
                normalized_prediction = float(max(0.0, min(1.0, raw_prediction)))
                
            except Exception as pred_error:
                logger.error(f"Prediction failed for {metric_name}: {pred_error}")
                return None
            
            # Log detailed prediction information
            logger.info(f"ML prediction for {metric_name}: {normalized_prediction:.4f} (raw: {raw_prediction:.4f}, stdev: {pred_std:.6f})")
            
            # Track prediction distributions for monitoring
            if not hasattr(self, 'prediction_stats'):
                self.prediction_stats = {}
            
            if metric_name not in self.prediction_stats:
                self.prediction_stats[metric_name] = {
                    'count': 0,
                    'sum': 0,
                    'sum_sq': 0,
                    'min': float('inf'),
                    'max': float('-inf')
                }
                
            stats = self.prediction_stats[metric_name]
            stats['count'] += 1
            stats['sum'] += normalized_prediction
            stats['sum_sq'] += normalized_prediction ** 2
            stats['min'] = min(stats['min'], normalized_prediction)
            stats['max'] = max(stats['max'], normalized_prediction)
            
            # Periodically log prediction distribution stats
            if stats['count'] % 100 == 0:
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                std = variance ** 0.5 if variance > 0 else 0
                logger.info(f"Prediction stats for {metric_name}: count={stats['count']}, "
                          f"mean={mean:.4f}, std={std:.4f}, min={stats['min']:.4f}, "
                          f"max={stats['max']:.4f}")
                
                # Alert if distribution is suspicious
                if std < 0.05 and stats['count'] > 10:
                    logger.warning(f"Low variance in {metric_name} predictions across {stats['count']} samples. "
                                 f"Model may need retraining.")
            
            return normalized_prediction
            
        except Exception as e:
            logger.error(f"Error making ML prediction for {metric_name}: {e}")
            return None
            
    def analyze_property(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a property for investment potential.
        
        Args:
            property_data: Dictionary containing property features
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Validate inputs
            self._validate_inputs(property_data, market_data)
            
            # Calculate base metrics
            base_metrics = self._calculate_base_metrics(property_data, market_data)
            
            # Calculate investment metrics
            investment_metrics = self._calculate_investment_metrics(property_data, market_data)
            
            # Calculate risk metrics (with ML integration)
            risk_metrics = self._calculate_risk_metrics(property_data, market_data)
            
            # Calculate market metrics (with ML integration)
            market_metrics = self._calculate_market_metrics(market_data)
            
            # Extract numeric metrics for investment score calculation
            risk_metrics_numeric = {k: v for k, v in risk_metrics.items() if k != 'metrics_source'}
            market_metrics_numeric = {k: v for k, v in market_metrics.items() if k != 'metrics_source'}
            
            # Calculate overall investment score
            investment_score = self._calculate_investment_score(
                base_metrics, investment_metrics, risk_metrics_numeric, market_metrics_numeric
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                property_data, market_data, base_metrics, investment_metrics, 
                risk_metrics_numeric, market_metrics_numeric, investment_score
            )
            
            # Return results with structured metrics
            return {
                'investment_score': investment_score,
                'base_metrics': base_metrics,
                'investment_metrics': investment_metrics,
                'risk_metrics': risk_metrics,  # Contains both metrics and sources
                'market_metrics': market_metrics,  # Contains both metrics and sources
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing property: {str(e)}")
            raise
            
    def _validate_inputs(self, property_data: Dict[str, Any], market_data: Dict[str, Any]):
        """Enhanced input validation with automatic fixing."""
        try:
            property_id = property_data.get('property_id', 'unknown')
            logger.debug(f"Validating inputs for property {property_id}")
            
            # Use enhanced input validator
            validator = _get_input_validator()
            
            # Validate and enrich property data
            enriched_property_data, property_log = validator.validate_and_enrich_property_data(
                property_data, market_data, property_id
            )
            
            # Validate market data
            validated_market_data, market_log = validator.validate_market_data(market_data)
            
            # Update the original dictionaries with validated data
            property_data.update(enriched_property_data)
            market_data.update(validated_market_data)
            
            # Log validation results
            if property_log:
                logger.info(f"Applied {len(property_log)} property data fixes for {property_id}")
            if market_log:
                logger.info(f"Applied {len(market_log)} market data fixes for {property_id}")
            
            # Perform drift detection if training stats available
            auditor = _get_training_inference_auditor()
            if auditor.training_stats:
                # Extract features for drift analysis
                extractor = _get_robust_feature_extractor()
                features = extractor.extract_features_safely(property_data, market_data, property_id)
                
                if features:
                    audit_report = auditor.audit_inference_data(property_data, market_data, features)
                    drift_issues = audit_report.get('validation_issues', [])
                    if drift_issues:
                        logger.warning(f"Drift detection issues for {property_id}: {drift_issues}")
            
            # Legacy validation for backward compatibility
            required_property_fields = [
                'price', 'square_feet'  # Reduced to essential fields
            ]
            
            # Define required market fields
            required_market_fields = [
                'median_list_price', 'median_dom', 'inventory_count', 
                'price_reduction_count', 'price_increase_count'
            ]
            
            # Set default values for missing market fields
            for field in required_market_fields:
                if field not in market_data or market_data[field] is None:
                    if field == 'median_list_price':
                        # Use a reasonable market median estimate instead of property price
                        # This prevents price_to_median from always being 1.0
                        zip_code = property_data.get('zip_code', 0)
                        if zip_code and 70000 <= zip_code <= 99999:  # US ZIP codes
                            # Estimate median based on ZIP code ranges (rough approximation)
                            if zip_code >= 90000:  # CA, high-cost areas
                                market_data[field] = 800000
                            elif zip_code >= 80000:  # CO, MT, etc.
                                market_data[field] = 500000
                            elif zip_code >= 70000:  # TX, OK, etc.
                                market_data[field] = 350000
                            else:
                                market_data[field] = 400000
                        else:
                            market_data[field] = 400000  # National median approximation
                    elif field == 'median_dom':
                        market_data[field] = 30  # Default to 30 days
                    elif field == 'inventory_count':
                        market_data[field] = 1000  # Default inventory
                    elif field in ['price_reduction_count', 'price_increase_count']:
                        market_data[field] = 100  # Default price changes
            
            # Validate numeric fields
            for field in required_property_fields:
                if not isinstance(property_data[field], (int, float)):
                    raise ValueError(f"Property field {field} must be numeric")
            
            for field in required_market_fields:
                if not isinstance(market_data[field], (int, float)):
                    market_data[field] = float(market_data[field]) if market_data[field] else 0.0
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise
            
    def _calculate_base_metrics(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate base property metrics."""
        try:
            metrics = {}
            
            # --------------------------------------------------------------
            # Price-related metrics – use .get() with sensible fallbacks so
            # this helper can be called with partially-filled *market_data*
            # dictionaries (e.g. straight from the market_trends endpoint).
            # --------------------------------------------------------------
            square_feet = property_data.get('square_feet', 0) or 1  # Prevent div-by-zero

            # "median_list_price" is the canonical key but the calling sites
            # are inconsistent.  Try multiple aliases before falling back to
            # a neutral value.
            median_price = (
                market_data.get('median_list_price')
                or market_data.get('median_listing_price')
                or market_data.get('median_price')
                or market_data.get('median_home_price')
                or 1  # final guard – avoids ZeroDivision
            )

            metrics['price_per_sqft'] = property_data.get('price', 0) / square_feet
            metrics['price_to_median'] = property_data.get('price', 0) / median_price
            
            # Size metrics
            bedrooms = property_data.get('bedrooms', 0) or 1  # Prevent div-by-zero
            bathrooms = property_data.get('bathrooms', 0) or 1  # Prevent div-by-zero
            metrics['sqft_per_bed'] = square_feet / bedrooms
            metrics['beds_baths_ratio'] = bedrooms / bathrooms
            
            # Age metrics
            current_year = datetime.now().year
            metrics['property_age'] = current_year - property_data.get('year_built', current_year)
            
            # Lot metrics
            lot_size = property_data.get('lot_size', square_feet) or square_feet  # Fallback
            metrics['lot_size_per_sqft'] = lot_size / square_feet
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating base metrics: {str(e)}")
            raise
            
    def _calculate_investment_metrics(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate investment-specific metrics."""
        try:
            metrics = {}
            
            # Get property price with fallback
            price = property_data['price'] or 1  # Prevent division by zero
            square_feet = property_data.get('square_feet', 0) or 1
            beds = property_data.get('bedrooms', 0) or 1
            baths = property_data.get('bathrooms', 0) or 1
            
            # Calculate estimated rent based on market data and property features
            base_rent = market_data.get('median_rent', 0) or (price * 0.008)  # 0.8% of property value as monthly rent
            rent_multiplier = (
                (square_feet / 2000) * 0.3 +  # Size factor
                (beds / 3) * 0.3 +  # Bedroom factor
                (baths / 2) * 0.2 +  # Bathroom factor
                (1 if property_data.get('has_garage', False) else 0.8) * 0.1 +  # Garage factor
                (1 if property_data.get('has_pool', False) else 0.9) * 0.1  # Pool factor
            )
            estimated_monthly_rent = base_rent * rent_multiplier
            annual_rent = estimated_monthly_rent * 12
            
            # Calculate operating expenses
            property_tax_rate = market_data.get('property_tax_rate', 0.02)  # Default 2%
            annual_property_tax = price * property_tax_rate
            annual_insurance = price * 0.005  # 0.5% of property value
            annual_maintenance = price * 0.01  # 1% of property value
            annual_management = annual_rent * 0.1  # 10% of rent
            annual_expenses = annual_property_tax + annual_insurance + annual_maintenance + annual_management
            
            # Calculate mortgage payments
            down_payment = price * 0.2  # 20% down payment
            loan_amount = price - down_payment
            interest_rate = market_data.get('mortgage_rate', 0.07)  # Default 7%
            loan_term = 30  # 30 years
            monthly_rate = interest_rate / 12
            num_payments = loan_term * 12
            monthly_mortgage = (loan_amount * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            annual_mortgage = monthly_mortgage * 12
            
            # Calculate investment metrics
            metrics['cap_rate'] = ((annual_rent - annual_expenses) / price) * 100
            
            # Cash on cash return
            annual_cash_flow = annual_rent - annual_mortgage - annual_expenses
            metrics['cash_on_cash'] = (annual_cash_flow / down_payment) * 100
            
            # Price to rent ratio
            metrics['price_to_rent'] = price / annual_rent
            
            # Gross rent multiplier
            metrics['gross_rent_multiplier'] = price / annual_rent
            
            # Return on investment (ROI)
            metrics['roi'] = (annual_cash_flow / price) * 100
            
            # Debt service coverage ratio
            metrics['dscr'] = annual_rent / annual_mortgage if annual_mortgage > 0 else float('inf')
            
            # Improved Days-on-market ratio calculation with validation
            metrics['dom_ratio'] = self._calculate_dom_ratio(property_data, market_data)
            
            # Calculate flip ROI
            renovation_cost = price * 0.15  # Assume 15% of property value for renovations
            holding_costs = price * 0.05  # 5% of property value for holding costs
            total_investment = down_payment + renovation_cost + holding_costs
            estimated_arv = price * 1.3  # Assume 30% appreciation after renovation
            flip_profit = estimated_arv - price - renovation_cost - holding_costs
            metrics['flip_roi'] = (flip_profit / total_investment) * 100
            
            # Calculate rental yield
            metrics['rental_yield'] = (annual_rent / price) * 100
            
            # Calculate cash flow metrics
            metrics['monthly_cash_flow'] = annual_cash_flow / 12
            metrics['annual_cash_flow'] = annual_cash_flow
            
            # Calculate leverage metrics
            metrics['leverage_ratio'] = loan_amount / price
            metrics['equity_multiplier'] = 1 / (1 - metrics['leverage_ratio'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating investment metrics: {str(e)}")
            raise
            
    def _calculate_dom_ratio(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate DOM ratio with proper validation and error handling."""
        try:
            # Get property DOM with validation
            prop_dom = property_data.get('days_on_market')
            if prop_dom is None or prop_dom < 0:
                logger.warning(f"Invalid property DOM value: {prop_dom}")
                return 1.0  # Neutral value for invalid data
                
            # Get market median DOM with validation - try multiple field names
            market_dom = (
                market_data.get('median_dom') or 
                market_data.get('median_days_on_market') or
                market_data.get('avg_days_on_market')
            )
            
            if market_dom is None or market_dom <= 0:
                logger.warning(f"Invalid market DOM value: {market_dom}")
                # Use a reasonable default based on market conditions
                market_dom = 45  # Industry average for balanced market
                
            # Validate reasonable ranges and log warnings
            if prop_dom > 365:  # Flag extremely old listings
                logger.warning(f"Property DOM > 1 year ({prop_dom} days) - may indicate stale listing")
            if market_dom > 180:  # Flag unusual market conditions
                logger.warning(f"Market median DOM > 6 months ({market_dom} days) - unusual market conditions")
                
            # Calculate ratio with bounds
            dom_ratio = prop_dom / market_dom
            
            # Normalize to reasonable range (0.1 to 10) to prevent extreme outliers
            dom_ratio = np.clip(dom_ratio, 0.1, 10.0)
            
            logger.debug(f"DOM ratio calculated: property={prop_dom}, market={market_dom}, ratio={dom_ratio:.2f}")
            return float(dom_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating DOM ratio: {str(e)}")
            return 1.0  # Return neutral value on error
            
    def _calculate_price_trend(self, market_data: Dict[str, Any]) -> float:
        """Calculate price trend using proper historical analysis with smoothing and normalization."""
        try:
            # Add debug logging for price trend calculation
            logger.debug(f"PRICE_TREND_DEBUG: market_data keys: {list(market_data.keys())}")
            
            # Try to get historical data from different sources
            historical_data = market_data.get('historical_data', {})
            logger.debug(f"PRICE_TREND_DEBUG: historical_data keys: {list(historical_data.keys()) if historical_data else 'None'}")
            logger.debug(f"PRICE_TREND_DEBUG: historical_data type: {type(historical_data)}")
            
            price_data = None
            price_dates = None
            
            if historical_data:
                # From structured historical data
                median_price_data = historical_data.get('median_listing_price', {})
                logger.debug(f"PRICE_TREND_DEBUG: median_price_data type: {type(median_price_data)}, keys: {list(median_price_data.keys()) if isinstance(median_price_data, dict) else 'N/A'}")
                if isinstance(median_price_data, dict) and 'values' in median_price_data:
                    price_data = median_price_data.get('values', [])
                    price_dates = median_price_data.get('dates', [])
                    logger.debug(f"PRICE_TREND_DEBUG: Found {len(price_data)} price values: {price_data[:5] if price_data else 'None'}")
                    
            # If no structured historical data, try direct fields
            if not price_data:
                price_data = market_data.get('price_history', []) or market_data.get('historical_prices', [])
                logger.debug(f"PRICE_TREND_DEBUG: Trying direct fields, found: {price_data}")
                
            # If still no data, try to use single-period change data as fallback
            if not price_data or len(price_data) < 2:
                logger.debug(f"PRICE_TREND_DEBUG: Insufficient price data ({len(price_data) if price_data else 0} points), trying fallback methods")
                
                # Try to use price change fields if available
                price_change_1y = market_data.get('price_change_1y', 0)
                price_change_3y = market_data.get('price_change_3y', 0)
            
                logger.debug(f"PRICE_TREND_DEBUG: Checking fallback values - price_change_1y: {price_change_1y}, price_change_3y: {price_change_3y}")
                
                if price_change_1y != 0:
                    logger.debug(f"PRICE_TREND_DEBUG: Using price_change_1y fallback: {price_change_1y}%")
                    # Convert percentage to decimal and apply dampening
                    dampening_factor = np.exp(-abs(price_change_1y) / 50)
                    dampened_pct = price_change_1y * (0.7 + 0.3 * dampening_factor)
                    result = dampened_pct / 100
                    logger.debug(f"PRICE_TREND_DEBUG: Fallback result from price_change_1y: {result}")
                    return result
                elif price_change_3y != 0:
                    logger.debug(f"PRICE_TREND_DEBUG: Using price_change_3y fallback: {price_change_3y}%")
                    # Annualize the 3-year change and apply dampening
                    annual_change = price_change_3y / 3
                    dampening_factor = np.exp(-abs(annual_change) / 50)
                    dampened_pct = annual_change * (0.7 + 0.3 * dampening_factor)
                    result = dampened_pct / 100
                    logger.debug(f"PRICE_TREND_DEBUG: Fallback result from price_change_3y: {result}")
                    return result
                else:
                    # Try to use market analysis price trends as a final fallback
                    market_analysis = market_data.get('market_analysis', {})
                    price_trends = market_analysis.get('price_trends', {})
                    
                    if price_trends:
                        logger.debug(f"PRICE_TREND_DEBUG: Trying market_analysis price_trends fallback")
                        logger.debug(f"PRICE_TREND_DEBUG: price_trends keys: {list(price_trends.keys())}")
                        
                        # Try different price trend fields
                        yoy_change = price_trends.get('yoy_change', 0)
                        short_term_trend = price_trends.get('short_term_trend', 0)
                        
                        if yoy_change != 0:
                            logger.debug(f"PRICE_TREND_DEBUG: Using market_analysis yoy_change: {yoy_change}%")
                            # Apply dampening and convert to decimal
                            dampening_factor = np.exp(-abs(yoy_change) / 50)
                            dampened_pct = yoy_change * (0.7 + 0.3 * dampening_factor)
                            result = dampened_pct / 100
                            logger.debug(f"PRICE_TREND_DEBUG: Market analysis yoy_change result: {result}")
                            return result
                        elif short_term_trend != 0:
                            logger.debug(f"PRICE_TREND_DEBUG: Using market_analysis short_term_trend: {short_term_trend}%")
                            # Apply dampening and convert to decimal
                            dampening_factor = np.exp(-abs(short_term_trend) / 50)
                            dampened_pct = short_term_trend * (0.7 + 0.3 * dampening_factor)
                            result = dampened_pct / 100
                            logger.debug(f"PRICE_TREND_DEBUG: Market analysis short_term_trend result: {result}")
                            return result
                    
                    logger.debug(f"PRICE_TREND_DEBUG: No historical price data or change data available for trend calculation")
                    logger.debug(f"PRICE_TREND_DEBUG: All available market_data keys: {list(market_data.keys())}")
                    return 0.0
                
            # Convert to pandas Series for easier manipulation
            prices_series = pd.Series([float(p) for p in price_data if p is not None and pd.notna(p)])
            
            if len(prices_series) < 2:
                logger.debug("Insufficient valid price data for trend calculation")
                return 0.0
                
            # Apply 3-month moving average smoothing if we have enough data
            if len(prices_series) >= 3:
                smoothed_prices = prices_series.rolling(window=3, min_periods=1).mean()
            else:
                smoothed_prices = prices_series
                
            # Calculate different time period trends
            latest_price = smoothed_prices.iloc[-1]
            
            # 12-month trend
            if len(smoothed_prices) >= 12:
                year_ago_price = smoothed_prices.iloc[-12]
                trend_period = 12
                if year_ago_price <= 0:
                    logger.warning(f"Invalid historical price value at 12-month mark: {year_ago_price}")
                    return 0.0
                price_change_pct = ((latest_price - year_ago_price) / year_ago_price) * 100
            # 6-month trend (fallback)
            elif len(smoothed_prices) >= 6:
                year_ago_price = smoothed_prices.iloc[-6]
                trend_period = 6
                if year_ago_price <= 0:
                    logger.warning(f"Invalid historical price value at 6-month mark: {year_ago_price}")
                    return 0.0
                # Annualize the 6-month trend correctly
                # Calculate monthly growth rate, then annualize
                monthly_growth_rate = (latest_price / year_ago_price)**(1/6) - 1
                price_change_pct = ((1 + monthly_growth_rate)**12 - 1) * 100
            # 3-month trend (last resort)
            elif len(smoothed_prices) >= 3:
                year_ago_price = smoothed_prices.iloc[-3]
                trend_period = 3
                if year_ago_price <= 0:
                    logger.warning(f"Invalid historical price value at 3-month mark: {year_ago_price}")
                    return 0.0
                # Annualize the 3-month trend correctly
                # Calculate monthly growth rate, then annualize
                monthly_growth_rate = (latest_price / year_ago_price)**(1/3) - 1
                price_change_pct = ((1 + monthly_growth_rate)**12 - 1) * 100
            else:
                # Use first vs last, and annualize for short periods
                year_ago_price = smoothed_prices.iloc[0]
                trend_period = len(smoothed_prices)
                if year_ago_price <= 0 or trend_period == 0:
                    logger.warning(f"Invalid historical price value or period in fallback: {year_ago_price}, {trend_period}")
                    return 0.0
                # Annualize for periods less than 3 months
                monthly_growth_rate = (latest_price / year_ago_price)**(1/trend_period) - 1
                price_change_pct = ((1 + monthly_growth_rate)**12 - 1) * 100

            # Only normalize if using ML model prediction
            if self.ml_models_loaded and 'price_trend' in self.ml_models:
                normalized_trend = self._get_scaled_normalized_trend(price_change_pct, trend_period)
                logger.debug(f"Calculated price trend (ML): {price_change_pct:.2f}% over {trend_period} months, normalized to {normalized_trend:.2f}")
                return normalized_trend
            else:
                # For heuristic calculation, return the raw percentage after dampening
                dampening_factor = np.exp(-abs(price_change_pct) / 50)  # Dampen extreme values
                dampened_pct = price_change_pct * (0.7 + 0.3 * dampening_factor)
                logger.debug(f"Calculated price trend (heuristic): raw={price_change_pct:.2f}%, dampened={dampened_pct:.2f}%")
                return dampened_pct / 100  # Convert to decimal for consistency with other metrics
            
        except Exception as e:
            logger.error(f"Error in _calculate_price_trend: {e}", exc_info=True)
            return 0.0

    def _get_scaled_normalized_trend(self, price_change_pct: float, trend_period: int) -> float:
        """
        Scales trend by period length and normalizes to a -1 to 1 range, targeting typical 
        real estate market ranges of ±5% to ±15% annually.
        """
        try:
            # Allow slightly wider range for initial clipping to account for adjustments
            max_annual_change = 20.0  # Maximum realistic annual change
            clipped_change = np.clip(price_change_pct, -max_annual_change, max_annual_change)
            
            # Enhanced confidence scaling based on data period length
            # Exponential decay for shorter periods to heavily discount short-term volatility
            base_confidence = 1.0 - np.exp(-trend_period / 4)  # Reaches ~0.95 at 12 months
            confidence_scale = 0.3 + (0.7 * base_confidence)  # Range [0.3, 1.0]
            
            # More aggressive dampening as we approach max ranges
            abs_change = abs(clipped_change)
            if abs_change <= 5:
                dampening = 1.0  # No dampening for small changes
            elif abs_change <= 10:
                dampening = 0.8  # Moderate dampening
            else:
                # Progressive dampening for larger changes
                dampening = 0.8 * np.exp(-(abs_change - 10) / 10)
            
            # Apply confidence and dampening
            adjusted_change = clipped_change * confidence_scale * dampening
            
            # Final scaling to target range
            target_max = 15.0  # Maximum target annual change
            normalized = adjusted_change / target_max  # Will be in [-1, 1] range
            
            # Smooth sigmoid transformation for final output
            def sigmoid_transform(x, steepness=2.0):
                return 2 / (1 + np.exp(-steepness * x)) - 1
            
            final_trend = sigmoid_transform(normalized)
            
            # Log intermediate values for debugging
            logger.debug(f"Price trend calculation:")
            logger.debug(f"  Raw change: {price_change_pct:.2f}%")
            logger.debug(f"  Clipped change: {clipped_change:.2f}%")
            logger.debug(f"  Confidence ({trend_period} months): {confidence_scale:.2f}")
            logger.debug(f"  Dampening factor: {dampening:.2f}")
            logger.debug(f"  Adjusted change: {adjusted_change:.2f}%")
            logger.debug(f"  Final normalized trend: {final_trend:.2f}")
            
            return final_trend
            
        except Exception as e:
            logger.error(f"Error in _get_scaled_normalized_trend: {e}")
            return 0.0  # Safe fallback
            
    def _calculate_risk_metrics(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk assessment metrics using ML models with heuristic fallback.
        
        Returns:
            Dict matching RiskMetricsResponse schema with float metrics and source tracking.
        """
        try:
            metrics = {
                'market_risk': 0.0,
                'property_risk': 0.0,
                'location_risk': 0.0,
                'overall_risk': 0.0
            }
            metrics_source = {}  # Track the source of each metric (ML or heuristic)
            
            # Skip if property_data is empty
            if not property_data:
                property_id = property_data.get('property_id', 'unknown')
                logger.warning(f"Empty property_data provided for property_id: {property_id}. Using fallback values for risk metrics.")
                return {
                    'market_risk': 0.5,
                    'property_risk': 0.5,
                    'location_risk': 0.5,
                    'overall_risk': 0.5,
                    'metrics_source': {
                        'market_risk': 'neutral_fallback',
                        'property_risk': 'neutral_fallback',
                        'location_risk': 'neutral_fallback',
                        'overall_risk': 'neutral_fallback'
                    }
                }
                
            # Handle common field name aliases in market_data
            field_aliases = {
                'price_reduced_count': 'price_reduction_count',
                'price_increased_count': 'price_increase_count',
                'median_dom': 'median_days_on_market',
                'median_listing_price': 'median_price'
            }
            
            # Copy aliases to standard field names if the standard is missing but alias exists
            for std_field, alias_field in field_aliases.items():
                if std_field not in market_data and alias_field in market_data:
                    market_data[std_field] = market_data[alias_field]
                
            # Extract features for ML models
            ml_features = self._extract_ml_features(property_data, market_data)
            
            # If feature extraction failed (returned empty dict), use heuristic fallback
            if not ml_features:
                logger.warning("Feature extraction failed for risk metrics. Using heuristic fallback.")
                metrics['market_risk'] = self._calculate_market_risk_heuristic(market_data)
                metrics['property_risk'] = self._calculate_property_risk_heuristic(property_data)
                metrics['location_risk'] = self._calculate_location_risk_heuristic(property_data)
                metrics['overall_risk'] = (
                    metrics['market_risk'] * 0.4 +
                    metrics['property_risk'] * 0.3 +
                    metrics['location_risk'] * 0.3
                )
                return {
                    'market_risk': metrics['market_risk'],
                    'property_risk': metrics['property_risk'],
                    'location_risk': metrics['location_risk'],
                    'overall_risk': metrics['overall_risk'],
                    'metrics_source': {
                        'market_risk': 'heuristic_fallback',
                        'property_risk': 'heuristic_fallback',
                        'location_risk': 'heuristic_fallback',
                        'overall_risk': 'weighted_average'
                    }
                }
            
            # Market risk - try ML first, fallback to heuristic
            # ML temporarily disabled
            # ml_market_risk = self._predict_with_ml_model('market_risk', ml_features)
            # if ml_market_risk is not None:
            #     metrics['market_risk'] = ml_market_risk
            #     metrics_source['market_risk'] = 'ml_model'
            #     logger.debug("Using ML prediction for market_risk")
            # else:
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                metrics['market_risk'] = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                metrics_source['market_risk'] = 'neutral_fallback'
                logger.debug("Using neutral fallback for market_risk")
            else:
                metrics['market_risk'] = self._calculate_market_risk_heuristic(market_data)
                metrics_source['market_risk'] = 'heuristic_fallback'
                logger.debug("Using heuristic fallback for market_risk")
            
            # Property risk - try ML first, fallback to heuristic
            # ML temporarily disabled
            # ml_property_risk = self._predict_with_ml_model('property_risk', ml_features)
            # if ml_property_risk is not None:
            #     metrics['property_risk'] = ml_property_risk
            #     metrics_source['property_risk'] = 'ml_model'
            #     logger.debug("Using ML prediction for property_risk")
            # else:
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                metrics['property_risk'] = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                metrics_source['property_risk'] = 'neutral_fallback'
                logger.debug("Using neutral fallback for property_risk")
            else:
                metrics['property_risk'] = self._calculate_property_risk_heuristic(property_data)
                metrics_source['property_risk'] = 'heuristic_fallback'
                logger.debug("Using heuristic fallback for property_risk")
            
            # Location risk - try ML first, fallback to heuristic
            # ML temporarily disabled
            # ml_location_risk = self._predict_with_ml_model('location_risk', ml_features)
            # if ml_location_risk is not None:
            #     metrics['location_risk'] = ml_location_risk
            #     metrics_source['location_risk'] = 'ml_model'
            #     logger.debug("Using ML prediction for location_risk")
            # else:
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                metrics['location_risk'] = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                metrics_source['location_risk'] = 'neutral_fallback'
                logger.debug("Using neutral fallback for location_risk")
            else:
                metrics['location_risk'] = self._calculate_location_risk_heuristic(property_data)
                metrics_source['location_risk'] = 'heuristic_fallback'
                logger.debug("Using heuristic fallback for location_risk")
            
            # Overall risk - try ML first, fallback to weighted average
            # ML temporarily disabled
            # ml_overall_risk = self._predict_with_ml_model('overall_risk', ml_features)
            # if ml_overall_risk is not None:
            #     metrics['overall_risk'] = ml_overall_risk
            #     metrics_source['overall_risk'] = 'ml_model'
            #     logger.debug("Using ML prediction for overall_risk")
            # else:
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                metrics['overall_risk'] = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                metrics_source['overall_risk'] = 'neutral_fallback'
                logger.debug("Using neutral fallback for overall_risk")
            else:
                metrics['overall_risk'] = (
                    metrics['market_risk'] * 0.4 +
                    metrics['property_risk'] * 0.3 +
                    metrics['location_risk'] * 0.3
                )
                metrics_source['overall_risk'] = 'weighted_average'
                logger.debug("Using weighted average for overall_risk")
            
            # Log detailed metrics information
            logger.info(f"Risk metrics: market={metrics['market_risk']:.4f} ({metrics_source.get('market_risk', 'unknown')}), "
                      f"property={metrics['property_risk']:.4f} ({metrics_source.get('property_risk', 'unknown')}), "
                      f"location={metrics['location_risk']:.4f} ({metrics_source.get('location_risk', 'unknown')}), "
                      f"overall={metrics['overall_risk']:.4f} ({metrics_source.get('overall_risk', 'unknown')})")
            
            # Return in new schema format
            return {
                'market_risk': metrics['market_risk'],
                'property_risk': metrics['property_risk'],
                'location_risk': metrics['location_risk'],
                'overall_risk': metrics['overall_risk'],
                'metrics_source': metrics_source
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            # Return fallback values instead of raising exception
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                neutral = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                return {
                    'market_risk': neutral,
                    'property_risk': neutral,
                    'location_risk': neutral,
                    'overall_risk': neutral,
                    'metrics_source': {
                        'market_risk': 'error_fallback',
                        'property_risk': 'error_fallback',
                        'location_risk': 'error_fallback',
                        'overall_risk': 'error_fallback'
                    }
                }
            else:
                return {
                    'market_risk': 0.4,
                    'property_risk': 0.5,
                    'location_risk': 0.5,
                    'overall_risk': 0.45,
                    'metrics_source': {
                        'market_risk': 'error_fallback',
                        'property_risk': 'error_fallback',
                        'location_risk': 'error_fallback',
                        'overall_risk': 'error_fallback'
                    }
                }
            
    def _calculate_market_risk_heuristic(self, market_data: Dict[str, Any]) -> float:
        """Calculate market risk score using real features only."""
        try:
            # Price trend risk - based on real price changes
            price_change_1y = float(market_data.get('price_change_1y', 0))
            price_change_3y = float(market_data.get('price_change_3y', 0))
            price_change_5y = float(market_data.get('price_change_5y', 0))
            
            # Negative price changes increase risk
            price_trend_risk = 0.0
            if price_change_1y < 0:
                price_trend_risk += abs(price_change_1y) * 0.02  # 2% weight per % decline
            if price_change_3y < 0:
                price_trend_risk += abs(price_change_3y) * 0.01  # 1% weight per % decline
            if price_change_5y < 0:
                price_trend_risk += abs(price_change_5y) * 0.005  # 0.5% weight per % decline
            price_trend_risk = min(price_trend_risk, 1.0)
            
            # Market activity risk
            median_dom = float(market_data.get('median_dom', 30))
            active_listing_count = float(market_data.get('active_listing_count', 100))
            
            # DOM risk: Higher days on market indicates higher risk
            dom_risk = min(median_dom / 90, 1.0)  # 90 days as benchmark
            
            # Supply risk: Too many or too few listings indicate risk
            supply_months = active_listing_count / max(1, float(market_data.get('monthly_sales', 10)))
            if supply_months <= 2:  # Too little supply
                supply_risk = 0.7
            elif supply_months >= 8:  # Too much supply
                supply_risk = min(supply_months / 12, 1.0)
            else:  # Healthy supply
                supply_risk = 0.3
            
            # Price volatility risk
            price_volatility = float(market_data.get('price_volatility', 0.1))
            volatility_risk = min(price_volatility * 3, 1.0)  # Scale volatility to risk
            
            # Price reduction risk
            total_listings = float(market_data.get('active_listing_count', 100))
            price_reduction_count = float(market_data.get('price_reduction_count', 0))
            if total_listings > 0:
                reduction_risk = min((price_reduction_count / total_listings) * 2, 1.0)
            else:
                reduction_risk = 0.5
            
            # Combined market risk with weighted components
            market_risk = (
                price_trend_risk * 0.25 +
                dom_risk * 0.20 +
                supply_risk * 0.20 +
                volatility_risk * 0.20 +
                reduction_risk * 0.15
            )
            
            # Log metrics for debugging
                    # Note: Verbose logging reduced to prevent console flooding
        # logger.debug(f"Market Risk Components: price_trend={price_trend_risk:.2f}, "
        #              f"dom={dom_risk:.2f}, supply={supply_risk:.2f}, " 
        #              f"volatility={volatility_risk:.2f}, reduction={reduction_risk:.2f}")
            
            return max(0.1, min(0.9, market_risk))
            
        except Exception as e:
            logger.error(f"Error calculating market risk: {str(e)} - Input data: {market_data}")
            return 0.5
            
    def _calculate_property_risk_heuristic(self, property_data: Dict[str, Any]) -> float:
        """Calculate property-specific risk using real features only."""
        try:
            # Price risk based on price per square foot
            price = float(property_data.get('price', 300000))
            square_feet = float(property_data.get('square_feet', 2000))
            price_per_sqft = price / max(1, square_feet)
            
            # Compare to market averages
            market_avg_price_per_sqft = float(property_data.get('market_avg_price_per_sqft', price_per_sqft))
            price_ratio = price_per_sqft / max(1, market_avg_price_per_sqft)
            
            # Price position risk
            if price_ratio > 1.3:  # Significantly overpriced
                price_risk = min(price_ratio - 1, 1.0)
            elif price_ratio < 0.7:  # Significantly underpriced
                price_risk = min((0.7 - price_ratio) * 2, 1.0)
            else:
                price_risk = 0.3  # Reasonable price range
            
            # Size risk - extreme sizes carry more risk
            size_percentile = float(property_data.get('size_percentile', 50))
            if size_percentile < 10 or size_percentile > 90:
                size_risk = 0.8  # Extreme sizes
            elif size_percentile < 20 or size_percentile > 80:
                size_risk = 0.6  # Unusual sizes
            else:
                size_risk = 0.3  # Common sizes
            
            # Price trend risk
            price_change = float(property_data.get('price_change_since_listing', 0))
            if price_change < -10:
                trend_risk = min(abs(price_change) / 20, 1.0)  # Significant price drops
            elif price_change > 10:
                trend_risk = min(price_change / 20, 1.0)  # Significant price increases
            else:
                trend_risk = 0.3  # Stable price
            
            # Days on market risk
            days_on_market = float(property_data.get('days_on_market', 30))
            dom_risk = min(days_on_market / 120, 1.0)  # Longer time = higher risk
            
            # Combined property risk
            property_risk = (
                price_risk * 0.35 +
                size_risk * 0.20 +
                trend_risk * 0.25 +
                dom_risk * 0.20
            )
            
            # Note: Verbose logging reduced to prevent console flooding
            # logger.debug(f"Property Risk Components: price={price_risk:.2f}, "
            #            f"size={size_risk:.2f}, trend={trend_risk:.2f}, "
            #            f"dom={dom_risk:.2f}")
            
            return max(0.1, min(0.9, property_risk))
            
        except Exception as e:
            logger.error(f"Error calculating property risk: {str(e)} - Input data: {property_data}")
            return 0.5
            
    def _calculate_location_risk_heuristic(self, property_data: Dict[str, Any]) -> float:
        """Calculate location-based risk using real market data only."""
        try:
            # Market trend risk
            market_price_trend = float(property_data.get('market_price_trend_1y', 0))
            if market_price_trend < -5:
                trend_risk = min(abs(market_price_trend) / 15, 1.0)  # Declining market
            elif market_price_trend > 15:
                trend_risk = min((market_price_trend - 15) / 15, 0.8)  # Overheated market
            else:
                trend_risk = 0.3  # Stable market
            
            # Market liquidity risk
            avg_dom = float(property_data.get('market_avg_dom', 45))
            if avg_dom > 90:
                liquidity_risk = 0.9  # Very illiquid market
            elif avg_dom > 60:
                liquidity_risk = 0.7  # Somewhat illiquid
            elif avg_dom < 15:
                liquidity_risk = 0.6  # Too hot market
            else:
                liquidity_risk = 0.3  # Healthy liquidity
            
            # Price volatility risk
            market_volatility = float(property_data.get('market_price_volatility', 0.1))
            volatility_risk = min(market_volatility * 4, 1.0)
            
            # Market competition risk
            inventory_ratio = float(property_data.get('market_inventory_ratio', 1.0))
            if inventory_ratio > 2.0:  # High supply
                competition_risk = min(inventory_ratio / 3, 1.0)
            elif inventory_ratio < 0.5:  # Low supply
                competition_risk = min((0.5 - inventory_ratio) * 2, 0.8)
            else:
                competition_risk = 0.3  # Balanced market
            
            # Combined location risk
            location_risk = (
                trend_risk * 0.3 +
                liquidity_risk * 0.25 +
                volatility_risk * 0.25 +
                competition_risk * 0.2
            )
            
            # Log metrics for debugging
                    # Note: Verbose logging reduced to prevent console flooding  
        # logger.debug(f"Location Risk Components: trend={trend_risk:.2f}, "
        #              f"liquidity={liquidity_risk:.2f}, volatility={volatility_risk:.2f}, "
        #              f"competition={competition_risk:.2f}")
            
            return max(0.1, min(0.9, location_risk))
            
        except Exception as e:
            logger.error(f"Error calculating location risk: {str(e)} - Input data: {property_data}")
            return 0.5
            
    def _calculate_market_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market performance metrics using ML models with heuristic fallback.
        
        Returns:
            Dict matching MarketMetricsResponse schema with float metrics and source tracking.
        """
        try:
            metrics = {
                'market_health': 0.0,
                'market_momentum': 0.0,
                'market_stability': 0.0,
                'price_growth_rate': 0.0,
                'price_trend': 0.0  # Add price trend metric
            }
            metrics_source = {}  # Track the source of each metric (ML or heuristic)
            
            # Skip if market_data is empty or missing critical fields
            if not market_data:
                logger.warning("Empty market_data provided. Using fallback values for market metrics.")
                return {
                    'market_health': 0.5,
                    'market_momentum': 0.5,
                    'market_stability': 0.5,
                    'price_growth_rate': 0.0,
                    'price_trend': 0.0,
                    'metrics_source': {
                        'market_health': 'neutral_fallback',
                        'market_momentum': 'neutral_fallback',
                        'market_stability': 'neutral_fallback',
                        'price_growth_rate': 'neutral_fallback',
                        'price_trend': 'neutral_fallback'
                    }
                }
                
            # Count missing critical market data fields
            critical_fields = ['median_listing_price', 'median_dom', 'active_listing_count', 
                              'price_reduced_count', 'price_increased_count']
            missing_fields = [field for field in critical_fields if field not in market_data]
            
            # Only log if there are missing fields that aren't aliases
            # Handle common field name aliases
            field_aliases = {
                'price_reduced_count': 'price_reduction_count',
                'price_increased_count': 'price_increase_count',
                'median_dom': 'median_days_on_market',
                'median_listing_price': 'median_price'
            }
            
            # Check if missing fields have aliases that are present
            real_missing = []
            for field in missing_fields:
                if field in field_aliases and field_aliases[field] in market_data:
                    # Field has an alias that's present, so it's not really missing
                    continue
                real_missing.append(field)
                
            if real_missing:
                logger.warning(f"Missing critical market data fields: {real_missing}")
                
            # Extract features for ML models (using empty property_data for market-only metrics)
            ml_features = self._extract_ml_features({}, market_data)
            
            # If feature extraction failed (returned empty dict), use heuristic fallback
            if not ml_features:
                logger.warning("Feature extraction failed for market metrics. Using heuristic fallback.")
                metrics['market_health'] = self._calculate_market_health_heuristic(market_data)
                metrics['market_momentum'] = self._calculate_market_momentum_heuristic(market_data)
                metrics['market_stability'] = self._calculate_market_stability_heuristic(market_data)
                # Calculate price growth rate from market momentum and stability
                metrics['price_growth_rate'] = metrics['market_momentum'] * metrics['market_stability'] * 0.1  # 10% max annual growth
                
                # Calculate price trend using historical data
                metrics['price_trend'] = self._calculate_price_trend(market_data)
                logger.debug(f"PRICE_TREND_DEBUG: Final price_trend result (heuristic path): {metrics['price_trend']}")
                
                return {
                    'market_health': metrics['market_health'],
                    'market_momentum': metrics['market_momentum'],
                    'market_stability': metrics['market_stability'],
                    'price_growth_rate': metrics['price_growth_rate'],
                    'price_trend': metrics['price_trend'],
                    'metrics_source': {
                        'market_health': 'heuristic_fallback',
                        'market_momentum': 'heuristic_fallback',
                        'market_stability': 'heuristic_fallback',
                        'price_growth_rate': 'heuristic_fallback',
                        'price_trend': 'heuristic_fallback'
                    }
                }
                
            # Market health - try ML first, fallback to heuristic
            # ML temporarily disabled
            # ml_market_health = self._predict_with_ml_model('market_health', ml_features)
            # if ml_market_health is not None:
            #     metrics['market_health'] = ml_market_health
            #     metrics_source['market_health'] = 'ml_model'
            #     logger.debug("Using ML prediction for market_health")
            # else:
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                metrics['market_health'] = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                metrics_source['market_health'] = 'neutral_fallback'
                logger.debug("Using neutral fallback for market_health")
            else:
                metrics['market_health'] = self._calculate_market_health_heuristic(market_data)
                metrics_source['market_health'] = 'heuristic_fallback'
                logger.debug("Using heuristic fallback for market_health")
            
            # Market momentum - try ML first, fallback to heuristic
            # ML temporarily disabled
            # ml_market_momentum = self._predict_with_ml_model('market_momentum', ml_features)
            # if ml_market_momentum is not None:
            #     metrics['market_momentum'] = ml_market_momentum
            #     metrics_source['market_momentum'] = 'ml_model'
            #     logger.debug("Using ML prediction for market_momentum")
            # else:
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                metrics['market_momentum'] = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                metrics_source['market_momentum'] = 'neutral_fallback'
                logger.debug("Using neutral fallback for market_momentum")
            else:
                metrics['market_momentum'] = self._calculate_market_momentum_heuristic(market_data)
                metrics_source['market_momentum'] = 'heuristic_fallback'
                logger.debug("Using heuristic fallback for market_momentum")
            
            # Market stability - try ML first, fallback to heuristic
            # ML temporarily disabled
            # ml_market_stability = self._predict_with_ml_model('market_stability', ml_features)
            # if ml_market_stability is not None:
            #     metrics['market_stability'] = ml_market_stability
            #     metrics_source['market_stability'] = 'ml_model'
            #     logger.debug("Using ML prediction for market_stability")
            # else:
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                metrics['market_stability'] = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                metrics_source['market_stability'] = 'neutral_fallback'
                logger.debug("Using neutral fallback for market_stability")
            else:
                metrics['market_stability'] = self._calculate_market_stability_heuristic(market_data)
                metrics_source['market_stability'] = 'heuristic_fallback'
                logger.debug("Using heuristic fallback for market_stability")
            
            # Calculate price growth rate based on momentum and stability
            price_growth_rate = metrics['market_momentum'] * metrics['market_stability'] * 0.1  # 10% max annual growth
            metrics['price_growth_rate'] = price_growth_rate
            metrics_source['price_growth_rate'] = 'weighted_average'
            
            # Calculate price trend using historical data
            metrics['price_trend'] = self._calculate_price_trend(market_data)
            metrics_source['price_trend'] = 'historical_analysis'
            logger.debug(f"PRICE_TREND_DEBUG: Final price_trend result: {metrics['price_trend']}")
            
            # Log detailed metrics information
            logger.info(f"Market metrics: health={metrics['market_health']:.4f} ({metrics_source.get('market_health', 'unknown')}), "
                      f"momentum={metrics['market_momentum']:.4f} ({metrics_source.get('market_momentum', 'unknown')}), "
                      f"stability={metrics['market_stability']:.4f} ({metrics_source.get('market_stability', 'unknown')}), "
                      f"growth_rate={metrics['price_growth_rate']:.4f} ({metrics_source.get('price_growth_rate', 'unknown')}), "
                      f"price_trend={metrics['price_trend']:.4f} ({metrics_source.get('price_trend', 'unknown')})")
            
            # Return in new schema format
            return {
                'market_health': metrics['market_health'],
                'market_momentum': metrics['market_momentum'],
                'market_stability': metrics['market_stability'],
                'price_growth_rate': metrics['price_growth_rate'],
                'price_trend': metrics['price_trend'],
                'metrics_source': metrics_source
            }
            
        except Exception as e:
            logger.error(f"Error calculating market metrics: {str(e)}")
            # Return more realistic default values based on configuration
            if settings.ML_MODEL_SETTINGS['USE_NEUTRAL_FALLBACK']:
                neutral = settings.ML_MODEL_SETTINGS['NEUTRAL_FALLBACK_VALUE']
                return {
                    'market_health': neutral,
                    'market_momentum': neutral,
                    'market_stability': neutral,
                    'price_growth_rate': neutral * 0.1,  # 10% max annual growth
                    'price_trend': 0.0,  # Neutral price trend
                    'metrics_source': {
                        'market_health': 'error_fallback',
                        'market_momentum': 'error_fallback',
                        'market_stability': 'error_fallback',
                        'price_growth_rate': 'error_fallback',
                        'price_trend': 'error_fallback'
                    }
                }
            else:
                return {
                    'market_health': 0.35,
                    'market_momentum': 0.25,
                    'market_stability': 0.45,
                    'price_growth_rate': 0.03,  # 3% conservative growth
                    'price_trend': 0.0,  # Neutral price trend
                    'metrics_source': {
                        'market_health': 'error_fallback',
                        'market_momentum': 'error_fallback',
                        'market_stability': 'error_fallback',
                        'price_growth_rate': 'error_fallback',
                        'price_trend': 'error_fallback'
                    }
                }
    
    def _calculate_market_health_heuristic(self, market_data: Dict[str, Any]) -> float:
        """Calculate market health using real market data with dynamic scaling."""
        try:
            # Price trends with continuous scaling
            price_change_1y = float(market_data.get('price_change_1y', 0))
            price_change_3y = float(market_data.get('price_change_3y', 0))
            
            # Dynamic price health calculation using sigmoid-like scaling
            def calculate_price_health(change):
                # Center point at 0% change, scale factor determines steepness
                scale_factor = 0.15  # Adjusts sensitivity to price changes
                return 0.5 + 0.4 * (2 / (1 + np.exp(-change * scale_factor)) - 1)
            
            price_health_1y = calculate_price_health(price_change_1y)
            price_health_3y = calculate_price_health(price_change_3y)
            
            # Market activity metrics
            median_dom = float(market_data.get('median_dom', 45))
            active_listings = float(market_data.get('active_listing_count', 100))
            monthly_sales = float(market_data.get('monthly_sales', 10))
            
            # DOM health with continuous scaling
            # Optimal DOM range: 15-45 days
            dom_scale = np.clip((60 - median_dom) / 45, -1, 1)
            dom_health = 0.5 + 0.4 * dom_scale
            
            # Supply-demand dynamics
            months_of_supply = active_listings / max(1, monthly_sales)
            # Optimal range: 4-6 months of supply
            supply_scale = np.clip(2 - abs(months_of_supply - 5) / 3, 0, 1)
            supply_health = 0.3 + 0.6 * supply_scale
            
            # Market velocity (sales rate relative to inventory)
            sales_ratio = monthly_sales / max(1, active_listings)
            velocity_health = 0.3 + 0.6 * min(sales_ratio * 2, 1.0)
            
            # Combined health score with dynamic weights
            market_health = (
                price_health_1y * 0.25 +
                price_health_3y * 0.15 +
                dom_health * 0.25 +
                supply_health * 0.2 +
                velocity_health * 0.15
            )
            
            # Ensure output range matches existing contract
            return max(0.1, min(0.9, market_health))
            
        except Exception as e:
            logger.error(f"Error calculating market health: {str(e)} - Input data: {market_data}")
            return 0.35
    
    def _calculate_market_momentum_heuristic(self, market_data: Dict[str, Any]) -> float:
        """Calculate market momentum using real market data with continuous scaling."""
        try:
            # Price momentum with exponential weighting
            price_change_1y = float(market_data.get('price_change_1y', 0))
            price_change_3y = float(market_data.get('price_change_3y', 0))
            price_change_5y = float(market_data.get('price_change_5y', 0))
            
            # Normalize price changes with continuous scaling
            def normalize_price_momentum(change):
                # Use sigmoid-like function centered at 0
                scale = 0.2  # Sensitivity to price changes
                base = 0.5 + 0.4 * np.tanh(change * scale)
                return max(0.1, min(0.9, base))
            
            # Weight recent changes more heavily
            price_momentum = (
                normalize_price_momentum(price_change_1y) * 0.5 +
                normalize_price_momentum(price_change_3y) * 0.3 +
                normalize_price_momentum(price_change_5y) * 0.2
            )
            
            # Sales momentum with continuous scaling
            current_sales = float(market_data.get('monthly_sales', 10))
            prev_sales = float(market_data.get('prev_monthly_sales', 10))
            sales_change_pct = ((current_sales - prev_sales) / max(1, prev_sales)) * 100
            
            # Continuous sales momentum scaling
            sales_scale = np.clip(sales_change_pct / 20, -1, 1)  # Normalize to [-1, 1]
            sales_momentum = 0.5 + 0.4 * sales_scale  # Scale to [0.1, 0.9]
            
            # Listing momentum with continuous scaling
            new_listings = float(market_data.get('new_listings', 0))
            total_listings = float(market_data.get('active_listing_count', 100))
            listing_ratio = new_listings / max(1, total_listings)
            
            # Optimal listing ratio around 0.15-0.25
            listing_scale = 1 - abs(listing_ratio - 0.2) * 2
            listing_momentum = 0.3 + 0.6 * np.clip(listing_scale, 0, 1)
            
            # Market velocity component
            monthly_sales = float(market_data.get('monthly_sales', 10))
            active_listings = float(market_data.get('active_listing_count', 100))
            velocity_ratio = monthly_sales / max(1, active_listings)
            velocity_momentum = 0.3 + 0.6 * min(velocity_ratio * 3, 1.0)
            
            # Combined momentum with dynamic weighting
            market_momentum = (
                price_momentum * 0.35 +
                sales_momentum * 0.25 +
                listing_momentum * 0.2 +
                velocity_momentum * 0.2
            )
            
            return max(0.1, min(0.9, market_momentum))
            
        except Exception as e:
            logger.error(f"Error calculating market momentum: {str(e)} - Input data: {market_data}")
            return 0.25
    
    def _calculate_market_stability_heuristic(self, market_data: Dict[str, Any]) -> float:
        """Calculate market stability using real market data with continuous scaling."""
        try:
            # Price stability with exponential decay
            price_volatility = float(market_data.get('price_volatility', 0.1))
            # Exponential decay function for smoother transition
            price_stability = 0.9 * np.exp(-3 * price_volatility) + 0.1
            
            # Inventory stability with continuous scaling
            current_inventory = float(market_data.get('active_listing_count', 100))
            prev_inventory = float(market_data.get('prev_active_listing_count', 100))
            inventory_change = abs((current_inventory - prev_inventory) / max(1, prev_inventory))
            
            # Gaussian-like stability score centered around 0% change
            inventory_stability = 0.9 * np.exp(-(inventory_change ** 2) / 0.04) + 0.1
            
            # DOM stability with continuous scaling
            dom_volatility = float(market_data.get('dom_volatility', 0.1))
            # Sigmoid-based stability score
            dom_stability = 0.1 + 0.8 / (1 + np.exp(5 * dom_volatility))
            
            # Price reduction stability with continuous scaling
            reduction_ratio = float(market_data.get('price_reduction_ratio', 0.1))
            # Optimal reduction ratio around 0.1-0.2
            reduction_scale = abs(reduction_ratio - 0.15)
            reduction_stability = 0.9 * np.exp(-5 * reduction_scale) + 0.1
            
            # Sales variance stability
            sales_volatility = float(market_data.get('sales_volatility', 0.15))
            sales_stability = 0.9 * np.exp(-4 * sales_volatility) + 0.1
            
            # Combined stability with dynamic weighting
            market_stability = (
                price_stability * 0.3 +
                inventory_stability * 0.2 +
                dom_stability * 0.2 +
                reduction_stability * 0.15 +
                sales_stability * 0.15
            )
            
            return max(0.1, min(0.9, market_stability))
            
        except Exception as e:
            logger.error(f"Error calculating market stability: {str(e)}")
            return 0.45
            
    def _calculate_investment_score(
        self,
        base_metrics: Dict[str, float],
        investment_metrics: Dict[str, float],
        risk_metrics: Dict[str, Any],  # Updated to handle new format
        market_metrics: Dict[str, Any]  # Updated to handle new format
    ) -> float:
        """Calculate overall investment score."""
        try:
            # Normalize metrics to 0-1 range with safe bounds
            def normalize(value, min_val, max_val, inverse=False):
                if max_val <= min_val:
                    return 0.5
                normalized = (value - min_val) / (max_val - min_val)
                if inverse:
                    normalized = 1 - normalized
                return max(0, min(1, normalized))
            
            # Base score components (25% weight)
            price_per_sqft = base_metrics.get('price_per_sqft', 0)
            price_to_median = base_metrics.get('price_to_median', 0)
            sqft_per_bed = base_metrics.get('sqft_per_bed', 0)
            beds_baths_ratio = base_metrics.get('beds_baths_ratio', 0)
            
            base_score = (
                normalize(price_per_sqft, 100, 500, inverse=True) * 0.3 +  # Lower price per sqft is better
                normalize(price_to_median, 0.5, 1.5, inverse=True) * 0.3 +  # Lower price to median is better
                normalize(sqft_per_bed, 200, 500) * 0.2 +  # Higher sqft per bed is better
                normalize(beds_baths_ratio, 1, 3) * 0.2  # Balanced ratio is better
            )
            
            # Investment score components (35% weight)
            cap_rate = investment_metrics.get('cap_rate', 0)
            cash_on_cash = investment_metrics.get('cash_on_cash', 0)
            roi = investment_metrics.get('roi', 0)
            dscr = investment_metrics.get('dscr', 0)
            rental_yield = investment_metrics.get('rental_yield', 0)
            
            investment_score = (
                normalize(cap_rate, 3, 10) * 0.25 +  # Higher cap rate is better
                normalize(cash_on_cash, 5, 15) * 0.25 +  # Higher cash on cash is better
                normalize(roi, 5, 15) * 0.2 +  # Higher ROI is better
                normalize(dscr, 1, 2) * 0.15 +  # Higher DSCR is better
                normalize(rental_yield, 5, 12) * 0.15  # Higher rental yield is better
            )
            
            # Risk score (20% weight)
            # Extract risk metrics, ignoring metrics_source
            market_risk = float(risk_metrics.get('market_risk', 0.5))
            property_risk = float(risk_metrics.get('property_risk', 0.5))
            location_risk = float(risk_metrics.get('location_risk', 0.5))
            overall_risk = float(risk_metrics.get('overall_risk', 0.5))
            
            # Calculate risk score using both individual risks and overall risk
            individual_risk_score = (
                (1 - market_risk) * 0.4 +  # Lower market risk is better
                (1 - property_risk) * 0.3 +  # Lower property risk is better
                (1 - location_risk) * 0.3  # Lower location risk is better
            )
            
            # Combine individual and overall risk scores
            risk_score = individual_risk_score * 0.7 + (1 - overall_risk) * 0.3
            
            # Market score (20% weight)
            # Extract market metrics, ignoring metrics_source
            market_health = float(market_metrics.get('market_health', 0.5))
            market_momentum = float(market_metrics.get('market_momentum', 0.5))
            market_stability = float(market_metrics.get('market_stability', 0.5))
            price_growth_rate = float(market_metrics.get('price_growth_rate', 0.03))  # 3% default growth
            
            # Calculate market score incorporating growth rate
            market_score = (
                market_health * 0.35 +
                market_momentum * 0.25 +
                market_stability * 0.25 +
                normalize(price_growth_rate, 0.02, 0.10) * 0.15  # Growth rate between 2% and 10%
            )
            
            # Calculate final weighted score
            final_score = (
                base_score * 0.25 +
                investment_score * 0.35 +
                risk_score * 0.20 +
                market_score * 0.20
            )
            
            # Add property-specific variations
            property_variations = []
            
            # Price position variation
            if price_to_median < 0.8:
                property_variations.append(0.1)  # Bonus for properties below median
            elif price_to_median > 1.2:
                property_variations.append(-0.1)  # Penalty for properties above median
            
            # Cap rate variation
            if cap_rate > 8:
                property_variations.append(0.1)  # Bonus for high cap rate
            elif cap_rate < 4:
                property_variations.append(-0.1)  # Penalty for low cap rate
            
            # Cash flow variation
            if cash_on_cash > 12:
                property_variations.append(0.1)  # Bonus for high cash on cash
            elif cash_on_cash < 6:
                property_variations.append(-0.1)  # Penalty for low cash on cash
            
            # Market momentum variation
            if market_momentum > 0.7:
                property_variations.append(0.1)  # Bonus for strong market momentum
            elif market_momentum < 0.3:
                property_variations.append(-0.1)  # Penalty for weak market momentum
            
            # Apply variations with moderate scaling to give meaningful boosts/penalties
            variation_factor = sum(property_variations) * 0.35  # was 0.2 (increase influence)
            
            # Add variations
            final_score = max(0, min(1, final_score + variation_factor))
            
            # -------------------------------------------------------------
            # Stretch distribution to improve nuance
            # Many properties were clustering between 0.3-0.6.  Applying a
            # conservative square-root based transformation increases
            # separation near the upper-mid range without letting poor
            # properties inflate past 0.7-0.8.  This keeps the system
            # conservative but provides more visible differentiation.
            # -------------------------------------------------------------
            import math
            final_score = 0.5 * final_score + 0.5 * math.sqrt(final_score)
            
            # Ensure within bounds after transformation
            final_score = max(0, min(1, final_score))
            
            # Add small random variation to break ties (±2%)
            import random
            random_variation = (random.random() - 0.5) * 0.04
            final_score = max(0, min(1, final_score + random_variation))
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating investment score: {str(e)}")
            return 0.5  # Return neutral score if calculation fails
            
    def _generate_recommendations(
        self,
        property_data: Dict[str, Any],
        market_data: Dict[str, Any],
        base_metrics: Dict[str, float],
        investment_metrics: Dict[str, float],
        risk_metrics: Dict[str, float],
        market_metrics: Dict[str, float],
        investment_score: float
    ) -> List[str]:
        """Generate investment recommendations based on analysis metrics."""
        recommendations = []
        
        # Investment score based recommendations
        if investment_score >= 0.8:
            recommendations.append("High investment potential. Consider quick action.")
        elif investment_score >= 0.6:
            recommendations.append("Good investment opportunity. Worth further investigation.")
        elif investment_score >= 0.4:
            recommendations.append("Moderate investment potential. Consider market conditions.")
        else:
            recommendations.append("Low investment potential. Consider other opportunities.")
        
        # Price metrics based recommendations
        if base_metrics['price_to_median'] > 1.2:
            recommendations.append("Property is priced significantly above market median. Consider negotiating.")
        elif base_metrics['price_to_median'] < 0.8:
            recommendations.append("Property is priced below market median. Good potential for value appreciation.")
        
        # Investment metrics based recommendations
        if investment_metrics['cap_rate'] < 4:
            recommendations.append("Low cap rate. Consider rental market conditions.")
        if investment_metrics['cash_on_cash'] < 8:
            recommendations.append("Low cash on cash return. Review financing options.")
        
        # Risk metrics based recommendations
        if risk_metrics['market_risk'] > 0.7:
            recommendations.append("High market risk. Consider market stability.")
        if risk_metrics['property_risk'] > 0.7:
            recommendations.append("High property risk. Review property condition and maintenance needs.")
        
        return recommendations

    async def get_market_trends(self, location: str) -> Dict[str, Any]:
        """
        Get market trends and analysis for a specific location.
        
        Args:
            location: Location string in format "City, State" or ZIP code
            
        Returns:
            Dictionary containing market trends and analysis
        """
        try:
            logger.info(f"Getting market trends for location: {location}")
            
            # Get market data using the instance variable
            if ',' in location:
                city, state = location.split(',')
                market_data = await self.market_data_service.get_metro_level_metrics(city.strip(), state.strip())
            else:
                market_data = await self.market_data_service.get_zip_level_metrics(location.strip())
            
            if "error" in market_data:
                logger.error(f"Error getting market data: {market_data['error']}")
                return {"error": market_data["error"]}
            
            # Analyze market trends
            market_analysis = self.market_data_service.analyze_market_trends(market_data)
            
            if "error" in market_analysis:
                logger.error(f"Error analyzing market trends: {market_analysis['error']}")
                return {"error": market_analysis["error"]}
            
            logger.info("Successfully retrieved and analyzed market trends")
            return {
                "location": location,
                "market_data": market_data,
                "market_analysis": market_analysis
            }
            
        except Exception as e:
            logger.error(f"Error getting market trends for {location}: {str(e)}")
            return {"error": str(e)}

    async def get_comparable_properties(self, property_id: str) -> List[Dict[str, Any]]:
        """
        Get comparable properties for a given property ID.
        
        Args:
            property_id: The ID of the property to find comparables for
            
        Returns:
            List of comparable property dictionaries
        """
        if not property_id:
            logger.error("get_comparable_properties called with empty property_id")
            return []
            
        logger.info(f"Starting comparable property search for property_id: {property_id}")
        
        try:
            # ---------------------------------------------------------
            # PRIMARY ENGINE – FAISS latent similarity search
            # ---------------------------------------------------------
            from app.services.comparable_property_service import ComparablePropertyService

            try:
                logger.info(f"Attempting FAISS-based comparable property search for {property_id}")
                
                if not hasattr(self, "_comparable_service"):
                    # Lazily initialise and cache on the instance so subsequent
                    # calls reuse the loaded model/index (costly to load).
                    logger.info("Initializing ComparablePropertyService with FAISS index and autoencoder")
                    self._comparable_service = ComparablePropertyService()
                    logger.info("ComparablePropertyService initialized successfully")

                comps = await self._comparable_service.get_comparable_properties(property_id)
                if comps and len(comps) > 0:
                    logger.info(f"FAISS comparable model SUCCESS: returned {len(comps)} comparable properties for {property_id}")
                    for i, comp in enumerate(comps):
                        logger.debug(f"FAISS comparable {i+1}: property_id={comp.get('property_id')}, "
                                   f"similarity_score={comp.get('similarity_score', 0):.4f}")
                    return comps
                else:
                    logger.warning(f"FAISS comparable model returned empty results for property {property_id}. "
                                 f"This could be due to:")
                    logger.warning("  1. Property details could not be fetched for encoding")
                    logger.warning("  2. All similar properties in FAISS index failed to fetch details")  
                    logger.warning("  3. No similar properties found in the trained model")
                    logger.warning("Falling back to rules-based comparable property search...")
                    
            except Exception as faiss_err:
                # Log detailed error information and fall back to rules-based method
                logger.error(f"FAISS comparable model FAILED for property {property_id} due to: {str(faiss_err)}")
                logger.error("Possible causes:")
                logger.error("  1. FAISS index or autoencoder model files missing/corrupted")
                logger.error("  2. Property feature encoding failed")
                logger.error("  3. FAISS search operation failed")


            
            logger.info(f"Starting rules-based comparable property search for {property_id}")

            # Get the target property data
            logger.debug(f"Fetching target property data for {property_id}")
            property_data = await self.data_collector.get_property_by_id(property_id)
            if not property_data:
                logger.error(f"Target property {property_id} not found - cannot generate comparable properties")
                raise ValueError(f"Property {property_id} not found")

            logger.info(f"Target property details: city={property_data.get('city')}, "
                       f"state={property_data.get('state')}, price=${property_data.get('price', 0):,}, "
                       f"beds={property_data.get('beds')}, baths={property_data.get('baths')}")

            # Get market data for the location (kept for parity with previous
            # implementation – may be useful for future extensions).
            location = f"{property_data.get('city', '')}, {property_data.get('state', '')}"
            logger.debug(f"Fetching market trends for location: {location}")
            _ = await self.market_data_service.get_market_trends(location)

            # Search for comparable properties using DataCollector filters
            search_params = {
                'city': property_data.get('city'),
                'state_code': property_data.get('state'),
                'zip_code': property_data.get('zip_code'),
                'min_price': int(property_data.get('price', 0) * 0.8),  # 20% below
                'max_price': int(property_data.get('price', 0) * 1.2),  # 20% above
                'beds': property_data.get('beds'),
                'baths': property_data.get('baths'),
                'property_type': property_data.get('property_type')
            }
            
            logger.info(f"Searching for comparable properties with criteria: {search_params}")
            comps = await self.data_collector.get_properties_by_location(**search_params)
            
            logger.info(f"Found {len(comps)} potential comparable properties before filtering")

            # Filter out the target property
            comps_before_filter = len(comps)
            comps = [comp for comp in comps if comp.get('property_id') != property_id]
            logger.debug(f"Filtered out target property, {len(comps)} comparable properties remaining")

            if not comps:
                logger.warning(f"No comparable properties found for {property_id} using rules-based search")
                logger.warning("Consider widening search criteria or checking if target property exists in the area")
                return []

            # Calculate rule-based similarity scores
            logger.debug(f"Calculating similarity scores for {len(comps)} comparable properties")
            for comp in comps:
                comp['similarity_score'] = self._calculate_similarity_score(property_data, comp)

            # Sort by similarity score and return top 5
            comps.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            top_comps = comps[:5]
            
            logger.info(f"Rules-based comparable property search SUCCESS: returning {len(top_comps)} properties for {property_id}")
            for i, comp in enumerate(top_comps):
                logger.debug(f"Rules-based comparable {i+1}: property_id={comp.get('property_id')}, "
                           f"similarity_score={comp.get('similarity_score', 0):.4f}")
            
            return top_comps

        except Exception as e:
            logger.error(f"FAISS comparable property search FAILED for {property_id}: {str(e)}")
            logger.error("This indicates a serious issue with FAISS setup or property data retrieval")
            logger.warning("Skipping Realtor fallback for similar properties (FAISS debugging phase)")
            return []  # Return empty list instead of raising exception

    def _calculate_similarity_score(self, target: Dict[str, Any], comp: Dict[str, Any]) -> float:
        """
        Calculate similarity score between target property and comparable.
        
        Args:
            target: Target property data
            comp: Comparable property data
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Price similarity (30% weight)
            price_diff = abs(target.get('price', 0) - comp.get('price', 0))
            price_similarity = 1 - (price_diff / max(target.get('price', 1), comp.get('price', 1)))
            
            # Size similarity (20% weight)
            size_diff = abs(target.get('square_feet', 0) - comp.get('square_feet', 0))
            size_similarity = 1 - (size_diff / max(target.get('square_feet', 1), comp.get('square_feet', 1)))
            
            # Bed/bath similarity (20% weight)
            bed_diff = abs(target.get('beds', 0) - comp.get('beds', 0))
            bath_diff = abs(target.get('baths', 0) - comp.get('baths', 0))
            bed_similarity = 1 - (bed_diff / max(target.get('beds', 1), comp.get('beds', 1)))
            bath_similarity = 1 - (bath_diff / max(target.get('baths', 1), comp.get('baths', 1)))
            
            # Location similarity (30% weight)
            # Calculate distance between properties using lat/long
            target_lat = target.get('latitude', 0)
            target_lon = target.get('longitude', 0)
            comp_lat = comp.get('latitude', 0)
            comp_lon = comp.get('longitude', 0)
            
            # Simple distance calculation (can be improved with proper geodetic calculations)
            lat_diff = abs(target_lat - comp_lat)
            lon_diff = abs(target_lon - comp_lon)
            location_similarity = 1 - min((lat_diff + lon_diff) / 0.1, 1)  # 0.1 degrees is roughly 11km
            
            # Calculate weighted average
            similarity_score = (
                price_similarity * 0.3 +
                size_similarity * 0.2 +
                (bed_similarity + bath_similarity) / 2 * 0.2 +
                location_similarity * 0.3
            )
            
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error calculating similarity score: {str(e)}")
            return 0.0

    def generate_predictions(
        self,
        property_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate price predictions and market forecasts for a property.
        
        Args:
            property_data: Dictionary containing property features
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # ----------------------------------------------------------
            # Normalise / flatten *market_data* in case the caller passed the
            # raw object returned by ``market_data_service.get_market_trends``
            # (which nests the actual numbers under 'market_data' →
            # 'current_metrics').  We extract the most relevant fields to avoid
            # KeyErrors like 'median_list_price'.
            # ----------------------------------------------------------
            if 'market_data' in market_data:
                md_current = market_data.get('market_data', {}).get('current_metrics', {})
                # Use same keys expected downstream
                flat = {
                    'median_list_price': md_current.get('median_price') or md_current.get('median_listing_price'),
                    'median_dom': md_current.get('avg_days_on_market'),
                }
                market_data = {**flat, **market_data}  # retain other keys

            # Ensure graceful defaults
            market_data.setdefault('median_list_price', market_data.get('median_price', 0))

            # Calculate base metrics
            base_metrics = self._calculate_base_metrics(property_data, market_data)
            
            # Calculate market metrics
            market_metrics = self._calculate_market_metrics(market_data)
            
            # Generate price predictions
            current_price = property_data["price"]
            market_momentum = market_metrics.get("market_momentum", 0.5)
            market_stability = market_metrics.get("market_stability", 0.5)
            
            # Calculate price changes based on market momentum and stability
            price_change_1y = current_price * (0.05 + (market_momentum - 0.5) * 0.1) * market_stability
            price_change_3y = current_price * (0.15 + (market_momentum - 0.5) * 0.2) * market_stability
            price_change_5y = current_price * (0.25 + (market_momentum - 0.5) * 0.3) * market_stability
            
            # Calculate predicted prices
            predicted_price_1y = current_price + price_change_1y
            predicted_price_3y = current_price + price_change_3y
            predicted_price_5y = current_price + price_change_5y
            
            # Calculate confidence scores
            confidence_1y = market_stability * 0.8
            confidence_3y = market_stability * 0.6
            confidence_5y = market_stability * 0.4
            
            return {
                "current_price": current_price,
                "predictions": {
                    "1_year": {
                        "price": predicted_price_1y,
                        "change": price_change_1y,
                        "confidence": confidence_1y
                    },
                    "3_year": {
                        "price": predicted_price_3y,
                        "change": price_change_3y,
                        "confidence": confidence_3y
                    },
                    "5_year": {
                        "price": predicted_price_5y,
                        "change": price_change_5y,
                        "confidence": confidence_5y
                    }
                },
                "market_metrics": market_metrics,
                "base_metrics": base_metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    # AI-powered insights
    async def generate_insights(self, property_data: Dict[str, Any], analysis_type: str = "full") -> Dict[str, Any]:
        """Generate hybrid rules-based + LLM expert insights for a property."""
        try:
            # Get market data for the property
            location = f"{property_data.get('city', '')}, {property_data.get('state', '')}"
            market_data = await self.market_data_service.get_market_trends(location)
            # Run rules-based analysis
            rules_output = self.analyze_property(property_data, market_data)
            # Get expert insights from LLM
            from app.services.service_manager import ServiceManager
            openai_service = ServiceManager.get_openai_service()
            hybrid_result = await openai_service.generate_property_insights(
                rules_output=rules_output,
                analysis_type=analysis_type
            )
            return hybrid_result
        except Exception as e:
            logger.error(f"Error generating hybrid insights: {str(e)}")
            return {"error": str(e)} 