import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path
from datetime import datetime
import logging
from sklearn.metrics import mean_squared_error, r2_score
import json
from app.core.config import settings
from app.core.logging import loggers
from sklearn.impute import SimpleImputer

logger = loggers['ml']

class MLPredictor:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize ML predictor with models and scalers."""
        if self._initialized:
            return
            
        self.model_dir = Path(settings.MODEL_DIR)
        self.value_model = None
        self.investment_model = None
        self.scalers = {}
        self.imputers = {}
        self.feature_names = None
        self._initialized = True
        
    def _ensure_models_loaded(self):
        """Ensure models are loaded before use."""
        if not self._models_loaded:
            self._load_models()
            self._models_loaded = True
            
    def _load_models(self):
        """Load trained models and their associated scalers."""
        try:
            # Find latest model version
            model_versions = sorted([d for d in self.model_dir.iterdir() if d.is_dir()])
            if not model_versions:
                raise FileNotFoundError("No trained models found")
            
            latest_version = model_versions[-1]
            logger.info(f"Loading models from version: {latest_version.name}")
            
            # Load metadata
            with open(latest_version / "metadata.json", 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
            
            # Load value model
            value_model_path = latest_version / "property_value_model.json"
            if not value_model_path.exists():
                raise FileNotFoundError(f"Value model not found at {value_model_path}")
            self.value_model = xgb.Booster()
            self.value_model.load_model(str(value_model_path))
            
            # Load investment model
            investment_model_path = latest_version / "investment_model.json"
            if not investment_model_path.exists():
                raise FileNotFoundError(f"Investment model not found at {investment_model_path}")
            self.investment_model = xgb.Booster()
            self.investment_model.load_model(str(investment_model_path))
            
            # Load scalers
            scaler_path = latest_version / "scalers.joblib"
            if scaler_path.exists():
                self.scalers = joblib.load(scaler_path)
            
            # Load imputers
            imputer_path = latest_version / "imputers.joblib"
            if imputer_path.exists():
                self.imputers = joblib.load(imputer_path)
            
            logger.info("Models and scalers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
            
    def predict(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions for a property.
        
        Args:
            property_data: Dictionary containing property features
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            self._ensure_models_loaded()
            
            # Convert to DataFrame
            df = pd.DataFrame([property_data])
            
            # Validate features
            self._validate_features(df)
            
            # Preprocess features
            X = self._preprocess_features(df)
            
            # Make predictions
            value_pred = self._predict_value(X)
            investment_pred = self._predict_investment(X)
            
            # Calculate confidence scores
            value_confidence = self._calculate_confidence(X, 'value')
            investment_confidence = self._calculate_confidence(X, 'investment')
            
            return {
                'predicted_value': float(value_pred),
                'predicted_investment_score': float(investment_pred),
                'value_confidence': float(value_confidence),
                'investment_confidence': float(investment_confidence),
                'feature_importance': self._get_feature_importance()
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def _validate_features(self, df: pd.DataFrame):
        """Validate input features."""
        try:
            # Check required features
            missing_features = set(settings.REQUIRED_FEATURES) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Check data types
            for feature in settings.REQUIRED_FEATURES:
                if not np.issubdtype(df[feature].dtype, np.number):
                    raise ValueError(f"Feature {feature} must be numeric")
            
            # Check for infinite values
            inf_values = np.isinf(df[settings.REQUIRED_FEATURES].select_dtypes(include=np.number)).sum()
            if inf_values.any():
                raise ValueError(f"Infinite values found in features: {inf_values[inf_values > 0]}")
            
        except Exception as e:
            logger.error(f"Feature validation failed: {str(e)}")
            raise
            
    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for prediction."""
        try:
            # Handle missing values
            for feature in settings.REQUIRED_FEATURES:
                if feature in self.imputers:
                    df[feature] = self.imputers[feature].transform(df[[feature]])
                else:
                    # Use median imputation as fallback
                    df[feature] = df[feature].fillna(df[feature].median())
            
            # Scale features
            for feature in settings.REQUIRED_FEATURES:
                if feature in self.scalers:
                    df[feature] = self.scalers[feature].transform(df[[feature]])
                else:
                    # Use robust scaling as fallback
                    scaler = RobustScaler()
                    df[feature] = scaler.fit_transform(df[[feature]])
            
            # Create derived features
            df = self._create_derived_features(df)
            
            return df[settings.REQUIRED_FEATURES]
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
            
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw data."""
        try:
            # Price per square foot
            df['price_per_sqft'] = df['price'] / df['square_feet']
            
            # Bedrooms to bathrooms ratio
            df['beds_baths_ratio'] = df['bedrooms'] / df['bathrooms']
            
            # Property age
            current_year = pd.Timestamp.now().year
            df['property_age'] = current_year - df['year_built']
            
            # Square feet per bedroom
            df['sqft_per_bed'] = df['square_feet'] / df['bedrooms']
            
            # Location score (weighted average of walk and transit scores)
            df['location_score'] = (df['walk_score'] * 0.6 + df['transit_score'] * 0.4) / 100
            
            # Market health score
            df['market_health'] = (
                (df['population_growth'] * 0.3) +
                (df['median_income'] * 0.3) +
                (df['housing_supply'] * 0.2) +
                (df['unemployment_rate'] * 0.2)
            )
            
            # Investment potential score
            df['investment_potential'] = (
                (df['cap_rate'] * 0.4) +
                (df['cash_on_cash_return'] * 0.4) +
                (df['rent_to_price_ratio'] * 0.2)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating derived features: {str(e)}")
            raise
            
    def _predict_value(self, X: pd.DataFrame) -> float:
        """Predict property value."""
        try:
            dmatrix = xgb.DMatrix(X)
            return self.value_model.predict(dmatrix)[0]
        except Exception as e:
            logger.error(f"Error predicting value: {str(e)}")
            raise
            
    def _predict_investment(self, X: pd.DataFrame) -> float:
        """Predict investment score."""
        try:
            dmatrix = xgb.DMatrix(X)
            return self.investment_model.predict(dmatrix)[0]
        except Exception as e:
            logger.error(f"Error predicting investment score: {str(e)}")
            raise
            
    def _calculate_confidence(self, X: pd.DataFrame, model_type: str) -> float:
        """Calculate prediction confidence score."""
        try:
            if model_type == 'value':
                model = self.value_model
            else:
                model = self.investment_model
                
            # Get feature importance
            importance = model.get_score(importance_type='gain')
            
            # Calculate weighted feature coverage
            coverage = 0
            total_importance = sum(importance.values())
            
            for feature, imp in importance.items():
                if feature in X.columns:
                    # Check if feature value is within expected range
                    if feature in self.scalers:
                        scaled_value = X[feature].iloc[0]
                        if -3 <= scaled_value <= 3:  # Within 3 standard deviations
                            coverage += imp / total_importance
            
            return coverage
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            raise
            
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        try:
            value_importance = self.value_model.get_score(importance_type='gain')
            investment_importance = self.investment_model.get_score(importance_type='gain')
            
            # Normalize importance scores
            def normalize_importance(importance_dict):
                total = sum(importance_dict.values())
                return {k: v/total for k, v in importance_dict.items()}
            
            return {
                'value_model': normalize_importance(value_importance),
                'investment_model': normalize_importance(investment_importance)
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train the ML models with proper evaluation and logging."""
        try:
            # Validate required features
            missing_features = set(self.feature_names) - set(training_data.columns)
            if missing_features:
                raise ValueError(f"Missing required features in training data: {missing_features}")
            
            # Split data into features and targets
            X = training_data.drop(['price', 'roi'], axis=1)
            y_value = training_data['price']
            y_investment = training_data['roi']
            
            # Split into train/test sets
            X_train, X_test, y_value_train, y_value_test = train_test_split(
                X, y_value, test_size=0.2, random_state=42
            )
            _, _, y_investment_train, y_investment_test = train_test_split(
                X, y_investment, test_size=0.2, random_state=42
            )
            
            # Fit scaler
            self.scalers = {feature: StandardScaler() for feature in X.columns}
            X_train_scaled = pd.DataFrame({feature: self.scalers[feature].fit_transform(X_train[[feature]]) for feature in X.columns})
            X_test_scaled = pd.DataFrame({feature: self.scalers[feature].transform(X_test[[feature]]) for feature in X.columns})
            
            # Train value model
            self.value_model.fit(X_train_scaled, y_value_train)
            
            # Train investment model
            self.investment_model.fit(X_train_scaled, y_investment_train)
            
            # Calculate evaluation metrics
            value_pred = self.value_model.predict(X_test_scaled)
            investment_pred = self.investment_model.predict(X_test_scaled)
            
            metrics = {
                'property_value': {
                    'mse': mean_squared_error(y_value_test, value_pred),
                    'r2': r2_score(y_value_test, value_pred)
                },
                'investment': {
                    'mse': mean_squared_error(y_investment_test, investment_pred),
                    'r2': r2_score(y_investment_test, investment_pred)
                }
            }
            
            # Save models and scalers
            self._save_models()
            self._save_scalers()
            self._save_metrics(metrics)
            
            logger.info(f"Model training completed with metrics: {metrics}")
            
            return {
                'status': 'success',
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def _save_models(self):
        """Save trained models to disk."""
        try:
            self.value_model.save_model(str(self.model_dir / "property_value_model.json"))
            self.investment_model.save_model(str(self.model_dir / "investment_model.json"))
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def _save_scalers(self):
        """Save feature scalers to disk."""
        try:
            joblib.dump(self.scalers, self.model_dir / "scalers.joblib")
            logger.info("Scalers saved successfully")
        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
            raise

    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save model evaluation metrics to disk."""
        try:
            metrics_path = self.model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

    def predict_property_value(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict property value and provide confidence metrics."""
        try:
            features = self._prepare_features(property_data, market_data)
            prediction = self.value_model.predict(features)[0]
            
            # Calculate prediction confidence (using model's feature importance)
            feature_importance = self.value_model.get_score(importance_type='gain')
            confidence = np.mean(feature_importance.values())
            
            return {
                'predicted_value': float(prediction),
                'confidence': float(confidence),
                'features_used': self.feature_names,
                'feature_importance': dict(zip(self.feature_names, feature_importance.values()))
            }
        except Exception as e:
            logger.error(f"Error predicting property value: {str(e)}")
            raise

    def score_investment_opportunity(self, property_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score investment opportunity and provide detailed analysis."""
        try:
            features = self._prepare_features(property_data, market_data)
            score = self.investment_model.predict(features)[0]
            
            # Calculate investment metrics
            price = float(property_data.get('price', 0))
            sqft = float(property_data.get('sqft', 0))
            market_price = float(market_data.get('median_list_price', 0))
            
            metrics = {
                'investment_score': float(score),
                'price_to_market_ratio': price / market_price if market_price > 0 else 0,
                'price_per_sqft': price / sqft if sqft > 0 else 0,
                'market_price_per_sqft': float(market_data.get('median_price_per_sqft', 0)),
                'days_on_market': float(market_data.get('median_dom', 0))
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            return {
                'metrics': metrics,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"Error scoring investment opportunity: {str(e)}")
            raise

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate investment recommendations based on metrics."""
        recommendations = []
        
        # Price analysis
        if metrics['price_to_market_ratio'] > 1.1:
            recommendations.append("Property is priced above market average. Consider negotiating.")
        elif metrics['price_to_market_ratio'] < 0.9:
            recommendations.append("Property is priced below market average. Good potential for value appreciation.")
        
        # Price per sqft analysis
        if metrics['price_per_sqft'] > metrics['market_price_per_sqft'] * 1.1:
            recommendations.append("Price per square foot is above market average. Verify property condition and upgrades.")
        elif metrics['price_per_sqft'] < metrics['market_price_per_sqft'] * 0.9:
            recommendations.append("Price per square foot is below market average. Good potential for value-add opportunities.")
        
        # Days on market analysis
        if metrics['days_on_market'] > 30:
            recommendations.append("Property has been on market for over 30 days. Investigate potential issues.")
        
        # Investment score analysis
        if metrics['investment_score'] > 0.7:
            recommendations.append("High investment potential. Consider quick action.")
        elif metrics['investment_score'] < 0.3:
            recommendations.append("Low investment potential. Consider other opportunities.")
        
        return recommendations 