import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class ModelUtils:
    """Utility functions for ML model operations."""
    
    @staticmethod
    def load_clean_dataset(file_path: str) -> pd.DataFrame:
        """
        Load the clean training dataset and validate its structure.
        
        Args:
            file_path: Path to the clean training dataset CSV
            
        Returns:
            Loaded and validated DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Validate required columns exist
            required_label_cols = [
                'market_risk_label', 'property_risk_label', 'location_risk_label',
                'overall_risk_label', 'market_health_label', 'market_momentum_label',
                'market_stability_label'
            ]
            
            missing_labels = [col for col in required_label_cols if col not in df.columns]
            if missing_labels:
                raise ValueError(f"Missing required label columns: {missing_labels}")
            
            logger.info(f"Dataset validation passed. Found {len(required_label_cols)} target labels.")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    @staticmethod
    def prepare_features_and_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Prepare feature matrix (X) and target vectors (y) from the dataset.
        
        Args:
            df: Clean training dataset
            
        Returns:
            Tuple of (features_df, targets_dict)
        """
        # Define columns to exclude from features
        exclude_cols = [
            # Labels
            'market_risk_label', 'property_risk_label', 'location_risk_label',
            'overall_risk_label', 'market_health_label', 'market_momentum_label',
            'market_stability_label',
            # Metadata
            'property_id', 'data_source', 'month_date', 'cbsa_code', 'cbsa_title'
        ]
        
        # Extract features (everything except labels and metadata)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # Extract targets
        target_cols = [
            'market_risk_label', 'property_risk_label', 'location_risk_label',
            'overall_risk_label', 'market_health_label', 'market_momentum_label',
            'market_stability_label'
        ]
        
        y_dict = {}
        for target in target_cols:
            if target in df.columns:
                y_dict[target] = df[target].copy()
        
        logger.info(f"Prepared features: {len(feature_cols)} columns")
        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Prepared targets: {len(y_dict)} labels")
        
        # Handle missing values with improved imputation
        if X.isnull().sum().sum() > 0:
            null_counts = X.isnull().sum()
            logger.warning(f"Found missing values in features: {dict(null_counts[null_counts > 0])}")
            
            # Use median for numeric columns, mode for categorical
            from sklearn.impute import SimpleImputer
            import numpy as np
            
            # Separate numeric and categorical columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='median')
                X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
                
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
                
            logger.info("Successfully imputed missing values")
        
        return X, y_dict
    
    @staticmethod
    def split_data(X: pd.DataFrame, y_dict: Dict[str, pd.Series], 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of target vectors
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train_dict, y_test_dict)
        """
        # Use the first target for stratification (to ensure consistent splits)
        first_target = list(y_dict.values())[0]
        
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        y_train_dict = {}
        y_test_dict = {}
        
        for target_name, y in y_dict.items():
            y_train = y.loc[X_train.index]
            y_test = y.loc[X_test.index]
            y_train_dict[target_name] = y_train
            y_test_dict[target_name] = y_test
        
        logger.info(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples")
        return X_train, X_test, y_train_dict, y_test_dict
    
    @staticmethod
    def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                           target_name: str, **rf_params) -> RandomForestRegressor:
        """
        Train a Random Forest Regressor for a specific target.
        
        Args:
            X_train: Training features
            y_train: Training target
            target_name: Name of the target for logging
            **rf_params: Additional parameters for RandomForestRegressor
            
        Returns:
            Trained RandomForestRegressor model
        """
        # Enhanced Random Forest parameters for better performance
        default_params = {
            'n_estimators': 300,    # More trees for better performance
            'max_depth': 20,        # Deeper trees to capture complex patterns
            'min_samples_split': 2, # More granular splits
            'min_samples_leaf': 1,  # Allow more detailed leaf nodes
            'max_features': 'sqrt', # Feature subsampling for generalization
            'bootstrap': True,      # Enable bootstrap sampling
            'oob_score': True,     # Out-of-bag scoring for evaluation
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(rf_params)
        
        logger.info(f"Training Random Forest for {target_name} with params: {default_params}")
        
        # Add comprehensive data quality checks before training
        if len(X_train) < 10:
            logger.error(f"Insufficient training data for {target_name}: {len(X_train)} samples")
            raise ValueError(f"Need at least 10 samples to train {target_name}")
        
        # Check for NaN values in training data
        if X_train.isnull().sum().sum() > 0:
            logger.error(f"Training data contains NaN values for {target_name}")
            raise ValueError(f"Training data for {target_name} contains NaN values. Please clean the data first.")
        
        if y_train.isnull().sum() > 0:
            logger.error(f"Target data contains NaN values for {target_name}")
            raise ValueError(f"Target data for {target_name} contains NaN values. Please clean the data first.")
        
        if y_train.std() < 0.001:
            logger.warning(f"Very low target variance for {target_name}: std={y_train.std():.6f}")
        
        model = RandomForestRegressor(**default_params)
        model.fit(X_train, y_train)
        
        # Log out-of-bag score if available
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            logger.info(f"Out-of-bag score for {target_name}: {model.oob_score_:.4f}")
        
        logger.info(f"Random Forest training completed for {target_name}")
        return model
    
    @staticmethod
    def normalize_predictions(y_pred: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Normalize predictions to a target range using min-max scaling.
        
        Args:
            y_pred: Raw predictions
            target_range: Desired output range (min, max)
            
        Returns:
            Normalized predictions
        """
        if len(y_pred) == 0:
            return y_pred
        
        # Min-max scaling to target range
        pred_min, pred_max = y_pred.min(), y_pred.max()
        
        if pred_max == pred_min:
            # If all predictions are the same, return middle of target range
            return np.full_like(y_pred, (target_range[0] + target_range[1]) / 2)
        
        # Scale to [0, 1] first
        normalized = (y_pred - pred_min) / (pred_max - pred_min)
        
        # Scale to target range
        target_min, target_max = target_range
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled
    
    @staticmethod
    def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series, 
                      target_name: str, normalize_output: bool = True) -> Dict[str, float]:
        """
        Evaluate a trained model and return metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            target_name: Name of the target for logging
            normalize_output: Whether to normalize predictions to [0, 1]
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Normalize predictions if requested
        if normalize_output:
            y_pred = ModelUtils.normalize_predictions(y_pred, (0.0, 1.0))
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pred_min': y_pred.min(),
            'pred_max': y_pred.max(),
            'pred_mean': y_pred.mean()
        }
        
        logger.info(f"Evaluation for {target_name}:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        
        return metrics
    
    @staticmethod
    def save_model(model: RandomForestRegressor, model_path: str, 
                   feature_names: List[str], target_name: str) -> None:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained model
            model_path: Path to save the model
            feature_names: List of feature column names
            target_name: Name of the target variable
        """
        from datetime import datetime
        
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'target_name': target_name,
            'model_type': 'RandomForestRegressor',
            'saved_at': datetime.now().isoformat(),
            'n_features': len(feature_names),
            'model_params': model.get_params(),
            'oob_score': getattr(model, 'oob_score_', None)
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model with metadata saved to {model_path}")
        logger.info(f"Features: {len(feature_names)}, Target: {target_name}, OOB Score: {model_data['oob_score']}")
    
    @staticmethod
    def load_model(model_path: str) -> Dict[str, Any]:
        """
        Load a saved model with metadata.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary containing model and metadata
        """
        model_data = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model_data
    
    @staticmethod
    def get_feature_importance(model: RandomForestRegressor, feature_names: List[str], 
                              top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from a trained Random Forest model.
        
        Args:
            model: Trained RandomForestRegressor
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    @staticmethod
    def create_prediction_comparison(y_true: pd.Series, y_pred: np.ndarray, 
                                   target_name: str, n_samples: int = 10) -> pd.DataFrame:
        """
        Create a comparison DataFrame of true vs predicted values.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            target_name: Name of the target
            n_samples: Number of samples to show
            
        Returns:
            DataFrame comparing predictions
        """
        comparison_df = pd.DataFrame({
            f'{target_name}_true': y_true.iloc[:n_samples],
            f'{target_name}_pred': y_pred[:n_samples],
            'difference': y_true.iloc[:n_samples] - y_pred[:n_samples],
            'abs_difference': np.abs(y_true.iloc[:n_samples] - y_pred[:n_samples])
        })
        
        return comparison_df 