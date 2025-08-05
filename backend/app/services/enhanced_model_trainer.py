"""
Enhanced model trainer with robust variance checking and automatic retraining.
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split, StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import logging
from pathlib import Path
from datetime import datetime
import joblib
import json
from app.ml_models.model_utils import ModelUtils  

logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Enhanced model trainer with variance monitoring and automatic retraining."""
    
    def __init__(self, model_dir: str = "models", min_prediction_std: float = 0.02):
        # Convert model_dir to absolute path if it's relative
        if not os.path.isabs(model_dir):
            # Get the backend directory path (2 levels up from this file)
            backend_dir = Path(__file__).resolve().parent.parent.parent
            model_dir = os.path.join(backend_dir, model_dir)
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.min_prediction_std = min_prediction_std
        self.metrics = {}
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def train_risk_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all risk models with enhanced variance checking.
        
        Returns:
            Dictionary with training results and model paths
        """
        logger.info("Starting enhanced risk model training")
        
        results = {
            'models_trained': [],
            'training_metrics': {},
            'variance_issues': [],
            'retrained_models': [],
            'model_paths': {}
        }
        
        # Define risk models to train
        risk_models = {
            'market_risk': {'type': 'regression', 'label': 'market_risk_label'},
            'property_risk': {'type': 'regression', 'label': 'property_risk_label'},
            'location_risk': {'type': 'regression', 'label': 'location_risk_label'},
            'overall_risk': {'type': 'regression', 'label': 'overall_risk_label'},
            'market_health': {'type': 'regression', 'label': 'market_health_label'},
            'market_momentum': {'type': 'regression', 'label': 'market_momentum_label'},
            'market_stability': {'type': 'regression', 'label': 'market_stability_label'}
        }
        
        # Prepare features
        feature_columns = [col for col in df.columns if not col.endswith('_label') 
                          and col not in ['property_id', 'data_source', 'month_date', 'cbsa_code', 'cbsa_title']]
        
        X = df[feature_columns].copy()
        
        # Handle categorical columns
        X = self._encode_categorical_columns(X)
        
        # Handle NaN values before training
        X = self._handle_missing_values(X)
        
        # Train each model
        for model_name, config in risk_models.items():
            if config['label'] not in df.columns:
                logger.warning(f"Label {config['label']} not found in dataset, skipping {model_name}")
                continue
                
            logger.info(f"Training {model_name} model...")
            
            y = df[config['label']].copy()
            
            # Train model with variance checking
            model_result = self._train_model_with_variance_check(
                X, y, model_name, config['type']
            )
            
            results['models_trained'].append(model_name)
            results['training_metrics'][model_name] = model_result['metrics']
            results['model_paths'][model_name] = model_result['model_path']
            
            if model_result['had_variance_issue']:
                results['variance_issues'].append(model_name)
            
            if model_result['was_retrained']:
                results['retrained_models'].append(model_name)
        
        # Generate summary
        logger.info(f"Training complete. Models trained: {len(results['models_trained'])}")
        logger.info(f"Models with variance issues: {len(results['variance_issues'])}")
        logger.info(f"Models retrained: {len(results['retrained_models'])}")
        
        return results
    
    def _train_model_with_variance_check(self, 
                                       X: pd.DataFrame, 
                                       y: pd.Series, 
                                       model_name: str,
                                       model_type: str) -> Dict[str, Any]:
        """
        Train a model with comprehensive variance checking and automatic retraining.
        """
        result = {
            'metrics': {},
            'model_path': None,
            'had_variance_issue': False,
            'was_retrained': False,
            'training_attempts': 1
        }
        
        # Check minimum dataset size
        n_samples = len(X)
        if n_samples < 10:
            logger.error(f"Dataset too small for meaningful ML training: {n_samples} samples")
            raise ValueError(f"Need at least 10 samples for training, got {n_samples}")
        
        # Adaptive train/test split based on dataset size
        if n_samples < 20:
            # For very small datasets, use larger training portion
            test_size = max(0.1, 2/n_samples)  # At least 2 samples for test, but not more than 10%
        elif n_samples < 50:
            test_size = 0.15  # 15% for small datasets
        else:
            test_size = 0.2   # Standard 20% for larger datasets
            
        logger.info(f"Using test_size={test_size:.2f} for {n_samples} samples")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Initial training attempt
        model, initial_metrics = self._train_single_model(X_train, y_train, X_test, y_test, model_name, model_type)
        
        # Check prediction variance
        y_pred = model.predict(X_test)
        pred_std = np.std(y_pred)
        
        logger.info(f"{model_name} initial training - prediction std: {pred_std:.6f}")
        
        if pred_std < self.min_prediction_std:
            logger.warning(f"{model_name} has low prediction variance (std={pred_std:.6f}). Attempting to retrain with enhanced parameters.")
            result['had_variance_issue'] = True
            
            # Try enhanced training parameters
            enhanced_model, enhanced_metrics = self._train_enhanced_model(
                X_train, y_train, X_test, y_test, model_name, model_type
            )
            
            enhanced_pred = enhanced_model.predict(X_test)
            enhanced_std = np.std(enhanced_pred)
            
            logger.info(f"{model_name} enhanced training - prediction std: {enhanced_std:.6f}")
            
            if enhanced_std > pred_std:
                logger.info(f"Enhanced model for {model_name} shows improved variance. Using enhanced model.")
                model = enhanced_model
                initial_metrics = enhanced_metrics
                result['was_retrained'] = True
                result['training_attempts'] = 2
            else:
                logger.warning(f"Enhanced model for {model_name} did not improve variance. Using original model.")
        
        # Perform k-fold cross-validation with variance analysis
        cv_results = self._perform_variance_aware_cv(X, y, model_name, model_type)
        
        # Save model with metadata using new format
        from app.ml_models.model_utils import ModelUtils
        import os
        
        # Convert to absolute path
        model_path = os.path.abspath(self.model_dir / f"{model_name}_rf.joblib")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Get feature names from the training data
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Save using new format that includes metadata
        ModelUtils.save_model(
            model=model,
            model_path=model_path,  # Using absolute path
            feature_names=feature_names,
            target_name=model_name
        )
        result['model_path'] = model_path  # Store absolute path
        
        # Combine metrics
        result['metrics'] = {
            **initial_metrics,
            'prediction_std': float(np.std(model.predict(X_test))),
            'cv_results': cv_results
        }
        
        logger.info(f"{model_name} training complete. Final prediction std: {result['metrics']['prediction_std']:.6f}")
        
        return result
    
    def _train_single_model(self, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series, 
                          X_test: pd.DataFrame, 
                          y_test: pd.Series,
                          model_name: str,
                          model_type: str) -> Tuple[Any, Dict[str, float]]:
        """Train a single model with standard parameters."""
        
        if model_type == 'regression':
            # Enhanced RandomForest parameters for better variance
            model = RandomForestRegressor(
                n_estimators=200,  # Increased from typical 100
                max_depth=15,      # Deeper trees
                min_samples_split=5,  # Smaller splits
                min_samples_leaf=2,   # Smaller leaves
                max_features='sqrt',  # Feature subsampling
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        
        if model_type == 'regression':
            metrics = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted'))
            }
        
        return model, metrics
    
    def _train_enhanced_model(self, 
                            X_train: pd.DataFrame, 
                            y_train: pd.Series, 
                            X_test: pd.DataFrame, 
                            y_test: pd.Series,
                            model_name: str,
                            model_type: str) -> Tuple[Any, Dict[str, float]]:
        """Train model with enhanced parameters to increase variance."""
        
        if model_type == 'regression':
            # More aggressive parameters for variance
            model = RandomForestRegressor(
                n_estimators=300,     # Even more trees
                max_depth=20,         # Deeper trees
                min_samples_split=2,  # Minimum splits
                min_samples_leaf=1,   # Minimum leaves
                max_features='sqrt',  # Feature subsampling
                bootstrap=True,       # Bootstrap sampling
                max_samples=0.8,      # Subsample training data
                random_state=None,    # Remove fixed random state for more variance
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                max_samples=0.8,
                random_state=None,
                n_jobs=-1
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        
        if model_type == 'regression':
            metrics = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted'))
            }
        
        return model, metrics
    
    def _perform_variance_aware_cv(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series, 
                                 model_name: str,
                                 model_type: str) -> Dict[str, Any]:
        """Perform cross-validation with variance analysis."""
        
        # Skip TimeSeriesSplit as it was overridden anyway
        
        fold_predictions = []
        # Calculate number of folds based on dataset size
        n_samples = len(X)
        if n_samples < 10:
            n_splits = 2  # Minimum splits for very small datasets
        elif n_samples < 20:
            n_splits = 3
        elif n_samples < 50:
            n_splits = 4
        else:
            n_splits = 5  # Maximum splits for larger datasets
            
        logger.info(f"Using {n_splits} folds for {n_samples} samples")
        
        fold_scores = []
        fold_variances = []
        fold_predictions = []
        
        # Use KFold for regression, StratifiedKFold for classification
        if model_type == 'regression':
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            model = RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_split=2,  # Reduced min_samples_split
                min_samples_leaf=1, random_state=42, n_jobs=-1  # Reduced min_samples_leaf
            )
            scoring = 'r2'
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=2,
                min_samples_leaf=1, random_state=42, n_jobs=-1
            )
            scoring = 'accuracy'
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold model
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Predict and analyze
            fold_pred = fold_model.predict(X_fold_val)
            fold_predictions.extend(fold_pred)
            
            # Calculate fold score
            if model_type == 'regression':
                fold_score = r2_score(y_fold_val, fold_pred)
            else:
                from sklearn.metrics import accuracy_score
                fold_score = accuracy_score(y_fold_val, fold_pred)
            
            fold_scores.append(fold_score)
            
            # Calculate fold variance
            fold_variance = np.std(fold_pred)
            fold_variances.append(fold_variance)
            
            logger.debug(f"{model_name} fold {fold+1}: score={fold_score:.4f}, pred_std={fold_variance:.6f}")
        
        # Overall CV results
        cv_results = {
            'mean_score': float(np.mean(fold_scores)),
            'std_score': float(np.std(fold_scores)),
            'mean_prediction_variance': float(np.mean(fold_variances)),
            'std_prediction_variance': float(np.std(fold_variances)),
            'overall_prediction_variance': float(np.std(fold_predictions)),
            'fold_scores': [float(s) for s in fold_scores],
            'fold_variances': [float(v) for v in fold_variances]
        }
        
        logger.info(f"{model_name} CV results: mean_score={cv_results['mean_score']:.4f}, "
                   f"mean_pred_variance={cv_results['mean_prediction_variance']:.6f}")
        
        return cv_results
    
    def _encode_categorical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns using LabelEncoder."""
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                logger.info(f"Encoding categorical column: {col}")
                le = LabelEncoder()
                
                # Handle missing values
                X_encoded[col] = X_encoded[col].fillna('unknown')
                
                # Fit and transform
                X_encoded[col] = le.fit_transform(X_encoded[col])
                
                # Save encoder for later use
                encoder_path = self.model_dir / f"{col}_encoder.joblib"
                joblib.dump(le, encoder_path)
                logger.debug(f"Saved encoder for {col} to {encoder_path}")
        
        return X_encoded
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature data."""
        if X.isnull().sum().sum() == 0:
            return X  # No missing values
            
        logger.info(f"Found missing values in {X.isnull().sum().sum()} cells. Imputing...")
        
        # Simple imputation strategy
        X_filled = X.copy()
        
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    # Use median for numeric columns
                    fill_value = X[col].median()
                    if pd.isna(fill_value):  # If median is still NaN, use 0
                        fill_value = 0
                else:
                    # Use mode for categorical columns  
                    fill_value = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown'
                
                X_filled[col] = X[col].fillna(fill_value)
                logger.debug(f"Filled {X[col].isnull().sum()} missing values in {col} with {fill_value}")
        
        logger.info("Missing value imputation completed")
        return X_filled
    
    def diagnose_constant_predictions(self, model_path: str, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Diagnose why a model is producing constant predictions."""
        
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return {'error': f"Could not load model: {e}"}
        
        # Make predictions
        predictions = model.predict(X_test)
        pred_std = np.std(predictions)
        pred_unique = len(np.unique(predictions))
        
        diagnosis = {
            'prediction_std': float(pred_std),
            'unique_predictions': int(pred_unique),
            'prediction_range': float(np.max(predictions) - np.min(predictions)),
            'is_constant': pred_std < self.min_prediction_std,
            'feature_importance': {},
            'recommendations': []
        }
        
        # Analyze feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns.tolist()
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance
            diagnosis['feature_importance'] = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Check if any features dominate
            max_importance = max(model.feature_importances_)
            if max_importance > 0.8:
                dominant_feature = feature_names[np.argmax(model.feature_importances_)]
                diagnosis['recommendations'].append(f"Single feature '{dominant_feature}' dominates (importance: {max_importance:.3f})")
        
        # Generate recommendations
        if diagnosis['is_constant']:
            diagnosis['recommendations'].extend([
                "Increase model complexity (more trees, deeper depth)",
                "Reduce min_samples_leaf and min_samples_split",
                "Check for data leakage or constant features",
                "Ensure sufficient label variability in training data",
                "Consider feature engineering or selection"
            ])
        
        return diagnosis
    
    def retrain_model_with_diagnosis(self, 
                                   model_name: str, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Retrain a model based on diagnostic results."""
        
        logger.info(f"Retraining {model_name} based on diagnosis")
        
        # Adjust parameters based on diagnosis
        if diagnosis.get('is_constant', False):
            # Use very aggressive parameters
            model = RandomForestRegressor(
                n_estimators=500,      # Many trees
                max_depth=None,        # No depth limit
                min_samples_split=2,   # Minimum splits
                min_samples_leaf=1,    # Minimum leaves
                max_features='log2',   # Different feature sampling
                bootstrap=True,
                max_samples=0.7,       # Subsample for variance
                random_state=None,     # No fixed seed
                n_jobs=-1
            )
        else:
            # Standard enhanced parameters
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        pred_std = np.std(y_pred)
        
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'prediction_std': float(pred_std),
            'improved_variance': pred_std > diagnosis.get('prediction_std', 0)
        }
        
        # Save retrained model with metadata
        model_path = self.model_dir / f"{model_name}_retrained_rf.joblib"
        
        # Get feature names from the training data
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Save using new format that includes metadata
        ModelUtils.save_model(
            model=model,
            model_path=str(model_path),  # Convert Path to string
            feature_names=feature_names,
            target_name=f"{model_name}_retrained"
        )
        
        logger.info(f"Retrained {model_name}: prediction_std={pred_std:.6f}, improved={metrics['improved_variance']}")
        
        return {
            'model_path': str(model_path),
            'metrics': metrics,
            'diagnosis_addressed': diagnosis.get('is_constant', False) and pred_std > self.min_prediction_std
        }