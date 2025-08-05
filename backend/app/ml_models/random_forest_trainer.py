import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor

from .model_utils import ModelUtils

logger = logging.getLogger(__name__)

class RandomForestTrainer:
    """
    Main trainer class for Random Forest models for all risk and market metrics.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the trainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Target metrics to train models for
        self.target_metrics = [
            'market_risk_label',
            'property_risk_label', 
            'location_risk_label',
            'overall_risk_label',
            'market_health_label',
            'market_momentum_label',
            'market_stability_label'
        ]
        
        # Store trained models and metadata
        self.trained_models = {}
        self.feature_names = []
        self.evaluation_results = {}
        
    def train_all_models(self, dataset_path: str, test_size: float = 0.2, 
                        **rf_params) -> Dict[str, Any]:
        """
        Train Random Forest models for all target metrics.
        
        Args:
            dataset_path: Path to the clean training dataset
            test_size: Proportion of data to use for testing
            **rf_params: Additional parameters for RandomForestRegressor
            
        Returns:
            Dictionary containing training results and evaluation metrics
        """
        logger.info("Starting Random Forest training for all metrics...")
        
        # Load and prepare data
        df = ModelUtils.load_clean_dataset(dataset_path)
        X, y_dict = ModelUtils.prepare_features_and_targets(df)
        
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train_dict, y_test_dict = ModelUtils.split_data(
            X, y_dict, test_size=test_size
        )
        
        # Train models for each target metric
        training_results = {
            'models_trained': [],
            'evaluation_metrics': {},
            'feature_importance': {},
            'prediction_comparisons': {}
        }
        
        for target_name in self.target_metrics:
            if target_name not in y_dict:
                logger.warning(f"Target {target_name} not found in dataset, skipping...")
                continue
                
            logger.info(f"\n{'='*50}")
            logger.info(f"Training model for: {target_name}")
            logger.info(f"{'='*50}")
            
            # Enhanced model training with better parameters
            enhanced_params = {
                'n_estimators': 300,  # More trees for better performance
                'max_depth': 20,      # Deeper trees to capture complex patterns
                'min_samples_split': 2,  # More granular splits
                'min_samples_leaf': 1,   # Allow more detailed leaf nodes
                'max_features': 'sqrt',  # Feature subsampling for generalization
                'bootstrap': True,       # Enable bootstrap sampling
                'oob_score': True,      # Out-of-bag scoring for evaluation
                'random_state': 42,
                'n_jobs': -1
            }
            enhanced_params.update(rf_params)  # Allow custom parameters to override
            
            # Train model with enhanced parameters
            model = ModelUtils.train_random_forest(
                X_train, y_train_dict[target_name], target_name, **enhanced_params
            )
            
            # Evaluate model
            metrics = ModelUtils.evaluate_model(
                model, X_test, y_test_dict[target_name], target_name, normalize_output=True
            )
            
            # Get feature importance
            importance_df = ModelUtils.get_feature_importance(
                model, self.feature_names, top_n=10
            )
            
            # Create prediction comparison
            y_pred = model.predict(X_test)
            y_pred_normalized = ModelUtils.normalize_predictions(y_pred, (0.0, 1.0))
            comparison_df = ModelUtils.create_prediction_comparison(
                y_test_dict[target_name], y_pred_normalized, target_name, n_samples=10
            )
            
            # Save model
            model_filename = f"{target_name.replace('_label', '')}_rf.joblib"
            model_path = self.models_dir / model_filename
            ModelUtils.save_model(model, str(model_path), self.feature_names, target_name)
            
            # Store results
            self.trained_models[target_name] = model
            self.evaluation_results[target_name] = metrics
            
            training_results['models_trained'].append(target_name)
            training_results['evaluation_metrics'][target_name] = metrics
            training_results['feature_importance'][target_name] = importance_df
            training_results['prediction_comparisons'][target_name] = comparison_df
            
            logger.info(f"Model for {target_name} completed and saved to {model_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info("ALL MODELS TRAINING COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total models trained: {len(training_results['models_trained'])}")
        logger.info(f"Models saved to: {self.models_dir}")
        
        return training_results
    
    def print_training_summary(self, training_results: Dict[str, Any]) -> None:
        """
        Print a comprehensive summary of training results.
        
        Args:
            training_results: Results from train_all_models()
        """
        print("\n" + "="*80)
        print("RANDOM FOREST TRAINING SUMMARY")
        print("="*80)
        
        # Overall summary
        models_trained = training_results['models_trained']
        print(f"✅ Successfully trained {len(models_trained)} Random Forest models")
        print(f"📁 Models saved to: {self.models_dir}")
        print(f"🔧 Features used: {len(self.feature_names)}")
        print(f"📊 Feature names: {', '.join(self.feature_names[:5])}{'...' if len(self.feature_names) > 5 else ''}")
        
        # Evaluation metrics summary
        print(f"\n{'='*60}")
        print("MODEL EVALUATION METRICS")
        print(f"{'='*60}")
        
        metrics_df_data = []
        for target_name in models_trained:
            metrics = training_results['evaluation_metrics'][target_name]
            metrics_df_data.append({
                'Model': target_name.replace('_label', '').replace('_', ' ').title(),
                'MAE': f"{metrics['mae']:.4f}",
                'RMSE': f"{metrics['rmse']:.4f}",
                'R²': f"{metrics['r2']:.4f}",
                'Pred Range': f"[{metrics['pred_min']:.3f}, {metrics['pred_max']:.3f}]"
            })
        
        metrics_df = pd.DataFrame(metrics_df_data)
        print(metrics_df.to_string(index=False))
        
        # Feature importance summary
        print(f"\n{'='*60}")
        print("TOP FEATURE IMPORTANCE (averaged across all models)")
        print(f"{'='*60}")
        
        # Calculate average feature importance across all models
        all_importance = {}
        for target_name in models_trained:
            importance_df = training_results['feature_importance'][target_name]
            for _, row in importance_df.iterrows():
                feature = row['feature']
                importance = row['importance']
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(importance)
        
        # Average and sort
        avg_importance = {
            feature: np.mean(scores) 
            for feature, scores in all_importance.items()
        }
        
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<25} {importance:.4f}")
        
        # Prediction comparisons
        print(f"\n{'='*60}")
        print("PREDICTION vs TRUE VALUES (First 5 test samples)")
        print(f"{'='*60}")
        
        for target_name in models_trained[:3]:  # Show first 3 models to save space
            model_name = target_name.replace('_label', '').replace('_', ' ').title()
            comparison_df = training_results['prediction_comparisons'][target_name]
            
            print(f"\n{model_name}:")
            print(comparison_df.head().to_string(index=False, float_format='%.4f'))
    
    def load_trained_model(self, target_name: str) -> Dict[str, Any]:
        """
        Load a previously trained model.
        
        Args:
            target_name: Name of the target metric (e.g., 'market_risk_label')
            
        Returns:
            Dictionary containing model and metadata
        """
        model_filename = f"{target_name.replace('_label', '')}_rf.joblib"
        model_path = self.models_dir / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return ModelUtils.load_model(str(model_path))
    
    def predict_single_sample(self, features: Dict[str, float], 
                            target_name: str) -> float:
        """
        Make a prediction for a single sample using a trained model.
        
        Args:
            features: Dictionary of feature values
            target_name: Name of the target metric
            
        Returns:
            Normalized prediction in range [0, 1]
        """
        # Load model if not already loaded
        if target_name not in self.trained_models:
            model_data = self.load_trained_model(target_name)
            model = model_data['model']
        else:
            model = self.trained_models[target_name]
        
        # Prepare feature vector
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                feature_vector.append(features[feature_name])
            else:
                logger.warning(f"Feature {feature_name} not provided, using 0")
                feature_vector.append(0.0)
        
        # Make prediction
        prediction = model.predict([feature_vector])[0]
        
        # Normalize to [0, 1] range
        normalized_prediction = ModelUtils.normalize_predictions(
            np.array([prediction]), (0.0, 1.0)
        )[0]
        
        return normalized_prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about trained models.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'models_dir': str(self.models_dir),
            'target_metrics': self.target_metrics,
            'feature_names': self.feature_names,
            'models_trained': list(self.trained_models.keys()),
            'evaluation_results': self.evaluation_results
        } 