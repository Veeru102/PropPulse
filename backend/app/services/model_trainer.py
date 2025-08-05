import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from app.core.logging import loggers
from app.core.config import settings
import json
from pathlib import Path
from datetime import datetime
import joblib
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import gc  # For garbage collection to free memory

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger = loggers['ml']
    logger.warning("FAISS not available. Autoencoder training will be skipped.")

logger = loggers['ml']

class ModelTrainer:
    def __init__(self):
        """Initialize model trainer with configuration."""
        self.model_dir = Path(settings.MODEL_DIR)
        self.metrics = {}
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _encode_categorical_columns(self, X: pd.DataFrame, top_n_categories: int = 50) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder.
        
        Args:
            X: DataFrame with features
            top_n_categories: For large categorical fields, keep only top N most frequent categories
            
        Returns:
            DataFrame with encoded categorical columns
        """
        try:
            X_encoded = X.copy()
            categorical_columns = []
            encoding_stats = {}
            
            # Memory optimization: Reduce top_n_categories for large datasets
            if len(X) > 100000:
                top_n_categories = min(25, top_n_categories)  # More aggressive bucketing for large datasets
                logger.info(f"Large dataset detected ({len(X)} rows). Reducing categorical limit to {top_n_categories} for memory efficiency.")
            
            # Identify categorical columns (object dtype)
            for col in X_encoded.columns:
                if X_encoded[col].dtype == "object":
                    categorical_columns.append(col)
            
            if not categorical_columns:
                logger.info("No categorical columns found to encode")
                return X_encoded
                
            logger.info(f"Found {len(categorical_columns)} categorical columns to encode: {categorical_columns}")
            
            for col in categorical_columns:
                try:
                    # Handle missing values first
                    X_encoded[col] = X_encoded[col].fillna('MISSING')
                    
                    # Get unique value counts
                    value_counts = X_encoded[col].value_counts()
                    unique_count = len(value_counts)
                    
                    # For large categorical fields, keep only top N categories
                    if unique_count > top_n_categories:
                        logger.warning(f"Column '{col}' has {unique_count} unique values. Keeping top {top_n_categories} and grouping rest as 'OTHER'")
                        top_categories = value_counts.head(top_n_categories).index.tolist()
                        X_encoded[col] = X_encoded[col].apply(lambda x: x if x in top_categories else 'OTHER')
                        encoding_stats[col] = {
                            'original_unique_count': unique_count,
                            'encoded_unique_count': top_n_categories + 1,  # +1 for 'OTHER'
                            'action': 'top_n_bucketing'
                        }
                    else:
                        encoding_stats[col] = {
                            'original_unique_count': unique_count,
                            'encoded_unique_count': unique_count,
                            'action': 'direct_encoding'
                        }
                    
                    # Apply LabelEncoder
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col])
                    
                    logger.info(f"Encoded column '{col}': {encoding_stats[col]['original_unique_count']} -> {encoding_stats[col]['encoded_unique_count']} categories ({encoding_stats[col]['action']})")
                    
                except Exception as e:
                    logger.error(f"Error encoding column '{col}': {str(e)}. Dropping column.")
                    X_encoded = X_encoded.drop(columns=[col])
                    continue
            
            # Final defensive check - ensure no object columns remain
            remaining_object_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
            if remaining_object_cols:
                logger.warning(f"Still found object columns after encoding: {remaining_object_cols}. Dropping them.")
                X_encoded = X_encoded.drop(columns=remaining_object_cols)
            
            # Log final statistics
            logger.info(f"Categorical encoding complete. Final dataset shape: {X_encoded.shape}")
            logger.info(f"Encoding summary: {len(categorical_columns)} columns processed")
            for col, stats in encoding_stats.items():
                logger.info(f"  - {col}: {stats['original_unique_count']} -> {stats['encoded_unique_count']} categories")
            
            return X_encoded
            
        except Exception as e:
            logger.error(f"Error in categorical encoding: {str(e)}")
            raise
        
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train and validate models with separate feature sets for value and investment models.
        
        Args:
            training_data: DataFrame containing prepared training data
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            logger.info("Starting model training process")
            
            # Validate data
            self._validate_training_data(training_data)
            
            # Prepare data with separate feature sets
            X_value, y_value, X_investment, y_investment = self._prepare_data(training_data)
            
            # Enhance feature engineering
            X_value = self._create_derived_features(X_value)
            X_investment = self._create_derived_features(X_investment)

            # CRITICAL: Encode categorical columns before XGBoost training
            logger.info("Encoding categorical columns for value model...")
            X_value = self._encode_categorical_columns(X_value)
            
            logger.info("Encoding categorical columns for investment model...")
            X_investment = self._encode_categorical_columns(X_investment)
            
            # Final defensive check - ensure all columns are numeric
            for col in X_value.columns:
                if X_value[col].dtype == "object":
                    logger.warning(f"Encoding object column in value model: {col}")
                    X_value = X_value.drop(columns=[col])
                    
            for col in X_investment.columns:
                if X_investment[col].dtype == "object":
                    logger.warning(f"Encoding object column in investment model: {col}")
                    X_investment = X_investment.drop(columns=[col])
            
            logger.info(f"Final value model feature count: {X_value.shape[1]}")
            logger.info(f"Final investment model feature count: {X_investment.shape[1]}")
            logger.info(f"Dataset sizes - Value: {X_value.shape}, Investment: {X_investment.shape}")
            
            # Memory optimization: Sample large datasets to prevent memory exhaustion
            max_sample_size = 50000  # Reasonable limit for consumer hardware
            
            if len(X_value) > max_sample_size:
                logger.warning(f"Value dataset is large ({len(X_value)} rows). Sampling {max_sample_size} rows to prevent memory issues.")
                sample_indices = np.random.choice(len(X_value), size=max_sample_size, replace=False)
                X_value = X_value.iloc[sample_indices].copy()
                y_value = y_value.iloc[sample_indices].copy()
                logger.info(f"Value dataset sampled to {X_value.shape}")
                
            if len(X_investment) > max_sample_size:
                logger.warning(f"Investment dataset is large ({len(X_investment)} rows). Sampling {max_sample_size} rows to prevent memory issues.")
                sample_indices = np.random.choice(len(X_investment), size=max_sample_size, replace=False)
                X_investment = X_investment.iloc[sample_indices].copy()
                y_investment = y_investment.iloc[sample_indices].copy()
                logger.info(f"Investment dataset sampled to {X_investment.shape}")

            # Implement hyperparameter tuning
            from sklearn.model_selection import GridSearchCV

            # Define parameter grid for value model
            value_param_grid = {
                'max_depth': [3, 4],              # Reduced from [3,4,5]
                'learning_rate': [0.01, 0.1],     # Removed middle value 0.05
                'n_estimators': [100, 500],       # Removed largest value 1000
                'subsample': [0.8, 1.0],          # Removed middle value 0.9
                'colsample_bytree': [0.8, 1.0]    # Removed middle value 0.9
            }

            # Calculate and log total combinations for value model
            value_combinations = (
                len(value_param_grid['max_depth']) *
                len(value_param_grid['learning_rate']) *
                len(value_param_grid['n_estimators']) *
                len(value_param_grid['subsample']) *
                len(value_param_grid['colsample_bytree'])
            )
            logger.info(f"Value model grid search space: {value_combinations} combinations")
            logger.info("Value model parameter grid:")
            for param, values in value_param_grid.items():
                logger.info(f"  - {param}: {values}")

            # Define parameter grid for investment model
            investment_param_grid = {
                'max_depth': [3, 4],              # Reduced from [3,4,5]
                'learning_rate': [0.01, 0.1],     # Removed middle value 0.05
                'n_estimators': [100, 500],       # Removed largest value 1000
                'subsample': [0.8, 1.0],          # Removed middle value 0.9
                'colsample_bytree': [0.8, 1.0],   # Removed middle value 0.9
                'scale_pos_weight': [1, 2]        # Removed value 3
            }

            # Calculate and log total combinations for investment model
            investment_combinations = (
                len(investment_param_grid['max_depth']) *
                len(investment_param_grid['learning_rate']) *
                len(investment_param_grid['n_estimators']) *
                len(investment_param_grid['subsample']) *
                len(investment_param_grid['colsample_bytree']) *
                len(investment_param_grid['scale_pos_weight'])
            )
            logger.info(f"Investment model grid search space: {investment_combinations} combinations")
            logger.info("Investment model parameter grid:")
            for param, values in investment_param_grid.items():
                logger.info(f"  - {param}: {values}")

            # Log total number of fits
            cv_folds = 3  # Reduced from 5 to 3
            total_fits = (value_combinations + investment_combinations) * cv_folds
            logger.info(f"Total number of fits to be performed: {total_fits} (= ({value_combinations} + {investment_combinations}) * {cv_folds} folds)")
            estimated_time = total_fits * 0.5  # Rough estimate: 30 seconds per fit
            logger.info(f"Estimated training time: {estimated_time/60:.1f} minutes (assuming ~30 seconds per fit)")

            # Memory optimization: Limit parallelism to prevent memory exhaustion
            # Use fewer cores for GridSearchCV to leave memory for XGBoost processes
            max_jobs = min(4, max(1, os.cpu_count() // 2)) if os.cpu_count() else 2
            logger.info(f"Using {max_jobs} parallel jobs (instead of all {os.cpu_count() or 'unknown'} cores) to prevent memory exhaustion")

            # Perform grid search for value model
            logger.info("Starting value model grid search...")
            value_grid_search = GridSearchCV(
                estimator=xgb.XGBRegressor(
                    tree_method='hist',  # More memory efficient than 'auto'
                    max_bin=256         # Reduce memory usage
                ),
                param_grid=value_param_grid,
                scoring='neg_mean_squared_error',
                cv=TimeSeriesSplit(n_splits=cv_folds),  # Reduced from 5 to 3
                verbose=1,
                n_jobs=max_jobs  # Limited parallelism instead of -1
            )
            value_grid_search.fit(X_value, y_value)
            value_model = value_grid_search.best_estimator_
            logger.info("Value model grid search completed successfully!")
            
            # Memory cleanup after intensive operation
            gc.collect()

            # Perform grid search for investment model
            logger.info("Starting investment model grid search...")
            investment_grid_search = GridSearchCV(
                estimator=xgb.XGBClassifier(
                    tree_method='hist',  # More memory efficient than 'auto'
                    max_bin=256         # Reduce memory usage
                ),
                param_grid=investment_param_grid,
                scoring='f1',
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True),  # Reduced from 5 to 3
                verbose=1,
                n_jobs=max_jobs  # Limited parallelism instead of -1
            )
            investment_grid_search.fit(X_investment, y_investment)
            investment_model = investment_grid_search.best_estimator_
            logger.info("Investment model grid search completed successfully!")
            
            # Memory cleanup after intensive operation
            gc.collect()

            # Update metrics with best parameters
            value_metrics = self._calculate_metrics(value_model, X_value, y_value)
            value_metrics['best_params'] = value_grid_search.best_params_
            
            # Add feature importance for value model
            try:
                from sklearn.inspection import permutation_importance
                
                # Memory optimization: Sample data for permutation importance if dataset is large
                if len(X_value) > 10000:
                    sample_size = min(5000, len(X_value))
                    sample_indices = np.random.choice(len(X_value), size=sample_size, replace=False)
                    X_sample = X_value.iloc[sample_indices]
                    y_sample = y_value.iloc[sample_indices]
                    logger.info(f"Sampling {sample_size} rows from {len(X_value)} for feature importance calculation")
                else:
                    X_sample = X_value
                    y_sample = y_value
                
                r_value = permutation_importance(
                    value_model, X_sample, y_sample,
                    n_repeats=3,  # Reduced from 5 for memory efficiency
                    random_state=42,
                    n_jobs=1  # Use single process to prevent memory exhaustion
                )
                feature_importance_value = dict(zip(X_sample.columns, r_value.importances_mean))
                top_features_value = dict(sorted(
                    feature_importance_value.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])  # Top 10 features
                value_metrics['top_features'] = top_features_value
                logger.info(f"Value model top features calculated: {len(top_features_value)} features")
            except Exception as e:
                logger.warning(f"Failed to calculate feature importance for value model: {str(e)}")
                value_metrics['top_features'] = {}  # Fallback to empty dict
            
            # Memory cleanup after feature importance calculation
            gc.collect()

            investment_metrics = self._calculate_metrics(investment_model, X_investment, y_investment)
            investment_metrics['best_params'] = investment_grid_search.best_params_
            
            # Add feature importance for investment model
            try:
                # Memory optimization: Sample data for permutation importance if dataset is large
                if len(X_investment) > 10000:
                    sample_size = min(5000, len(X_investment))
                    sample_indices = np.random.choice(len(X_investment), size=sample_size, replace=False)
                    X_sample = X_investment.iloc[sample_indices]
                    y_sample = y_investment.iloc[sample_indices]
                    logger.info(f"Sampling {sample_size} rows from {len(X_investment)} for feature importance calculation")
                else:
                    X_sample = X_investment
                    y_sample = y_investment
                
                r_investment = permutation_importance(
                    investment_model, X_sample, y_sample,
                    n_repeats=3,  # Reduced from 5 for memory efficiency
                    random_state=42,
                    n_jobs=1  # Use single process to prevent memory exhaustion
                )
                feature_importance_investment = dict(zip(X_sample.columns, r_investment.importances_mean))
                top_features_investment = dict(sorted(
                    feature_importance_investment.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])  # Top 10 features
                investment_metrics['top_features'] = top_features_investment
                logger.info(f"Investment model top features calculated: {len(top_features_investment)} features")
            except Exception as e:
                logger.warning(f"Failed to calculate feature importance for investment model: {str(e)}")
                investment_metrics['top_features'] = {}  # Fallback to empty dict
            
            # Memory cleanup after feature importance calculation
            gc.collect()
            
            # Save models and metrics
            self._save_models(value_model, investment_model)
            self._save_metrics({
                'property_value': value_metrics,
                'investment': investment_metrics,
                'model_version': self.model_version,
                'training_date': datetime.now().isoformat(),
                'feature_importance': {
                    'property_value': value_metrics['top_features'],
                    'investment': investment_metrics['top_features']
                }
            })
            
            logger.info("Model training completed successfully")
            
            return {
                'status': 'success',
                'metrics': {
                    'property_value': value_metrics,
                    'investment': investment_metrics
                },
                'model_version': self.model_version,
                'data_stats': {
                    'value_model_samples': len(X_value),
                    'investment_model_samples': len(X_investment),
                    'original_samples': len(training_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def _validate_training_data(self, data: pd.DataFrame):
        """Validate training data quality."""
        try:
            # Check required features
            missing_features = set(settings.REQUIRED_FEATURES) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Check for missing values
            missing_values = data[settings.REQUIRED_FEATURES].isnull().sum()
            if missing_values.any():
                raise ValueError(f"Missing values found in features: {missing_values[missing_values > 0]}")
            
            # Check for infinite values
            inf_values = np.isinf(data[settings.REQUIRED_FEATURES].select_dtypes(include=np.number)).sum()
            if inf_values.any():
                raise ValueError(f"Infinite values found in features: {inf_values[inf_values > 0]}")
            
            # Check data types
            for feature in settings.REQUIRED_FEATURES:
                if not np.issubdtype(data[feature].dtype, np.number):
                    raise ValueError(f"Feature {feature} must be numeric")
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise

    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing values."""
        try:
            # Analyze missing values in target variables
            price_missing = data['price'].isna()
            roi_missing = data['roi'].isna()
            
            # Calculate missing value statistics
            missing_stats = {
                'total_samples': len(data),
                'price_missing': {
                    'count': price_missing.sum(),
                    'percentage': (price_missing.sum() / len(data)) * 100
                },
                'roi_missing': {
                    'count': roi_missing.sum(),
                    'percentage': (roi_missing.sum() / len(data)) * 100
                },
                'both_missing': {
                    'count': (price_missing & roi_missing).sum(),
                    'percentage': ((price_missing & roi_missing).sum() / len(data)) * 100
                }
            }
            
            # Analyze missing values by region
            if 'region_name' in data.columns:
                region_stats = data.groupby('region_name').agg({
                    'price': lambda x: x.isna().sum(),
                    'roi': lambda x: x.isna().sum()
                }).reset_index()
                missing_stats['by_region'] = region_stats.to_dict('records')
            
            # Analyze missing values by time period
            if 'month_date_yyyymm' in data.columns:
                time_stats = data.groupby('month_date_yyyymm').agg({
                    'price': lambda x: x.isna().sum(),
                    'roi': lambda x: x.isna().sum()
                }).reset_index()
                missing_stats['by_time'] = time_stats.to_dict('records')
            
            # Log the analysis results
            logger.info("Missing value analysis:")
            logger.info(f"Total samples: {missing_stats['total_samples']}")
            logger.info(f"Price missing: {missing_stats['price_missing']['count']} ({missing_stats['price_missing']['percentage']:.2f}%)")
            logger.info(f"ROI missing: {missing_stats['roi_missing']['count']} ({missing_stats['roi_missing']['percentage']:.2f}%)")
            logger.info(f"Both missing: {missing_stats['both_missing']['count']} ({missing_stats['both_missing']['percentage']:.2f}%)")
            
            if 'by_region' in missing_stats:
                logger.info("\nMissing values by region:")
                for region in missing_stats['by_region']:
                    logger.info(f"Region {region['region_name']}: Price missing: {region['price']}, ROI missing: {region['roi']}")
            
            if 'by_time' in missing_stats:
                logger.info("\nMissing values by time period:")
                for period in missing_stats['by_time']:
                    logger.info(f"Period {period['month_date_yyyymm']}: Price missing: {period['price']}, ROI missing: {period['roi']}")
            
            return missing_stats
            
        except Exception as e:
            logger.error(f"Error analyzing missing values: {str(e)}")
            raise

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling infinite values and extreme outliers."""
        try:
            # Make a copy to avoid modifying the original
            X_clean = X.copy()
            
            # Handle infinite values
            for col in X_clean.select_dtypes(include=[np.number]).columns:
                # Replace inf with NaN
                X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # Calculate robust statistics
                q1 = X_clean[col].quantile(0.25)
                q3 = X_clean[col].quantile(0.75)
                iqr = q3 - q1
                
                # Define bounds
                lower_bound = q1 - 5 * iqr
                upper_bound = q3 + 5 * iqr
                
                # Clip values
                X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
                
                # Fill remaining NaN with median
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            
            # Verify no infinite values remain
            inf_mask = np.isinf(X_clean.select_dtypes(include=[np.number]))
            if inf_mask.any().any():
                logger.warning("Infinite values found after cleaning, replacing with NaN")
                X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
                X_clean = X_clean.fillna(X_clean.median())
            
            return X_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for training with more conservative outlier handling."""
        try:
            initial_count = len(data)
            logger.info(f"Initial data count: {initial_count}")
            
            # Analyze missing values before any processing
            missing_stats = self._analyze_missing_values(data)
            
            # Split features and targets
            X = data.drop(['price', 'roi'], axis=1)
            y_value = data['price']
            
            # Create investment label using price per sqft comparison
            required_cols = ['price', 'sqft', 'zip_code']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for investment label creation: {missing_cols}")
                logger.warning("Falling back to ROI-based investment label")
                
                # Fallback to ROI-based label if available
                if 'roi' in data.columns:
                    # Create binary label based on ROI threshold
                    y_investment = (data['roi'] > data['roi'].median()).astype(int)
                    
                    # Add ROI-related features
                    X['roi_relative_to_median'] = data['roi'] / data['roi'].median()
                    if 'est_rent' in data.columns:
                        X['est_rent_yield'] = (data['est_rent'] * 12) / data['price']
                else:
                    logger.error("Neither price/sqft nor ROI data available for investment label")
                    raise ValueError("No suitable data available for investment label creation")
            else:
                # Calculate price per sqft
                data['price_per_sqft'] = data['price'] / data['sqft']
                
                # Calculate ZIP-level median price per sqft
                zip_medians = data.groupby('zip_code')['price_per_sqft'].transform('median')
                
                # Create binary investment label
                y_investment = (data['price_per_sqft'] < zip_medians).astype(int)
                
                # Add price per sqft as a feature
                X['price_per_sqft'] = data['price_per_sqft']
                X['price_relative_to_zip_avg'] = data['price_per_sqft'] / zip_medians
                
                # Add estimated rent yield if available
                if 'est_rent' in data.columns:
                    X['est_rent_yield'] = (data['est_rent'] * 12) / data['price']
            
            # Sort by date if available
            if 'date' in X.columns:
                X = X.sort_values('date')
                y_value = y_value[X.index]
                y_investment = y_investment[X.index]
            
            # Clean the data
            X = self._clean_data(X)
            
            # Create derived features
            X = self._create_derived_features(X)
            
            # Clean again after feature creation
            X = self._clean_data(X)
            
            # Handle target variables separately
            # For price (value model)
            value_mask = ~(np.isnan(y_value) | np.isinf(y_value))
            X_value = X[value_mask].copy()
            y_value = y_value[value_mask]
            
            # For investment model (binary classification)
            investment_mask = ~(np.isnan(y_investment) | np.isinf(y_investment))
            X_investment = X[investment_mask].copy()
            y_investment = y_investment[investment_mask]
            
            # Log data loss statistics
            logger.info(f"Value model data count: {len(X_value)} ({len(X_value)/initial_count*100:.2f}% of original)")
            logger.info(f"Investment model data count: {len(X_investment)} ({len(X_investment)/initial_count*100:.2f}% of original)")
            
            # Clip extreme values in target variables using more conservative percentiles
            y_value = y_value.clip(
                lower=y_value.quantile(0.005),  # More conservative
                upper=y_value.quantile(0.995)   # More conservative
            )
            
            # Add ZIP-level contextual features if zip_code is available
            if 'zip_code' in X_investment.columns:
                X_investment = self._add_zip_context_features(X_investment)
                # Clean again after adding ZIP features
                X_investment = self._clean_data(X_investment)
            
            # Ensure all required features exist
            for feature in settings.REQUIRED_FEATURES:
                if feature not in X_value.columns:
                    X_value[feature] = 0
                if feature not in X_investment.columns:
                    X_investment[feature] = 0
            
            return X_value, y_value, X_investment, y_investment
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def _add_zip_context_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add ZIP code level contextual features."""
        try:
            if 'zip_code' not in X.columns:
                return X
                
            # Group by ZIP code to calculate contextual features
            zip_stats = X.groupby('zip_code').agg({
                'median_dom': 'median',
                'median_list_price_per_sqft': 'median',
                'inventory': 'median',
                'active_listing_count_mm': 'median',
                'pending_listing_count': 'median'
            }).reset_index()
            
            # Merge ZIP stats back to main dataframe
            X = X.merge(zip_stats, on='zip_code', suffixes=('', '_zip_median'))
            
            # Create relative features
            X['dom_relative_to_zip'] = X['median_dom'] / X['median_dom_zip_median']
            X['price_relative_to_zip'] = X['median_list_price_per_sqft'] / X['median_list_price_per_sqft_zip_median']
            X['inventory_relative_to_zip'] = X['inventory'] / X['inventory_zip_median']
            X['market_activity_relative_to_zip'] = (
                X['active_listing_count_mm'] / X['active_listing_count_mm_zip_median']
            )
            
            # Drop intermediate columns
            drop_cols = [col for col in X.columns if col.endswith('_zip_median')]
            X = X.drop(columns=drop_cols)
            
            return X
            
        except Exception as e:
            logger.error(f"Error adding ZIP context features: {str(e)}")
            return X

    def _safe_division(self, numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """Safely divide two series, handling division by zero and infinite values."""
        try:
            # Replace zeros in denominator with NaN to avoid division by zero
            denominator = denominator.replace(0, np.nan)
            # Perform division
            result = numerator / denominator
            # Replace inf and -inf with fill_value
            result = result.replace([np.inf, -np.inf], fill_value)
            # Fill NaN values with fill_value
            result = result.fillna(fill_value)
            return result
        except Exception as e:
            logger.error(f"Error in safe division: {str(e)}")
            return pd.Series(fill_value, index=numerator.index)

    def _create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features to improve model performance."""
        try:
            # Create price-related ratios if price data is available
            if 'median_list_price' in X.columns and 'median_list_price_per_sqft' in X.columns:
                X['price_to_sqft_ratio'] = self._safe_division(
                    X['median_list_price'],
                    X['median_list_price_per_sqft']
                )
            
            # Create market trend features
            if 'month_date_yyyymm' in X.columns:
                X['month'] = pd.to_datetime(X['month_date_yyyymm'], format='%Y%m').dt.month
                X['year'] = pd.to_datetime(X['month_date_yyyymm'], format='%Y%m').dt.year
                X['quarter'] = pd.to_datetime(X['month_date_yyyymm'], format='%Y%m').dt.quarter
                
                # Add seasonality features
                X['is_summer'] = X['month'].isin([6, 7, 8]).astype(int)
                X['is_winter'] = X['month'].isin([12, 1, 2]).astype(int)
                X['is_spring'] = X['month'].isin([3, 4, 5]).astype(int)
                X['is_fall'] = X['month'].isin([9, 10, 11]).astype(int)
            
            # Create inventory-based features
            if 'inventory' in X.columns:
                X['inventory_per_sqft'] = self._safe_division(
                    X['inventory'],
                    X['sqft'] if 'sqft' in X.columns else pd.Series(1, index=X.index)
                )
                if 'active_listing_count_mm' in X.columns:
                    X['inventory_turnover'] = self._safe_division(
                        X['active_listing_count_mm'],
                        X['inventory']
                    )
                    X['market_velocity'] = self._safe_division(
                        X['active_listing_count_mm'],
                        X['total_listing_count'] if 'total_listing_count' in X.columns else X['inventory']
                    )
            
            # Create time-based features
            if 'median_dom' in X.columns:
                X['dom_to_inventory_ratio'] = self._safe_division(
                    X['median_dom'],
                    X['inventory'] if 'inventory' in X.columns else pd.Series(1, index=X.index)
                )
                if 'pending_listing_count' in X.columns:
                    X['pending_to_dom_ratio'] = self._safe_division(
                        X['pending_listing_count'],
                        X['median_dom']
                    )
                    X['pending_velocity'] = self._safe_division(
                        X['pending_listing_count'],
                        X['total_listing_count'] if 'total_listing_count' in X.columns else X['inventory']
                    )
            
            # Create price trend features
            if all(col in X.columns for col in ['median_listing_price_mm', 'median_listing_price_yy', 'median_list_price']):
                X['price_trend_mm'] = self._safe_division(
                    X['median_listing_price_mm'],
                    X['median_list_price']
                ) - 1
                X['price_trend_yy'] = self._safe_division(
                    X['median_listing_price_yy'],
                    X['median_list_price']
                ) - 1
                
                # Add price momentum features
                X['price_momentum'] = X['price_trend_mm'] - X['price_trend_yy']
                X['price_acceleration'] = self._safe_division(
                    X['price_trend_mm'],
                    X['price_trend_yy']
                ) - 1
            
            # Create market activity features
            if all(col in X.columns for col in ['new_listing_count', 'pending_listing_count', 'total_listing_count']):
                X['market_activity_ratio'] = self._safe_division(
                    X['new_listing_count'] + X['pending_listing_count'],
                    X['total_listing_count']
                )
                X['new_listing_ratio'] = self._safe_division(
                    X['new_listing_count'],
                    X['total_listing_count']
                )
                X['pending_listing_ratio'] = self._safe_division(
                    X['pending_listing_count'],
                    X['total_listing_count']
                )
            
            # Create price adjustment features
            if all(col in X.columns for col in ['price_increased_count', 'price_reduced_count', 'total_listing_count']):
                X['price_adjustment_ratio'] = self._safe_division(
                    X['price_increased_count'] + X['price_reduced_count'],
                    X['total_listing_count']
                )
                X['price_increase_ratio'] = self._safe_division(
                    X['price_increased_count'],
                    X['total_listing_count']
                )
                X['price_reduction_ratio'] = self._safe_division(
                    X['price_reduced_count'],
                    X['total_listing_count']
                )
                
                # Add price adjustment momentum
                X['price_adjustment_momentum'] = X['price_increase_ratio'] - X['price_reduction_ratio']
            
            # Create market efficiency features
            if all(col in X.columns for col in ['median_dom', 'inventory', 'active_listing_count_mm']):
                X['market_efficiency'] = self._safe_division(
                    X['active_listing_count_mm'],
                    X['median_dom'] * X['inventory']
                )
                X['listing_velocity'] = self._safe_division(
                    X['active_listing_count_mm'],
                    X['median_dom']
                )
            
            # Create price volatility features
            if all(col in X.columns for col in ['price_increased_count', 'price_reduced_count', 'total_listing_count']):
                X['price_volatility'] = self._safe_division(
                    X['price_increased_count'] + X['price_reduced_count'],
                    X['total_listing_count']
                )
            
            # Clip all numeric columns to handle any remaining extreme values
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col] = X[col].clip(
                    lower=X[col].quantile(0.001),
                    upper=X[col].quantile(0.999)
                )
            
            return X
            
        except Exception as e:
            logger.error(f"Error creating derived features: {str(e)}")
            raise

    def _handle_outliers(self, data: pd.DataFrame, iqr_multiplier: float) -> pd.DataFrame:
        """Handle outliers in numeric features using IQR method with a given multiplier."""
        try:
            total_clipped = 0
            for col in data.select_dtypes(include=np.number).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                # Count values that will be clipped
                below_lower = (data[col] < lower_bound).sum()
                above_upper = (data[col] > upper_bound).sum()
                total_clipped += below_lower + above_upper
                
                data[col] = data[col].clip(lower_bound, upper_bound)
                
                if below_lower > 0 or above_upper > 0:
                    logger.info(f"Feature {col}: {below_lower} values below lower bound, {above_upper} values above upper bound")
            
            logger.info(f"Total values clipped across all features: {total_clipped}")
            return data
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise

    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        target_name: str
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """Train a single model with validation and uncertainty estimation."""
        try:
            logger.info(f"Training {model_name}")
            
            # Import required modules at the top of the method
            from sklearn.model_selection import train_test_split, StratifiedKFold
            from sklearn.metrics import make_scorer, f1_score
            
            # Create time series cross-validation with stratification for investment model
            if model_name == 'investment_model':
                cv = StratifiedKFold(n_splits=5, shuffle=True)  # Enable shuffle for stratified split
            else:
                cv = TimeSeriesSplit(n_splits=5)
            
            # Initialize base model with parameters from settings
            model_params = settings.MODEL_PARAMS[model_name].copy()
            model_params['enable_categorical'] = True
            
            # Enhanced base parameters for all models
            model_params.update({
                'max_depth': 8,  # Increased for more complex patterns
                'learning_rate': 0.05,  # Balanced learning rate
                'n_estimators': 2000,  # More iterations
                'min_child_weight': 3,  # More balanced
                'subsample': 0.8,  # Prevent overfitting
                'colsample_bytree': 0.8,  # Prevent overfitting
                'gamma': 0.1,  # Minimum loss reduction for split
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1,  # L2 regularization
            })
            
            # Adjust parameters based on model type
            if model_name == 'investment_model':
                # Calculate class weights
                n_pos = (y == 1).sum()
                n_neg = (y == 0).sum()
                scale_pos_weight = n_neg / n_pos
                
                # Binary classification parameters for investment model with enhanced variability
                model_params.update({
                    'objective': 'binary:logistic',
                    'max_depth': 8,  # Increased for more complex patterns
                    'learning_rate': 0.05,  # Balanced learning rate
                    'n_estimators': 2000,  # More iterations
                    'min_child_weight': 3,  # More balanced
                    'subsample': 0.8,  # Prevent overfitting
                    'colsample_bytree': 0.8,  # Prevent overfitting
                    'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
                    'max_delta_step': 1,  # Help with class imbalance
                    'gamma': 0.1,  # Minimum loss reduction for split
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 1,  # L2 regularization
                    'min_child_weight': 3,  # More conservative for imbalanced data
                })
            
            # Train base model for predictions
            base_model = xgb.XGBRegressor(**model_params)
            
            # Convert categorical columns to category type
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                X[col] = X[col].astype('category')
            
            # Perform cross-validation with appropriate scoring
            if model_name == 'investment_model':
                # Create a proper scorer function that matches scikit-learn's expected signature
                def custom_f1_scorer(y_true, y_pred):
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    return f1_score(y_true, y_pred_binary)
                
                cv_scores = cross_val_score(
                    base_model, X, y,
                    cv=cv,
                    scoring=make_scorer(custom_f1_scorer, needs_proba=False)
                )
            else:
                cv_scores = cross_val_score(
                    base_model, X, y,
                    cv=cv,
                    scoring='neg_mean_squared_error'
                )
            
            # Train models with early stopping
            if model_name == 'investment_model':
                # Use stratified split for investment model with shuffle enabled
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y, shuffle=True  # Enable shuffle for stratified split
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
            
            # Train base model
            base_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,  # More patience
                verbose=False
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(base_model, X, y)
            
            # Calculate prediction intervals and check for constant predictions
            y_pred = base_model.predict(X)
            pred_std = np.std(y_pred)
            
            # Check for constant predictions and retrain if needed
            if pred_std < 0.02:  # Threshold for "constant" predictions
                logger.warning(f"Model {model_name} has low prediction variance (std={pred_std:.6f}). Retraining with adjusted parameters...")
                
                # Adjust parameters to increase variability
                model_params.update({
                    'max_depth': min(model_params['max_depth'] + 2, 12),  # Increase complexity
                    'learning_rate': model_params['learning_rate'] * 2,  # Increase learning rate
                    'gamma': max(model_params['gamma'] - 0.05, 0),  # Reduce split threshold
                    'min_child_weight': max(model_params['min_child_weight'] - 1, 1),  # Allow smaller leaf nodes
                })
                
                # Retrain model with adjusted parameters
                base_model = xgb.XGBRegressor(**model_params)
                base_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                # Recalculate metrics and predictions
                metrics = self._calculate_metrics(base_model, X, y)
                y_pred = base_model.predict(X)
                pred_std = np.std(y_pred)
                
                logger.info(f"After retraining: prediction std={pred_std:.6f}")
            
            # For investment model, convert predictions to binary for metrics
            if model_name == 'investment_model':
                y_pred_binary = (y_pred > 0.5).astype(int)
                metrics['binary_predictions'] = {
                    'positive_ratio': float(y_pred_binary.mean()),
                    'total_predictions': len(y_pred_binary)
                }
            
            # Calculate uncertainty using standard deviation of predictions
            pred_std = np.std(y_pred)
            target_std = np.std(y)
            uncertainty_score = 1 - (pred_std / target_std)
            
            # Calculate feature importance using permutation importance
            from sklearn.inspection import permutation_importance
            r = permutation_importance(
                base_model, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Get top features
            feature_importance = dict(zip(X.columns, r.importances_mean))
            top_features = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 features
            
            # Add cross-validation metrics
            if model_name == 'investment_model':
                metrics.update({
                    'cv_f1_mean': np.mean(cv_scores),
                    'cv_f1_std': np.std(cv_scores),
                })
            else:
                metrics.update({
                    'cv_mse': -np.mean(cv_scores),
                    'cv_mse_std': np.std(cv_scores),
                })
            
            metrics.update({
                'best_iteration': base_model.best_iteration,
                'uncertainty_score': uncertainty_score,
                'prediction_std': pred_std,
                'top_features': top_features
            })
            
            logger.info(f"{model_name} training completed with metrics: {metrics}")
            
            return base_model, metrics
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def _calculate_metrics(
        self,
        model: xgb.XGBRegressor,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        try:
            predictions = model.predict(X)
            
            # For binary classification (investment model)
            if model.get_params()['objective'] == 'binary:logistic':
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score,
                    f1_score, roc_auc_score, confusion_matrix
                )
                
                # Convert probabilities to binary predictions
                binary_predictions = (predictions > 0.5).astype(int)
                
                metrics = {
                    'accuracy': accuracy_score(y, binary_predictions),
                    'precision': precision_score(y, binary_predictions),
                    'recall': recall_score(y, binary_predictions),
                    'f1': f1_score(y, binary_predictions),
                    'roc_auc': roc_auc_score(y, predictions),
                    'confusion_matrix': confusion_matrix(y, binary_predictions).tolist()
                }
                
                # Add class distribution
                metrics['class_distribution'] = {
                    'positive': int(y.sum()),
                    'negative': int(len(y) - y.sum()),
                    'positive_ratio': float(y.mean())
                }
                
            # For regression (value model)
            else:
                metrics = {
                    'mse': mean_squared_error(y, predictions),
                    'rmse': np.sqrt(mean_squared_error(y, predictions)),
                    'mae': mean_absolute_error(y, predictions),
                    'r2': r2_score(y, predictions)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def _save_models(self, value_model: xgb.XGBRegressor, investment_model: xgb.XGBRegressor):
        """Save trained models to disk with versioning."""
        try:
            # Create version directory
            version_dir = self.model_dir / self.model_version
            version_dir.mkdir(exist_ok=True)
            
            # Save value model
            value_model_path = version_dir / "property_value_model.json"
            value_model.save_model(str(value_model_path))
            logger.info(f"Saved value model to {value_model_path}")
            
            # Save investment model
            investment_model_path = version_dir / "investment_model.json"
            investment_model.save_model(str(investment_model_path))
            logger.info(f"Saved investment model to {investment_model_path}")
            
            # Save model metadata
            metadata = {
                'version': self.model_version,
                'training_date': datetime.now().isoformat(),
                'feature_names': list(value_model.feature_names_in_)
            }
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save model metrics to disk with versioning."""
        try:
            def convert_numpy_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            # Convert numpy types to Python native types
            metrics = convert_numpy_types(metrics)
            
            # Save to version directory
            metrics_path = self.model_dir / self.model_version / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved metrics to {metrics_path}")
            
            # Update latest metrics
            latest_path = self.model_dir / "latest_metrics.json"
            with open(latest_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

    async def train_autoencoder_and_faiss(self, property_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train autoencoder and build FAISS index for property similarity search.
        
        Args:
            property_data: List of property dictionaries with complete feature data
            
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            if not FAISS_AVAILABLE:
                logger.error("FAISS is not available. Cannot train autoencoder and build index.")
                return {"status": "error", "message": "FAISS not available"}
                
            logger.info(f"Starting autoencoder training with {len(property_data)} properties")
            
            # Convert property data to DataFrame
            df = pd.DataFrame(property_data)
            
            # Extract features for autoencoder training
            feature_columns = ['price', 'beds', 'baths', 'area', 'year_built', 'lot_size', 'latitude', 'longitude']
            
            # Fill missing values with defaults
            df['year_built'] = df.get('year_built', datetime.now().year - 20)
            df['lot_size'] = df.get('lot_size', df['area'] * 2)  # Default to 2x house area
            
            # Create feature matrix
            X = df[feature_columns].copy()
            
            # Handle missing values
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median() if not X[col].isna().all() else 0)
            
            # Remove rows with any remaining NaN or invalid values
            valid_mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
            X = X[valid_mask]
            property_ids = df[valid_mask]['property_id'].tolist()
            
            logger.info(f"Training with {len(X)} valid properties after cleaning")
            
            if len(X) < 10:
                logger.error("Insufficient valid property data for training")
                return {"status": "error", "message": "Insufficient data"}
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Import AutoEncoder model
            from app.models.autoencoder import AutoEncoder
            
            # Initialize autoencoder
            input_dim = X_scaled.shape[1]
            latent_dim = min(16, input_dim // 2)  # Adaptive latent dimension
            autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop
            autoencoder.train()
            train_losses = []
            best_loss = float('inf')
            patience_counter = 0
            
            logger.info("Starting autoencoder training...")
            for epoch in range(200):  # Maximum epochs
                epoch_loss = 0.0
                for batch_idx, (data, target) in enumerate(dataloader):
                    optimizer.zero_grad()
                    reconstructed, encoded = autoencoder(data)
                    loss = criterion(reconstructed, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                train_losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = autoencoder.state_dict().copy()
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
                
                # Early stopping
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            autoencoder.load_state_dict(best_model_state)
            autoencoder.eval()
            
            logger.info(f"Autoencoder training completed. Best loss: {best_loss:.6f}")
            
            # Generate embeddings for all properties
            logger.info("Generating property embeddings...")
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                _, embeddings = autoencoder(X_tensor)
                embeddings_np = embeddings.numpy()
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            embeddings_np = embeddings_np / np.clip(norms, a_min=1e-8, a_max=None)
            
            # Build FAISS index
            logger.info("Building FAISS index...")
            dimension = embeddings_np.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            index.add(embeddings_np.astype('float32'))
            
            logger.info(f"FAISS index built with {index.ntotal} vectors")
            
            # Create property ID mapping
            property_id_map = {str(prop_id): idx for idx, prop_id in enumerate(property_ids)}
            
            # Save all artifacts
            version_dir = self.model_dir / self.model_version
            version_dir.mkdir(exist_ok=True)
            
            # Save autoencoder model
            autoencoder_path = version_dir / "autoencoder.pt"
            torch.save(autoencoder, str(autoencoder_path))
            logger.info(f"Saved autoencoder to {autoencoder_path}")
            
            # Save feature scaler
            scaler_path = version_dir / "feature_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved feature scaler to {scaler_path}")
            
            # Save FAISS index
            faiss_path = version_dir / "property_embeddings.faiss"
            faiss.write_index(index, str(faiss_path))
            logger.info(f"Saved FAISS index to {faiss_path}")
            
            # Save property ID mapping
            id_map_path = version_dir / "property_id_map.json"
            with open(id_map_path, 'w') as f:
                json.dump(property_id_map, f, indent=2)
            logger.info(f"Saved property ID mapping to {id_map_path}")
            
            # Save training metadata
            metadata = {
                'model_version': self.model_version,
                'training_date': datetime.now().isoformat(),
                'total_properties': len(property_data),
                'valid_properties': len(property_ids),
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'feature_columns': feature_columns,
                'best_loss': best_loss,
                'epochs_trained': len(train_losses)
            }
            
            metadata_path = version_dir / "autoencoder_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved autoencoder metadata to {metadata_path}")
            
            logger.info("Autoencoder and FAISS training completed successfully")
            
            return {
                'status': 'success',
                'model_version': self.model_version,
                'total_properties': len(property_data),
                'valid_properties': len(property_ids),
                'best_loss': best_loss,
                'faiss_index_size': index.ntotal,
                'feature_dimensions': input_dim,
                'latent_dimensions': latent_dim
            }
            
        except Exception as e:
            logger.error(f"Error in autoencoder and FAISS training: {str(e)}")
            raise 