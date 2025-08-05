"""
Comprehensive data quality validation for ML training pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Validates and ensures data quality for ML training."""
    
    def __init__(self, 
                 variance_threshold: float = 0.01,
                 missing_threshold: float = 0.20,
                 correlation_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
    
    def validate_training_data(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive validation and cleaning of training data.
        
        Returns:
            (is_valid, cleaned_df, validation_report)
        """
        logger.info(f"Starting data quality validation for dataset with shape: {df.shape}")
        
        validation_report = {
            'original_shape': df.shape,
            'issues_found': [],
            'fixes_applied': [],
            'final_statistics': {}
        }
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # 1. Ensure all expected columns exist
        df_clean, column_fixes = self._ensure_required_columns(df_clean)
        validation_report['fixes_applied'].extend(column_fixes)
        
        # 2. Handle missing values
        df_clean, missing_fixes = self._handle_missing_values(df_clean)
        validation_report['fixes_applied'].extend(missing_fixes)
        
        # 3. Remove duplicate/correlated features safely
        df_clean, correlation_fixes = self._handle_correlated_features(df_clean)
        validation_report['fixes_applied'].extend(correlation_fixes)
        
        # 4. Apply feature transformations
        df_clean, transform_fixes = self._apply_feature_transforms(df_clean)
        validation_report['fixes_applied'].extend(transform_fixes)
        
        # 5. Validate feature variance
        df_clean, variance_fixes = self._validate_feature_variance(df_clean)
        validation_report['fixes_applied'].extend(variance_fixes)
        
        # 6. Validate label quality
        is_valid, label_issues = self._validate_labels(df_clean)
        validation_report['issues_found'].extend(label_issues)
        
        # 7. Generate final statistics
        validation_report['final_statistics'] = self._generate_statistics(df_clean)
        validation_report['final_shape'] = df_clean.shape
        
        logger.info(f"Data validation complete. Final shape: {df_clean.shape}")
        logger.info(f"Applied {len(validation_report['fixes_applied'])} fixes")
        if validation_report['issues_found']:
            logger.warning(f"Found {len(validation_report['issues_found'])} issues")
        
        return is_valid, df_clean, validation_report
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Ensure all required columns exist, create missing ones with defaults."""
        fixes = []
        
        # Define expected columns and their default values
        expected_columns = {
            'price': 0.0,
            'square_feet': 0.0,
            'days_on_market': 0.0,
            'active_listing_count': 0.0,
            'price_reduced_count': 0.0,
            'price_increased_count': 0.0,
            'total_listing_count': 0.0,
            'median_days_on_market': 0.0,
            'median_listing_price': 0.0,
            'price_per_sqft': 0.0,
            'price_volatility': 0.1,  # Default volatility
            'price_change_1y': 0.0,
            'price_change_3y': 0.0,
            'price_change_5y': 0.0,
            'price_reduction_ratio': 0.0,
            'price_increase_ratio': 0.0
        }
        
        for col, default_val in expected_columns.items():
            if col not in df.columns:
                df[col] = default_val
                fixes.append(f"Added missing column '{col}' with default value {default_val}")
                logger.warning(f"Added missing column '{col}' with default value {default_val}")
        
        return df, fixes
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values in the dataset."""
        fixes = []
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.endswith('_label'):
                continue  # Don't impute labels
                
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(df)
                
                if missing_pct > self.missing_threshold:
                    # Too many missing values, use default
                    if 'price' in col.lower() or 'count' in col.lower():
                        default_val = 0.0
                    elif 'ratio' in col.lower():
                        default_val = 0.0
                    elif 'volatility' in col.lower():
                        default_val = 0.1
                    else:
                        default_val = df[col].median()
                    
                    df[col] = df[col].fillna(default_val)
                    fixes.append(f"Filled {missing_count} missing values in '{col}' with {default_val}")
                else:
                    # Use median imputation for reasonable missing values
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    fixes.append(f"Imputed {missing_count} missing values in '{col}' with median {median_val}")
        
        return df, fixes
    
    def _handle_correlated_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove duplicate/highly correlated features safely."""
        fixes = []
        
        # Define features to drop if they exist (keeping the preferred version)
        drop_if_exists = []
        
        # Only drop if both columns exist
        if 'median_listing_price' in df.columns and 'price' in df.columns:
            drop_if_exists.append('median_listing_price')
        
        if 'median_days_on_market' in df.columns and 'days_on_market' in df.columns:
            drop_if_exists.append('median_days_on_market')
        
        # Drop the identified columns
        for col in drop_if_exists:
            if col in df.columns:
                df = df.drop(columns=[col])
                fixes.append(f"Dropped correlated feature '{col}'")
        
        return df, fixes
    
    def _apply_feature_transforms(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Apply feature transformations to improve model performance."""
        fixes = []
        
        # Apply log transformation to heavily skewed features
        skewed_features = [
            'price', 'active_listing_count', 'price_reduced_count', 'price_increased_count',
            'total_listing_count', 'price_per_sqft', 'price_volatility'
        ]
        
        for col in skewed_features:
            if col in df.columns:
                # Check if the feature is actually skewed
                skewness = stats.skew(df[col].dropna())
                if abs(skewness) > 1.0:  # Only transform if significantly skewed
                    # Add small constant to handle zeros
                    min_val = df[col].min()
                    offset = 1 if min_val >= 0 else abs(min_val) + 1
                    df[f'{col}_log'] = np.log1p(df[col] + offset)
                    fixes.append(f"Applied log transformation to '{col}' (skewness: {skewness:.2f})")
        
        # Normalize price changes to reduce extreme values
        change_features = ['price_change_1y', 'price_change_3y', 'price_change_5y']
        for col in change_features:
            if col in df.columns:
                # Clip extreme values and normalize
                df[f'{col}_norm'] = np.clip(df[col] / 100, -1, 1)
                fixes.append(f"Normalized '{col}' to [-1, 1] range")
        
        return df, fixes
    
    def _validate_feature_variance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate that features have sufficient variance."""
        fixes = []
        
        # Get feature columns (exclude labels and metadata)
        feature_columns = [col for col in df.columns if not col.endswith('_label') 
                          and col not in ['property_id', 'data_source', 'month_date', 'cbsa_code', 'cbsa_title']]
        
        constant_features = []
        near_constant_features = []
        
        for col in feature_columns:
            if col not in df.columns:
                continue  # Skip if column doesn't exist
                
            # Check for constant features
            if df[col].nunique() <= 1:
                constant_features.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                # Check variance
                if df[col].std() < self.variance_threshold:
                    near_constant_features.append(col)
                
                # Check if >95% of values are the same
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 0 and value_counts.iloc[0] > 0.95:
                    near_constant_features.append(col)
        
        # Drop constant features
        if constant_features:
            df = df.drop(columns=constant_features)
            fixes.append(f"Dropped constant features: {constant_features}")
            logger.warning(f"Dropped constant features: {constant_features}")
        
        # Warn about near-constant features but keep them
        if near_constant_features:
            fixes.append(f"Flagged near-constant features: {near_constant_features}")
            logger.warning(f"Near-constant features detected: {near_constant_features}")
        
        return df, fixes
    
    def _validate_labels(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that labels have sufficient variability."""
        issues = []
        is_valid = True
        
        label_columns = [col for col in df.columns if col.endswith('_label')]
        
        for col in label_columns:
            if col not in df.columns:
                continue
                
            # Check for constant labels
            if df[col].nunique() <= 1:
                issues.append(f"Label '{col}' has constant value: {df[col].iloc[0]}")
                is_valid = False
            else:
                # Check variance
                std_val = df[col].std()
                if std_val < 0.02:  # Very low variance threshold for labels
                    issues.append(f"Label '{col}' has very low variance (std={std_val:.4f})")
                
                # Check distribution
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 0 and value_counts.iloc[0] > 0.9:
                    issues.append(f"Label '{col}' has {value_counts.iloc[0]*100:.1f}% same value")
                
                # Log statistics
                stats_info = df[col].describe()
                logger.info(f"Label '{col}' statistics: mean={stats_info['mean']:.4f}, "
                           f"std={stats_info['std']:.4f}, min={stats_info['min']:.4f}, "
                           f"max={stats_info['max']:.4f}")
        
        return is_valid, issues
    
    def _generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics for the dataset."""
        stats_dict = {
            'shape': df.shape,
            'feature_count': len([col for col in df.columns if not col.endswith('_label') 
                                 and col not in ['property_id', 'data_source', 'month_date', 'cbsa_code', 'cbsa_title']]),
            'label_count': len([col for col in df.columns if col.endswith('_label')]),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_features': {},
            'label_distributions': {}
        }
        
        # Statistics for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not col.endswith('_label'):
                stats_dict['numeric_features'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'nunique': df[col].nunique()
                }
        
        # Statistics for labels
        label_cols = [col for col in df.columns if col.endswith('_label')]
        for col in label_cols:
            if col in df.columns:
                stats_dict['label_distributions'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'nunique': df[col].nunique()
                }
        
        return stats_dict