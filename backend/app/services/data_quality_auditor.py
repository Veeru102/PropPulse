"""Data quality auditor for ML training pipeline."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DataQualityAuditor:
    """Performs comprehensive data quality checks on training datasets."""
    
    def __init__(self, 
                 variance_threshold: float = 0.01,
                 missing_threshold: float = 0.20,
                 correlation_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
    
    def audit_dataset(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Performs comprehensive data quality audit.
        Returns (passed_audit, audit_results).
        """
        audit_results = {}
        passed_audit = True
        
        # 1. Check for missing values, NaNs, infinities
        missing_stats = self._check_missing_values(df)
        audit_results['missing_values'] = missing_stats
        
        # Flag features with too many missing values
        high_missing = [col for col, pct in missing_stats.items() 
                       if pct > self.missing_threshold]
        if high_missing:
            passed_audit = False
            audit_results['high_missing_features'] = high_missing
            logger.error(f"Features with >{self.missing_threshold*100}% missing values: {high_missing}")
        
        # 2. Check for near-constant features
        constant_stats = self._check_constant_features(df)
        audit_results['constant_features'] = constant_stats
        
        if constant_stats['constant'] or constant_stats['near_constant']:
            passed_audit = False
            logger.error("Found constant or near-constant features:")
            if constant_stats['constant']:
                logger.error(f"Constant features: {constant_stats['constant']}")
            if constant_stats['near_constant']:
                logger.error(f"Near-constant features: {constant_stats['near_constant']}")
        
        # 3. Check for correlations
        corr_pairs = self._check_correlations(df)
        audit_results['correlated_pairs'] = corr_pairs
        
        if corr_pairs:
            logger.warning(f"Found {len(corr_pairs)} highly correlated feature pairs:")
            for pair, corr in corr_pairs:
                logger.warning(f"  {pair[0]} vs {pair[1]}: r={corr:.3f}")
        
        # 4. Compute descriptive statistics
        desc_stats = self._compute_descriptive_stats(df)
        audit_results['descriptive_stats'] = desc_stats
        
        # 5. Check label variability
        label_stats = self._check_label_variability(df)
        audit_results['label_stats'] = label_stats
        
        if label_stats['low_variance_labels']:
            passed_audit = False
            logger.error("Labels with insufficient variance:")
            for label, stats in label_stats['low_variance_labels'].items():
                logger.error(f"  {label}: std={stats['std']:.6f}, range={stats['range']:.6f}")
        
        return passed_audit, audit_results
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check for missing values and return percentage missing per column."""
        # Check for null values
        missing_pcts = (df.isnull().sum() / len(df)).to_dict()
        
        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            inf_pct = df[col].isin([np.inf, -np.inf]).sum() / len(df)
            missing_pcts[col] = missing_pcts.get(col, 0) + inf_pct
            
        return missing_pcts
    
    def _check_constant_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for constant and near-constant features."""
        constant = []
        near_constant = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'bool']:
                nunique = df[col].nunique()
                if nunique <= 1:
                    constant.append(col)
                else:
                    value_counts = df[col].value_counts(normalize=True)
                    if value_counts.iloc[0] > 0.95:  # More than 95% same value
                        near_constant.append(col)
                        
        return {
            'constant': constant,
            'near_constant': near_constant
        }
    
    def _check_correlations(self, df: pd.DataFrame) -> List[Tuple[Tuple[str, str], float]]:
        """Find highly correlated feature pairs."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        # Get pairs of features with high correlation
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                    pair = (corr_matrix.columns[i], corr_matrix.columns[j])
                    high_corr_pairs.append((pair, corr_matrix.iloc[i, j]))
                    
        return high_corr_pairs
    
    def _compute_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute detailed descriptive statistics for each feature."""
        stats_dict = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            col_stats = df[col].describe()
            stats_dict[col] = {
                'mean': col_stats['mean'],
                'std': col_stats['std'],
                'min': col_stats['min'],
                'max': col_stats['max'],
                'skewness': stats.skew(df[col].dropna()),
                'kurtosis': stats.kurtosis(df[col].dropna()),
                'nunique': df[col].nunique(),
                'missing_pct': df[col].isnull().mean()
            }
            
        return stats_dict
    
    def _check_label_variability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check variability in label columns."""
        label_cols = [col for col in df.columns if col.endswith('_label')]
        label_stats = {'all_labels': {}, 'low_variance_labels': {}}
        
        for col in label_cols:
            stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min(),
                'skewness': stats.skew(df[col].dropna()),
                'kurtosis': stats.kurtosis(df[col].dropna())
            }
            label_stats['all_labels'][col] = stats
            
            # Check for low variance
            if stats['std'] < self.variance_threshold:
                label_stats['low_variance_labels'][col] = stats
                
        return label_stats