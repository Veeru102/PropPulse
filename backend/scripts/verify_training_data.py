#!/usr/bin/env python3
"""
Verify and prepare training data for XGBoost models.
This script:
1. Loads and validates the training data
2. Checks for sufficient variability in features and labels
3. Reports data quality metrics
4. Saves the validated dataset for model training
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from scipy import stats

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.data_quality_auditor import DataQualityAuditor
from app.core.logging import loggers
from app.core.config import settings

logger = loggers['ml']

def analyze_feature_distributions(df: pd.DataFrame) -> dict:
    """Analyze the distribution of each feature."""
    stats_dict = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skew': stats.skew(df[col].dropna()),
            'kurtosis': stats.kurtosis(df[col].dropna()),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100
        }
    
    return stats_dict

def verify_training_data(data_path: str) -> bool:
    """
    Verify the quality of training data.
    Returns True if data passes all quality checks.
    """
    logger.info(f"Verifying training data from: {data_path}")
    
    try:
        # Load the data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Drop non-numeric identifier columns
        id_cols = ['property_id', 'data_source', 'month_date', 'cbsa_code', 'cbsa_title']
        df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        # Convert all remaining columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Identify feature and label columns
        label_cols = [col for col in df.columns if col.endswith('_label')]
        feature_cols = [col for col in df.columns if not col.endswith('_label')]
        
        logger.info(f"Found {len(label_cols)} label columns and {len(feature_cols)} feature columns")
        
        # Run data quality audit
        auditor = DataQualityAuditor(
            variance_threshold=0.01,
            missing_threshold=0.20,
            correlation_threshold=0.95
        )
        
        passed_audit, audit_results = auditor.audit_dataset(df, label_cols)
        
        # Analyze feature distributions
        dist_stats = analyze_feature_distributions(df)
        
        # Log detailed statistics
        logger.info("\nFeature Distribution Analysis:")
        for col, stats in dist_stats.items():
            logger.info(f"\n{col}:")
            for stat_name, value in stats.items():
                logger.info(f"  {stat_name}: {value:.4f}")
        
        # Check for specific issues
        issues_found = []
        
        # 1. Check for insufficient label variance
        low_variance_labels = [col for col in label_cols if dist_stats[col]['std'] < 0.02]
        if low_variance_labels:
            issues_found.append(f"Labels with insufficient variance: {low_variance_labels}")
        
        # 2. Check for highly correlated features
        if 'high_correlations' in audit_results:
            issues_found.append(f"Highly correlated features: {audit_results['high_correlations']}")
        
        # 3. Check for missing data
        high_missing = [col for col, stats in dist_stats.items() 
                       if stats['missing_pct'] > 20]
        if high_missing:
            issues_found.append(f"Features with >20% missing data: {high_missing}")
        
        # 4. Check for extreme skewness
        extreme_skew = [col for col, stats in dist_stats.items() 
                       if abs(stats['skew']) > 3]
        if extreme_skew:
            issues_found.append(f"Features with extreme skewness: {extreme_skew}")
        
        # Log all issues
        if issues_found:
            logger.error("\nData Quality Issues Found:")
            for issue in issues_found:
                logger.error(f"- {issue}")
        else:
            logger.info("\nNo major data quality issues found.")
        
        return len(issues_found) == 0
        
    except Exception as e:
        logger.error(f"Error verifying training data: {e}", exc_info=True)
        return False

def main():
    """Main entry point."""
    data_dir = Path(settings.BASE_DIR) / "data"
    training_file = data_dir / "training_dataset_clean.csv"
    
    if not training_file.exists():
        logger.error(f"Training data file not found: {training_file}")
        return False
    
    return verify_training_data(str(training_file))

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)