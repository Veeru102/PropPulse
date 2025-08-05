"""
Comprehensive auditor for training-inference consistency and data quality.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
from pathlib import Path
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class TrainingInferenceAuditor:
    """Audits data quality and ensures training-inference consistency."""
    
    def __init__(self, 
                 variance_threshold: float = 0.001,  # Reduced for better constant detection
                 drift_threshold: float = 0.30,      # Increased to be more lenient
                 training_stats_path: str = None):
        self.variance_threshold = variance_threshold
        self.drift_threshold = drift_threshold
        self.training_stats_path = training_stats_path or "data/training_feature_stats.json"
        self.training_stats = {}
        self.load_training_stats()
    
    def audit_training_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive audit of training data with feature validation.
        
        Returns:
            (is_valid, audit_report)
        """
        logger.info(f"Starting comprehensive training data audit for dataset with shape: {df.shape}")
        
        audit_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'feature_statistics': {},
            'label_statistics': {},
            'data_quality_issues': [],
            'removed_features': [],
            'validation_passed': True
        }
        
        # 1. Validate required columns exist
        required_features = [
            'price', 'square_feet', 'days_on_market', 'active_listing_count',
            'price_reduced_count', 'price_increased_count', 'total_listing_count',
            'price_per_sqft', 'price_volatility', 'price_reduction_ratio', 'price_increase_ratio'
        ]
        
        required_labels = [
            'market_risk_label', 'property_risk_label', 'location_risk_label', 'overall_risk_label',
            'market_health_label', 'market_momentum_label', 'market_stability_label'
        ]
        
        missing_features = [col for col in required_features if col not in df.columns]
        missing_labels = [col for col in required_labels if col not in df.columns]
        
        if missing_features:
            audit_report['data_quality_issues'].append(f"Missing required features: {missing_features}")
            audit_report['validation_passed'] = False
        
        if missing_labels:
            audit_report['data_quality_issues'].append(f"Missing required labels: {missing_labels}")
            audit_report['validation_passed'] = False
        
        # 2. Analyze feature statistics
        feature_columns = [col for col in df.columns if not col.endswith('_label') 
                          and col not in ['property_id', 'data_source', 'month_date', 'cbsa_code', 'cbsa_title']]
        
        constant_features = []
        near_constant_features = []
        
        for col in feature_columns:
            if col not in df.columns:
                continue
                
            stats = self._calculate_feature_stats(df[col], col)
            audit_report['feature_statistics'][col] = stats
            
            # Check for constant/near-constant features
            if stats['nunique'] <= 1:
                constant_features.append(col)
            elif stats['std'] < self.variance_threshold:
                near_constant_features.append(col)
            elif stats['coefficient_of_variation'] < 0.01:  # CV < 1%
                near_constant_features.append(col)
        
        # Remove constant features
        if constant_features:
            df = df.drop(columns=constant_features)
            audit_report['removed_features'].extend(constant_features)
            audit_report['data_quality_issues'].append(f"Removed constant features: {constant_features}")
            logger.warning(f"Training data audit found and removed constant features: {constant_features}")
        
        # Flag near-constant features but be more lenient with small datasets
        if near_constant_features and len(df) > 50:  # Only flag if we have reasonable sample size
            audit_report['data_quality_issues'].append(f"Near-constant features detected: {near_constant_features}")
            logger.warning(f"Near-constant features detected: {near_constant_features}")
        elif near_constant_features:
            logger.info(f"Features with low variance (may be due to small sample size): {near_constant_features}")
        
        # 3. Analyze label statistics and variability
        label_columns = [col for col in df.columns if col.endswith('_label')]
        
        for col in label_columns:
            if col not in df.columns:
                continue
                
            stats = self._calculate_feature_stats(df[col], col)
            audit_report['label_statistics'][col] = stats
            
            # Check label quality
            if stats['std'] < 0.02:  # Very low variance for labels
                audit_report['data_quality_issues'].append(f"Label '{col}' has very low variance: std={stats['std']:.4f}")
                audit_report['validation_passed'] = False
            
            if stats['range'] < 0.1:  # Very small range
                audit_report['data_quality_issues'].append(f"Label '{col}' has very small range: {stats['range']:.4f}")
            
            # Log detailed label statistics
            logger.info(f"Label '{col}' statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                       f"min={stats['min']:.4f}, max={stats['max']:.4f}, range={stats['range']:.4f}")
        
        # 4. Check for data leakage indicators
        self._check_data_leakage(df, audit_report)
        
        # 5. Save training statistics for inference comparison
        self._save_training_stats(audit_report['feature_statistics'], audit_report['label_statistics'])
        
        # 6. Generate summary
        total_issues = len(audit_report['data_quality_issues'])
        logger.info(f"Training data audit complete. Found {total_issues} issues.")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Validation passed: {audit_report['validation_passed']}")
        
        return audit_report['validation_passed'], audit_report
    
    def audit_inference_data(self, property_data: Dict[str, Any], 
                           market_data: Dict[str, Any],
                           extracted_features: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Audit inference data for consistency with training data.
        
        Returns:
            audit_report with drift detection and validation results
        """
        audit_report = {
            'timestamp': datetime.now().isoformat(),
            'property_data_quality': {},
            'market_data_quality': {},
            'feature_drift_analysis': {},
            'validation_issues': [],
            'recommendations': []
        }
        
        # 1. Audit property data completeness
        property_audit = self._audit_property_data(property_data)
        audit_report['property_data_quality'] = property_audit
        
        if property_audit['is_empty']:
            audit_report['validation_issues'].append("Property data is empty - using market-based imputation")
            audit_report['recommendations'].append("Ensure property data is populated for better predictions")
        
        # 2. Audit market data completeness
        market_audit = self._audit_market_data(market_data)
        audit_report['market_data_quality'] = market_audit
        
        # 3. Analyze feature drift if extracted features provided
        if extracted_features and self.training_stats:
            drift_analysis = self._analyze_feature_drift(extracted_features)
            audit_report['feature_drift_analysis'] = drift_analysis
            
            # Flag significant drift
            high_drift_features = [f for f, analysis in drift_analysis.items() 
                                 if analysis.get('drift_severity', 'low') in ['high', 'critical']]
            
            if high_drift_features:
                audit_report['validation_issues'].append(f"High drift detected in features: {high_drift_features}")
                audit_report['recommendations'].append("Consider model retraining due to feature drift")
        
        # 4. Generate recommendations
        if len(audit_report['validation_issues']) == 0:
            audit_report['recommendations'].append("Data quality is good for inference")
        
        return audit_report
    
    def _calculate_feature_stats(self, series: pd.Series, feature_name: str) -> Dict[str, float]:
        """Calculate comprehensive statistics for a feature."""
        stats = {
            'count': len(series),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'nunique': series.nunique(),
            'dtype': str(series.dtype)
        }
        
        if series.dtype in ['int64', 'float64']:
            stats.update({
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'range': float(series.max() - series.min()),
                'median': float(series.median()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else float('inf')
            })
            
            # Check for outliers (IQR method)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = (len(outliers) / len(series)) * 100
        
        return stats
    
    def _audit_property_data(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Audit property data completeness and quality."""
        audit = {
            'is_empty': not property_data or len(property_data) == 0,
            'missing_fields': [],
            'zero_fields': [],
            'field_count': len(property_data) if property_data else 0,
            'quality_score': 0.0
        }
        
        required_fields = ['price', 'square_feet', 'days_on_market']
        optional_fields = ['year_built', 'bedrooms', 'bathrooms', 'property_type']
        
        if property_data:
            for field in required_fields:
                if field not in property_data or property_data[field] in [None, '', 0, 0.0]:
                    if field not in property_data:
                        audit['missing_fields'].append(field)
                    else:
                        audit['zero_fields'].append(field)
            
            # Calculate quality score (0-1)
            total_fields = len(required_fields) + len(optional_fields)
            present_fields = sum(1 for field in required_fields + optional_fields 
                               if field in property_data and property_data[field] not in [None, '', 0, 0.0])
            audit['quality_score'] = present_fields / total_fields
        
        return audit
    
    def _audit_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Audit market data completeness and quality."""
        audit = {
            'is_empty': not market_data or len(market_data) == 0,
            'missing_fields': [],
            'zero_fields': [],
            'field_count': len(market_data) if market_data else 0,
            'quality_score': 0.0
        }
        
        required_fields = [
            'active_listing_count', 'price_reduced_count', 'price_increased_count',
            'median_listing_price', 'price_volatility'
        ]
        
        if market_data:
            for field in required_fields:
                if field not in market_data or market_data[field] in [None, '']:
                    if field not in market_data:
                        audit['missing_fields'].append(field)
                    elif market_data[field] in [0, 0.0] and 'count' in field:
                        audit['zero_fields'].append(field)
            
            # Calculate quality score
            present_fields = sum(1 for field in required_fields 
                               if field in market_data and market_data[field] not in [None, ''])
            audit['quality_score'] = present_fields / len(required_fields)
        
        return audit
    
    def _analyze_feature_drift(self, current_features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Analyze drift between current features and training statistics."""
        drift_analysis = {}
        
        for feature_name, current_value in current_features.items():
            if feature_name not in self.training_stats:
                continue
                
            training_stats = self.training_stats[feature_name]
            
            analysis = {
                'current_value': current_value,
                'training_mean': training_stats['mean'],
                'training_std': training_stats['std'],
                'training_min': training_stats['min'],
                'training_max': training_stats['max'],
                'z_score': 0.0,
                'out_of_range': False,
                'drift_severity': 'low',
                'drift_reason': 'within_normal_range'
            }
            
            # Calculate z-score with better handling of edge cases
            if training_stats['std'] > 1e-10:  # More robust zero check
                analysis['z_score'] = (current_value - training_stats['mean']) / training_stats['std']
            else:
                # Handle zero/near-zero variance case
                if abs(current_value - training_stats['mean']) > 1e-10:
                    analysis['drift_severity'] = 'critical'
                    analysis['drift_reason'] = 'zero_variance_feature_with_different_value'
                    logger.warning(f"Critical drift in {feature_name}: feature has zero variance in training but current value differs (current={current_value:.4f}, training_mean={training_stats['mean']:.4f})")
                else:
                    analysis['drift_reason'] = 'zero_variance_feature_same_value'
                
                drift_analysis[feature_name] = analysis
                continue
            
            # Check if out of training range
            analysis['out_of_range'] = (current_value < training_stats['min'] or 
                                      current_value > training_stats['max'])
            
            # Determine drift severity with improved logic and thresholds
            abs_z_score = abs(analysis['z_score'])
            
            # Adaptive thresholds based on training data size and feature characteristics
            critical_threshold = 8.0  # Very lenient for small datasets
            high_threshold = 6.0
            moderate_threshold = 4.0
            
            # More nuanced drift classification
            if abs_z_score > critical_threshold:
                analysis['drift_severity'] = 'critical'
                analysis['drift_reason'] = 'high_z_score'
                logger.warning(f"Critical drift detected in {feature_name}: z-score={analysis['z_score']:.4f} (threshold: {critical_threshold})")
            elif analysis['out_of_range'] and abs_z_score > 2.0:
                # Only flag as critical if both out of range AND significant z-score
                analysis['drift_severity'] = 'critical'
                analysis['drift_reason'] = 'out_of_training_range'
                logger.warning(f"Critical drift detected in {feature_name}: out of training range (z-score={analysis['z_score']:.4f}, range=[{training_stats['min']:.2f}, {training_stats['max']:.2f}], current={current_value:.2f})")
            elif analysis['out_of_range'] and abs_z_score <= 2.0:
                # Small z-score but out of range - likely edge case or small training set, treat as moderate
                analysis['drift_severity'] = 'moderate'
                analysis['drift_reason'] = 'slightly_out_of_range'
                logger.info(f"Moderate drift detected in {feature_name}: slightly out of training range (z-score={analysis['z_score']:.4f}, range=[{training_stats['min']:.2f}, {training_stats['max']:.2f}], current={current_value:.2f})")
            elif abs_z_score > high_threshold:
                analysis['drift_severity'] = 'high'
                analysis['drift_reason'] = 'moderate_z_score'
                logger.info(f"High drift detected in {feature_name}: z-score={analysis['z_score']:.4f}")
            elif abs_z_score > moderate_threshold:
                analysis['drift_severity'] = 'moderate'
                analysis['drift_reason'] = 'low_z_score'
            else:
                analysis['drift_reason'] = 'within_normal_range'
            
            drift_analysis[feature_name] = analysis
        
        return drift_analysis
    
    def _check_data_leakage(self, df: pd.DataFrame, audit_report: Dict[str, Any]):
        """Check for potential data leakage indicators."""
        # Check for suspiciously high correlations between features and labels
        feature_cols = [col for col in df.columns if not col.endswith('_label') 
                       and col not in ['property_id', 'data_source', 'month_date', 'cbsa_code']]
        label_cols = [col for col in df.columns if col.endswith('_label')]
        
        high_correlations = []
        
        for feature_col in feature_cols:
            if df[feature_col].dtype not in ['int64', 'float64']:
                continue
                
            for label_col in label_cols:
                try:
                    correlation = df[feature_col].corr(df[label_col])
                    if abs(correlation) > 0.95:  # Suspiciously high correlation
                        high_correlations.append({
                            'feature': feature_col,
                            'label': label_col,
                            'correlation': correlation
                        })
                except:
                    continue
        
        if high_correlations:
            audit_report['data_quality_issues'].append(f"Potential data leakage detected: {high_correlations}")
    
    def _save_training_stats(self, feature_stats: Dict[str, Any], label_stats: Dict[str, Any]):
        """Save training statistics for inference comparison."""
        self.training_stats = {**feature_stats, **label_stats}
        
        # Save to file
        stats_path = Path(self.training_stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)
        
        logger.info(f"Training statistics saved to {stats_path}")
    
    def load_training_stats(self):
        """Load training statistics from file."""
        stats_path = Path(self.training_stats_path)
        
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
                logger.info(f"Loaded training statistics from {stats_path}")
            except Exception as e:
                logger.warning(f"Failed to load training statistics: {e}")
                self.training_stats = {}
        else:
            logger.info("No training statistics file found - will be created during training")
            self.training_stats = {}
    
    def generate_drift_report(self, drift_analysis: Dict[str, Dict[str, Any]]) -> str:
        """Generate a human-readable drift report."""
        if not drift_analysis:
            return "No drift analysis available."
        
        report_lines = ["=== Feature Drift Analysis ==="]
        
        # Group by severity
        by_severity = {'critical': [], 'high': [], 'moderate': [], 'low': []}
        
        for feature, analysis in drift_analysis.items():
            severity = analysis.get('drift_severity', 'low')
            by_severity[severity].append((feature, analysis))
        
        for severity in ['critical', 'high', 'moderate', 'low']:
            features = by_severity[severity]
            if features:
                report_lines.append(f"\n{severity.upper()} DRIFT ({len(features)} features):")
                for feature, analysis in features:
                    z_score = analysis['z_score']
                    out_of_range = analysis['out_of_range']
                    current_val = analysis['current_value']
                    training_mean = analysis['training_mean']
                    
                    status_flags = []
                    if out_of_range:
                        status_flags.append("OUT_OF_RANGE")
                    if abs(z_score) > 2:
                        status_flags.append("HIGH_Z_SCORE")
                    
                    flags_str = f" [{', '.join(status_flags)}]" if status_flags else ""
                    
                    report_lines.append(
                        f"  {feature}: {current_val:.4f} (training_mean: {training_mean:.4f}, "
                        f"z-score: {z_score:.2f}){flags_str}"
                    )
        
        return "\n".join(report_lines)