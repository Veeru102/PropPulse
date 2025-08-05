#!/usr/bin/env python3
"""
Test script for the enhanced ML pipeline with focus on location_risk constant predictions.
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pandas as pd
import numpy as np
import logging
from app.services.training_data_generator import TrainingDataGenerator
from app.services.enhanced_model_trainer import EnhancedModelTrainer
from app.services.property_analyzer import PropertyAnalyzer
from app.services.training_inference_auditor import TrainingInferenceAuditor
from app.services.input_data_validator import InputDataValidator
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_training_data_generation():
    """Test the enhanced training data generation."""
    logger.info("=== Testing Enhanced Training Data Generation ===")
    
    try:
        generator = TrainingDataGenerator()
        
        # Generate a reasonably sized dataset for meaningful ML training
        output_path = await generator.generate_training_dataset(
            sample_size=100500,  # Increased sample size for more robust testing
            output_file="test_training_dataset.csv"
        )
        
        logger.info(f"Training dataset generated: {output_path}")
        
        # Load and analyze the dataset
        df = pd.read_csv(output_path)
        logger.info(f"Dataset shape: {df.shape}")
        
        # Analyze label distributions
        label_cols = [col for col in df.columns if col.endswith('_label')]
        logger.info("Label distribution analysis:")
        
        for col in label_cols:
            if col in df.columns:
                stats = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'nunique': df[col].nunique()
                }
                logger.info(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                           f"range={stats['max']-stats['min']:.4f}, unique={stats['nunique']}")
        
        return df, output_path
        
    except Exception as e:
        logger.error(f"Error in training data generation test: {e}")
        raise

def test_model_training(df):
    """Test the enhanced model training."""
    logger.info("=== Testing Enhanced Model Training ===")
    
    try:
        # Check dataset size before training
        if len(df) < 10:
            logger.warning(f"Dataset too small for ML training: {len(df)} samples")
            logger.info("Skipping ML training test due to insufficient data")
            return {'models_trained': [], 'skipped_reason': 'insufficient_data'}
        
        trainer = EnhancedModelTrainer()
        
        # Train models with variance checking
        results = trainer.train_risk_models(df)
        
        logger.info("Training results summary:")
        logger.info(f"Models trained: {results['models_trained']}")
        logger.info(f"Models with variance issues: {results['variance_issues']}")
        logger.info(f"Models retrained: {results['retrained_models']}")
        
        # Analyze specific model performance
        for model_name in results['models_trained']:
            metrics = results['training_metrics'][model_name]
            logger.info(f"{model_name} metrics:")
            r2 = metrics.get('r2', 'N/A')
            pred_std = metrics.get('prediction_std', 'N/A')
            if isinstance(r2, (int, float)):
                logger.info(f"  R2: {r2:.4f}")
            else:
                logger.info(f"  R2: {r2}")
            if isinstance(pred_std, (int, float)):
                logger.info(f"  Prediction std: {pred_std:.6f}")
            else:
                logger.info(f"  Prediction std: {pred_std}")
        
        # Seamlessly integrate new models by copying them to the base model directory
        # This ensures PropertyAnalyzer will load the new format models instead of legacy ones
        logger.info("=== Integrating New Models into Pipeline ===")
        import shutil
        import os
        import joblib
        from pathlib import Path
        
        # Get absolute path to base model directory
        base_model_dir = os.path.abspath(Path("models"))
        os.makedirs(base_model_dir, exist_ok=True)
        
        integrated_count = 0
        for model_name, model_path in results.get('model_paths', {}).items():
            source_path = Path(model_path)
            dest_path = Path(base_model_dir) / f"{model_name}_rf.joblib"
            
            try:
                if source_path.exists():
                    # Backup existing model if it exists
                    if dest_path.exists():
                        backup_path = dest_path.with_suffix('.joblib.bak')
                        shutil.copy2(dest_path, backup_path)
                        logger.info(f"Backed up existing {model_name} model to: {backup_path}")
                    
                    # Copy new model
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"Integrated new {model_name} model into pipeline: {dest_path}")
                    integrated_count += 1
                else:
                    logger.warning(f"Model file not found: {source_path}")
            except Exception as e:
                logger.error(f"Error integrating {model_name} model: {e}")
        
        if integrated_count > 0:
            logger.info(f"Model integration complete. {integrated_count} legacy models have been replaced with new format models.")
            
            # Verify the integrated models are in the new format
            from app.ml_models.model_utils import ModelUtils
            for model_name in results.get('models_trained', []):
                model_path = Path(base_model_dir) / f"{model_name}_rf.joblib"
                if model_path.exists():
                    try:
                        model_data = joblib.load(model_path)
                        if isinstance(model_data, dict) and 'feature_names' in model_data:
                            logger.info(f"Verified {model_name} is using new format with metadata")
                        else:
                            logger.warning(f"{model_name} may still be in legacy format")
                    except Exception as e:
                        logger.error(f"Error verifying {model_name} format: {e}")
        else:
            logger.warning("No models were integrated. Check model paths and permissions.")
        
        return results
        
    except ValueError as e:
        if "Need at least 10 samples" in str(e):
            logger.warning(f"Dataset size validation failed: {e}")
            logger.info("This is expected for very small test datasets")
            return {'models_trained': [], 'skipped_reason': 'dataset_too_small'}
        else:
            logger.error(f"ValueError in model training test: {e}")
            raise
    except Exception as e:
        logger.error(f"Error in model training test: {e}")
        raise

def test_location_risk_investigation(df):
    """Specifically investigate location_risk constant predictions."""
    logger.info("=== Investigating Location Risk Constant Predictions ===")
    
    try:
        # Analyze location_risk_label distribution
        if 'location_risk_label' in df.columns:
            location_risk = df['location_risk_label']
            
            logger.info("Location risk label analysis:")
            logger.info(f"  Unique values: {location_risk.nunique()}")
            logger.info(f"  Value counts (top 10):")
            value_counts = location_risk.value_counts().head(10)
            for value, count in value_counts.items():
                percentage = (count / len(location_risk)) * 100
                logger.info(f"    {value:.6f}: {count} ({percentage:.1f}%)")
            
            # Check for the specific constant value mentioned in logs
            constant_value = 0.61676465
            matches = (location_risk == constant_value).sum()
            if matches > 0:
                logger.warning(f"Found {matches} instances of constant value {constant_value}")
            
            # Analyze by data source
            if 'data_source' in df.columns:
                logger.info("Location risk by data source:")
                for source in df['data_source'].unique():
                    source_data = df[df['data_source'] == source]['location_risk_label']
                    logger.info(f"  {source}: mean={source_data.mean():.4f}, "
                               f"std={source_data.std():.6f}, unique={source_data.nunique()}")
            
            # Check correlation with features
            feature_cols = [col for col in df.columns if not col.endswith('_label') 
                           and col not in ['property_id', 'data_source', 'month_date', 'cbsa_code']]
            
            logger.info("Location risk correlation with top features:")
            correlations = []
            for col in feature_cols:
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        corr = location_risk.corr(df[col])
                        if not np.isnan(corr):
                            correlations.append((col, abs(corr)))
                    except:
                        continue
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            for col, corr in correlations[:10]:
                logger.info(f"  {col}: {corr:.4f}")
        
    except Exception as e:
        logger.error(f"Error in location risk investigation: {e}")
        raise

def test_inference_validation():
    """Test inference-time validation and drift detection."""
    logger.info("=== Testing Inference Validation ===")
    
    try:
        # Test with empty property data
        validator = InputDataValidator()
        
        empty_property_data = {}
        market_data = {
            'active_listing_count': 100,
            'median_listing_price': 300000,
            'median_dom': 45,
            'price_volatility': 0.15
        }
        
        enriched_data, log = validator.validate_and_enrich_property_data(
            empty_property_data, market_data, "test_property"
        )
        
        logger.info("Empty property data validation:")
        logger.info(f"  Applied {len(log)} fixes")
        logger.info(f"  Enriched data keys: {list(enriched_data.keys())}")
        logger.info(f"  Price: ${enriched_data.get('price', 0):,.0f}")
        logger.info(f"  Square feet: {enriched_data.get('square_feet', 0):,.0f}")
        
        # Test drift detection
        auditor = TrainingInferenceAuditor()
        if auditor.training_stats:
            logger.info("Testing drift detection with training stats")
            # Create some test features
            test_features = {
                'price': 500000,  # High price
                'square_feet': 1200,  # Small house
                'days_on_market': 200  # Long DOM
            }
            
            audit_report = auditor.audit_inference_data(enriched_data, market_data, test_features)
            
            drift_analysis = audit_report.get('feature_drift_analysis', {})
            if drift_analysis:
                logger.info("Drift analysis results:")
                for feature, analysis in drift_analysis.items():
                    severity = analysis.get('drift_severity', 'unknown')
                    z_score = analysis.get('z_score', 0)
                    logger.info(f"  {feature}: {severity} drift (z-score: {z_score:.2f})")
        else:
            logger.info("No training statistics available for drift detection")
        
    except Exception as e:
        logger.error(f"Error in inference validation test: {e}")
        raise

def test_property_analysis():
    """Test property analysis with enhanced validation."""
    logger.info("=== Testing Property Analysis ===")
    
    try:
        analyzer = PropertyAnalyzer()
        
        # Test with minimal property data (should trigger validation/imputation)
        minimal_property_data = {
            'property_id': 'test_property_123',
            'price': 350000
        }
        
        market_data = {
            'active_listing_count': 150,
            'median_listing_price': 325000,
            'median_dom': 35,
            'price_volatility': 0.12,
            'price_change_1y': 5.2
        }
        
        # This should trigger our enhanced validation
        result = analyzer.analyze_property(minimal_property_data, market_data)
        
        logger.info("Property analysis with minimal data:")
        logger.info(f"  Analysis completed: {'error' not in result}")
        
        if 'data_quality' in result:
            dq = result['data_quality']
            logger.info(f"  Property imputation applied: {dq.get('property_imputation_applied', False)}")
            logger.info(f"  Market validation applied: {dq.get('market_validation_applied', False)}")
            logger.info(f"  Data quality score: {dq.get('data_quality_score', 0):.2f}")
        
        if 'risk_metrics' in result:
            rm = result['risk_metrics']
            logger.info("Risk metrics:")
            for metric, value in rm.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error in property analysis test: {e}")
        raise

async def main():
    """Run all enhanced pipeline tests."""
    logger.info("Starting Enhanced ML Pipeline Tests")
    
    try:
        # 1. Test training data generation
        df, dataset_path = await test_training_data_generation()
        
        # 2. Investigate location_risk issue
        test_location_risk_investigation(df)
        
        # 3. Test model training
        training_results = test_model_training(df)
        
        # 4. Test inference validation
        test_inference_validation()
        
        # 5. Test property analysis
        test_property_analysis()
        
        logger.info("=== All Tests Completed Successfully ===")
        
        # Summary
        logger.info("Summary:")
        logger.info(f"  Training dataset: {dataset_path}")
        logger.info(f"  Dataset shape: {df.shape}")
        logger.info(f"  Models trained: {len(training_results['models_trained'])}")
        logger.info(f"  Models with variance issues: {len(training_results['variance_issues'])}")
        
        if training_results['variance_issues']:
            logger.warning(f"Models with variance issues: {training_results['variance_issues']}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())