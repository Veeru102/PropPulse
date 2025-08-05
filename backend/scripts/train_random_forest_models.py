#!/usr/bin/env python3
"""
Script to train Random Forest models for all risk and market health metrics.

This script:
1. Loads the clean training dataset from Phase 1.5
2. Trains RandomForestRegressor models for each target metric
3. Evaluates models with MAE, RMSE, and R² metrics
4. Normalizes predictions to [0.0, 1.0] range using min-max scaling
5. Saves trained models as .joblib files
6. Provides comprehensive training summary and feature importance analysis

Usage:
    python scripts/train_random_forest_models.py [--dataset data/training_dataset_clean.csv] [--test-size 0.2]
"""

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path

# Add the parent directory to Python path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml_models.random_forest_trainer import RandomForestTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description='Train Random Forest models for risk metrics')
    parser.add_argument('--dataset', type=str, default='data/training_dataset_clean.csv',
                       help='Path to clean training dataset (default: data/training_dataset_clean.csv)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in Random Forest (default: 100)')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of trees (default: 10)')
    parser.add_argument('--min-samples-split', type=int, default=5,
                       help='Minimum samples required to split node (default: 5)')
    parser.add_argument('--min-samples-leaf', type=int, default=2,
                       help='Minimum samples required at leaf node (default: 2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PropPulse Random Forest Model Training")
    print("="*80)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔀 Test size: {args.test_size}")
    print(f"📁 Models directory: {args.models_dir}")
    print(f"🌳 Random Forest parameters:")
    print(f"   - n_estimators: {args.n_estimators}")
    print(f"   - max_depth: {args.max_depth}")
    print(f"   - min_samples_split: {args.min_samples_split}")
    print(f"   - min_samples_leaf: {args.min_samples_leaf}")
    print(f"   - random_state: {args.random_state}")
    print()
    
    # Check if dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print("Please run the training data generation script first:")
        print("  python scripts/generate_training_data.py --sample-size 1000")
        sys.exit(1)
    
    try:
        # Initialize trainer
        trainer = RandomForestTrainer(models_dir=args.models_dir)
        
        # Prepare Random Forest parameters
        rf_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'random_state': args.random_state,
            'n_jobs': -1  # Use all available cores
        }
        
        # Train all models
        print("🚀 Starting model training...")
        training_results = trainer.train_all_models(
            dataset_path=str(dataset_path),
            test_size=args.test_size,
            **rf_params
        )
        
        # Print comprehensive summary
        trainer.print_training_summary(training_results)
        
        # Additional detailed analysis
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS")
        print(f"{'='*80}")
        
        models_trained = training_results['models_trained']
        
        # Show detailed metrics for each model
        print(f"\n📊 DETAILED EVALUATION METRICS:")
        print("-" * 60)
        
        for target_name in models_trained:
            model_name = target_name.replace('_label', '').replace('_', ' ').title()
            metrics = training_results['evaluation_metrics'][target_name]
            
            print(f"\n{model_name}:")
            print(f"  • Mean Absolute Error (MAE): {metrics['mae']:.6f}")
            print(f"  • Root Mean Square Error (RMSE): {metrics['rmse']:.6f}")
            print(f"  • R-squared (R²): {metrics['r2']:.6f}")
            print(f"  • Prediction Range: [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}]")
            print(f"  • Prediction Mean: {metrics['pred_mean']:.4f}")
        
        # Show feature importance for each model
        print(f"\n🔍 FEATURE IMPORTANCE BY MODEL:")
        print("-" * 60)
        
        for target_name in models_trained[:3]:  # Show first 3 to save space
            model_name = target_name.replace('_label', '').replace('_', ' ').title()
            importance_df = training_results['feature_importance'][target_name]
            
            print(f"\n{model_name} - Top 5 Features:")
            for i, (_, row) in enumerate(importance_df.head().iterrows(), 1):
                print(f"  {i}. {row['feature']:<20} {row['importance']:.4f}")
        
        # Model files summary
        print(f"\n💾 SAVED MODEL FILES:")
        print("-" * 60)
        models_dir = Path(args.models_dir)
        for target_name in models_trained:
            model_filename = f"{target_name.replace('_label', '')}_rf.joblib"
            model_path = models_dir / model_filename
            file_size = model_path.stat().st_size / 1024  # KB
            print(f"  📄 {model_filename:<30} ({file_size:.1f} KB)")
        
        # Usage instructions
        print(f"\n🎯 NEXT STEPS:")
        print("-" * 60)
        print("1. Models are ready to replace rule-based heuristics in PropertyAnalyzer")
        print("2. Each model outputs continuous scores in range [0.0, 1.0]")
        print("3. Models can be loaded using:")
        print("   from app.ml_models import RandomForestTrainer")
        print("   trainer = RandomForestTrainer()")
        print("   model_data = trainer.load_trained_model('market_risk_label')")
        print("4. For integration, update PropertyAnalyzer to use ML predictions")
        
        print(f"\n✅ SUCCESS: All Random Forest models trained successfully!")
        print(f"📁 Models saved to: {Path(args.models_dir).resolve()}")
        
    except Exception as e:
        print(f"❌ ERROR: Training failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 