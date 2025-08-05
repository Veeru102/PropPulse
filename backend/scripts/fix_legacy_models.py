#!/usr/bin/env python3
"""
Quick script to retrain legacy models with correct feature alignment.
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import asyncio
import logging
import pandas as pd
from app.services.training_data_generator import TrainingDataGenerator
from app.ml_models.random_forest_trainer import RandomForestTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def retrain_legacy_models():
    """Retrain all legacy models with consistent features."""
    try:
        logger.info("=== Retraining Legacy Models ===")
        
        # Generate sufficient training data
        generator = TrainingDataGenerator()
        dataset_path = await generator.generate_training_dataset(
            sample_size=200,  # Reasonable size for stable training
            output_file="legacy_fix_dataset.csv"
        )
        
        logger.info(f"Generated training dataset: {dataset_path}")
        
        # Load and check dataset
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        if len(df) < 50:
            logger.error(f"Dataset too small: {len(df)} samples. Need at least 50.")
            return False
            
        # Train Random Forest models with proper features
        trainer = RandomForestTrainer(models_dir="models")
        results = trainer.train_all_models(dataset_path)
        
        logger.info("=== Training Results ===")
        logger.info(f"Models trained: {results.get('models_trained', [])}")
        
        # Print model performance summary
        if 'evaluation_metrics' in results:
            for model_name, metrics in results['evaluation_metrics'].items():
                r2 = metrics.get('r2', 'N/A')
                logger.info(f"{model_name}: R²={r2:.4f}" if isinstance(r2, (int, float)) else f"{model_name}: R²={r2}")
        
        logger.info("Legacy model retraining completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in legacy model retraining: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(retrain_legacy_models())
    sys.exit(0 if success else 1)