#!/usr/bin/env python3
"""
Quick model retraining script to fix legacy model format issues.
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import asyncio
import logging
from app.ml_models.random_forest_trainer import RandomForestTrainer
from app.services.training_data_generator import TrainingDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def retrain_models():
    """Retrain all ML models with updated pipeline."""
    try:
        logger.info("=== Quick Model Retraining ===")
        
        # Generate training data with reasonable size
        generator = TrainingDataGenerator()
        dataset_path = await generator.generate_training_dataset(
            sample_size=200,  # Reasonable size for quick training
            output_file="retrain_dataset.csv"
        )
        
        logger.info(f"Generated training dataset: {dataset_path}")
        
        # Train Random Forest models
        trainer = RandomForestTrainer(models_dir="models")
        results = trainer.train_all_models(dataset_path)
        
        logger.info("=== Training Results ===")
        logger.info(f"Models trained: {results['models_trained']}")
        logger.info(f"Training completed successfully!")
        
        # Print model performance summary
        for model_name in results['models_trained']:
            metrics = results['evaluation_metrics'][model_name]
            logger.info(f"{model_name}: R²={metrics.get('r2', 'N/A'):.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(retrain_models())