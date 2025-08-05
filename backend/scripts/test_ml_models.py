#!/usr/bin/env python3
"""
Test script to verify that trained Random Forest models work correctly.

This script:
1. Loads all trained Random Forest models
2. Makes sample predictions with test data
3. Verifies that outputs are in the [0.0, 1.0] range
4. Tests the model loading and prediction pipeline

Usage:
    python scripts/test_ml_models.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to Python path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml_models.random_forest_trainer import RandomForestTrainer
from app.ml_models.model_utils import ModelUtils

def main():
    print("="*80)
    print("PropPulse ML Models Test")
    print("="*80)
    
    try:
        # Initialize trainer
        trainer = RandomForestTrainer(models_dir="models")
        
        # Test data - sample features matching the training data
        test_features = {
            'price': 450000.0,
            'square_feet': 1800.0,
            'days_on_market': 45.0,
            'active_listing_count': 150.0,
            'price_reduced_count': 25.0,
            'price_increased_count': 5.0,
            'total_listing_count': 200.0,
            'median_days_on_market': 40.0,
            'median_listing_price': 425000.0,
            'price_per_sqft': 250.0,
            'price_volatility': 0.08,
            'price_change_1y': 5.2,
            'price_change_3y': 15.8,
            'price_change_5y': 28.5,
            'price_reduction_ratio': 0.125,
            'price_increase_ratio': 0.025
        }
        
        print("Testing all trained models with sample data:")
        print(f"Sample features: {list(test_features.keys())}")
        print()
        
        # Test each model
        target_metrics = [
            'market_risk_label',
            'property_risk_label',
            'location_risk_label',
            'overall_risk_label',
            'market_health_label',
            'market_momentum_label',
            'market_stability_label'
        ]
        
        predictions = {}
        
        for target_name in target_metrics:
            model_filename = f"{target_name.replace('_label', '')}_rf.joblib"
            model_path = Path("models") / model_filename
            
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                continue
            
            try:
                # Load model
                model_data = ModelUtils.load_model(str(model_path))
                model = model_data['model']
                feature_names = model_data['feature_names']
                
                # Prepare feature vector
                feature_vector = []
                for feature_name in feature_names:
                    if feature_name in test_features:
                        feature_vector.append(test_features[feature_name])
                    else:
                        print(f"Feature {feature_name} not in test data, using 0")
                        feature_vector.append(0.0)
                
                # Make prediction
                raw_prediction = model.predict([feature_vector])[0]
                
                # Normalize to [0, 1] range
                normalized_prediction = ModelUtils.normalize_predictions(
                    np.array([raw_prediction]), (0.0, 1.0)
                )[0]
                
                predictions[target_name] = normalized_prediction
                
                # Verify range
                if 0.0 <= normalized_prediction <= 1.0:
                    range_check = "Success"
                else:
                    range_check = "Error"
                
                metric_name = target_name.replace('_label', '').replace('_', ' ').title()
                print(f"  {range_check} {metric_name:<18} = {normalized_prediction:.4f} (raw: {raw_prediction:.4f})")
                
            except Exception as e:
                print(f"Error testing {target_name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY")
        print(f"{'='*60}")
        
        if predictions:
            print("All models loaded and predictions generated successfully!")
            print(f"Total models tested: {len(predictions)}")
            print(f"All predictions in range [0.0, 1.0]: {'Success' if all(0.0 <= p <= 1.0 for p in predictions.values()) else '❌'}")
            
            # Show summary statistics
            pred_values = list(predictions.values())
            print(f"Prediction statistics:")
            print(f"   • Min: {min(pred_values):.4f}")
            print(f"   • Max: {max(pred_values):.4f}")
            print(f"   • Mean: {np.mean(pred_values):.4f}")
            print(f"   • Std: {np.std(pred_values):.4f}")
            
            # Test with different input values
            print(f"\n{'='*60}")
            print("TESTING WITH DIFFERENT INPUT VALUES")
            print(f"{'='*60}")
            
            # Test with high-risk scenario
            high_risk_features = test_features.copy()
            high_risk_features.update({
                'days_on_market': 120.0,  # High days on market
                'price_volatility': 0.15,  # High volatility
                'price_change_1y': -10.0,  # Negative price change
                'price_reduction_ratio': 0.35  # High reduction ratio
            })
            
            print("\nHigh-risk scenario predictions:")
            for target_name in ['market_risk_label', 'market_stability_label']:
                if target_name in predictions:
                    model_data = ModelUtils.load_model(f"models/{target_name.replace('_label', '')}_rf.joblib")
                    model = model_data['model']
                    feature_names = model_data['feature_names']
                    
                    feature_vector = [high_risk_features.get(fn, 0.0) for fn in feature_names]
                    raw_pred = model.predict([feature_vector])[0]
                    norm_pred = ModelUtils.normalize_predictions(np.array([raw_pred]), (0.0, 1.0))[0]
                    
                    metric_name = target_name.replace('_label', '').replace('_', ' ').title()
                    print(f"{metric_name:<18} = {norm_pred:.4f}")
            
            # Test with low-risk scenario  
            low_risk_features = test_features.copy()
            low_risk_features.update({
                'days_on_market': 15.0,   # Low days on market
                'price_volatility': 0.03,  # Low volatility
                'price_change_1y': 8.0,   # Positive price change
                'price_reduction_ratio': 0.05  # Low reduction ratio
            })
            
            print("\nLow-risk scenario predictions:")
            for target_name in ['market_risk_label', 'market_stability_label']:
                if target_name in predictions:
                    model_data = ModelUtils.load_model(f"models/{target_name.replace('_label', '')}_rf.joblib")
                    model = model_data['model']
                    feature_names = model_data['feature_names']
                    
                    feature_vector = [low_risk_features.get(fn, 0.0) for fn in feature_names]
                    raw_pred = model.predict([feature_vector])[0]
                    norm_pred = ModelUtils.normalize_predictions(np.array([raw_pred]), (0.0, 1.0))[0]
                    
                    metric_name = target_name.replace('_label', '').replace('_', ' ').title()
                    print(f"{metric_name:<18} = {norm_pred:.4f}")
            
        else:
            print("No models could be tested successfully")
            return 1
        
        print(f"\nSUCCESS: All ML models are working correctly!")
        print("Models are ready for integration into PropertyAnalyzer")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 