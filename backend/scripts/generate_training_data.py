#!/usr/bin/env python3
"""
Script to generate clean training dataset for ML-powered metric calculations.

This script:
1. Audits available real features from Realtor API and CSV data sources (excludes placeholders)
2. Generates clean training dataset with only real features and computed derived features
3. Filters out records with missing required features
4. Computes current heuristic scores as labels
5. Saves the cleaned dataset to CSV for ML model training

Usage:
    python scripts/generate_training_data.py [--sample-size 1000] [--output training_dataset_clean.csv]
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add the parent directory to Python path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.training_data_generator import TrainingDataGenerator


async def main():
    parser = argparse.ArgumentParser(description='Generate clean training dataset for ML models')
    parser.add_argument('--sample-size', type=int, default=1000, 
                       help='Number of training samples to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='training_dataset_clean.csv',
                       help='Output CSV filename (default: training_dataset_clean.csv)')
    parser.add_argument('--audit-only', action='store_true',
                       help='Only run feature audit, do not generate dataset')
    
    args = parser.parse_args()
    
    print("=== PropPulse ML Clean Training Data Generator ===")
    print(f"Sample size: {args.sample_size}")
    print(f"Output file: {args.output}")
    print("NOTE: This version filters out placeholder features and computes derived features from real data only.")
    print()
    
    generator = TrainingDataGenerator()
    
    # Always run feature audit first
    print("=== FEATURE AUDIT (REAL FEATURES ONLY) ===")
    feature_audit = generator.audit_available_features()
    
    for metric, features in feature_audit.items():
        print(f"\n{metric.upper().replace('_', ' ')}:")
        for feature in features:
            print(f"  ✓ {feature}")
    
    print("\n" + "="*60)
    print("EXCLUDED PLACEHOLDER FEATURES:")
    print("  ✗ crime_rate (always 50)")
    print("  ✗ school_rating (always 5)")  
    print("  ✗ walk_score (always 50)")
    print("  ✗ unemployment_rate (always 5)")
    print("  ✗ flood_zone (always False)")
    print("  ✗ needs_renovation (always False)")
    print("  ✗ recently_renovated (always False)")
    print("  ✗ population_growth (always 0)")
    print("  ✗ employment_growth (always 0)")
    print("  ✗ income_growth (always 0)")
    print("  ✗ inventory_volatility (always 0.1)")
    print("  ✗ latitude/longitude (often 0.0 from API)")
    print("="*60)
    
    if args.audit_only:
        print("Audit complete. Exiting (--audit-only flag set).")
        return
    
    # Generate clean training dataset
    print("\n=== GENERATING CLEAN TRAINING DATASET ===")
    try:
        output_path = await generator.generate_training_dataset(
            sample_size=args.sample_size,
            output_file=args.output
        )
        
        print(f"\nSUCCESS: Clean training dataset generated!")
        print(f"Location: {output_path}")
        
        # Show basic dataset info
        import pandas as pd
        df = pd.read_csv(output_path)
        print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show feature columns (excluding labels and metadata)
        feature_cols = [col for col in df.columns if not col.endswith('_label') 
                       and col not in ['property_id', 'data_source', 'month_date', 'cbsa_code']]
        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        
        # Show label columns
        label_cols = [col for col in df.columns if col.endswith('_label')]
        print(f"Label columns ({len(label_cols)}): {label_cols}")
        
        # Show sample of data with key features
        key_features = ['price', 'square_feet', 'active_listing_count', 'price_volatility', 
                       'price_change_1y', 'market_risk_label', 'property_risk_label']
        available_features = [f for f in key_features if f in df.columns]
        
        print(f"\n Sample of generated clean data:")
        print(df[available_features].head())
        
        # Show feature statistics
        print(f"\nFeature Statistics:")
        for col in feature_cols[:10]:  # Show first 10 features
            if df[col].dtype in ['int64', 'float64']:
                print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                     f"mean={df[col].mean():.2f}, unique_values={df[col].nunique()}")
        
        if len(feature_cols) > 10:
            print(f"  ... and {len(feature_cols) - 10} more features")
        
    except Exception as e:
        print(f"ERROR: Failed to generate clean training dataset: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 