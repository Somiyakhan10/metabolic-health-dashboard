"""
Complete RMR Prediction Pipeline
Run this if Jupyter notebooks fail.
"""

import sys
import os
import pandas as pd
import numpy as np
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_loader import load_config, download_and_merge, save_merged
from feature_engineering import engineer_all_features
from preprocess import run_full_preprocessing
from train_models import train_all_models, select_and_save_best_model
from evaluate import evaluate_all_models
from explain import (compute_shap_values, plot_shap_summary, 
                     compute_permutation_importance)

def main():
    print("="*70)
    print("🧬 RMR Prediction Pipeline - NHANES 2017-2018")
    print("="*70)
    
    # Step 1: Load config
    print("\n[1/6] Loading configuration...")
    config = load_config('config.yaml')
    print(f"✓ Config loaded from config.yaml")
    
    # Step 2: Download and merge data
    print("\n[2/6] Downloading and merging NHANES data...")
    print("   (This requires internet access and may take a few minutes)")
    try:
        merged_df = download_and_merge(config)
        save_merged(merged_df, config)
        print(f"✓ Data merged: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("   Make sure you have internet connection or files in data/raw/")
        return
    
    # Step 3: Feature engineering
    print("\n[3/6] Engineering features...")
    df = engineer_all_features(merged_df, config)
    print(f"✓ Features engineered: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 4: Preprocessing
    print("\n[4/6] Preprocessing data...")
    splits = run_full_preprocessing(df, 'RMR_kcal_day', config)
    
    # Save splits for later
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(splits, 'data/processed/splits.pkl')
    print(f"✓ Preprocessing complete")
    print(f"   Train: {len(splits['X_train'])} samples")
    print(f"   Validation: {len(splits['X_val'])} samples")
    print(f"   Test: {len(splits['X_test'])} samples")
    
    # Step 5: Train models
    print("\n[5/6] Training models (this may take a few minutes)...")
    fitted_models, leaderboard = train_all_models(splits, config)
    
    print("\n   Model Leaderboard (CV R²):")
    print("-"*50)
    for i, row in leaderboard.iterrows():
        print(f"   {i+1}. {row['model']}: {row['mean_cv_r2']:.4f} (+/- {row['std_cv_r2']:.4f})")
    print("-"*50)
    
    # Step 6: Save best model and evaluate
    print("\n[6/6] Saving best model and evaluating...")
    best_path = select_and_save_best_model(fitted_models, leaderboard, config)
    best_name = leaderboard.iloc[0]['model']
    best_model = fitted_models[best_name]
    print(f"✓ Best model: {best_name}")
    print(f"✓ Saved to: {best_path}")
    
    # Evaluate on test set
    test_metrics = evaluate_all_models(fitted_models, splits, config)
    
    print("\n   Test Set Performance:")
    print("-"*50)
    for metric in ['R2', 'MAE', 'RMSE', 'MAPE']:
        value = test_metrics.loc[best_name, metric]
        print(f"   {metric}: {value:.4f}")
    print("-"*50)
    
    # Bonus: SHAP explanation
    print("\n   Generating SHAP explanations...")
    try:
        X_test_key = 'X_test_minmax' if best_name in ('linear_regression','elastic_net') else 'X_test_standard'
        X_bg = splits[X_test_key].sample(min(100, len(splits[X_test_key])), random_state=42)
        X_explain = splits[X_test_key].sample(min(500, len(splits[X_test_key])), random_state=42)
        
        shap_values = compute_shap_values(best_model, X_bg, X_explain)
        os.makedirs('outputs/figures', exist_ok=True)
        plot_shap_summary(shap_values, X_explain, 'outputs/figures/shap_summary.png')
        print("   ✓ SHAP summary plot saved to outputs/figures/shap_summary.png")
        
        # Top features
        perm_imp = compute_permutation_importance(best_model, splits[X_test_key], splits['y_test'], config)
        print("\n   Top 5 Important Features:")
        print("-"*50)
        for i, row in perm_imp.head(5).iterrows():
            print(f"   {i+1}. {row['feature']}: {row['importance_mean']:.4f}")
        print("-"*50)
        
    except Exception as e:
        print(f"   ⚠ SHAP explanation failed: {e}")
    
    print("\n" + "="*70)
    print("✅ Pipeline Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Launch dashboard: streamlit run dashboard/app.py")
    print("2. View report: report/manuscript.md")
    print("3. Check outputs: outputs/figures/ and outputs/reports/")

if __name__ == "__main__":
    main()