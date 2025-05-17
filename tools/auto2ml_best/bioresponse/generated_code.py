"""
This script uses AutoGluon Tabular to train a binary classification model on the bioresponse dataset.
It loads training data, trains a model, and generates predictions on test data.

Usage:
    python autogluon_bioresponse.py

Additional installation requirements:
    pip install autogluon.tabular
"""

import os
import time

import pandas as pd
from autogluon.tabular import TabularPredictor

if __name__ == "__main__":
    # Define paths
    input_dir = '/media/agent/maab/datasets/bioresponse/training'
    output_dir = "./"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training and test data
    print("Loading training data...")
    train_data = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    print("Loading test data...")
    test_data = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Check for missing values in the target column
    missing_target = train_data['target'].isna().sum()
    if missing_target > 0:
        print(f"Found {missing_target} rows with missing target values. Removing them...")
        train_data = train_data.dropna(subset=['target'])
    
    # Create a model directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    
    # Train the model
    print("Training model...")
    predictor = TabularPredictor(
        label='target',
        path=model_dir,
        problem_type='binary',  # Explicitly set as binary classification
        eval_metric='roc_auc'   # Good metric for binary classification
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Print model performance summary
    print("Model training completed. Leaderboard:")
    leaderboard = predictor.leaderboard()
    print(leaderboard)
    
    # Generate predictions
    print("Generating predictions on test data...")
    test_pred = predictor.predict(test_data)
    
    # Create results file
    results = test_data.copy()
    results['target'] = test_pred
    
    # Save results
    results_path = os.path.join(output_dir, 'results.csv')
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Save feature importance if available
    try:
        feature_importance = predictor.feature_importance(train_data)
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'))
        print("Feature importance saved.")
    except:
        print("Could not generate feature importance.")
    
    print("Process completed successfully.")