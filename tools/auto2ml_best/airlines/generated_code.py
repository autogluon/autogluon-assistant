#!/usr/bin/env python
"""
Flight Delay Prediction Script

This script uses AutoGluon to predict whether a flight will be delayed.
It loads training and test data, trains a binary classification model,
and generates predictions for the test set.

Installation requirements:
pip install autogluon.tabular
"""

import os
from datetime import datetime

import pandas as pd
from autogluon.tabular import TabularPredictor

if __name__ == "__main__":
    # Define paths
    base_dir = "/media/agent/maab/datasets/airlines/training"
    output_dir = "./"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    
    print("Loading training data from:", train_path)
    train_data = pd.read_csv(train_path)
    
    print("Loading test data from:", test_path)
    test_data = pd.read_csv(test_path)
    
    # Check data
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Data preprocessing
    # Check for missing labels in training data
    print("Checking for missing labels in training data...")
    missing_labels = train_data['Delay'].isna()
    if missing_labels.any():
        print(f"Removing {missing_labels.sum()} rows with missing labels")
        train_data = train_data[~missing_labels]
    
    # Generate model save path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f"model_{timestamp}")
    
    # Train the model
    print("Training model...")
    predictor = TabularPredictor(
        label='Delay',
        path=model_save_path,
        problem_type='binary',  # Binary classification: delay or no delay
        eval_metric='accuracy'
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Model evaluation and information
    print("Model training completed. Leaderboard:")
    leaderboard = predictor.leaderboard()
    print(leaderboard)
    
    # Make predictions on test data
    print("Generating predictions on test data...")
    predictions = predictor.predict(test_data)
    
    # Save predictions to the output directory
    results = test_data.copy()
    results['Delay'] = predictions
    
    results_path = os.path.join(output_dir, "results.csv")
    print(f"Saving predictions to {results_path}")
    results.to_csv(results_path, index=False)
    
    print("Prediction completed successfully!")