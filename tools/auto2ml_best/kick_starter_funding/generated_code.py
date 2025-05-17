#!/usr/bin/env python3
"""
AutoGluon script for predicting the final status of Kickstarter projects.

This script:
1. Loads training and test data
2. Preprocesses the data by removing unnecessary columns
3. Trains an AutoGluon TabularPredictor model
4. Makes predictions on the test data
5. Saves the predictions to a results file

Installation requirements:
pip install autogluon.tabular
pip install pandas
"""

import os
import time

import pandas as pd
from autogluon.tabular import TabularPredictor

if __name__ == "__main__":
    # Define paths
    output_dir = "./"
    model_dir = os.path.join(output_dir, f"model_{int(time.time())}")
    
    # Load the data
    train_path = "/media/agent/maab/datasets/kick_starter_funding/training/train.csv"
    test_path = "/media/agent/maab/datasets/kick_starter_funding/training/inference.csv"
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    
    # Check for missing labels in training data
    print("Missing labels in training data:", train_data['final_status'].isna().sum())
    
    # Remove rows with missing labels if any
    if train_data['final_status'].isna().sum() > 0:
        train_data = train_data.dropna(subset=['final_status'])
        print("After removing missing labels, train data shape:", train_data.shape)
    
    # Check if there's an index column to remove
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    
    # Determine problem type
    unique_values = train_data['final_status'].nunique()
    if unique_values == 2:
        problem_type = 'binary'
    elif unique_values > 2:
        problem_type = 'multiclass'
    else:
        problem_type = 'regression'
    
    print(f"Detected problem type: {problem_type}")
    
    # Train the model
    print("Training the model...")
    predictor = TabularPredictor(
        label='final_status',
        path=model_dir,
        problem_type=problem_type
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Print model leaderboard
    print("Model leaderboard:")
    leaderboard = predictor.leaderboard(train_data)
    print(leaderboard)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(test_data)
    
    # Create results dataframe
    results = test_data.copy()
    results['final_status'] = predictions
    
    # Save results
    results_path = os.path.join(output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Print feature importance if available
    try:
        print("Feature importance:")
        feature_importance = predictor.feature_importance(train_data)
        print(feature_importance)
    except:
        print("Could not calculate feature importance")