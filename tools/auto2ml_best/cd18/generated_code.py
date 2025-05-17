#!/usr/bin/env python
"""
AutoGluon script for predicting cellphone prices based on specifications.

This script:
1. Loads training and test data
2. Preprocesses the data (removes unnecessary columns, handles missing values)
3. Trains a model using AutoGluon TabularPredictor
4. Makes predictions on test data
5. Saves the predictions to a results file

Installation requirements:
pip install autogluon.tabular
"""

import os
from datetime import datetime

import pandas as pd
from autogluon.tabular import TabularPredictor

if __name__ == "__main__":
    # Define paths
    output_dir = "./"
    model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f"model_{model_timestamp}")
    
    # Load data
    train_data_path = "/media/agent/maab/datasets/cd18/training/train.csv"
    test_data_path = "/media/agent/maab/datasets/cd18/training/inference.csv"
    
    print("Loading training data...")
    train_data = pd.read_csv(train_data_path)
    
    print("Loading test data...")
    test_data = pd.read_csv(test_data_path)
    
    # Data exploration
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Check for missing values in the target variable
    print(f"Missing values in Price column: {train_data['Price'].isna().sum()}")
    
    # Remove samples without valid labels
    train_data = train_data.dropna(subset=['Price'])
    print(f"Training data shape after removing invalid labels: {train_data.shape}")
    
    # Check if there's an unnecessary index column
    # The data doesn't seem to have an explicit index column to remove
    
    # Feature analysis
    # Check if there are features in train but not in test or vice versa
    train_cols = set(train_data.columns)
    test_cols = set(test_data.columns)
    
    print(f"Columns in train but not in test: {train_cols - test_cols}")
    print(f"Columns in test but not in train: {test_cols - train_cols}")
    
    # Train AutoGluon model
    print("Training model with AutoGluon...")
    predictor = TabularPredictor(
        label='Price',
        path=model_save_path,
        problem_type='regression',  # Price prediction is a regression task
        eval_metric='root_mean_squared_error'  # Common metric for regression
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Print model leaderboard
    print("Model leaderboard:")
    leaderboard = predictor.leaderboard(train_data)
    print(leaderboard)
    
    # Make predictions on test data
    print("Making predictions on test data...")
    predictions = predictor.predict(test_data)
    
    # Create results dataframe
    results = test_data.copy()
    results['Price'] = predictions
    
    # Save results to CSV
    results_path = os.path.join(output_dir, "results.csv")
    print(f"Saving results to {results_path}")
    results.to_csv(results_path, index=False)
    
    print("Prediction completed successfully!")