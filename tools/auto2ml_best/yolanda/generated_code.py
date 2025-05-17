#!/usr/bin/env python
"""
AutoGluon Tabular Regression Script for Yolanda Dataset

This script trains a regression model using AutoGluon's TabularPredictor to predict column "101"
in the dataset. It handles data preprocessing, model training, and prediction generation.

Usage:
    python autogluon_regression.py

Additional installation requirements:
    pip install autogluon.tabular
"""

import os
from datetime import datetime

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

if __name__ == "__main__":
    # Define paths
    input_dir = "/media/agent/maab/datasets/yolanda/training"
    output_dir = "./"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    
    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv(os.path.join(input_dir, "train.csv"))
    print(f"Training data shape: {train_data.shape}")
    
    # Check for missing values in the target column
    target_column = '101'
    print(f"Number of missing values in target column: {train_data[target_column].isna().sum()}")
    
    # Remove rows with missing target values
    train_data = train_data.dropna(subset=[target_column])
    print(f"Training data shape after removing missing targets: {train_data.shape}")
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv(os.path.join(input_dir, "test.csv"))
    print(f"Test data shape: {test_data.shape}")
    
    # Convert to TabularDataset for AutoGluon
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)
    
    # Train the model
    print("Training the model...")
    predictor = TabularPredictor(
        label=target_column,
        path=model_dir,
        problem_type='regression',  # Explicitly set as regression
        eval_metric='rmse'  # Root Mean Squared Error is a common metric for regression
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Print model performance summary
    print("Model training completed. Summary:")
    print(predictor.fit_summary())
    
    # Show feature importance
    try:
        importance = predictor.feature_importance(train_data)
        print("\nFeature importance:")
        print(importance.head(10))  # Show top 10 important features
    except:
        print("Could not calculate feature importance.")
    
    # Make predictions on test data
    print("Making predictions on test data...")
    predictions = predictor.predict(test_data)
    
    # Create results dataframe with same format as test data
    results = test_data.copy()
    results[target_column] = predictions
    
    # Save predictions to output directory
    results_path = os.path.join(output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    print("Process completed successfully.")