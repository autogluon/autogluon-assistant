"""
AutoGluon Tabular Regression Script for Predicting Number of Rings

This script uses AutoGluon's TabularPredictor to train a regression model
on the provided dataset and make predictions on test data.

Installation requirements:
pip install autogluon.tabular
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from autogluon.tabular import TabularPredictor

# Define working directory
WORKING_DIR = "dncakl"
os.makedirs(WORKING_DIR, exist_ok=True)

if __name__ == "__main__":
    # Load training data
    train_data = pd.read_csv("train.csv")
    
    # Load test data
    test_data = pd.read_csv("test.csv")
    
    # Check if there's an index column to remove
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop('Unnamed: 0', axis=1)
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop('Unnamed: 0', axis=1)
    
    # Remove training samples without valid labels
    train_data = train_data.dropna(subset=['Class_number_of_rings'])
    
    # Define label column
    label = 'Class_number_of_rings'
    
    # Create a timestamp for the model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(WORKING_DIR, f"model_{timestamp}")
    
    # Train the model
    print(f"Training regression model for {label}...")
    predictor = TabularPredictor(
        label=label,
        path=model_dir,
        problem_type='regression',  # Explicitly set as regression
        eval_metric='rmse'  # Root Mean Squared Error as specified
    ).fit(
        train_data=train_data,
        time_limit=1800,  # 30 minutes
        presets="medium_quality"
    )
    
    # Print model leaderboard
    print("Model Leaderboard:")
    leaderboard = predictor.leaderboard(train_data)
    print(leaderboard)
    
    # Make predictions on test data
    print("Making predictions on test data...")
    predictions = predictor.predict(test_data)
    
    # Create results dataframe
    results = pd.DataFrame({label: predictions})
    
    # Save results to file with same format as test data
    results_path = os.path.join(WORKING_DIR, "results.csv")
    results.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")