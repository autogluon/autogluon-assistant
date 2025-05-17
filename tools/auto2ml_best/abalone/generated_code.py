"""
AutoGluon Tabular Regression Script for Abalone Dataset

This script uses AutoGluon's TabularPredictor to train a regression model on the abalone dataset
to predict the 'Class_number_of_rings' target variable. It handles data preprocessing,
model training, and generates predictions on test data.

Usage:
    python abalone_regression.py

Requirements:
    - pip install autogluon.tabular
    - pip install pandas
"""

import os
import time

import pandas as pd
from autogluon.tabular import TabularPredictor

if __name__ == "__main__":
    # Define paths
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the model directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"abalone_model_{timestamp}")
    
    # Load training data
    train_data_path = "/media/agent/maab/datasets/abalone/training/train.csv"
    train_data = pd.read_csv(train_data_path)
    
    # Load test data
    test_data_path = "/media/agent/maab/datasets/abalone/training/test.csv"
    test_data = pd.read_csv(test_data_path)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Check for missing values in the target column
    missing_target = train_data['Class_number_of_rings'].isna().sum()
    if missing_target > 0:
        print(f"Removing {missing_target} rows with missing target values")
        train_data = train_data.dropna(subset=['Class_number_of_rings'])
    
    # Check if there's an unnecessary index column (there isn't in this case, but included for completeness)
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    
    # Print data information
    print("Training data columns:", train_data.columns.tolist())
    print("Target variable statistics:")
    print(train_data['Class_number_of_rings'].describe())
    
    # Create and train the predictor
    print("Training AutoGluon model...")
    predictor = TabularPredictor(
        label='Class_number_of_rings',
        problem_type='regression',  # Explicitly set as regression
        path=model_dir,
        eval_metric='rmse'  # Root Mean Square Error
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets='best_quality'
    )
    
    # Print model performance and leaderboard
    print("Model training completed")
    print("Model performance summary:")
    leaderboard = predictor.leaderboard(train_data)
    print(leaderboard)
    
    # Make predictions on test data
    print("Making predictions on test data...")
    test_predictions = predictor.predict(test_data)
    
    # Create results dataframe with the same structure as test data
    results = test_data.copy()
    results['Class_number_of_rings'] = test_predictions
    
    # Save results to the output directory
    results_path = os.path.join(output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    print(f"Model saved to {model_dir}")