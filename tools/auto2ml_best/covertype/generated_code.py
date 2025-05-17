#!/usr/bin/env python
"""
AutoGluon Tabular script for Forest Cover Type Classification.

This script uses autogluon.tabular to train a model that predicts forest cover type
from cartographic variables. It loads training data, trains a model, makes predictions
on test data, and saves the results.

Installation requirements:
    pip install autogluon.tabular
    pip install pandas
    pip install numpy
"""

import os
import time

import pandas as pd
from autogluon.tabular import TabularPredictor

if __name__ == "__main__":
    # Define paths
    output_dir = "./"
    model_dir = os.path.join(output_dir, f"model_{int(time.time())}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_path = "/media/agent/maab/datasets/covertype/training/train.csv"
    test_path = "/media/agent/maab/datasets/covertype/training/test.csv"
    
    print("Loading training data...")
    train_data = pd.read_csv(train_path)
    
    print("Loading test data...")
    test_data = pd.read_csv(test_path)
    
    # Check for any missing labels in training data
    print(f"Training data shape before cleaning: {train_data.shape}")
    train_data = train_data.dropna(subset=['class'])
    print(f"Training data shape after cleaning: {train_data.shape}")
    
    # Check if there's an index column to remove
    # For this dataset, there isn't a separate index column to remove
    
    # Identify features and label
    label = 'class'
    features = [col for col in train_data.columns if col != label]
    
    print(f"Number of features: {len(features)}")
    print(f"Target variable: {label}")
    
    # Check class distribution
    print("Class distribution in training data:")
    print(train_data[label].value_counts())
    
    # Train the model
    print("Training model...")
    predictor = TabularPredictor(
        label=label,
        path=model_dir,
        problem_type='multiclass',  # Explicitly set as multiclass classification
        eval_metric='accuracy'  # Appropriate for multiclass classification
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Print model performance and leaderboard
    print("Model training completed!")
    print("Leaderboard:")
    leaderboard = predictor.leaderboard()
    print(leaderboard)
    
    # Make predictions on test data
    print("Making predictions on test data...")
    predictions = predictor.predict(test_data)
    
    # Create results dataframe with the same structure as test data
    results = test_data.copy()
    results['class'] = predictions
    
    # Save results
    results_path = os.path.join(output_dir, "results.csv")
    print(f"Saving results to {results_path}")
    results.to_csv(results_path, index=False)
    
    print("Process completed successfully!")