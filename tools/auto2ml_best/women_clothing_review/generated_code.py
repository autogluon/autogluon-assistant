"""
Women's Clothing Review Rating Prediction

This script uses AutoGluon Multimodal to predict ratings for women's clothing reviews.
It processes training data, trains a model, and makes predictions on test data.

Usage:
    python women_clothing_review_prediction.py

Additional installation requirements:
    pip install autogluon.multimodal pandas pyarrow
"""

import datetime
import os

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

if __name__ == "__main__":
    # Define paths
    base_output_dir = "./"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create a timestamped folder for the model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_output_dir, f"model_{timestamp}")
    
    # Load data
    train_data = pd.read_parquet("/media/agent/maab/datasets/women_clothing_review/training/train.pq")
    dev_data = pd.read_parquet("/media/agent/maab/datasets/women_clothing_review/training/dev.pq")
    test_data = pd.read_parquet("/media/agent/maab/datasets/women_clothing_review/training/inference.pq")
    
    # Data preprocessing
    # 1. Remove the unnecessary index column
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in dev_data.columns:
        dev_data = dev_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    
    # 2. Remove training data samples without valid labels
    train_data = train_data.dropna(subset=['Rating'])
    dev_data = dev_data.dropna(subset=['Rating'])
    
    # Print data information
    print(f"Train data shape: {train_data.shape}")
    print(f"Dev data shape: {dev_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize MultiModalPredictor
    predictor = MultiModalPredictor(
        label='Rating',
        path=model_dir,
        problem_type='regression',  # Ratings are typically numeric values
        eval_metric='rmse'  # Root Mean Squared Error is appropriate for rating prediction
    )
    
    # Train the model
    predictor.fit(time_limit=24*3600,
        train_data=train_data,
        tuning_data=dev_data,  # Use validation data for tuning
        presets="best_quality"  # As specified
    )
    
    # Make predictions on test data
    predictions = predictor.predict(test_data)
    
    # Create results dataframe
    results = test_data.copy()
    results['Rating'] = predictions
    
    # Save results to the specified output directory
    # Using the same format as the test data (parquet)
    results_path = os.path.join(base_output_dir, "results.pq")
    results.to_parquet(results_path, index=False)
    
    print(f"Model saved to: {model_dir}")
    print(f"Predictions saved to: {results_path}")