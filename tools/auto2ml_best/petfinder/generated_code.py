"""
PetFinder Adoption Speed Prediction using AutoGluon Multimodal

This script trains a multimodal model to predict pet adoption speed using the PetFinder dataset.
It processes tabular data along with pet images to make predictions.

Required packages:
- autogluon.multimodal
- pandas
- numpy
- os
- datetime
- To install: pip install autogluon.multimodal pandas numpy
"""

import datetime
import os

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

if __name__ == "__main__":
    # Define paths
    input_dir = "/media/agent/maab/datasets/petfinder/training"
    output_dir = "./"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped model directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    train_data = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(input_dir, "inference.csv"))
    
    # Data preprocessing
    # 1. Remove the unnecessary index column
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    
    # 2. Remove training data samples without valid labels
    train_data = train_data.dropna(subset=['AdoptionSpeed'])
    
    # 3. Process image paths - convert relative paths to absolute paths
    def expand_image_path(image_path):
        # Extract the first image if multiple are provided
        if isinstance(image_path, str) and ';' in image_path:
            image_path = image_path.split(';')[0]
        
        # Convert to absolute path
        if isinstance(image_path, str):
            return os.path.join(input_dir, image_path)
        return image_path
    
    train_data['Images'] = train_data['Images'].apply(expand_image_path)
    test_data['Images'] = test_data['Images'].apply(expand_image_path)
    
    # Define label column
    label_column = 'AdoptionSpeed'
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize and train the model
    predictor = MultiModalPredictor(
        label=label_column,
        path=model_dir
    )
    
    # Train the model with the specified parameters
    predictor.fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Make predictions on test data
    predictions = predictor.predict(test_data)
    
    # Prepare the results DataFrame
    results = test_data.copy()
    results[label_column] = predictions
    
    # Save the results to the output directory
    results_path = os.path.join(output_dir, "results.csv")
    results.to_csv(results_path, index=False)
    
    print(f"Model trained and saved to: {model_dir}")
    print(f"Predictions saved to: {results_path}")