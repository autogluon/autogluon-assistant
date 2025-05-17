#!/usr/bin/env python
"""
Hateful Meme Classification using AutoGluon Multimodal

This script trains a model to classify memes as hateful (1) or not hateful (0)
using both image and text data. It uses AutoGluon's MultiModalPredictor to
handle the multimodal nature of the data.

Usage:
    python hateful_meme_classifier.py

Requirements:
    - pip install autogluon.multimodal
    - pip install pandas
    - pip install numpy
    - pip install matplotlib
"""

import os
from datetime import datetime

import pandas as pd
from autogluon.multimodal import MultiModalPredictor


def expand_image_paths(df, base_path):
    """
    Expand relative image paths to absolute paths
    """
    df['img'] = df['img'].apply(lambda x: os.path.join(base_path, x))
    return df

if __name__ == "__main__":
    # Define paths
    data_path = "/media/agent/maab/datasets/hateful_meme/training"
    output_path = "./"
    
    # Create timestamp for model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, f"model_{timestamp}")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Load training data
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    
    # Check for and remove invalid labels (if any)
    initial_count = len(train_df)
    train_df = train_df.dropna(subset=['label'])
    if len(train_df) < initial_count:
        print(f"Removed {initial_count - len(train_df)} samples with missing labels")
    
    # Expand image paths
    train_df = expand_image_paths(train_df, data_path)
    
    # Load test data
    test_df = pd.read_csv(os.path.join(data_path, "inference.csv"))
    test_df = expand_image_paths(test_df, data_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize and train the model
    predictor = MultiModalPredictor(
        label='label',
        path=model_path
    )
    
    print("Training model...")
    predictor.fit(time_limit=24*3600,
        train_data=train_df,
        presets="best_quality"
    )
    
    # Make predictions on test data
    print("Making predictions...")
    predictions = predictor.predict(test_df)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['label'] = predictions
    
    # Save results
    results_path = os.path.join(output_path, "results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    print(f"Model saved to {model_path}")