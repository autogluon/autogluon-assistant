"""
German News Article Classification using AutoGluon Multimodal

This script trains a text classification model on German news articles using AutoGluon's
multimodal capabilities. It processes training data, trains a model, and makes predictions
on test data.

Usage:
    python german_news_classification.py

Additional installation requirements:
    pip install autogluon.multimodal
    pip install pandas
"""

import datetime
import os

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

if __name__ == "__main__":
    # Define output directory
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the model directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    
    # Load the data
    train_path = "/media/agent/maab/datasets/gnad10/training/train.csv"
    val_path = "/media/agent/maab/datasets/gnad10/training/validation.csv"
    test_path = "/media/agent/maab/datasets/gnad10/training/inference.csv"
    
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    # Data preprocessing
    # Check for missing labels in training data
    print(f"Training data shape before cleaning: {train_data.shape}")
    train_data = train_data.dropna(subset=['label'])
    print(f"Training data shape after cleaning: {train_data.shape}")
    
    # Check for missing labels in validation data
    print(f"Validation data shape before cleaning: {val_data.shape}")
    val_data = val_data.dropna(subset=['label'])
    print(f"Validation data shape after cleaning: {val_data.shape}")
    
    # Initialize the predictor
    predictor = MultiModalPredictor(
        label='label',
        path=model_dir
    )
    
    # Train the model with AutoGluon
    predictor.fit(time_limit=24*3600,
        train_data=train_data,
        tuning_data=val_data,  # Use validation data for tuning
        presets="best_quality"
    )
    
    # Make predictions on test data
    predictions = predictor.predict(test_data)
    
    # Save predictions to output file
    results_df = pd.DataFrame({'text': test_data['text'], 'label': predictions})
    results_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"Model saved to: {model_dir}")
    print(f"Predictions saved to: {results_path}")
    
    # Evaluate model on validation data
    eval_metrics = predictor.evaluate(val_data)
    print(f"Validation metrics: {eval_metrics}")