"""
HAM10000 Skin Lesion Classification using AutoGluon Multimodal

This script trains an image classification model to diagnose skin lesions from the HAM10000 dataset.
It processes training data, trains a model using AutoGluon's MultiModalPredictor, and makes predictions
on test data.

Usage:
    python ham10000_classification.py

Required packages:
    - autogluon.multimodal
    - pandas
    - numpy
    - os
    - datetime
"""

# Additional installation steps if needed:
# pip install autogluon.multimodal pandas numpy

import os
from datetime import datetime

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

if __name__ == "__main__":
    # Define paths
    output_dir = "./"
    train_img_dir = "/media/agent/maab/datasets/ham10000/training/train"
    test_img_dir = "/media/agent/maab/datasets/ham10000/training/test/ISIC2018_Task3_Test_Images"
    train_annotations_path = "/media/agent/maab/datasets/ham10000/training/ham10000_train_annotations.csv"
    inference_path = "/media/agent/maab/datasets/ham10000/training/inference.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    
    # Load training data
    train_df = pd.read_csv(train_annotations_path)
    
    # Check for and remove samples without valid labels
    print(f"Training data shape before cleaning: {train_df.shape}")
    train_df = train_df.dropna(subset=['dx'])
    print(f"Training data shape after removing invalid labels: {train_df.shape}")
    
    # Add full image paths to training data
    train_df['image_path'] = train_df['ImageID'].apply(lambda x: os.path.join(train_img_dir, x))
    
    # Verify all training images exist
    missing_train_images = [img for img in train_df['image_path'].tolist() if not os.path.exists(img)]
    if missing_train_images:
        print(f"Warning: {len(missing_train_images)} training images are missing")
        # Remove rows with missing images
        train_df = train_df[~train_df['image_path'].isin(missing_train_images)]
    
    # Load test data
    test_df = pd.read_csv(inference_path)
    
    # Add full image paths to test data
    test_df['image_path'] = test_df['image_id'].apply(lambda x: os.path.join(test_img_dir, x))
    
    # Verify all test images exist
    missing_test_images = [img for img in test_df['image_path'].tolist() if not os.path.exists(img)]
    if missing_test_images:
        print(f"Warning: {len(missing_test_images)} test images are missing")
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize and train the model
    print("Initializing and training the model...")
    predictor = MultiModalPredictor(
        label='dx',  # Target column to predict
        path=model_dir,  # Path to save the model
        problem_type='multiclass',  # Classification type
    )
    
    # Train the model with the specified parameters
    predictor.fit(time_limit=24*3600,
        train_data=train_df,
        presets="best_quality"
    )
    
    # Make predictions on the test data
    print("Making predictions on test data...")
    predictions = predictor.predict(test_df)
    
    # Add predictions to the test dataframe
    test_df['dx'] = predictions
    
    # Save the results to the output directory
    results_path = os.path.join(output_dir, "results.csv")
    test_df.to_csv(results_path, index=False)
    
    print(f"Predictions saved to {results_path}")
    print("Done!")