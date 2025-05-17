"""
Document Classification Script using AutoGluon MultiModal

This script trains a document image classifier on the RVL-CDIP dataset, which contains
document images across 16 different categories. The model is trained using AutoGluon's
MultiModalPredictor and then used to make predictions on test data.

Installation requirements:
pip install autogluon.multimodal pandas numpy pillow
"""

import os
from datetime import datetime

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

if __name__ == "__main__":
    # Define paths
    BASE_PATH = "/media/agent/maab/datasets/rvl_cdip/training"
    OUTPUT_PATH = "./"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Generate a timestamp for model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(OUTPUT_PATH, f"model_{timestamp}")
    
    print("Loading data...")
    # Load training, validation, and inference data
    train_df = pd.read_csv(os.path.join(BASE_PATH, "train.txt"), sep=" ")
    val_df = pd.read_csv(os.path.join(BASE_PATH, "val.txt"), sep=" ")
    test_df = pd.read_csv(os.path.join(BASE_PATH, "inference.txt"))
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Check if there are any invalid labels in training data
    if 'label' in train_df.columns:
        # Remove samples without valid labels
        train_df = train_df.dropna(subset=['label'])
        print(f"Training data shape after removing invalid labels: {train_df.shape}")
    
    # Prepare full paths for images
    print("Preparing image paths...")
    
    def expand_path(img_path):
        # Add the full path to the image
        return os.path.join(BASE_PATH, "images", img_path)
    
    # Apply path expansion to all datasets
    train_df['document'] = train_df['document'].apply(expand_path)
    val_df['document'] = val_df['document'].apply(expand_path)
    test_df['document'] = test_df['document'].apply(expand_path)
    
    # AutoGluon expects 'image' column for image paths
    train_df = train_df.rename(columns={'document': 'image'})
    val_df = val_df.rename(columns={'document': 'image'})
    test_df = test_df.rename(columns={'document': 'image'})
    
    print("Training model...")
    # Initialize and train the predictor
    predictor = MultiModalPredictor(
        label='label',
        path=model_path
    )
    
    # Train the model with validation data
    predictor.fit(time_limit=24*3600,
        train_data=train_df,
        tuning_data=val_df,
        presets="best_quality"
    )
    
    print("Making predictions...")
    # Make predictions on test data
    predictions = predictor.predict(test_df)
    
    # Prepare results dataframe
    results_df = pd.DataFrame()
    # Extract the original document path from the full path
    results_df['document'] = test_df['image'].apply(lambda x: x.replace(os.path.join(BASE_PATH, "images/"), ''))
    results_df['label'] = predictions
    
    # Save results
    results_path = os.path.join(OUTPUT_PATH, "results.txt")
    results_df.to_csv(results_path, sep=' ', index=False)
    
    print(f"Results saved to {results_path}")
    print("Document classification completed successfully!")