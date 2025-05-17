"""
European Flood Depth Image Classification Script using AutoGluon Multimodal

This script processes a dataset of flood images and trains a model to classify
whether an image is useful for determining flood depth or not. It handles data
preprocessing, model training, and generating predictions on test data.

Additional installation requirements:
    pip install autogluon.multimodal
    pip install pandas
    pip install numpy
    pip install opencv-python
"""

import datetime
import os

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

# Define paths
OUTPUT_DIR = "./"
DATA_DIR = "/media/agent/maab/datasets/europeanflooddepth/training"
IMAGES_DIR = os.path.join(DATA_DIR, "europeanflooddepth")
TRAIN_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "europeanflooddepth_train_annotations.csv")
INFERENCE_PATH = os.path.join(DATA_DIR, "inference.csv")

if __name__ == "__main__":
    # Create timestamp for model directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(OUTPUT_DIR, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load training annotations and inference data
    train_df = pd.read_csv(TRAIN_ANNOTATIONS_PATH)
    inference_df = pd.read_csv(INFERENCE_PATH)
    
    # Check if there are any missing labels in training data
    print(f"Original training data shape: {train_df.shape}")
    train_df = train_df.dropna(subset=['LabelName'])
    print(f"Training data shape after removing invalid labels: {train_df.shape}")
    
    # Prepare training data in the format expected by AutoGluon
    # We need an 'image' column with full paths and a 'label' column
    train_data = pd.DataFrame({
        'image': [os.path.join(IMAGES_DIR, img_id) for img_id in train_df['ImageID']],
        'label': train_df['LabelName']
    })
    
    # Prepare test data
    test_data = pd.DataFrame({
        'image': [os.path.join(IMAGES_DIR, img_id) for img_id in inference_df['ImageID']]
    })
    
    print(f"Training on {len(train_data)} images")
    print(f"Testing on {len(test_data)} images")
    
    # Initialize and train the predictor
    predictor = MultiModalPredictor(
        label='label',
        path=model_dir
    )
    
    # Train the model with specified parameters
    predictor.fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Make predictions on test data
    predictions = predictor.predict(test_data)
    print(f"Generated predictions for {len(predictions)} test images")
    
    # Create results dataframe with the same structure as the inference file
    # but adding the LabelName column with our predictions
    results_df = inference_df.copy()
    results_df['LabelName'] = predictions
    
    # Save results to the output directory
    results_path = os.path.join(OUTPUT_DIR, "results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    print("Model training and prediction completed successfully!")