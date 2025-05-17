"""
Camouflaged Object Semantic Segmentation using AutoGluon MultiModal

This script performs semantic segmentation to identify and delineate camouflaged objects 
that visually blend with their surroundings, outputting pixel-level binary masks.

Installation requirements:
pip install autogluon.multimodal
pip install opencv-python
pip install pandas
"""

import os
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')

# Set environment variable for single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prepare_data(train_csv, val_csv, inference_csv, base_path):
    """
    Prepare training, validation, and test data by removing unnecessary columns
    and ensuring all paths are absolute.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        inference_csv: Path to inference CSV file
        base_path: Base path for relative paths in CSV files
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Load data
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)
    test_data = pd.read_csv(inference_csv)
    
    # Remove unnecessary index column if present
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in val_data.columns:
        val_data = val_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    
    # Convert relative paths to absolute paths
    def convert_to_abs_path(path):
        return os.path.abspath(os.path.join(base_path, path))
    
    train_data['image'] = train_data['image'].apply(convert_to_abs_path)
    train_data['label'] = train_data['label'].apply(convert_to_abs_path)
    
    val_data['image'] = val_data['image'].apply(convert_to_abs_path)
    val_data['label'] = val_data['label'].apply(convert_to_abs_path)
    
    test_data['image'] = test_data['image'].apply(convert_to_abs_path)
    
    # Remove samples without valid labels (if any)
    train_data = train_data[train_data['label'].apply(lambda x: os.path.exists(x))]
    val_data = val_data[val_data['label'].apply(lambda x: os.path.exists(x))]
    
    return train_data, val_data, test_data

def save_prediction_masks(predictions, test_data, output_dir):
    """
    Save prediction masks as grayscale JPG images and update the test dataframe
    with paths to these masks.
    
    Args:
        predictions: Predictions from the model
        test_data: Test dataframe
        output_dir: Directory to save masks
    
    Returns:
        Updated test dataframe with mask paths
    """
    # Create directory for predicted masks
    mask_dir = os.path.join(output_dir, "predicted_mask")
    os.makedirs(mask_dir, exist_ok=True)
    
    # Create a copy of test_data to add label column
    result_df = test_data.copy()
    result_df['label'] = ''
    
    # Process each prediction and save as grayscale JPG
    for i, (idx, row) in enumerate(test_data.iterrows()):
        # Get the prediction mask
        mask = predictions[i]
        
        # Squeeze to remove any extra dimensions
        mask = np.squeeze(mask)
        
        # Ensure mask is binary (0 or 255)
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Create filename based on original image name
        img_name = os.path.basename(row['image'])
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.jpg")
        
        # Save mask as grayscale JPG
        cv2.imwrite(mask_path, mask)
        
        # Add absolute path to mask in result dataframe
        result_df.at[idx, 'label'] = os.path.abspath(mask_path)
    
    return result_df

def train_and_predict(train_data, val_data, test_data, output_dir):
    """
    Train a semantic segmentation model and make predictions on test data.
    
    Args:
        train_data: Training dataframe
        val_data: Validation dataframe
        test_data: Test dataframe
        output_dir: Directory to save model and results
    
    Returns:
        Result dataframe with predictions
    """
    # Create a timestamp for the model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    
    # Initialize the predictor
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        label="label",
        path=model_dir,
        hyperparameters={
            "model.sam.checkpoint_name": "facebook/sam-vit-base",
        },
    )
    
    # Train the model
    predictor.fit(time_limit=24*3600,
        train_data=train_data,
        tuning_data=val_data,
        presets="best_quality"
    )
    
    # Make predictions on test data
    predictions = predictor.predict(test_data)
    
    # Save prediction masks and get updated dataframe
    result_df = save_prediction_masks(predictions, test_data, output_dir)
    
    return result_df

if __name__ == "__main__":
    # Define paths
    base_path = "/media/agent/maab/datasets/camo_sem_seg/training"
    train_csv = os.path.join(base_path, "train.csv")
    val_csv = os.path.join(base_path, "val.csv")
    inference_csv = os.path.join(base_path, "inference.csv")
    output_dir = "./"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_data, val_data, test_data = prepare_data(train_csv, val_csv, inference_csv, base_path)
    
    # Train model and make predictions
    print("Training model and making predictions...")
    result_df = train_and_predict(train_data, val_data, test_data, output_dir)
    
    # Save results to CSV
    result_path = os.path.join(output_dir, "results.csv")
    result_df.to_csv(result_path, index=True)
    print(f"Results saved to {result_path}")