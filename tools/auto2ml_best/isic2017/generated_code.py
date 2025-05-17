#!/usr/bin/env python
"""
ISIC 2017 Skin Lesion Segmentation using AutoGluon MultiModal

This script trains a semantic segmentation model using autogluon.multimodal
to segment skin lesions in dermoscopic images. The model is trained on the
ISIC 2017 dataset and evaluated on test images.

Usage:
    python script.py

Additional installation requirements:
    pip install autogluon.multimodal
    python -m mim install "mmcv==2.1.0"
    python -m pip install "mmdet==3.2.0"
    python -m pip install "mmengine>=0.10.6"
    pip install opencv-python
"""

import datetime
import os

import cv2
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

# Define output directory
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a model directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = os.path.join(OUTPUT_DIR, f"model_{timestamp}")
os.makedirs(MODEL_DIR, exist_ok=True)

# Create directory for predicted masks
PRED_MASKS_DIR = os.path.join(OUTPUT_DIR, "predicted_mask")
os.makedirs(PRED_MASKS_DIR, exist_ok=True)

def preprocess_data(train_csv, inference_csv):
    """
    Preprocess the training and inference data.
    
    Args:
        train_csv: Path to the training CSV file
        inference_csv: Path to the inference CSV file
        
    Returns:
        Processed training and inference DataFrames
    """
    # Load data
    train_data = pd.read_csv(train_csv)
    inference_data = pd.read_csv(inference_csv)
    
    # Remove unnecessary index column
    if 'Unnamed: 0' in train_data.columns:
        train_data = train_data.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in inference_data.columns:
        inference_data = inference_data.drop(columns=['Unnamed: 0'])
    
    # Remove samples without valid labels (if any)
    train_data = train_data.dropna(subset=['label'])
    
    # Convert relative paths to absolute paths
    base_dir = os.path.dirname(train_csv)
    
    # Process training data
    train_data['image'] = train_data['image'].apply(
        lambda x: os.path.abspath(os.path.join(base_dir, x))
    )
    train_data['label'] = train_data['label'].apply(
        lambda x: os.path.abspath(os.path.join(base_dir, x))
    )
    
    # Process inference data
    inference_data['image'] = inference_data['image'].apply(
        lambda x: os.path.abspath(os.path.join(base_dir, x))
    )
    
    return train_data, inference_data

def save_masks(predictions, inference_data):
    """
    Save predicted segmentation masks and update the results DataFrame
    
    Args:
        predictions: Predictions from the model
        inference_data: Original inference DataFrame
        
    Returns:
        Updated DataFrame with mask paths
    """
    results = inference_data.copy()
    results['label'] = ""
    
    for i, pred in enumerate(predictions):
        # Get the image name from the path
        img_path = inference_data.iloc[i]['image']
        img_name = os.path.basename(img_path).split('.')[0]
        
        # Create mask filename
        mask_filename = f"{img_name}_segmentation.jpg"
        mask_path = os.path.join(PRED_MASKS_DIR, mask_filename)
        
        # Convert prediction to grayscale mask (0 or 255)
        mask = pred.squeeze()
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Save the mask
        cv2.imwrite(mask_path, mask)
        
        # Update results DataFrame with absolute path to mask
        results.iloc[i, results.columns.get_loc('label')] = os.path.abspath(mask_path)
    
    return results

if __name__ == "__main__":
    # Set environment variable to use only one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Define paths to data
    train_csv = "/media/agent/maab/datasets/isic2017/training/train.csv"
    inference_csv = "/media/agent/maab/datasets/isic2017/training/inference.csv"
    
    # Preprocess data
    train_data, inference_data = preprocess_data(train_csv, inference_csv)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Inference data shape: {inference_data.shape}")
    
    # Initialize and train the model
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        label="label",
        path=MODEL_DIR,
        hyperparameters={
            "model.sam.checkpoint_name": "facebook/sam-vit-base",
        },
    )
    
    # Train the model
    predictor.fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Process test data in batches to avoid memory issues
    batch_size = 10
    total_samples = len(inference_data)
    results_list = []
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}: samples {i} to {end_idx-1}")
        
        batch_data = inference_data.iloc[i:end_idx].reset_index(drop=True)
        batch_predictions = predictor.predict(batch_data)
        batch_results = save_masks(batch_predictions, batch_data)
        results_list.append(batch_results)
    
    # Combine all batch results
    results = pd.concat(results_list, ignore_index=True)
    
    # Save results to CSV
    results.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    
    print(f"Training and prediction completed. Results saved to {OUTPUT_DIR}")