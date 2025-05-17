"""
Road Segmentation with AutoGluon MultiModal

This script performs semantic segmentation on road images using AutoGluon's MultiModalPredictor.
It processes training data, trains a model, and generates predictions on test data.

Additional installation requirements:
    pip install autogluon.multimodal
    python -m mim install mmcv
    pip install mmdet==3.1.0
"""

import os
import time
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')

# Set single GPU for semantic segmentation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def expand_paths(df, base_path):
    """
    Expand relative paths to absolute paths
    
    Args:
        df: DataFrame containing image and label paths
        base_path: Base path to prepend to relative paths
    
    Returns:
        DataFrame with absolute paths
    """
    if 'image' in df.columns:
        df['image'] = df['image'].apply(lambda x: os.path.join(base_path, x))
    
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: os.path.join(base_path, x) if isinstance(x, str) else x)
    
    return df

def save_predictions(predictor, test_df, output_dir):
    """
    Generate and save predictions
    
    Args:
        predictor: Trained MultiModalPredictor
        test_df: DataFrame with test data
        output_dir: Directory to save predictions
    
    Returns:
        DataFrame with predictions
    """
    # Create directory for predicted masks
    mask_dir = os.path.join(output_dir, "predicted_mask")
    os.makedirs(mask_dir, exist_ok=True)
    
    # Make predictions
    predictions = predictor.predict(test_df)
    
    # Process and save each prediction
    results_df = test_df.copy()
    results_df['label'] = None
    
    for i, row in results_df.iterrows():
        # Get the predicted mask
        pred_mask = predictions[i]
        
        # Squeeze to remove extra dimensions if needed
        if len(pred_mask.shape) > 2:
            pred_mask = np.squeeze(pred_mask)
        
        # Convert to uint8 if needed
        if pred_mask.dtype != np.uint8:
            pred_mask = (pred_mask * 255).astype(np.uint8)
        
        # Save the mask as grayscale JPG
        img_name = os.path.basename(row['image'])
        mask_path = os.path.join(mask_dir, f"{os.path.splitext(img_name)[0]}_mask.jpg")
        cv2.imwrite(mask_path, pred_mask)
        
        # Update the label column with the absolute path
        results_df.at[i, 'label'] = mask_path
    
    return results_df

if __name__ == "__main__":
    # Define paths
    base_path = "/media/agent/maab/datasets/road_segmentation/training"
    output_dir = "./"
    
    # Create timestamp for model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    
    # Load data
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(base_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "inference.csv"))
    
    # Remove unnecessary index column
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in val_df.columns:
        val_df = val_df.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop(columns=['Unnamed: 0'])
    
    # Remove training samples without valid labels (if any)
    train_df = train_df.dropna(subset=['label'])
    
    # Expand paths to absolute paths
    train_df = expand_paths(train_df, base_path)
    val_df = expand_paths(val_df, base_path)
    test_df = expand_paths(test_df, base_path)
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Validation data: {len(val_df)} samples")
    print(f"Testing data: {len(test_df)} samples")
    
    # Initialize predictor for semantic segmentation
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        label="label",
        path=model_dir,
        hyperparameters={
            "model.sam.checkpoint_name": "facebook/sam-vit-base",
        },
    )
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    predictor.fit(time_limit=24*3600,
        train_data=train_df,
        tuning_data=val_df,
        presets="best_quality"
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")
    
    # Evaluate the model if validation data is available
    if len(val_df) > 0:
        scores = predictor.evaluate(val_df, metrics=["iou"])
        print(f"Validation IoU: {scores['iou']}")
    
    # Generate predictions on test data
    print("Generating predictions...")
    results_df = save_predictions(predictor, test_df, output_dir)
    
    # Save results to CSV
    results_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    print("Done!")