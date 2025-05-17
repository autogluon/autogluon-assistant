"""
Meme Classification with AutoGluon MultiModal

This script trains a model to predict five attributes of memes: humor, sarcasm, offensive, 
motivational, and overall_sentiment using both image and text data.

Installation requirements:
pip install autogluon.multimodal
pip install pandas
pip install numpy
"""

import datetime
import os

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor


def expand_image_paths(df, image_col, base_folder):
    """
    Expands relative image paths to absolute paths and handles different file extensions.
    
    Args:
        df: DataFrame containing image IDs
        image_col: Column name containing image IDs
        base_folder: Base folder containing images
        
    Returns:
        DataFrame with expanded image paths
    """
    def find_image_path(image_id, base_folder):
        # Check common extensions
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.jpe']
        for ext in extensions:
            # Try with original extension if it exists
            if os.path.exists(os.path.join(base_folder, image_id)):
                return os.path.join(base_folder, image_id)
            
            # Try with different extensions
            base_name = os.path.splitext(image_id)[0]
            potential_path = os.path.join(base_folder, base_name + ext)
            if os.path.exists(potential_path):
                return potential_path
        
        # If no file found, return the original path and log a warning
        print(f"Warning: Image file not found for {image_id}")
        return os.path.join(base_folder, image_id)
    
    df['image_path'] = df[image_col].apply(lambda x: find_image_path(x, base_folder))
    return df

if __name__ == "__main__":
    # Define paths
    train_data_path = '/media/agent/maab/datasets/memotion/training/memotion_train.csv'
    test_data_path = '/media/agent/maab/datasets/memotion/training/inference.csv'
    image_folder = '/media/agent/maab/datasets/memotion/training/train'
    output_folder = './'
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate timestamp for model folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(output_folder, f"memotion_model_{timestamp}")
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(train_data_path)
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    
    # Check for missing values in training data
    print(f"Training data shape before cleaning: {train_df.shape}")
    train_df = train_df.dropna(subset=['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment'])
    print(f"Training data shape after cleaning: {train_df.shape}")
    
    # Expand image paths
    print("Expanding image paths...")
    train_df = expand_image_paths(train_df, 'ImageID', image_folder)
    test_df = expand_image_paths(test_df, 'ImageID', image_folder)
    
    # Define label columns
    label_columns = ['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment']
    
    # Create a dictionary to store the models
    models = {}
    
    # Train a model for each label column
    for label in label_columns:
        print(f"\nTraining model for {label}...")
        
        # Determine problem type based on number of unique values
        unique_values = train_df[label].nunique()
        problem_type = 'binary' if unique_values == 2 else 'multiclass'
        print(f"Detected {unique_values} unique values for {label}, using {problem_type} classification")
        
        # Initialize predictor
        predictor = MultiModalPredictor(
            label=label,
            problem_type=problem_type,
            path=os.path.join(model_folder, label),
        )
        
        # Train model
        predictor.fit(
            train_data=train_df[['image_path', 'text_corrected', label]],
<<<<<<< HEAD
            time_limit=3600*24,  # 24 hours per model
=======
            time_limit=3600 * 6,
>>>>>>> a14002d (update visualizations)
            presets="best_quality"
        )
        
        # Store model
        models[label] = predictor
    
    # Make predictions on test data
    print("\nMaking predictions on test data...")
    results_df = test_df[['ImageID', 'text_corrected']].copy()
    
    for label in label_columns:
        print(f"Predicting {label}...")
        predictions = models[label].predict(test_df[['image_path', 'text_corrected']])
        results_df[label] = predictions
    
    # Save results
    results_path = os.path.join(output_folder, 'results.csv')
    print(f"Saving results to {results_path}")
    results_df.to_csv(results_path, index=False)
    
    print("Done!")
