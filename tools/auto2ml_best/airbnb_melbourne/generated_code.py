"""
AutoGluon Tabular Model for Airbnb Melbourne Price Prediction

This script trains an AutoGluon TabularPredictor on the Melbourne Airbnb dataset
to predict price categories. It loads training data, trains a model, and generates
predictions on test data.

Requirements:
- pip install autogluon.tabular
- pip install pyarrow (for parquet file support)
"""

import datetime
import os

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


def create_output_dir():
    """Create output directory if it doesn't exist"""
    output_dir = "./"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def generate_model_dir():
    """Generate a directory name with timestamp for model storage"""
    output_dir = create_output_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    return model_dir

if __name__ == "__main__":
    # Define paths
    train_path = "/media/agent/maab/datasets/airbnb_melbourne/training/train.pq"
    test_path = "/media/agent/maab/datasets/airbnb_melbourne/training/inference.pq"
    output_dir = create_output_dir()
    model_dir = generate_model_dir()
    
    print("Loading training data...")
    train_data = TabularDataset(train_path)
    
    # Check if there's a target column
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data columns: {train_data.columns.tolist()}")
    
    # Look for a potential target column (price_label)
    if 'price_label' in train_data.columns:
        label_column = 'price_label'
        print(f"Found target column: {label_column}")
    else:
        # If no clear target is found, we need to examine the data more closely
        print("No 'price_label' column found. Examining data for potential target columns...")
        
        # Look for columns that might be target variables
        potential_targets = [col for col in train_data.columns if 'price' in col.lower() or 'label' in col.lower()]
        if potential_targets:
            print(f"Potential target columns: {potential_targets}")
            label_column = potential_targets[0]
            print(f"Using {label_column} as the target column")
        else:
            raise ValueError("Could not identify a target column. Please specify the target column manually.")
    
    # Check for index column and remove if necessary
    if 'Unnamed: 0' in train_data.columns:
        print("Removing unnecessary index column 'Unnamed: 0'")
        train_data = train_data.drop(columns=['Unnamed: 0'])
    
    # Remove samples without valid labels
    if train_data[label_column].isna().any():
        print(f"Removing {train_data[label_column].isna().sum()} samples without valid labels")
        train_data = train_data.dropna(subset=[label_column])
    
    # Determine problem type based on target variable
    unique_values = train_data[label_column].nunique()
    if unique_values == 2:
        problem_type = 'binary'
    elif unique_values > 2:
        problem_type = 'multiclass'
    else:
        problem_type = 'regression'
    
    print(f"Detected problem type: {problem_type}")
    print(f"Number of unique values in target: {unique_values}")
    
    # Train the model
    print(f"Training AutoGluon model with {problem_type} problem type...")
    predictor = TabularPredictor(
        label=label_column,
        path=model_dir,
        problem_type=problem_type,
        eval_metric=None  # AutoGluon will select an appropriate metric based on problem_type
    ).fit(time_limit=24*3600,
        train_data=train_data,
        presets="best_quality"
    )
    
    # Print model leaderboard
    print("Model training complete. Leaderboard:")
    leaderboard = predictor.leaderboard(train_data)
    print(leaderboard)
    
    # Load test data and make predictions
    print("Loading test data and making predictions...")
    test_data = TabularDataset(test_path)
    
    # Remove index column from test data if it exists
    if 'Unnamed: 0' in test_data.columns:
        test_data = test_data.drop(columns=['Unnamed: 0'])
    
    # Make predictions
    if problem_type == 'binary' or problem_type == 'multiclass':
        predictions = predictor.predict(test_data)
    else:  # regression
        predictions = predictor.predict(test_data)
    
    # Create results dataframe
    results = pd.DataFrame()
    results[label_column] = predictions
    
    # Save results to the output directory
    results_path = os.path.join(output_dir, "results.pq")
    results.to_parquet(results_path)
    print(f"Predictions saved to {results_path}")