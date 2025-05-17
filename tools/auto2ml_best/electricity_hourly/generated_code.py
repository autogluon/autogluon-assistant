#!/usr/bin/env python
"""
AutoGluon TimeSeriesPredictor for Electricity Hourly Forecasting

This script loads electricity consumption hourly data, trains a TimeSeriesPredictor model,
and generates 24-hour forecasts. It handles data preprocessing, model training with
appropriate parameters, and saves the results in the required format.

Usage:
    python electricity_forecasting.py

Additional installation requirements:
    pip install autogluon.timeseries
    pip install pandas
    pip install matplotlib
"""

import gzip
import json
import os
from datetime import datetime

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def load_jsonl_gz(file_path):
    """Load a gzipped JSONL file into a list of dictionaries."""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_timeseries_df(data_list):
    """Convert list of dictionaries to TimeSeriesDataFrame."""
    # Create a list to store the expanded data
    rows = []
    
    for item in data_list:
        item_id = item['item_id']
        start_time = pd.Timestamp(item['start'])
        target_values = item['target']
        
        # Create a row for each target value
        for i, value in enumerate(target_values):
            timestamp = start_time + pd.Timedelta(hours=i)
            rows.append({
                'item_id': item_id,
                'timestamp': timestamp,
                'target': value
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Convert to TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column='item_id',
        timestamp_column='timestamp'
    )
    
    return ts_df

def save_predictions_to_jsonl(predictions, output_path):
    """Save predictions to JSONL format."""
    # Group predictions by item_id
    result = []
    for item_id in predictions.index.get_level_values('item_id').unique():
        item_preds = predictions.loc[item_id]
        
        # Create the result dictionary
        result_item = {
            'item_id': item_id,
            'start': item_preds.index[0].strftime('%Y-%m-%d %H:%M:%S'),
            'target': item_preds['mean'].tolist()
        }
        
        result.append(result_item)
    
    # Save to gzipped JSONL
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    # Define paths
    data_path = "/media/agent/maab/datasets/electricity_hourly/training/data.json.gz"
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a model directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    raw_data = load_jsonl_gz(data_path)
    
    # Convert to TimeSeriesDataFrame
    print("Converting data to TimeSeriesDataFrame...")
    ts_data = convert_to_timeseries_df(raw_data)
    
    # Check for missing values and handle if necessary
    print("Checking for missing values...")
    if ts_data.isna().any().any():
        print("Filling missing values...")
        ts_data = ts_data.fill_missing_values()
    
    # Set prediction length to 24 hours as specified in the task
    prediction_length = 24
    
    # Split data for evaluation
    print("Splitting data for training and validation...")
    train_data, val_data = ts_data.train_test_split(prediction_length)
    
    # Initialize and train the predictor
    print("Initializing TimeSeriesPredictor...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_dir,
        eval_metric='MASE'
    )
    
    print("Training the model...")
    predictor.fit(time_limit=24*3600,
        train_data=train_data,
        tuning_data=val_data,
        presets="best_quality",
    )
    
    # Show model leaderboard
    print("Model leaderboard:")
    leaderboard = predictor.leaderboard(val_data)
    print(leaderboard)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predictor.predict(ts_data)
    
    # Ensure the output column is named correctly
    if 'mean' in predictions.columns:
        # No need to rename, already correct
        pass
    else:
        # Find the median quantile (usually '0.5') and rename it to 'mean'
        median_col = [col for col in predictions.columns if '0.5' in str(col)]
        if median_col:
            predictions.rename(columns={median_col[0]: 'mean'}, inplace=True)
        else:
            # If no median quantile, use the first column
            predictions.rename(columns={predictions.columns[0]: 'mean'}, inplace=True)
    
    # Save predictions
    print("Saving predictions...")
    results_path = os.path.join(output_dir, "results.json.gz")
    save_predictions_to_jsonl(predictions, results_path)
    
    print(f"Done! Results saved to {results_path}")