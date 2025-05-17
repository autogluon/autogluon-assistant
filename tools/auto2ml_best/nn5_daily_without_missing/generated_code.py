#!/usr/bin/env python
"""
Time Series Forecasting using AutoGluon for NN5 Daily Dataset

This script trains an AutoGluon TimeSeriesPredictor on the NN5 daily time series data
and generates forecasts for the next 56 days. The script handles data loading, preprocessing,
model training, and prediction.

Installation requirements:
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

if __name__ == "__main__":
    # Define paths
    input_data_path = '/media/agent/maab/datasets/nn5_daily_without_missing/training/data.json.gz'
    output_dir = "./"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for the model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Loading data...")
    
    # Load the JSONL data
    time_series_data = []
    with gzip.open(input_data_path, 'rt') as f:
        for line in f:
            time_series_data.append(json.loads(line))
    
    # Process the JSON data into a format suitable for TimeSeriesDataFrame
    all_data = []
    
    for ts in time_series_data:
        item_id = ts['item_id']
        start_date = pd.Timestamp(ts['start'])
        target_values = ts['target']
        
        # Create a date range starting from the start date
        dates = pd.date_range(start=start_date, periods=len(target_values), freq='D')
        
        # Create a DataFrame for this time series
        series_df = pd.DataFrame({
            'item_id': item_id,
            'timestamp': dates,
            'target': target_values
        })
        
        all_data.append(series_df)
    
    # Combine all time series into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Data loaded. Shape: {combined_df.shape}")
    print(f"Number of unique time series: {combined_df['item_id'].nunique()}")
    
    # Convert to TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        combined_df,
        id_column='item_id',
        timestamp_column='timestamp'
    )
    
    print("Data converted to TimeSeriesDataFrame.")
    
    # Define prediction length (56 days as specified in the task)
    prediction_length = 56
    
    # Create and train the TimeSeriesPredictor
    print("Training the model...")
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_dir,
        eval_metric='MASE'
    )
    
    # Train the model with best_quality preset and time limit of 1800 seconds
    predictor.fit(time_limit=24*3600,
        train_data=ts_df,
        presets="best_quality",
    )
    
    print("Model training completed.")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predictor.predict(ts_df)
    
    # Rename the 'mean' column to 'target' to match the input format
    predictions = predictions.rename(columns={'mean': 'target'})
    
    # Convert predictions to the required JSONL format
    print("Converting predictions to JSONL format...")
    result_data = []
    
    for item_id in predictions.index.get_level_values('item_id').unique():
        item_predictions = predictions.loc[item_id]['target'].tolist()
        
        # Get the start date for predictions (which is the day after the last training data point)
        item_last_date = ts_df.loc[item_id].index[-1]
        prediction_start = item_last_date + pd.Timedelta(days=1)
        
        # Create the JSON object
        json_obj = {
            'target': item_predictions,
            'start': prediction_start.strftime('%Y-%m-%d %H:%M:%S'),
            'item_id': item_id
        }
        
        # Add static categorical features if they exist in the original data
        for ts in time_series_data:
            if ts['item_id'] == item_id and 'feat_static_cat' in ts:
                json_obj['feat_static_cat'] = ts['feat_static_cat']
                break
        
        result_data.append(json_obj)
    
    # Save the results to a JSONL file
    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        for item in result_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Predictions saved to {output_file}")
    print("Process completed successfully.")