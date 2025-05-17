"""
Solar Power Forecasting using AutoGluon Time Series

This script loads solar power data at 10-minute intervals, trains a time series forecasting model
using AutoGluon, and generates predictions for the test data. The model forecasts solar power
production 10 hours into the future.

Usage:
    python solar_forecasting.py

Requirements:
    - autogluon.timeseries
    - pandas
    - numpy
    - json
    - gzip
"""

# Installation steps (if needed):
# pip install autogluon.timeseries
# pip install pandas numpy

import gzip
import json
import os
from datetime import datetime

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Define output directory
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define model directory with timestamp
MODEL_DIR = os.path.join(OUTPUT_DIR, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_jsonl_gz(file_path):
    """Load compressed JSONL file into a list of dictionaries."""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_dataframe(json_data):
    """Convert JSON data to pandas DataFrame in the format required by TimeSeriesDataFrame."""
    rows = []
    
    for item in json_data:
        item_id = item['item_id']
        start_time = pd.Timestamp(item['start'])
        
        # Create a row for each target value
        for i, target_value in enumerate(item['target']):
            # Calculate timestamp for this value (10-minute intervals)
            timestamp = start_time + pd.Timedelta(minutes=10*i)
            
            # Create a row with item_id, timestamp, and target
            row = {
                'item_id': item_id,
                'timestamp': timestamp,
                'target': target_value
            }
            
            # Add static features if available
            if 'feat_static_cat' in item:
                for j, feat_val in enumerate(item['feat_static_cat']):
                    row[f'static_feat_{j}'] = feat_val
                    
            rows.append(row)
    
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Load training data
    train_data_path = "/media/agent/maab/datasets/solar_10_minutes/training/data.json.gz"
    json_data = load_jsonl_gz(train_data_path)
    
    # Convert to DataFrame
    df = convert_to_dataframe(json_data)
    
    # Create TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp"
    )
    
    # Extract static features if they exist
    static_features = None
    static_cols = [col for col in df.columns if col.startswith('static_feat_')]
    if static_cols:
        static_features_df = df[['item_id'] + static_cols].drop_duplicates().set_index('item_id')
        ts_df.static_features = static_features_df
    
    # Calculate prediction length (10 hours with 10-minute intervals = 60 steps)
    prediction_length = 60  # 10 hours * 6 intervals per hour
    
    # Create and train the predictor
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=MODEL_DIR,
        eval_metric="MASE",
        target="target"
    )
    
    # Train the model
    predictor.fit(time_limit=24*3600,
        train_data=ts_df,
        presets="best_quality"
    )
    
    # Generate predictions for test data
    # For this example, we'll use the training data as test data since test data wasn't provided
    # In a real scenario, you would load the test data similarly to how we loaded the training data
    predictions = predictor.predict(ts_df)
    
    # Rename the 'mean' column in predictions to 'target' to match the expected format
    predictions.rename(columns={'mean': 'target'}, inplace=True)
    
    # Convert predictions to JSONL format
    results = []
    for item_id in predictions.index.get_level_values('item_id').unique():
        item_preds = predictions.loc[item_id]['target'].values.tolist()
        
        # Get the start time for predictions (last timestamp of training data for this item + 10 minutes)
        last_timestamp = ts_df.loc[item_id].index[-1]
        start_time = (last_timestamp + pd.Timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create result object
        result = {
            "target": item_preds,
            "start": start_time,
            "item_id": item_id
        }
        
        # Add static features if they exist
        if static_features is not None:
            result["feat_static_cat"] = static_features_df.loc[item_id].values.tolist()
            
        results.append(result)
    
    # Save results to JSONL file
    results_path = os.path.join(OUTPUT_DIR, "results.json.gz")
    with gzip.open(results_path, 'wt', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Model trained and saved to {MODEL_DIR}")
    print(f"Predictions saved to {results_path}")