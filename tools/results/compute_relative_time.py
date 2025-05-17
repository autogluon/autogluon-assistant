import numpy as np
import pandas as pd

# Path to the CSV file
file_path = '/media/agent/AutoMLAgent/tools/results/maab_aggregated_results.csv'

# Read the CSV file
def calculate_relative_time_usage(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter out rows with missing time_used values
    df = df[pd.notna(df['time_used'])]
    
    # Get unique datasets and models
    datasets = df['dataset_name'].unique()
    models = df['model_name'].unique()
    
    # Dictionary to store relative time usage for each model on each dataset
    relative_time_usage = {model: [] for model in models if model != 'auto2ml_def'}
    
    # Calculate relative time usage for each dataset
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        
        # Get auto2ml_def time for this dataset (reference model)
        auto2ml_time = dataset_df[dataset_df['model_name'] == 'auto2ml_def']['time_used'].values
        
        # Skip dataset if auto2ml_def time is not available
        if len(auto2ml_time) == 0:
            continue
            
        # Use average instead of first value
        auto2ml_time = np.mean(auto2ml_time)
        
        # Calculate relative time for each model on this dataset
        for model in models:
            if model == 'auto2ml_def':
                continue
                
            model_time = dataset_df[dataset_df['model_name'] == model]['time_used'].values
            
            # Skip if model time is not available for this dataset
            if len(model_time) == 0:
                continue
                
            # Use average instead of first value
            model_time = np.mean(model_time)
            
            # Calculate ratio: model_time / auto2ml_time
            ratio = model_time / auto2ml_time
            
            # Add to the model's list of ratios
            relative_time_usage[model].append(ratio)
    
    # Calculate average relative time usage for each model
    average_relative_time = {}
    for model, ratios in relative_time_usage.items():
        if ratios:  # Check if list is not empty
            average_relative_time[model] = np.mean(ratios)
    
    # Add auto2ml_def (reference model) with value 1.0
    average_relative_time['auto2ml_def'] = 1.0
    
    return average_relative_time

# Run the calculation
average_relative_time = calculate_relative_time_usage(file_path)

# Print results
print("Average Relative Time Usage (auto2ml_def = 1.0):")
for model, value in sorted(average_relative_time.items(), key=lambda x: x[1]):
    print(f"{model}: {value:.2f}")