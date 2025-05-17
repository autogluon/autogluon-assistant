import glob
import os

import pandas as pd

# Define the expected datasets
EXPECTED_DATASETS = [
    'abalone', 'airbnb_melbourne', 'airlines', 'bioresponse', 'camo_sem_seg',
    'cd18', 'climate_fever', 'covertype', 'electricity_hourly', 'europeanflooddepth',
    'fiqabeir', 'gnad10', 'ham10000', 'hateful_meme', 'isic2017',
    'kick_starter_funding', 'memotion', 'mldoc', 'nn5_daily_without_missing', 'petfinder',
    'road_segmentation', 'rvl_cdip', 'solar_10_minutes', 'women_clothing_review', 'yolanda'
]

# Define datasets that can only be null or positive
POSITIVE_OR_NULL_DATASETS = [
    'abalone', 'yolanda', 'electricity_hourly', 
    'nn5_daily_without_missing', 'solar_10_minutes'
]

# Define datasets where lower values are better (higher_is_better = False)
LOWER_IS_BETTER_DATASETS = [
    'abalone', 'yolanda', 'electricity_hourly', 
    'nn5_daily_without_missing', 'solar_10_minutes'
]

# Function to validate a single CSV file
def validate_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check 1: Assert that exactly the expected datasets exist
    datasets = df['dataset_name'].tolist()
    assert set(datasets) == set(EXPECTED_DATASETS), f"File {file_path} has incorrect datasets"
    assert len(datasets) == len(EXPECTED_DATASETS), f"File {file_path} has duplicate datasets"
    
    # Check 2: Check positive or null values for specific datasets
    for dataset in POSITIVE_OR_NULL_DATASETS:
        dataset_row = df[df['dataset_name'] == dataset]
        performance = dataset_row['performance'].values[0]
        assert performance == -9999 or performance > 0, f"Dataset {dataset} in {file_path} has invalid performance value: {performance}"
    
    # Check 3: If performance is null, time_used should also be null
    null_performance = df[df['performance'] == -9999]
    for _, row in null_performance.iterrows():
        assert row['time_used'] == -9999, f"Dataset {row['dataset_name']} in {file_path} has null performance but non-null time_used"
    
    # Extract agent_name and config_name from filename
    filename = os.path.basename(file_path)
    parts = filename.replace('.csv', '').split('_')
    
    # Handle different file name formats
    if 'run' in parts[-1]:
        run_num = parts[-1].replace('run', '')
        if '+' in filename:
            # For cases like 'maab_aide_+ext_run1.csv'
            agent_name = parts[1]
            config_name = '_'.join(parts[2:-1])
        else:
            # For other cases
            agent_name = parts[1]
            config_name = '_'.join(parts[2:-1])
    else:
        # Handle any other naming pattern if needed
        agent_name = parts[1]
        config_name = '_'.join(parts[2:-1])
        run_num = "unknown"
    
    model_name = f"{agent_name}_{config_name}"
    
    # Add model_name and run_num columns
    df['model_name'] = model_name
    df['run_num'] = run_num
    
    # Add higher_is_better column
    df['higher_is_better'] = df['dataset_name'].apply(
        lambda x: False if x in LOWER_IS_BETTER_DATASETS else True
    )
    
    return df

# Main function to aggregate all CSV files
def aggregate_maab_results():
    # Find all CSV files matching the pattern
    csv_files = glob.glob("maab_*_run*.csv")
    
    if not csv_files:
        print("No CSV files found matching the pattern")
        return
    
    # Validate and process each file
    all_dfs = []
    for file_path in csv_files:
        try:
            df = validate_csv(file_path)
            all_dfs.append(df)
            print(f"Successfully validated and processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise
    
    # Concatenate all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs)
        
        # Replace -9999 with empty string in the final output
        combined_df = combined_df.replace(-9999, '')
        
        # Rearrange columns: dataset_name, model_name, run_num, performance, time_used, higher_is_better
        combined_df = combined_df[['dataset_name', 'model_name', 'run_num', 'performance', 'time_used', 'higher_is_better']]
        
        # Write to the output file
        combined_df.to_csv('maab_aggregated_results.csv', index=False)
        print(f"Successfully created maab_aggregated_results.csv with {len(combined_df)} entries.")
    else:
        print("No valid files were processed.")

# Run the aggregation
if __name__ == "__main__":
    aggregate_maab_results()