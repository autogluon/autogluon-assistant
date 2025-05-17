import argparse
import glob

import numpy as np
import pandas as pd


def process_maab_results(model_name, model_config):
    # Construct file pattern using provided model name and config
    file_pattern = f"/media/agent/AutoMLAgent/tools/results/maab_{model_name}_{model_config}_run*.csv"
    all_files = glob.glob(file_pattern)
    
    if len(all_files) != 3:
        print(f"Incorrect number of files ({len(all_files)}) found matching the pattern: {file_pattern}")
        return
    
    # List of all 25 datasets
    all_datasets = [
        "abalone", "airbnb_melbourne", "airlines", "bioresponse", "camo_sem_seg",
        "cd18", "climate_fever", "covertype", "electricity_hourly", "europeanflooddepth",
        "fiqabeir", "gnad10", "ham10000", "hateful_meme", "isic2017",
        "kick_starter_funding", "memotion", "mldoc", "nn5_daily_without_missing", "petfinder",
        "road_segmentation", "rvl_cdip", "solar_10_minutes", "women_clothing_review", "yolanda"
    ]
    
    # Initialize dictionaries to store performance and time data
    performance_data = {dataset: [] for dataset in all_datasets}
    time_data = {dataset: [] for dataset in all_datasets}
    
    # Initialize success counts for each run
    success_counts = [0, 0, 0]  # For run1, run2, run3
    
    # Process each run file
    for run in range(1, 4):
        run_file = f"/media/agent/AutoMLAgent/tools/results/maab_{model_name}_{model_config}_run{run}.csv"
        
        try:
            print(run_file)
            df = pd.read_csv(run_file)

            # NEW: Assert that the DataFrame columns are exactly 'dataset_name', 'performance', and 'time_used'
            expected_columns = ['dataset_name', 'performance', 'time_used']
            assert list(df.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(df.columns)}"
            
            # NEW: Filter out datasets not in all_datasets
            invalid_datasets = df[~df['dataset_name'].isin(all_datasets)]['dataset_name'].unique()
            if len(invalid_datasets) > 0:
                raise ValueError(f"Warning: Run {run} contains invalid datasets: {invalid_datasets}")
                # Remove invalid datasets from dataframe
                df = df[df['dataset_name'].isin(all_datasets)]
            
            # Step 1: Mark time as invalid for invalid performance
            df.loc[df['performance'] == -9999, 'time_used'] = -9999
            
            # Update success counts for this run
            run_idx = run - 1
            success_counts[run_idx] = len(df[df['performance'] != -9999])
            
            # Store performance and time data for each dataset
            for dataset in all_datasets:
                dataset_row = df[df['dataset_name'] == dataset]
                
                if not dataset_row.empty:
                    perf = dataset_row['performance'].values[0]
                    time_used = dataset_row['time_used'].values[0]
                    
                    performance_data[dataset].append(perf)
                    time_data[dataset].append(time_used)
                else:
                    # If dataset not found in this run, add invalid values
                    performance_data[dataset].append(-9999)
                    time_data[dataset].append(-9999)
                    
        except FileNotFoundError:
            print(f"Warning: File {run_file} not found.")
    
    # Calculate statistics
    stats = {}
    
    # Step 2: Calculate performance and time stats for each dataset
    for dataset in all_datasets:
        # Performance stats
        valid_perfs = [p for p in performance_data[dataset] if p != -9999]
        if valid_perfs:
            perf_mean = np.mean(valid_perfs)
            perf_std = np.std(valid_perfs) if len(valid_perfs) > 1 else 0
        else:
            perf_mean = -9999
            perf_std = -9999
        
        # Time stats
        valid_times = [t for t in time_data[dataset] if t != -9999]
        if valid_times:
            time_mean = np.mean(valid_times)
            time_std = np.std(valid_times) if len(valid_times) > 1 else 0
        else:
            time_mean = -9999
            time_std = -9999
        
        # Store the stats
        stats[f"{dataset}_performance_mean"] = perf_mean
        stats[f"{dataset}_performance_std"] = perf_std
        stats[f"{dataset}_time_used_mean"] = time_mean
        stats[f"{dataset}_time_used_std"] = time_std
    
    # Step 3: Calculate success rate stats
    total_datasets = len(all_datasets)
    success_rates = [count / total_datasets for count in success_counts]
    success_rate_mean = np.mean(success_rates)
    success_rate_std = np.std(success_rates)
    
    stats["success_rate_mean"] = success_rate_mean
    stats["success_rate_std"] = success_rate_std
    
    # Step 4: Write stats to CSV
    output_file = f"/media/agent/AutoMLAgent/tools/results/maab_{model_name}_{model_config}_stats.csv"
    
    # Prepare data for CSV
    rows = []
    for key, value in stats.items():
        if key.endswith("_mean"):
            base_key = key[:-5]  # Remove "_mean"
            std_key = f"{base_key}_std"
            if std_key in stats:
                rows.append([base_key, value, stats[std_key]])
    
    # Add success rate row
    rows.append(["success_rate", success_rate_mean, success_rate_std])
    
    # Create DataFrame and save to CSV
    stats_df = pd.DataFrame(rows, columns=["metric", "mean", "std"])
    stats_df.to_csv(output_file, index=False)
    
    print(f"Statistics saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process MAAB results and calculate statistics.')
    parser.add_argument('-n', '--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('-c', '--model_config', type=str, required=True, help='Configuration of the model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process results with provided arguments
    process_maab_results(args.model_name, args.model_config)