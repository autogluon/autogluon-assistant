import pandas as pd

# Define the method names from the LaTeX table
methods = [
    "Auto²ML (def)", "Auto²ML (8B)", "Auto²ML (-ext)", "Auto²ML (-epi)", 
    "Codex CLI (def)", "Codex CLI (+rea)", 
    "AIDE (def)", "AIDE (+ext)",
    "DS-Agent (def)", "DS-Agent (zero-shot)", 
    "AK (def)"
]

# Dataset names from the LaTeX table
datasets = [
    "abalone", "airbnb", "airlines", "bio", "camoseg", "cd18", "climate", 
    "covertype", "electric(H)", "flood", "fiqa", "gnad10", "ham10000", 
    "hateful", "isic2017", "funding", "memotion", "mldoc", "nn5(D)", 
    "petfinder", "roadseg", "rvlcdip", "solar(10m)", "clothing", "yolanda"
]

# Performance data from the LaTeX table, using -9999 for missing values (×)
# Performance data already adjusted with negative sign for metrics where lower is better
performance_data = [
    # Auto²ML (def), Auto²ML (8B), Auto²ML (-ext), Auto²ML (-epi), 
    # Codex CLI (def), Codex CLI (+rea), 
    # AIDE (def), AIDE (+ext),
    # DS-Agent (def), DS-Agent (zero-shot), 
    # AK (def)
    [-2.13, -2.09, -2.19, -2.13, -2.23, -2.27, -2.16, -2.18, -2.24, -2.36, -9999],  # abalone
    [0.43, 0.42, 0.24, 0.42, -9999, 0.39, 0.39, 0.39, -9999, 0.31, 0.32],  # airbnb
    [0.66, 0.69, 0.63, 0.66, -9999, 0.63, -9999, 0.65, -9999, 0.61, -9999],  # airlines
    [0.81, 0.80, 0.83, 0.80, 0.79, 0.85, 0.87, 0.84, 0.79, 0.79, -9999],  # bio
    [0.84, -9999, 0.46, 0.84, -9999, -9999, -9999, -9999, -9999, -9999, -9999],  # camoseg
    [0.46, -0.21, -1.57, 0.51, -0.94, -1.44, -9999, -0.05, -1.94, -0.64, -1.84],  # cd18
    [0.48, -9999, 0.24, 0.48, -9999, 0.20, -9999, -9999, -9999, -9999, -9999],  # climate
    [0.98, 0.98, 0.88, 0.98, 0.96, 0.92, 0.88, 0.86, 0.96, 0.95, 0.94],  # covertype
    [-1.42, -9999, -1.75, -1.40, -9999, -9999, -9999, -9999, -9999, -11.66, -9999],  # electric(H)
    [0.69, 0.69, 0.60, 0.68, -9999, 0.44, 0.71, 0.68, -9999, -9999, 0.58],  # flood
    [0.50, -9999, 0.22, 0.46, -9999, 0.20, -9999, -9999, -9999, -9999, -9999],  # fiqa
    [0.86, 0.85, 0.88, 0.83, 0.82, 0.85, 0.90, 0.58, -9999, 0.80, 0.11],  # gnad10
    [0.63, 0.57, 0.67, 0.67, 0.48, 0.47, 0.81, 0.81, -9999, -9999, -9999],  # ham10000
    [0.59, 0.57, 0.35, 0.59, -9999, 0.48, 0.51, 0.49, -9999, -9999, 0.36],  # hateful
    [0.75, -9999, -9999, -9999, -9999, 0.11, -9999, -9999, -9999, -9999, -9999],  # isic2017
    [0.45, 0.41, 0.36, 0.44, -9999, 0.34, -9999, 0.44, -9999, -9999, 0.24],  # funding
    [0.50, -9999, 0.83, 0.51, 0.53, 0.76, 0.47, -9999, -9999, -9999, -9999],  # memotion
    [0.95, 0.95, 0.94, 0.95, 0.32, 0.82, 0.96, 0.94, 0.95, -9999, -9999],  # mldoc
    [-0.76, -9999, -1.14, -0.76, -9999, -9999, -9999, -9999, -4.68, -9999, -9999],  # nn5(D)
    [0.39, 0.40, 0.38, 0.39, -9999, 0.40, 0.34, 0.38, 0.36, 0.27, 0.39],  # petfinder
    [0.47, -9999, 0.31, 0.60, -9999, -9999, -9999, -9999, -9999, -9999, -9999],  # roadseg
    [0.87, -9999, 0.89, 0.87, -9999, -9999, -9999, -9999, -9999, -9999, -9999],  # rvlcdip
    [-1.49, -9999, -9999, -1.29, -9999, -1.29, -1.05, -9999, -9999, -9999, -9999],  # solar(10m)
    [0.75, 0.61, 0.66, 0.72, -9999, 0.35, -9999, 0.75, -9999, 0.35, -9999],  # clothing
    [-8.53, -8.54, -8.93, -8.53, -9999, -9.43, -9999, -8.79, -9999, -9999, -9999]   # yolanda
]

def main():
    # Create a pandas DataFrame
    df = pd.DataFrame(performance_data, index=datasets, columns=methods)
    
    # Get ranks for each dataset (row)
    # For all datasets, higher values are better after our transformations
    ranks_df = df.rank(axis=1, method='average', ascending=False)
    
    # Calculate average rank for each method (column)
    avg_ranks = ranks_df.mean()
    
    # Calculate success rate for each method
    success_mask = (df != -9999)
    success_rate = success_mask.mean() * 100
    
    # Print results
    print("Average Ranks (lower is better):")
    for method, avg_rank in avg_ranks.items():
        print(f"{method}: {avg_rank:.2f} (Success rate: {success_rate[method]:.1f}%)")
    
    # Create a results DataFrame for better visualization
    results_df = pd.DataFrame({
        'Method': methods,
        'Avg Rank': avg_ranks.values,
        'Success Rate (%)': success_rate.values
    })
    results_df = results_df.sort_values('Avg Rank')
    
    print("\nSorted by Average Rank (best to worst):")
    print(results_df.to_string(index=False))
    
    # Display the number of datasets
    print(f"\nTotal number of datasets: {len(datasets)}")
    
    # Print the ranks for each dataset for verification
    print("\nRanks for each dataset (1 is best):")
    print(ranks_df)

if __name__ == "__main__":
    main()