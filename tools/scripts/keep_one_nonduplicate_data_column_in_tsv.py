import pandas as pd
import shutil
from pathlib import Path

def clean_csv(csv_path, columns_to_remove):
    """
    Process a TSV file by creating a backup, removing specified columns, 
    and keeping only the first occurrence of each query-id.
    
    Args:
        csv_path (str): Path to the input TSV file
        columns_to_remove (list): List of column names to remove
    """
    # Convert to Path object for easier path manipulation
    path = Path(csv_path)
    
    # Create backup path by adding '_backup' suffix
    backup_path = path.with_name(f"{path.stem}_backup{path.suffix}")
    
    # Create backup of original file
    shutil.copy2(csv_path, backup_path)
    
    # Read the TSV file
    df = pd.read_csv(csv_path, sep='\t')
    
    # Keep only the first occurrence of each query-id
    df = df.drop_duplicates(subset=['query-id'], keep='first')
    
    # Remove specified columns
    df.drop(columns=columns_to_remove, errors='ignore', inplace=True)
    
    # Save the cleaned dataframe back to original path, maintaining TSV format
    df.to_csv(csv_path, sep='\t', index=False)

# Example usage:
clean_csv("/media/agent/maab/datasets/climate_fever/training/test.tsv", ["score", "corpus-id"])
