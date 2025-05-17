import pandas as pd
import shutil
from pathlib import Path

def clean_csv(csv_path, columns_to_remove):
    """
    Process a CSV file by creating a backup, removing specified columns, and saving the cleaned data.
    
    Args:
        csv_path (str): Path to the input CSV file
        columns_to_remove (list): List of column names to remove
    """
    # Convert to Path object for easier path manipulation
    path = Path(csv_path)
    
    # Create backup path by adding '_backup' suffix
    backup_path = path.with_name(f"{path.stem}_backup{path.suffix}")
    
    # Create backup of original file
    shutil.copy2(csv_path, backup_path)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Remove specified columns
    df.drop(columns=columns_to_remove, errors='ignore', inplace=True)
    
    # Save the cleaned dataframe back to original path
    df.to_csv(csv_path, index=False)

# Example usage:
# csv_path = "your_file.csv"
# columns_to_remove = ["column1", "column2"]
# clean_csv(csv_path, columns_to_remove)

clean_csv("/media/agent/maab/datasets/yolanda/training/test.csv", ["101"])
