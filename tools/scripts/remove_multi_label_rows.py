import pandas as pd

def remove_multilabel_rows(input_path, output_path=None):
    """
    Remove rows with multiple labels from a CSV file.
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the filtered CSV. 
            If None, will overwrite the input file.
    
    Returns:
        int: Number of rows removed
    """
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Get initial row count
    initial_count = len(df)
    
    # Keep only rows where the labels column doesn't contain a comma
    df_filtered = df[~df['labels'].str.contains(',', na=False)]
    
    # Calculate number of removed rows
    removed_count = initial_count - len(df_filtered)
    
    # Determine output path
    final_path = output_path if output_path else input_path
    
    # Save the filtered dataframe
    df_filtered.to_csv(final_path, index=False)
    
    return removed_count

if __name__ == "__main__":
    # Define the input path
    input_path = "/media/agent/maab/unfinished_datasets/fsdkaggle2019/fsd2019_single/eval/groundtruth.csv"  # Replace with your actual file path
    
    # Remove multi-label rows and save to the same file
    removed = remove_multilabel_rows(input_path)
    print(f"Removed {removed} rows with multiple labels")
    print(f"Updated file saved to: {input_path}")
