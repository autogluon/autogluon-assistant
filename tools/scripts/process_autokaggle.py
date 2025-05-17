import os
import json
import shutil
import pandas as pd
import argparse

def process_dataset(dataset_path, train_filename=None, test_filename=None):
    """
    Process a dataset according to the specified structure.
    Args:
        dataset_path: Path to the dataset folder
        train_filename: Name of the training file (default: train.csv)
        test_filename: Name of the test file (default: test.csv)
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"Processing dataset: {dataset_name}")

    # Create autokaggle directory
    autokaggle_dir = os.path.join(dataset_path, "autokaggle")
    os.makedirs(autokaggle_dir, exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    label_column = metadata.get("label_column")

    # Process training directory
    training_dir = os.path.join(dataset_path, "training")

    # Copy description.txt to overview.txt
    description_path = os.path.join(training_dir, "descriptions.txt")
    overview_path = os.path.join(autokaggle_dir, "overview.txt")
    shutil.copy2(description_path, overview_path)

    # Set default filenames if not provided
    if train_filename is None:
        train_filename = "train.csv"
    if test_filename is None:
        test_filename = "test.csv"

    # Process train file
    train_file = os.path.join(training_dir, train_filename)
    train_output = os.path.join(autokaggle_dir, "train.csv")
    
    if train_file.endswith('.pq'):
        train_df = pd.read_parquet(train_file)
    else:
        train_df = pd.read_csv(train_file)
    
    # Add index column to train data
    train_df['index'] = range(len(train_df))
    train_df.to_csv(train_output, index=False)
    
    # Store the last index used in training data
    last_train_index = len(train_df) - 1

    # Process test file
    test_file = os.path.join(training_dir, test_filename)
    test_output = os.path.join(autokaggle_dir, "test.csv")
    
    if test_file.endswith('.pq'):
        test_df = pd.read_parquet(test_file)
    else:
        test_df = pd.read_csv(test_file)
        
    # Add index column to test data, starting after the last train index
    test_df['index'] = range(last_train_index + 10, last_train_index + 10 + len(test_df))
    test_df.to_csv(test_output, index=False)

    # Create submission file
    submission_output = os.path.join(autokaggle_dir, "sample_submission.csv")
    
    # Get first label value from training data to use as default
    first_label_value = train_df[label_column].iloc[0]
    
    # Create submission DataFrame
    submission_df = pd.DataFrame()
    submission_df['index'] = test_df['index']
    submission_df[label_column] = first_label_value
    submission_df.to_csv(submission_output, index=False)

    print(f"Successfully processed dataset: {dataset_name}")

def main():
    """
    Main function to process a single dataset with command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process a dataset for AutoKaggle format')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset folder')
    parser.add_argument('--train', type=str, help='Training file name (default: train.csv)')
    parser.add_argument('--test', type=str, help='Test file name (default: test.csv)')
    
    args = parser.parse_args()

    # Check if the path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Path '{args.dataset_path}' does not exist.")
        return

    # Check if it's a valid dataset
    if not os.path.exists(os.path.join(args.dataset_path, "metadata.json")):
        print(f"Error: '{args.dataset_path}' is not a valid dataset folder (missing metadata.json).")
        return

    # Process the dataset
    process_dataset(args.dataset_path, args.train, args.test)
    print("Dataset processed successfully!")

if __name__ == "__main__":
    main()
