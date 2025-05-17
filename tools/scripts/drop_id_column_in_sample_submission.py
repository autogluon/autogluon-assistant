import os

import pandas as pd


def process_dataset(dataset_path):
    train_path = os.path.join(dataset_path, "training", "train.csv")
    sample_submission_path = os.path.join(
        dataset_path, "training", "sample_submission.csv"
    )

    # Check if both files exist
    if not (os.path.exists(train_path) and os.path.exists(sample_submission_path)):
        print(f"Skipping {dataset_path}: Missing required files")
        return

    # Read the CSV files
    train_df = pd.read_csv(train_path)
    sample_submission_df = pd.read_csv(sample_submission_path)

    # Get the columns from train.csv
    train_columns = set(train_df.columns)

    # Filter the columns in sample_submission.csv
    columns_to_keep = [
        col for col in sample_submission_df.columns if col in train_columns
    ]
    pruned_submission_df = sample_submission_df[columns_to_keep]

    # Save the pruned sample_submission.csv
    pruned_submission_df.to_csv(sample_submission_path, index=False)
    print(f"Processed {dataset_path}")


def main(root_dir):
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)
        if os.path.isdir(dataset_path):
            process_dataset(dataset_path)


if __name__ == "__main__":
    root_dir = "/media/deephome/maab/datasets"
    main(root_dir)
