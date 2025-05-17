import os

import pandas as pd


def process_folder(folder_path):
    aga_format_path = os.path.join(folder_path, "aga_format")

    # Read id_n_label.txt
    with open(os.path.join(aga_format_path, "id_n_label.txt"), "r") as f:
        id_column = f.readline().strip()
        label_column = f.readline().strip()

    # Read train.csv and test.csv
    train_df = pd.read_csv(os.path.join(aga_format_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(aga_format_path, "test.csv"))

    # Add id column if necessary
    if not id_column:
        train_df.insert(0, "id", range(1, len(train_df) + 1))
        test_df.insert(
            0, "id", range(len(train_df) + 1, len(train_df) + len(test_df) + 1)
        )
        id_column = "id"
    else:
        print(f"id column exists in {aga_format_path}")

    # Check if label column exists in test file
    if label_column not in test_df.columns:
        print(
            f"Error: Label column '{label_column}' not found in test.csv for folder {folder_path}"
        )
        return

    # Create sample_submission.csv
    sample_submission = test_df[[id_column, label_column]].head(10)
    sample_submission.to_csv(
        os.path.join(aga_format_path, "sample_submission.csv"), index=False
    )


def main(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            try:
                process_folder(folder_path)
            except Exception as e:
                print(f"Error processing folder {folder_path}: {str(e)}")


if __name__ == "__main__":
    root_dir = "/media/deephome/data/aga_benchmark"
    main(root_dir)
