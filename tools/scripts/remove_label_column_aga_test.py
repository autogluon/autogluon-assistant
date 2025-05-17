import os

import pandas as pd


def process_folder(folder_path):
    aga_format_path = os.path.join(folder_path, "aga_format")

    # Read id_n_label.txt
    with open(os.path.join(aga_format_path, "id_n_label.txt"), "r") as f:
        id_column = f.readline().strip()
        label_column = f.readline().strip()

    test_file = os.path.join(aga_format_path, "test.csv")

    # Read test.csv
    test_df = pd.read_csv(test_file)

    # Remove label column in test file
    if label_column in test_df.columns:
        test_df = test_df.drop(columns=[label_column])
        test_df.to_csv(test_file, index=False)


def main(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            try:
                process_folder(folder_path)
            except Exception as e:
                print(f"Error processing folder {folder_path}: {str(e)}")


if __name__ == "__main__":
    root_dir = "/media/deephome/maab/datasets"
    main(root_dir)
