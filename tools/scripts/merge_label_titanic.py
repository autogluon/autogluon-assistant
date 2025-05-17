import os

import pandas as pd


def merge_csv_files(root_dir, output_dir):
    # Construct file paths
    answers_path = os.path.join(root_dir, "answer.csv")
    test_path = os.path.join(root_dir, "test.csv")

    # Read CSV files
    answers_df = pd.read_csv(answers_path)
    test_df = pd.read_csv(test_path)

    # Extract the 'Transported' column from answers_df
    transported_column = answers_df["Transported"]

    # Add the 'Transported' column to test_df
    test_df["Transported"] = transported_column

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save merged dataframe to output directory
    output_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(output_path, index=False)

    print(f"Merged CSV file saved to: {output_path}")


# Example usage
root_dir = "/media/deephome/data/aga_benchmark/spaceship-titanic/env"
output_dir = "/media/deephome/data/aga_benchmark/spaceship-titanic/aga_format"
merge_csv_files(root_dir, output_dir)
