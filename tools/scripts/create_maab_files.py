import json
import os
import shutil


def create_folder_structure(root_dir):
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)

        if not os.path.isdir(dataset_path):
            continue

        # Create training folder
        training_path = os.path.join(dataset_path, "training")
        os.makedirs(training_path, exist_ok=True)

        # Copy files to training folder
        shutil.copy(
            os.path.join(dataset_path, "aga_format", "data.txt"),
            os.path.join(training_path, "descriptions.txt"),
        )
        shutil.copy(
            os.path.join(dataset_path, "aga_format", "sample_submission.csv"),
            os.path.join(training_path, "sample_submission.csv"),
        )
        shutil.copy(
            os.path.join(dataset_path, "env", "train.csv"),
            os.path.join(training_path, "train.csv"),
        )
        shutil.copy(
            os.path.join(dataset_path, "env", "test.csv"),
            os.path.join(training_path, "test.csv"),
        )

        # Create eval folder
        eval_path = os.path.join(dataset_path, "eval")
        os.makedirs(eval_path, exist_ok=True)

        # Copy file to eval folder
        shutil.copy(
            os.path.join(dataset_path, "aga_format", "test.csv"),
            os.path.join(eval_path, "ground_truth.csv"),
        )

        # Create metadata.json
        metadata = {
            "dataset_name": dataset_name,
            "metric_name": "",
            "problem_type": "",
            "label_column": "",
        }

        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


# Usage
root_dir = (
    "/media/deephome/data/aga_benchmark"
)  # Replace with the actual root directory path
create_folder_structure(root_dir)
