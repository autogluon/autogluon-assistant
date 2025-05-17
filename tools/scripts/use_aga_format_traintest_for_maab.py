import os
import shutil


def copy_traintest_from_aga(root_dir):
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)

        if not os.path.isdir(dataset_path):
            continue

        training_path = os.path.join(dataset_path, "training")
        aga_path = os.path.join(dataset_path, "aga_format")

        # Copy files to training folder
        shutil.copy(
            os.path.join(aga_path, "train.csv"),
            os.path.join(training_path, "train.csv"),
        )
        shutil.copy(
            os.path.join(aga_path, "test.csv"), os.path.join(training_path, "test.csv")
        )


# Usage
root_dir = (
    "/media/deephome/maab/datasets"
)  # Replace with the actual root directory path
copy_traintest_from_aga(root_dir)
