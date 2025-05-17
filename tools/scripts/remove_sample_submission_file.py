import os


def process_dataset(dataset_path):
    sample_submission_path = os.path.join(
        dataset_path, "training", "sample_submission.csv"
    )

    # Delete sample_submission.csv if it exists
    if os.path.exists(sample_submission_path):
        os.remove(sample_submission_path)
        print(f"Deleted sample_submission.csv in {dataset_path}")
    else:
        print(f"No sample_submission.csv found in {dataset_path}")

    print(f"Processed {dataset_path}")


def main(root_dir):
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)
        if os.path.isdir(dataset_path):
            process_dataset(dataset_path)


if __name__ == "__main__":
    root_dir = "/media/deephome/maab/datasets"
    main(root_dir)
