import os
import shutil


def process_folders(root_dir):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)

        if os.path.isdir(folder_path):
            env_folder = os.path.join(folder_path, "env")
            scripts_folder = os.path.join(folder_path, "scripts")
            aga_format_folder = os.path.join(folder_path, "aga_format")

            # Create aga_format folder if it doesn't exist
            os.makedirs(aga_format_folder, exist_ok=True)

            # Copy train.csv and test.csv from env to aga_format
            for file in ["train.csv", "test.csv"]:
                src = os.path.join(env_folder, file)
                dst = os.path.join(aga_format_folder, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"Warning: {src} not found")

            # Copy and rename research_problem.txt to data.txt
            src = os.path.join(scripts_folder, "research_problem.txt")
            dst = os.path.join(aga_format_folder, "data.txt")
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: {src} not found")

            # Create competition_files.txt
            competition_files_path = os.path.join(
                aga_format_folder, "competition_files.txt"
            )
            with open(competition_files_path, "w") as f:
                f.write("train.csv\ntest.csv\nsample_submission.csv")


if __name__ == "__main__":
    root_dir = "/media/deephome/data/aga_benchmark"
    process_folders(root_dir)
    print("Processing complete.")
