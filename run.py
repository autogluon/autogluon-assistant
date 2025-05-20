import argparse
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from autogluon.assistant.coding_agent import run_agent
from autogluon.assistant.utils import extract_archives

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate and execute code using AutoML Agent")
    parser.add_argument("-i", "--input_data_folder", required=True, help="Path to the input data folder")
    parser.add_argument("-e", "--extract_archives_to", default=None, help="Extract the archives.")
    parser.add_argument("-o", "--output_dir", default=None, help="Path to output directory")
    parser.add_argument("-c", "--config_path", default=None, help="Path to configuration file")
    parser.add_argument(
        "-n",
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of iterations for code generation",
    )
    parser.add_argument(
        "--need_user_input",
        action="store_true",
        help="Enable user input between iterations",
    )
    parser.add_argument("-u", "--initial_user_input", default=None, help="Initial user input (Optional)")

    args = parser.parse_args()

    if args.output_dir is None or not args.output_dir:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get current date in YYYYMMDD format
        current_date = datetime.now().strftime("%Y%m%d")
        # Generate a random UUID4
        random_uuid = uuid.uuid4()
        # Create the folder name using the pattern
        folder_name = f"mlzero-{current_date}-{random_uuid}"

        # Create the full path for the new folder
        args.output_dir = os.path.join(current_dir, "runs", folder_name)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    if args.extract_archives_to is not None:
        if args.extract_archives_to and args.extract_archives_to != args.input_data_folder:
            # TODO: copy all from args.input_data_folder to args.extract_archives_to
            import shutil

            # Create the destination directory if it doesn't exist
            os.makedirs(args.extract_archives_to, exist_ok=True)

            # Walk through all files and directories in the source folder
            for root, dirs, files in os.walk(args.input_data_folder):
                # Calculate the relative path from the source folder
                rel_path = os.path.relpath(root, args.input_data_folder)

                # Create the corresponding directory structure in the destination
                if rel_path != ".":
                    dest_dir = os.path.join(args.extract_archives_to, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                else:
                    dest_dir = args.extract_archives_to

                # Copy all files in the current directory
                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)  # copy2 preserves metadata

            args.input_data_folder = args.extract_archives_to
            print(
                f"Note: we strongly recommend using data without archived files. Extracting archived files under {args.input_data_folder}..."
            )
            extract_archives(args.input_data_folder)

    # Generate and execute code
    run_agent(
        input_data_folder=args.input_data_folder,
        tutorial_link=None,  # TODO: Only needed if we use RAG
        output_folder=str(output_dir),
        config_path=args.config_path,
        max_iterations=args.max_iterations,
        need_user_input=args.need_user_input,
        initial_user_input=args.initial_user_input,
    )


if __name__ == "__main__":
    main()
