import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from autogluon.assistant.coding_agent import run_agent

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
        "--enable_per_iteration_instruction",
        "--need_user_input",
        dest="enable_per_iteration_instruction",
        action="store_true",
        help="Prompt for a user instruction between iterations",
    )
    parser.add_argument("-t", "--initial_user_input", default=None, help="Initial user instruction (optional)")

    args = parser.parse_args()

    # Generate and execute code
    run_agent(
        input_data_folder=args.input_data_folder,
        output_folder=args.output_dir,
        config_path=args.config_path,
        max_iterations=args.max_iterations,
        enable_per_iteration_instruction=args.enable_per_iteration_instruction,
        initial_user_input=args.initial_user_input,
        extract_archives_to=args.extract_archives_to,
    )


if __name__ == "__main__":
    main()
