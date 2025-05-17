import argparse
import logging
import os

from automlagent.llm import ChatLLMFactory
from automlagent.prompt import generate_data_prompt_with_llm
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate data prompt from folder contents using LLM")

    # Add arguments
    parser.add_argument("folder_path", type=str, help="Path to the folder to analyze")
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=1000,
        help="Maximum characters to read from each file",
    )
    parser.add_argument(
        "-p",
        "--config-path",
        type=str,
        default="/media/agent/AutoMLAgent/automlagent/src/automlagent/configs/default.yaml",
        help="Path to LLM configuration file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save the output (if not specified, prints to stdout)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate folder path
    if not os.path.isdir(args.folder_path):
        logger.error(f"Error: '{args.folder_path}' is not a valid directory")
        exit(1)

    # Initialize LLM

    config = OmegaConf.load(args.config_path)

    # Generate the data prompt
    logger.info("Generating data prompt...")
    prompt = generate_data_prompt_with_llm(
        input_data_folder=args.folder_path,
        max_chars_per_file=args.max_chars_per_file,
        llm_config=config.file_reader
    )

    # Output the result
    if args.output_file:
        logger.info(f"Writing output to: {args.output_file}")
        with open(args.output_file, 'w') as f:
            f.write(prompt)
    else:
        print(prompt)
    
    logger.info("Done!")
    
    # Log token usage if available
    try:
        token_usage = ChatLLMFactory.get_total_token_usage()
        logger.info(f"Total token usage: {token_usage}")
    except:
        pass


if __name__ == "__main__":
    main()
