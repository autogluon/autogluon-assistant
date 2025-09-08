"""
MCTS-based search for AutoGluon Assistant.

This module provides an alternative approach to the sequential iteration-based
execution in AutoGluon Assistant. It implements a tree-based search strategy
using Monte Carlo Tree Search (MCTS) to explore the solution space more effectively.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .constants import DEFAULT_CONFIG_PATH
from .rich_logging import configure_logging
from .utils import extract_archives

logger = logging.getLogger(__name__)


def run_agent(
    input_data_folder,
    output_folder=None,
    config_path=None,
    max_iterations=10,  # Default higher for MCTS search
    continuous_improvement=None,
    enable_meta_prompting=None,
    enable_per_iteration_instruction=False,
    initial_user_input=None,
    extract_archives_to=None,
    verbosity=1,
):
    """
    Run the AutoGluon Assistant with MCTS-based search strategy.

    Args:
        input_data_folder: Path to input data directory
        output_folder: Path to output directory
        config_path: Path to configuration file
        max_iterations: Maximum number of iterations
        continuous_improvement: Whether to continue after finding a valid solution
        enable_meta_prompting: Whether to enable meta-prompting
        enable_per_iteration_instruction: Whether to ask for user input at each iteration
        initial_user_input: Initial user instruction
        extract_archives_to: Path to extract archives to
        verbosity: Verbosity level

    Returns:
        None
    """
    # Get the directory of the current file
    current_file_dir = Path(__file__).parent

    if output_folder is None or not output_folder:
        working_dir = os.path.join(current_file_dir.parent.parent.parent, "runs")
        # Get current date in YYYYMMDD format
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Generate a random UUID4
        random_uuid = uuid.uuid4()
        # Create the folder name using the pattern
        folder_name = f"mcts-{current_datetime}-{random_uuid}"

        # Create the full path for the new folder
        output_folder = os.path.join(working_dir, folder_name)

    # Create output directory
    output_dir = Path(output_folder).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=True)

    configure_logging(verbosity=verbosity, output_dir=output_dir)
    from .managers.node_manager import NodeManager

    if extract_archives_to is not None:
        if extract_archives_to and extract_archives_to != input_data_folder:
            import shutil

            # Create the destination directory if it doesn't exist
            os.makedirs(extract_archives_to, exist_ok=True)

            # Walk through all files and directories in the source folder
            for root, dirs, files in os.walk(input_data_folder):
                # Calculate the relative path from the source folder
                rel_path = os.path.relpath(root, input_data_folder)

                # Create the corresponding directory structure in the destination
                if rel_path != ".":
                    dest_dir = os.path.join(extract_archives_to, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                else:
                    dest_dir = extract_archives_to

                # Copy all files in the current directory
                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)  # copy2 preserves metadata

            input_data_folder = extract_archives_to
            logger.warning(
                f"Note: we strongly recommend using data without archived files. Extracting archived files under {input_data_folder}..."
            )
            extract_archives(input_data_folder)

    # Always load default config first
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Default config file not found: {DEFAULT_CONFIG_PATH}")

    config = OmegaConf.load(DEFAULT_CONFIG_PATH)

    # If config_path is provided, merge it with the default config
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)

    if continuous_improvement is not None:
        config.continuous_improvement = continuous_improvement
    if enable_meta_prompting is not None:
        config.enable_meta_prompting = enable_meta_prompting

    # Add specific MCTS configuration parameters
    config.exploration_constant = getattr(config, "exploration_constant", 1.414)
    config.max_debug_depth = getattr(config, "max_debug_depth", 5)
    config.max_evolve_attempts = getattr(config, "max_evolve_attempts", 3)
    config.max_debug_attempts = getattr(config, "max_debug_attempts", 3)
    config.metric_improvement_threshold = getattr(config, "metric_improvement_threshold", 0.01)

    # Create a new NodeManager instance
    manager = NodeManager(
        input_data_folder=input_data_folder,
        output_folder=output_folder,
        config=config,
        enable_per_iteration_instruction=enable_per_iteration_instruction,
        initial_user_input=initial_user_input,
    )

    # Initialize the manager (generate initial prompts)
    manager.initialize()

    # Variables for tracking best solutions
    best_success_found = False
    successive_failures = 0
    max_successive_failures = 5

    # Execute the MCTS search
    iteration = 0
    start_time = time.time()

    while iteration < max_iterations:
        # Log the current iteration
        logger.brief(f"Starting MCTS iteration {iteration + 1}/{max_iterations}")

        # Perform one step of the Monte Carlo Tree Search
        success = manager.step()

        if success:
            # Reset the successive failure counter
            successive_failures = 0

            # Flag that we've found at least one successful solution
            best_success_found = True

            # Create a best run copy when we find a successful solution
            manager.create_best_run_copy()

            # If not in continuous improvement mode, we can stop
            if not config.continuous_improvement:
                logger.brief("Stopping search - solution found and continuous improvement is disabled")
                break
        elif success is None:
            logger.brief("Stopping search - all nodes are terminal.")
            break
        else:
            # Increment successive failure counter
            successive_failures += 1

            # If we've had too many successive failures, but we have a successful solution, stop
            if successive_failures >= max_successive_failures and best_success_found:
                logger.warning(
                    f"Stopping search after {successive_failures} successive failures with a successful solution found"
                )
                break

        # Increment iteration counter
        iteration += 1

        # Check if we've exceeded the maximum iterations
        if iteration >= max_iterations:
            logger.warning(f"[bold red]Warning: Reached maximum iterations ({max_iterations})[/bold red]")

    manager.visualize_results()
    # Report token usage and validation score summary
    manager.cleanup()

    # Log summary
    elapsed_time = time.time() - start_time
    logger.brief(f"MCTS search completed in {elapsed_time:.2f} seconds")
    logger.brief(f"Total nodes explored: {manager.time_step + 1}")
    logger.brief(f"Best validation score: {manager.best_validation_score}")
    logger.brief(f"Tools used: {', '.join(manager.used_tools)}")
    logger.brief(f"Tools not used: {', '.join(set(manager.available_tools) - manager.used_tools)}")
    logger.brief(f"Output saved in {output_dir}")


if __name__ == "__main__":
    # This allows for direct execution of the script
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoGluon Assistant with MCTS search")
    parser.add_argument("-i", "--input", required=True, help="Path to data folder")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help=f"YAML config file (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument("-n", "--max-iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--continuous_improvement", action="store_true", help="Enable continuous improvement")
    parser.add_argument("--enable-meta-prompting", action="store_true", help="Enable meta-prompting")
    parser.add_argument(
        "--enable-per-iteration-instruction", action="store_true", help="Enable per-iteration instruction"
    )
    parser.add_argument("-t", "--initial-instruction", help="Initial instruction")
    parser.add_argument("-e", "--extract-to", help="Extract archives to this directory")
    parser.add_argument("-v", "--verbosity", type=int, default=1, help="Verbosity level (0-4)")
    parser.add_argument("--exploration", type=float, default=1.414, help="MCTS exploration constant")
    parser.add_argument("--max-debug", type=int, default=5, help="Maximum debug attempts for a node")
    parser.add_argument("--max-evolve", type=int, default=3, help="Maximum evolve attempts for a node")

    args = parser.parse_args()

    run_agent(
        input_data_folder=args.input,
        output_folder=args.output,
        config_path=args.config,
        max_iterations=args.max_iterations,
        continuous_improvement=args.continuous_improvement,
        enable_meta_prompting=args.enable_meta_prompting,
        enable_per_iteration_instruction=args.enable_per_iteration_instruction,
        initial_user_input=args.initial_instruction,
        extract_archives_to=args.extract_to,
        verbosity=args.verbosity,
    )
