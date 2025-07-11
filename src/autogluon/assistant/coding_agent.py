import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .rich_logging import configure_logging
from .utils import extract_archives

logger = logging.getLogger(__name__)


def run_agent(
    input_data_folder,
    output_folder=None,
    config_path=None,
    max_iterations=5,
    need_user_input=False,
    initial_user_input=None,
    extract_archives_to=None,
    manager=None,
    verbosity=1,
):
    # Get the directory of the current file
    current_file_dir = Path(__file__).parent

    if output_folder is None or not output_folder:
        working_dir = os.path.join(current_file_dir.parent.parent.parent, "runs")
        # Get current date in YYYYMMDD format
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Generate a random UUID4
        random_uuid = uuid.uuid4()
        # Create the folder name using the pattern
        folder_name = f"mlzero-{current_datetime}-{random_uuid}"

        # Create the full path for the new folder
        output_folder = os.path.join(working_dir, folder_name)

    # Create output directory
    output_dir = Path(output_folder).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=True)

    configure_logging(verbosity=verbosity, output_dir=output_dir)
    from .managers import Manager

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
    default_config_path = current_file_dir / "configs" / "default.yaml"
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")

    config = OmegaConf.load(default_config_path)

    # If config_path is provided, merge it with the default config
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)

    if manager is None:
        manager = Manager(
            input_data_folder=input_data_folder,
            output_folder=output_folder,
            config=config,
        )

    manager.set_initial_user_input(need_user_input=need_user_input, initial_user_input=initial_user_input)

    while manager.time_step + 1 < max_iterations:
        logger.brief(f"Starting iteration {manager.time_step + 1}!")

        manager.step()

        # Generate code
        manager.update_python_code()
        manager.update_bash_script()

        successful = manager.execute_code()
        if successful:
            break

        if manager.time_step + 1 >= max_iterations:
            logger.warning(
                f"[bold red]Warning: Reached maximum iterations ({max_iterations}) without success[/bold red]"
            )

    manager.report_token_usage()
    logger.brief(f"output saved in {output_dir}.")
    manager.cleanup()
