# automlagent/src/automlagent/cli/commands/run_cmd.py

import logging
from pathlib import Path

from autogluon.assistant.coding_agent import run_agent

log = logging.getLogger(__name__)


def run_cmd(
    input_data_folder: str,
    output_dir: str,
    config_path: str,
    max_iterations: int = 5,
    need_user_input: bool = False,
    initial_user_input: str | None = None,
    extract_archives_to: str | None = None,
) -> None:

    # Ensure the parent directory of output_dir exists
    out_path = Path(output_dir).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Output directory (to be created by run_agent): %s", out_path)

    # Delegate the entire pipeline to run_agent, including:
    # 1. Copying and extracting archives (if extract_archives_to is set)
    # 2. Creating output folder
    # 3. Merging configuration
    # 4. Running the full prompt → code → execute → iterate loop
    run_agent(
        input_data_folder=input_data_folder,
        output_folder=str(out_path),
        tutorial_link=None,
        config_path=config_path,
        max_iterations=max_iterations,
        need_user_input=need_user_input,
        initial_user_input=initial_user_input,
        extract_archives_to=extract_archives_to,
    )
