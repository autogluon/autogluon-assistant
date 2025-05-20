import logging
import os
import shutil
from pathlib import Path

import typer

from autogluon.assistant.coding_agent import run_agent
from autogluon.assistant.utils import extract_archives

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
    """
    Migrated main() logic from original run.py:
    1. Create output directory
    2. Optional copy + extraction
    3. Call run_agent to enter core iteration loop
    """
    # 0. Create output directory
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_path)

    # 1. Copy & extract (if needed)
    if extract_archives_to:
        dst = Path(extract_archives_to).expanduser().resolve()
        log.info("Copying data to %s …", dst)
        dst.mkdir(parents=True, exist_ok=True)

        for root, _, files in os.walk(input_data_folder):
            rel = os.path.relpath(root, input_data_folder)
            target = dst / rel if rel != "." else dst
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(Path(root) / f, target / f)

        input_data_folder = str(dst)
        typer.echo(f"[bold yellow]Notice:[/] extracted data to {input_data_folder}")
        extract_archives(input_data_folder)

    # 2. Call core run_agent
    run_agent(
        input_data_folder=input_data_folder,
        tutorial_link=None,
        output_folder=str(out_path),
        config_path=config_path,
        max_iterations=max_iterations,
        need_user_input=need_user_input,
        initial_user_input=initial_user_input,
    )
