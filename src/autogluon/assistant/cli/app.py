#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

import typer

from autogluon.assistant.coding_agent import run_agent
from autogluon.assistant.constants import BRIEF_LEVEL, MODEL_INFO_LEVEL

from .. import __file__ as assistant_file
from ..rich_logging import configure_logging

PACKAGE_ROOT = Path(assistant_file).parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    # === Run parameters ===
    input_data_folder: str = typer.Option(..., "-i", "--input", help="Path to data folder"),
    output_dir: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory (if omitted, auto-generated under runs/)",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "-c",
        "--config",
        help=f"YAML config file (default: {DEFAULT_CONFIG_PATH})",
    ),
    max_iterations: int = typer.Option(5, "-n", "--max-iterations", help="Max iteration count"),
    need_user_input: bool = typer.Option(False, "--need-user-input", help="Whether to prompt user each iteration"),
    initial_user_input: str | None = typer.Option(None, "-u", "--user-input", help="Initial user input"),
    extract_archives_to: str | None = typer.Option(
        None, "-e", "--extract-to", help="Directory in which to unpack any archives"
    ),
    # === Logging parameters ===
    verbosity: int = typer.Option(1, "-v", "--verbosity", help="Verbosity level (0–4)"),
):
    """
    mlzero: a CLI for running the AutoMLAgent pipeline.
    """

    # 1) Configure logging
    match verbosity:
        case 0:
            level = logging.ERROR     # Only errors
        case 1:
            level = BRIEF_LEVEL       # Brief summaries
        case 2:
            level = logging.INFO      # Standard info
        case 3:
            level = MODEL_INFO_LEVEL  # Model details
        case _:  # 4+
            level = logging.DEBUG     # Full debug info
    configure_logging(level)

    # 2) If the user specified output_dir, ensure its parent directory exists;
    #    otherwise pass None to let run_agent auto-generate the output path
    if output_dir:
        out_path = output_dir.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_folder_arg = str(out_path)
        logging.getLogger(__name__).info("Output directory to be created: %s", out_path)
    else:
        output_folder_arg = None

    # 3) Invoke the core run_agent function
    run_agent(
        input_data_folder=input_data_folder,
        output_folder=output_folder_arg,
        tutorial_link=None,
        config_path=str(config_path),
        max_iterations=max_iterations,
        need_user_input=need_user_input,
        initial_user_input=initial_user_input,
        extract_archives_to=extract_archives_to,
    )


if __name__ == "__main__":
    app()
