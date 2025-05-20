#!/usr/bin/env python3
from __future__ import annotations

import logging

import typer

from .commands.run_cmd import run_cmd
from .rich_logging import (
    BRIEF_LEVEL,
    MODEL_INFO_LEVEL,
    configure_logging,
)

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    # ============== Run parameters ==============
    input_data_folder: str = typer.Option(..., "-i", "--input", help="Path to data folder"),
    output_dir: str = typer.Option(..., "-o", "--output", help="Output directory"),
    config_path: str = typer.Option(..., "-c", "--config", help="YAML config"),
    max_iterations: int = typer.Option(5, "-n", "--max-iterations", help="Max iteration count"),
    need_user_input: bool = typer.Option(False, "--need-user-input", help="Whether to prompt user each iteration"),
    initial_user_input: str | None = typer.Option(None, "-u", "--user-input", help="Initial user input"),
    extract_archives_to: str | None = typer.Option(
        None, "-e", "--extract-to", help="Directory in which to unpack any archives"
    ),
    # ============ Logging parameters ============
    verbosity: int = typer.Option(0, "-v", "--verbosity", count=True, help="-v => INFO, -vv => DEBUG"),
    model_info: bool = typer.Option(False, "-m", "--model-info", help="Show MODEL_INFO level logs"),
):
    """
    mlzero: a CLI for running the AutoMLAgent pipeline.
    """
    # 1) configure logging based on verbosity / model_info
    if model_info:
        level = MODEL_INFO_LEVEL
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = BRIEF_LEVEL
    configure_logging(level)

    # 2) delegate to your existing run_cmd (which calls run_agent)
    run_cmd(
        input_data_folder=input_data_folder,
        output_dir=output_dir,
        config_path=config_path,
        max_iterations=max_iterations,
        need_user_input=need_user_input,
        initial_user_input=initial_user_input,
        extract_archives_to=extract_archives_to,
    )


if __name__ == "__main__":
    app()
