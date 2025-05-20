#!/usr/bin/env python3
from __future__ import annotations
import logging
import typer

from .rich_logging import (
    configure_logging,
    BRIEF_LEVEL,
    MODEL_INFO_LEVEL,
)

from .commands.run_cmd import run_cmd

app = typer.Typer(add_completion=False)

@app.callback(invoke_without_command=True)
def _global_options(
    verbosity: int = typer.Option(
        0, "-v", "--verbosity", count=True,
        help="-v => INFO, -vv => DEBUG"
    ),
    model_info: bool = typer.Option(
        False, "-m", "--model-info",
        help="Show MODEL_INFO level logs"
    ),
):
    """
    Determine the root logger level based on user parameters
    """
    if model_info:
        level = MODEL_INFO_LEVEL
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = BRIEF_LEVEL          # Default
    configure_logging(level)

# ---------------- run subcommand -----------------
@app.command("run")
def run(
    input_data_folder: str = typer.Option(..., "-i", help="Path to data folder"),
    output_dir:        str = typer.Option(..., "-o", help="Output directory"),
    config_path:       str = typer.Option(..., "-c", help="YAML config"),
    max_iterations:    int = typer.Option(5, "-n", help="Max iteration count"),
    need_user_input:   bool = typer.Option(False, "--need-user-input"),
    initial_user_input:str|None = typer.Option(None, "-u"),
    extract_archives_to:str|None= typer.Option(None, "-e"),
):
    run_cmd(
        input_data_folder=input_data_folder,
        output_dir=output_dir,
        config_path=config_path,
        max_iterations=max_iterations,
        need_user_input=need_user_input,
        initial_user_input=initial_user_input,
        extract_archives_to=extract_archives_to,
    )

if __name__ == "__main__":   # Allows usage like: python -m automlagent.cli.app
    app()
