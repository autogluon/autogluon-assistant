#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing.resource_tracker
from pathlib import Path

import typer

from autogluon.assistant.coding_agent import run_agent


def _noop(*args, **kwargs):
    pass


multiprocessing.resource_tracker.register = _noop
multiprocessing.resource_tracker.unregister = _noop
multiprocessing.resource_tracker.ensure_running = _noop

PACKAGE_ROOT = Path(__file__).parent.parent
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
    max_iterations: int = typer.Option(
        5,
        "-n",
        "--max-iterations",
        help="Max iteration count. If the task hasn’t succeeded after this many iterations, it will terminate.",
    ),
    need_user_input: bool = typer.Option(
        False,
        "--enable-per-iteration-instruction",
        help="If enabled, provide an instruction at the start of each iteration (except the first, which uses the initial instruction). The process suspends until you provide it.",
    ),
    initial_user_input: str | None = typer.Option(
        None, "-t", "--initial-instruction", help="You can provide the initial instruction here."
    ),
    extract_archives_to: str | None = typer.Option(
        None,
        "-e",
        "--extract-to",
        help="Copy input data to specified directory and automatically extract all .zip archives. ",
    ),
    # === Logging parameters ===
    verbosity: int = typer.Option(
        1,
        "-v",
        "--verbosity",
        help=(
            "-v 0: Only includes error messages\n"
            "-v 1: Contains key essential information\n"
            "-v 2: Includes brief information plus detailed information such as file save locations\n"
            "-v 3: Includes info-level information plus all model training related information\n"
            "-v 4: Includes full debug information"
        ),
    ),
):
    """
    mlzero: a CLI for running the AutoMLAgent pipeline.
    """

    # 3) Invoke the core run_agent function
    run_agent(
        input_data_folder=input_data_folder,
        output_folder=output_dir,
        config_path=str(config_path),
        max_iterations=max_iterations,
        need_user_input=need_user_input,
        initial_user_input=initial_user_input,
        extract_archives_to=extract_archives_to,
        verbosity=verbosity,
    )


if __name__ == "__main__":
    app()
