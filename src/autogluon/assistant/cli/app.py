#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing.resource_tracker
from pathlib import Path

import typer

from autogluon.assistant.chatting_agent import run_chat_agent
from autogluon.assistant.coding_agent import run_agent
from autogluon.assistant.constants import DEFAULT_CONFIG_PATH


def _noop(*args, **kwargs):
    pass


multiprocessing.resource_tracker.register = _noop
multiprocessing.resource_tracker.unregister = _noop
multiprocessing.resource_tracker.ensure_running = _noop


app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    # === Run parameters ===
    input_data_folder: str | None = typer.Option(None, "-i", "--input", help="Path to data folder"),
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
    llm_provider: str = typer.Option(
        "bedrock",
        "--provider",
        help="LLM provider to use (bedrock, openai, anthropic, sagemaker). Overrides config file.",
    ),
    max_iterations: int = typer.Option(
        5,
        "-n",
        "--max-iterations",
        help="Max iteration count. If the task hasn’t succeeded after this many iterations, it will terminate.",
    ),
    continuous_improvement: bool = typer.Option(
        False,
        "--continuous_improvement",
        help="If enabled, the system will continue optimizing even after finding a valid solution. Instead of stopping at the first successful run, it will keep searching for better solutions until reaching the maximum number of iterations. This allows the system to potentially find higher quality solutions at the cost of additional computation time.",
    ),
    enable_per_iteration_instruction: bool = typer.Option(
        False,
        "--enable-per-iteration-instruction",
        help="If enabled, provide an instruction at the start of each iteration (except the first, which uses the initial instruction). The process suspends until you provide it.",
    ),
    enable_meta_prompting: bool = typer.Option(
        False,
        "-m",
        "--enable-meta-prompting",
        help="If enabled, the system will refine the prompts itself based on user instruction and given data.",
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
    mlzero: a CLI for running the AutoGluon Assistant.

    Use 'mlzero' for code generation and execution (coding agent).
    Use 'mlzero chat' for conversational Q&A without code execution.
    """

    # If a subcommand (like 'chat') is invoked, skip the main function
    if ctx.invoked_subcommand is not None:
        return

    # Check if input_data_folder is required for coding agent
    if input_data_folder is None:
        typer.echo("Error: Missing option '-i' / '--input' for coding agent.", err=True)
        typer.echo("Use 'mlzero -i /path/to/data' for coding agent, or 'mlzero chat' for chatting agent.", err=True)
        raise typer.Exit(1)

    # 3) Invoke the core run_agent function
    # Override config path if provider is specified and config path is default
    provider_config_path = config_path
    if llm_provider in ["bedrock", "openai", "anthropic", "sagemaker"] and config_path == DEFAULT_CONFIG_PATH:
        provider_config_path = Path(DEFAULT_CONFIG_PATH).parent / f"{llm_provider}.yaml"
        if not provider_config_path.exists():
            provider_config_path = DEFAULT_CONFIG_PATH

    run_agent(
        input_data_folder=input_data_folder,
        output_folder=output_dir,
        config_path=str(provider_config_path),
        max_iterations=max_iterations,
        continuous_improvement=continuous_improvement,
        enable_per_iteration_instruction=enable_per_iteration_instruction,
        enable_meta_prompting=enable_meta_prompting,
        initial_user_input=initial_user_input,
        extract_archives_to=extract_archives_to,
        verbosity=verbosity,
    )


@app.command()
def chat(
    # === Run parameters ===
    input_data_folder: str | None = typer.Option(
        None,
        "-i",
        "--input",
        help="Optional path to data folder (provides context for questions)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output directory for session logs (if omitted, auto-generated under chat_sessions/)",
    ),
    config_path: Path | None = typer.Option(
        None,
        "-c",
        "--config",
        help="YAML config file (default: uses chat_config.yaml)",
    ),
    llm_provider: str = typer.Option(
        "bedrock",
        "--provider",
        help="LLM provider to use (bedrock, openai, anthropic, sagemaker). Overrides config file.",
    ),
    session_id: str | None = typer.Option(
        None,
        "-s",
        "--session-id",
        help="Session ID to resume a previous chat session",
    ),
    # === Logging parameters ===
    verbosity: int = typer.Option(
        1,
        "-v",
        "--verbosity",
        help=(
            "-v 0: Only includes error messages\n"
            "-v 1: Contains key essential information\n"
            "-v 2: Includes brief information plus detailed information\n"
            "-v 3: Includes info-level information\n"
            "-v 4: Includes full debug information"
        ),
    ),
):
    """
    Start a chat session with the AutoGluon Assistant.

    This mode allows you to ask questions about data analysis, machine learning,
    and best practices without executing code. If you provide an input data folder,
    the assistant will have context about your data.
    """
    # Override config path if provider is specified
    provider_config_path = config_path
    if config_path is None:
        # Use chat-specific config
        chat_config_path = Path(__file__).parent.parent / "configs" / "chat_config.yaml"
        if chat_config_path.exists():
            provider_config_path = chat_config_path

    if llm_provider in ["bedrock", "openai", "anthropic", "sagemaker"] and provider_config_path:
        provider_specific = Path(provider_config_path).parent / f"{llm_provider}.yaml"
        if provider_specific.exists():
            provider_config_path = provider_specific

    run_chat_agent(
        input_data_folder=input_data_folder,
        output_folder=output_dir,
        config_path=str(provider_config_path) if provider_config_path else None,
        session_id=session_id,
        verbosity=verbosity,
        interactive=True,
    )


if __name__ == "__main__":
    app()
