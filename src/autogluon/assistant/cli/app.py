#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing.resource_tracker
from pathlib import Path

import typer

from autogluon.assistant.coding_agent import run_agent
from autogluon.assistant.run_mcts import run_mcts_agent
from autogluon.assistant.constants import DEFAULT_CONFIG_PATH


def _noop(*args, **kwargs):
    pass


multiprocessing.resource_tracker.register = _noop
multiprocessing.resource_tracker.unregister = _noop
multiprocessing.resource_tracker.ensure_running = _noop


app = typer.Typer(add_completion=False)

# Create subcommands
main_app = typer.Typer()
app.add_typer(main_app, name="")

mcts_app = typer.Typer()
app.add_typer(mcts_app, name="mcts", help="Run with Monte Carlo Tree Search for exploring solution space")


@main_app.callback(invoke_without_command=True)
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
    """

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


@mcts_app.callback(invoke_without_command=True)
def mcts_command(
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
    llm_provider: str = typer.Option(
        "bedrock",
        "--provider",
        help="LLM provider to use (bedrock, openai, anthropic, sagemaker). Overrides config file.",
    ),
    max_iterations: int = typer.Option(
        10,
        "-n",
        "--max-iterations",
        help="Max iteration count. Default is higher for MCTS search.",
    ),
    continuous_improvement: bool = typer.Option(
        True,  # Default to True for MCTS
        "--continuous_improvement",
        help="If enabled, the system will continue optimizing even after finding a valid solution.",
    ),
    enable_per_iteration_instruction: bool = typer.Option(
        False,
        "--enable-per-iteration-instruction",
        help="If enabled, provide an instruction at the start of each iteration.",
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
    # === MCTS parameters ===
    exploration_constant: float = typer.Option(
        1.414,
        "--exploration-constant",
        help="MCTS exploration constant that balances exploration vs exploitation.",
    ),
    max_debug_depth: int = typer.Option(
        5,
        "--max-debug-depth",
        help="Maximum depth of debugging attempts for a single node.",
    ),
    max_evolve_attempts: int = typer.Option(
        3,
        "--max-evolve-attempts",
        help="Maximum number of evolution attempts for a single node.",
    ),
    metric_improvement_threshold: float = typer.Option(
        0.01,
        "--metric-improvement-threshold",
        help="Minimum improvement in validation score required to consider an evolution successful.",
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
    Run the AutoGluon Assistant with MCTS-based search strategy.
    """
    # Override config path if provider is specified and config path is default
    provider_config_path = config_path
    if llm_provider in ["bedrock", "openai", "anthropic", "sagemaker"] and config_path == DEFAULT_CONFIG_PATH:
        provider_config_path = Path(DEFAULT_CONFIG_PATH).parent / f"{llm_provider}.yaml"
        if not provider_config_path.exists():
            provider_config_path = DEFAULT_CONFIG_PATH
    
    # Add MCTS specific parameters to config (will be passed to NodeManager)
    import os
    from omegaconf import OmegaConf
    config = OmegaConf.load(provider_config_path)
    config.exploration_constant = exploration_constant
    config.max_debug_depth = max_debug_depth
    config.max_evolve_attempts = max_evolve_attempts
    config.max_debug_attempts = max_debug_depth  # Use the same value as max_debug_depth
    config.metric_improvement_threshold = metric_improvement_threshold
    
    # Save modified config
    temp_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_mcts_config.yaml")
    with open(temp_config_path, "w") as f:
        OmegaConf.save(config, f)
    
    run_mcts_agent(
        input_data_folder=input_data_folder,
        output_folder=output_dir,
        config_path=temp_config_path,
        max_iterations=max_iterations,
        continuous_improvement=continuous_improvement,
        enable_per_iteration_instruction=enable_per_iteration_instruction,
        enable_meta_prompting=enable_meta_prompting,
        initial_user_input=initial_user_input,
        extract_archives_to=extract_archives_to,
        verbosity=verbosity,
    )
    
    # Clean up temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)


if __name__ == "__main__":
    app()
