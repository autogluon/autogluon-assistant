import datetime
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from omegaconf import OmegaConf
from rich import print as rprint
from typing_extensions import Annotated

from autogluon.assistant.assistant import TabularPredictionAssistant
from autogluon.assistant.constants import DEFAULT_QUALITY, NO_ID_COLUMN_IDENTIFIED, PRESETS
from autogluon.assistant.task import TabularPredictionTask
from autogluon.assistant.utils import load_config

try:
    __version__ = version("autogluon.assistant")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"


__all__ = ["TabularPredictionAssistant", "TabularPredictionTask", "run_assistant", "__version__"]

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)


def get_task(path: Path) -> TabularPredictionTask:
    """Get a task from a path."""

    return TabularPredictionTask.from_path(path)


def make_prediction_outputs(task: TabularPredictionTask, predictions: pd.DataFrame) -> pd.DataFrame:
    # Convert predictions to DataFrame if it's a Series
    if isinstance(predictions, pd.Series):
        outputs = predictions.to_frame()
    else:
        outputs = predictions.copy()

    # Ensure we only keep required output columns from predictions
    common_cols = [col for col in task.output_columns if col in outputs.columns]
    outputs = outputs[common_cols]

    # Handle specific test ID column if provided and detected
    if task.test_id_column is not None and task.test_id_column != NO_ID_COLUMN_IDENTIFIED:
        test_ids = task.test_data[task.test_id_column]
        output_ids = task.sample_submission_data[task.output_id_column]

        if not test_ids.equals(output_ids):
            print("Warning: Test IDs and output IDs do not match!")

        # Ensure test ID column is included
        if task.test_id_column not in outputs.columns:
            outputs = pd.concat([task.test_data[task.test_id_column], outputs], axis="columns")

    # Handle undetected ID columns
    missing_columns = [col for col in task.output_columns if col not in outputs.columns]
    if missing_columns:
        print(
            f"Warning: The following columns are not in predictions and will be treated as ID columns: {missing_columns}"
        )

        for col in missing_columns:
            if col in task.test_data.columns:
                # Copy from test data if available
                outputs[col] = task.test_data[col]
                print(f"Warning: Copied from test data for column '{col}'")
            else:
                # Generate unique integer values
                outputs[col] = range(len(outputs))
                print(f"Warning: Generated unique integer values for column '{col}' as it was not found in test data")

    # Ensure columns are in the correct order
    outputs = outputs[task.output_columns]

    return outputs


def get_ui_path() -> str:
    """Get the absolute path to the UI directory using package resources"""
    try:
        from importlib.resources import files

        package_paths = list(files("autogluon.assistant.ui").iterdir())
        ui_path = next(str(p.parent) for p in package_paths if "app.py" in str(p))
        return ui_path
    except Exception:
        # Fallback for development environment
        return str(Path(__file__).parent / "ui")


def launch_ui(port: int = typer.Option(8501, help="Port to run the UI on")):
    """Launch the AutoGluon Assistant Web UI"""
    try:
        import streamlit
    except Exception as e:
        rprint(f"[red]Error UI not installed: {str(e)}[/red]")
        sys.exit(1)

    ui_dir = get_ui_path()
    app_path = os.path.join(ui_dir, "app.py")

    if not os.path.exists(app_path):
        rprint(f"[red]Error: UI file not found at {app_path}[/red]")
        sys.exit(1)

    # Change working directory to UI directory before running streamlit
    original_dir = os.getcwd()
    os.chdir(ui_dir)

    cmd = ["streamlit", "run", "app.py", "--server.port", str(port)]  # Use relative path since we changed directory

    try:
        rprint(f"[green]Launching AutoGluon Assistant UI on port {port}...[/green]")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        rprint("\n[yellow]Shutting down UI server...[/yellow]")
    except Exception as e:
        rprint(f"[red]Error launching UI: {str(e)}[/red]")
    finally:
        # Change back to original directory
        os.chdir(original_dir)
        sys.exit(1)


@dataclass
class TimingContext:
    start_time: float
    total_time_limit: float

    @property
    def time_elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def time_remaining(self) -> float:
        return self.total_time_limit - self.time_elapsed


@contextmanager
def time_block(description: str, timer: TimingContext):
    """Context manager for timing code blocks and logging the duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logging.info(
            f"It took {duration:.2f} seconds {description}. "
            f"Time remaining: {timer.time_remaining:.2f}/{timer.total_time_limit:.2f}"
        )


def run_assistant(
    task_path: Annotated[str, typer.Argument(help="Directory where task files are included")],
    presets: Annotated[
        Optional[str],
        typer.Option("--presets", "-p", help="Presets"),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option("--config-path", "-c", help="Path to the configuration file (config.yaml)"),
    ] = None,
    config_overrides: Annotated[
        Optional[List[str]],
        typer.Option(
            "--config_overrides",
            "-o",
            help="Override config values. Format: key=value or key.nested=value. Can be used multiple times.",
        ),
    ] = None,
    output_filename: Annotated[Optional[str], typer.Option(help="Output File")] = "",
) -> str:
    start_time = time.time()

    logging.info("Starting AutoGluon-Assistant")

    if presets is None or presets not in PRESETS:
        logging.info(f"Presets is not provided or invalid: {presets}")
        presets = DEFAULT_QUALITY
        logging.info(f"Using default presets: {presets}")
    logging.info(f"Presets: {presets}")

    # Load config with all overrides
    try:
        config = load_config(presets, config_path, config_overrides)
        logging.info("Successfully loaded config")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise

    timer = TimingContext(start_time=start_time, total_time_limit=config.time_limit)
    with time_block("initializing components", timer):
        rprint("🤖 [bold red] Welcome to AutoGluon-Assistant [/bold red]")

        rprint("Will use task config:")
        rprint(OmegaConf.to_container(config))

        task_path = Path(task_path).resolve()
        assert task_path.is_dir(), "Task path does not exist, please provide a valid directory."
        rprint(f"Task path: {task_path}")

        task = TabularPredictionTask.from_path(task_path)

        rprint("[green]Task loaded![/green]")
        rprint(task)

        assistant = TabularPredictionAssistant(config)

    with time_block("preprocessing task", timer):
        task = assistant.preprocess_task(task)

    with time_block("training model", timer):
        rprint("Model training starts...")

        assistant.fit_predictor(task, time_limit=timer.time_remaining)

        rprint("[green]Model training complete![/green]")

    with time_block("making predictions", timer):
        rprint("Prediction starts...")

        predictions = assistant.predict(task)

        if not output_filename:
            output_filename = f"aga-output-{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(output_filename, "w") as fp:
            make_prediction_outputs(task, predictions).to_csv(fp, index=False)

        rprint(f"[green]Prediction complete! Outputs written to {output_filename}[/green]")

    if config.save_artifacts.enabled:
        # Determine the artifacts_dir with or without timestamp
        artifacts_dir_name = f"{task.metadata['name']}_artifacts"
        if config.save_artifacts.append_timestamp:
            current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir_name = f"{task.metadata['name']}_artifacts_{current_timestamp}"

        full_save_path = f"{config.save_artifacts.path.rstrip('/')}/{artifacts_dir_name}"

        task.save_artifacts(
            full_save_path, assistant.predictor, task.train_data, task.test_data, task.sample_submission_data
        )

        rprint(f"[green]Artifacts including transformed datasets and trained model saved at {full_save_path}[/green]")

    return output_filename


def main():
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command("run")(run_assistant)  # CLI Mode
    app.command("ui")(launch_ui)  # UI Mode
    app()


if __name__ == "__main__":
    main()
