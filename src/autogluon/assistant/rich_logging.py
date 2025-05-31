import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .constants import BRIEF_LEVEL, MODEL_INFO_LEVEL

# ── Custom log levels ─────────────────────────────
logging.addLevelName(MODEL_INFO_LEVEL, "MODEL_INFO")
logging.addLevelName(BRIEF_LEVEL, "BRIEF")


def model_info(self, msg, *args, **kw):
    if self.isEnabledFor(MODEL_INFO_LEVEL):
        self._log(MODEL_INFO_LEVEL, msg, args, **kw)


def brief(self, msg, *args, **kw):
    if self.isEnabledFor(BRIEF_LEVEL):
        self._log(BRIEF_LEVEL, msg, args, **kw)


logging.Logger.model_info = model_info  # type: ignore
logging.Logger.brief = brief  # type: ignore
# ─────────────────────────────────────────


def _configure_logging(console_level: int, output_dir: Path = None) -> None:
    """
    Globally initialize logging with separate levels for console and file

    Args:
        console_level: Logging level for terminal output
        output_dir: If provided, creates both debug and info level file loggers in this directory
    """
    console = Console()

    # Set root logger level to DEBUG to allow file handlers to capture all logs
    root_level = logging.DEBUG

    # Create RichHandler with the specified console level
    console_handler = RichHandler(console=console, markup=True, rich_tracebacks=True)
    console_handler.setLevel(console_level)

    handlers = [console_handler]

    # Add file handlers if output_dir is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Debug log file (captures everything DEBUG and above)
        debug_log_path = output_dir / "debugging_logs.txt"
        debug_handler = logging.FileHandler(str(debug_log_path), mode="w", encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        debug_handler.setFormatter(debug_formatter)
        handlers.append(debug_handler)

        # Info log file (captures INFO and above only)
        info_log_path = output_dir / "info_logs.txt"
        info_handler = logging.FileHandler(str(info_log_path), mode="w", encoding="utf-8")
        info_handler.setLevel(logging.INFO)
        info_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        info_handler.setFormatter(info_formatter)
        handlers.append(info_handler)

    logging.basicConfig(
        level=root_level,
        format="%(message)s",
        handlers=handlers,
        force=True,  # Ensure override
    )


def configure_logging(verbosity: int, output_dir: Path = None) -> None:
    match verbosity:
        case 0:
            level = logging.ERROR  # Only errors
        case 1:
            level = BRIEF_LEVEL  # Brief summaries
        case 2:
            level = logging.INFO  # Standard info
        case 3:
            level = MODEL_INFO_LEVEL  # Model details
        case _:  # 4+
            level = logging.DEBUG  # Full debug info
    _configure_logging(console_level=level, output_dir=output_dir)
