import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .constants import BRIEF_LEVEL, CONSOLE_HANDLER, DETAIL_LEVEL

# ── Custom log levels ─────────────────────────────
logging.addLevelName(DETAIL_LEVEL, "DETAIL")
logging.addLevelName(BRIEF_LEVEL, "BRIEF")


def detail(self, msg, *args, **kw):
    if self.isEnabledFor(DETAIL_LEVEL):
        # Add stacklevel=2 to skip one frame up the call stack
        kw.setdefault("stacklevel", 2)
        self._log(DETAIL_LEVEL, msg, args, **kw)


def brief(self, msg, *args, **kw):
    if self.isEnabledFor(BRIEF_LEVEL):
        # Add stacklevel=2 to skip one frame up the call stack
        kw.setdefault("stacklevel", 2)
        self._log(BRIEF_LEVEL, msg, args, **kw)


logging.Logger.detail = detail  # type: ignore
logging.Logger.brief = brief  # type: ignore
# ─────────────────────────────────────────


def _configure_logging(console_level: int, output_dir: Path = None) -> None:
    """
    Globally initialize logging with separate levels for console and file

    Args:
        console_level: Logging level for terminal output
        output_dir: If provided, creates both debug and info level file loggers in this directory
    """

    # Set root logger level to DEBUG to allow file handlers to capture all logs
    root_level = logging.DEBUG

    if sys.stdout.isatty():
        console = Console(file=sys.stderr)
        console_handler = RichHandler(console=console, markup=True, rich_tracebacks=True)
        console_handler.setLevel(console_level)
        console_handler.name = CONSOLE_HANDLER
        handlers = [console_handler]
    else:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(console_level)
        stdout_fmt = logging.Formatter("%(levelname)s %(message)s")
        stdout_handler.setFormatter(stdout_fmt)
        handlers = [stdout_handler]

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

        # Detail log file (captures DETAIL and above only)
        detail_log_path = output_dir / "detail_logs.txt"
        detail_handler = logging.FileHandler(str(detail_log_path), mode="w", encoding="utf-8")
        detail_handler.setLevel(DETAIL_LEVEL)
        detail_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        detail_handler.setFormatter(detail_formatter)
        handlers.append(detail_handler)

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

        # Console log file (captures same level as console output)
        console_log_path = output_dir / "logs.txt"
        console_file_handler = logging.FileHandler(str(console_log_path), mode="w", encoding="utf-8")
        console_file_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_file_handler.setFormatter(console_formatter)
        handlers.append(console_file_handler)

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
            level = DETAIL_LEVEL  # Model details
        case _:  # 4+
            level = logging.DEBUG  # Full debug info
    _configure_logging(console_level=level, output_dir=output_dir)


def show_progress_bar():
    root_logger = logging.getLogger()
    console_handler_level = None
    for handler in root_logger.handlers:
        if hasattr(handler, "name") and handler.name == CONSOLE_HANDLER:
            console_handler_level = handler.level

    if console_handler_level is None:
        return False
    return console_handler_level > DETAIL_LEVEL
