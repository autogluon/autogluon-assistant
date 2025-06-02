import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from autogluon.assistant.constants import BRIEF_LEVEL, MODEL_INFO_LEVEL

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


def configure_logging(level: int) -> None:
    """
    Globally initialize logging (overrides any basicConfig set by other modules)
    """
    console = Console()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
        force=True,  # Ensure override
    )


def attach_file_logger(output_dir: Path):
    """
    Create a logs.txt file under output_dir to record all logs at DEBUG level and above.
    """
    log_path = output_dir / "debugging_logs.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)

    logging.getLogger().addHandler(fh)
