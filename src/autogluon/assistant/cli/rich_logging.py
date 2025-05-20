import logging
from rich.logging import RichHandler
from rich.console import Console

# ── Custom log levels ─────────────────────────────
MODEL_INFO_LEVEL = 19
BRIEF_LEVEL       = 25

logging.addLevelName(MODEL_INFO_LEVEL, "MODEL_INFO")
logging.addLevelName(BRIEF_LEVEL,       "BRIEF")

def model_info(self, msg, *args, **kw):
    if self.isEnabledFor(MODEL_INFO_LEVEL):
        self._log(MODEL_INFO_LEVEL, msg, args, **kw)

def brief(self, msg, *args, **kw):
    if self.isEnabledFor(BRIEF_LEVEL):
        self._log(BRIEF_LEVEL, msg, args, **kw)

logging.Logger.model_info = model_info    # type: ignore
logging.Logger.brief      = brief         # type: ignore
# ─────────────────────────────────────────

def configure_logging(level: int) -> None:
    """
    Globally initialize logging (overrides any basicConfig set by other modules)
    """
    console = Console()
    logging.basicConfig(
        level   = level,
        format  = "%(message)s",
        handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
        force   = True,        # Ensure override
    )