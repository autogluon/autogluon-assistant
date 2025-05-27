import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from autogluon.assistant.constants import MODEL_INFO_LEVEL, BRIEF_LEVEL

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
    在 output_dir 下创建一个 logs.txt 文件，记录 DEBUG 及以上所有日志。
    """
    log_path = output_dir / "logs.txt"
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建 handler
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # 你可以自定义格式，这里我们简单用时间+级别+logger name+消息
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)

    # 把 handler 挂到 root logger
    logging.getLogger().addHandler(fh)
