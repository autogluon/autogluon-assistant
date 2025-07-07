"""
Constants for MCP (Model Control Protocol) module
"""

from pathlib import Path

# ==================== File Handling ====================
# File size limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB total

# Directory patterns
UPLOAD_DIR_PATTERN = "upload_{timestamp}_{uuid}"
CONFIG_DIR_PATTERN = "config_{timestamp}_{uuid}"
OUTPUT_DIR_PATTERN = "mlzero-{datetime}-{uuid}"

# ==================== Task Management ====================
# Task states
TASK_STATES = {
    "IDLE": "idle",
    "QUEUED": "queued",
    "RUNNING": "running",
    "WAITING_INPUT": "waiting_for_input",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled"
}

# Log levels (MCP specific)
LOG_LEVELS = ["ERROR", "BRIEF", "INFO", "DETAIL", "DEBUG"]

# Allowed file extensions for config
CONFIG_EXTENSIONS = [".yaml", ".yml"]

# Default values
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_VERBOSITY = "INFO"
DEFAULT_PROVIDER = "bedrock"
