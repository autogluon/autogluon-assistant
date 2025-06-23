"""
Constants for MCP server
"""

# API endpoint
API_URL = "http://localhost:5000/api"

# Verbosity mapping
VERBOSITY_MAP = {
    "DETAIL": "3",
    "INFO": "2",
    "BRIEF": "1",
    "ERROR": "0",
    "DEBUG": "4"
}

# Provider defaults
PROVIDER_DEFAULTS = {
    "bedrock": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "openai": "gpt-4o-2024-08-06",
    "anthropic": "claude-3-7-sonnet-20250219",
}

# File size limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB total

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

# Log levels
LOG_LEVELS = ["ERROR", "BRIEF", "INFO", "DETAIL", "DEBUG"]

# Special log markers (from webui)
WEBUI_INPUT_REQUEST = "###WEBUI_INPUT_REQUEST###"
WEBUI_INPUT_MARKER = "###WEBUI_USER_INPUT###"
WEBUI_OUTPUT_DIR = "###WEBUI_OUTPUT_DIR###"

# Directory patterns
UPLOAD_DIR_PATTERN = "upload_{timestamp}_{uuid}"
CONFIG_DIR_PATTERN = "config_{timestamp}_{uuid}"
OUTPUT_DIR_PATTERN = "mlzero-{datetime}-{uuid}"

# Allowed file extensions for config
CONFIG_EXTENSIONS = [".yaml", ".yml"]

# Default values
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_VERBOSITY = "INFO"
DEFAULT_PROVIDER = "bedrock"
