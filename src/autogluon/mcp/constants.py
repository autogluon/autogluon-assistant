"""
Constants for MCP (Model Control Protocol) module
"""

# File size limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Directory patterns
UPLOAD_DIR_PATTERN = "upload_{timestamp}_{uuid}"
CONFIG_DIR_PATTERN = "config_{timestamp}_{uuid}"
OUTPUT_DIR_PATTERN = "mlzero-{datetime}-{uuid}"

# API endpoint
API_URL = "http://localhost:5000/api"

# Verbosity mapping
VERBOSITY_MAP = {"DETAIL": "3", "INFO": "2", "BRIEF": "1", "ERROR": "0", "DEBUG": "4"}

TASK_STATES = {
    "IDLE": "idle",
    "QUEUED": "queued",
    "RUNNING": "running",
    "WAITING_INPUT": "waiting_for_input",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
}

# Log levels (MCP specific)
LOG_LEVELS = ["ERROR", "BRIEF", "INFO", "DETAIL", "DEBUG"]

# Allowed file extensions for config
CONFIG_EXTENSIONS = [".yaml", ".yml"]

# Default values
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_VERBOSITY = "INFO"
DEFAULT_PROVIDER = "bedrock"

DEFAULT_AWS_REGION = "us-west-2"

DEFAULT_PIPELINE_PORT = 8005

MCP_BEDROCK_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
