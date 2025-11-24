from pathlib import Path

### PATHs

PACKAGE_ROOT = Path(__file__).parent  # /src/autogluon/assistant
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"
LOGO_DAY_PATH = PACKAGE_ROOT / "webui" / "static" / "sidebar_logo_blue.png"
LOGO_NIGHT_PATH = PACKAGE_ROOT / "webui" / "static" / "sidebar_icon.png"
LOGO_PATH = PACKAGE_ROOT / "webui" / "static" / "page_icon.png"

### Default Library
DEFAULT_LIBRARY = "machine learning"

### WebUI

VALID_CODING_LANGUAGES = ["python", "bash"]
DEMO_URL = "https://youtu.be/kejJ3QJPW7E"
DETAIL_LEVEL = 19
BRIEF_LEVEL = 25
CONSOLE_HANDLER = "console_handler"

API_URL = "http://localhost:5000/api"

# Special markers for WebUI communication
WEBUI_INPUT_REQUEST = "###WEBUI_INPUT_REQUEST###"
WEBUI_INPUT_MARKER = "###WEBUI_USER_INPUT###"
WEBUI_OUTPUT_DIR = "###WEBUI_OUTPUT_DIR###"

# Node tree visualization file patterns
NODE_TREE_VISUALIZATION_PATTERN = "node_tree_iteration_*.pdf"
NODE_FULL_VISUALIZATION_FILE = "node_visualization.pdf"

# Success message displayed after task completion
SUCCESS_MESSAGE = """🎉🎉 Task completed successfully! If you found this useful, please consider:
⭐ [Starring our repository](https://github.com/autogluon/autogluon-assistant)
⭐ [Citing our paper](https://arxiv.org/abs/2505.13941)"""

# TODO
IGNORED_MESSAGES = [
    "Too many requests, please wait before trying again",
]

VERBOSITY_MAP = {
    "DETAIL": "3",
    "INFO": "2",
    "BRIEF": "1",
}

# Provider defaults
PROVIDER_DEFAULTS = {
    "bedrock": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "openai": "gpt-4o-2024-08-06",
    "anthropic": "claude-sonnet-4-5-20250929",
}

### MLZero

ENV_FOLDER_NAME = "conda_env"
