VALID_CODING_LANGUAGES = ["python", "bash"]
LOGO_PATH = "static/page_icon.png"
DEMO_URL = "https://youtu.be/kejJ3QJPW7E"
DETAIL_LEVEL = 19
BRIEF_LEVEL = 25
CONSOLE_HANDLER = "console_handler"

API_URL = "http://localhost:5000/api"

# Special markers for WebUI communication
WEBUI_INPUT_REQUEST = "###WEBUI_INPUT_REQUEST###"
WEBUI_INPUT_MARKER = "###WEBUI_USER_INPUT###"
WEBUI_OUTPUT_DIR = "###WEBUI_OUTPUT_DIR###"

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
    "bedrock": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "openai": "gpt-4o-2024-08-06",
    "anthropic": "claude-3-7-sonnet-20250219",
}
