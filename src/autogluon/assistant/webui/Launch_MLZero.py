# src/autogluon/assistant/webui/pages/Launch_MLZero.py

import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests
import streamlit as st
import streamlit.components.v1 as components
import yaml
from botocore.exceptions import ClientError, NoCredentialsError
from streamlit_theme import st_theme

from autogluon.assistant.constants import API_URL, LOGO_PATH, PROVIDER_DEFAULTS, SUCCESS_MESSAGE, VERBOSITY_MAP

# Import prompt classes for default templates
from autogluon.assistant.prompts import (
    DescriptionFileRetrieverPrompt,
    ErrorAnalyzerPrompt,
    ExecuterPrompt,
    PythonCoderPrompt,
    PythonReaderPrompt,
    RerankerPrompt,
    RetrieverPrompt,
    TaskDescriptorPrompt,
    ToolSelectorPrompt,
)
from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.webui.log_processor import messages, process_logs, render_task_logs

PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "assistant" / "configs" / "default.yaml"
logo_day_path = PACKAGE_ROOT / "assistant" / "webui" / "static" / "sidebar_logo_blue.png"
logo_night_path = PACKAGE_ROOT / "assistant" / "webui" / "static" / "sidebar_icon.png"

# Agent list for template setter
AGENTS_LIST = [
    "coder",
    "executer",
    "reader",
    "error_analyzer",
    "retriever",
    "reranker",
    "description_file_retriever",
    "task_descriptor",
    "tool_selector",
]

# Agent to Prompt class mapping
AGENT_PROMPT_MAPPING = {
    "coder": PythonCoderPrompt,  # Note: This is for Python coder
    "executer": ExecuterPrompt,
    "reader": PythonReaderPrompt,
    "error_analyzer": ErrorAnalyzerPrompt,
    "retriever": RetrieverPrompt,
    "reranker": RerankerPrompt,
    "description_file_retriever": DescriptionFileRetrieverPrompt,
    "task_descriptor": TaskDescriptorPrompt,
    "tool_selector": ToolSelectorPrompt,
}


# ==================== Data Classes ====================
@dataclass
class Message:
    """Chat message"""

    role: str
    type: str
    content: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text(cls, text: str, role: str = "assistant") -> "Message":
        return cls(role=role, type="text", content={"text": text})

    @classmethod
    def user_summary(cls, summary: str, input_dir: Optional[str] = None) -> "Message":
        content = {"summary": summary}
        if input_dir:
            content["input_dir"] = input_dir
        return cls(role="user", type="user_summary", content=content)

    @classmethod
    def command(cls, command: str) -> "Message":
        return cls(role="assistant", type="command", content={"command": command})

    @classmethod
    def task_log(
        cls,
        run_id: str,
        phase_states: Dict,
        max_iter: int,
        output_dir: Optional[str] = None,
        input_dir: Optional[str] = None,
    ) -> "Message":
        content = {
            "run_id": run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase_states": phase_states,
            "max_iter": max_iter,
        }
        if output_dir:
            content["output_dir"] = output_dir
        if input_dir:
            content["input_dir"] = input_dir
        return cls(role="assistant", type="task_log", content=content)

    @classmethod
    def task_results(cls, run_id: str, output_dir: str) -> "Message":
        return cls(role="assistant", type="task_results", content={"run_id": run_id, "output_dir": output_dir})

    @classmethod
    def queue_status(cls, task_id: str, position: int) -> "Message":
        return cls(role="assistant", type="queue_status", content={"task_id": task_id, "position": position})

    @classmethod
    def debug_config(cls, config_path: str, config_content: str) -> "Message":
        """Debug message for showing final config content"""
        return cls(role="assistant", type="debug_config", content={"path": config_path, "content": config_content})


@dataclass
class TaskConfig:
    """Task configuration"""

    uploaded_config: Any
    max_iter: int
    control: bool
    log_verbosity: str
    provider: str
    model: str
    credentials: Optional[Dict[str, str]] = None


# ==================== Helper Functions ====================
def get_default_template(agent_name: str) -> str:
    """Get default template for an agent"""
    prompt_class = AGENT_PROMPT_MAPPING.get(agent_name)
    if not prompt_class:
        return ""

    try:
        # Create mock objects for initialization
        mock_config = type(
            "MockConfig",
            (),
            {
                "template": None,
                "max_stdout_length": 8192,
                "max_stderr_length": 2048,
                "max_description_files_length_to_show": 1024,
                "max_description_files_length_for_summarization": 16384,
            },
        )()
        mock_manager = type("MockManager", (), {})()

        # All agents use the same initialization pattern based on BasePrompt
        instance = prompt_class(llm_config=mock_config, manager=mock_manager, template=None)

        return instance.default_template()
    except Exception as e:
        # Log the error for debugging
        print(f"Error getting default template for {agent_name}: {str(e)}")
        # Return a fallback message instead of error
        return f"# Default template for {agent_name}\n# Unable to load default template"


# ==================== Credentials Validators ====================
class CredentialsValidator:
    """Base credentials validator"""

    @staticmethod
    def parse_credentials(credentials_text: str, required_vars: List[str]) -> Optional[Dict[str, str]]:
        """Parse credentials text for environment variables"""
        if not credentials_text:
            return None

        credentials = {}
        lines = credentials_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]  # Remove 'export '

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")  # Remove quotes if present

                if key in required_vars:
                    credentials[key] = value

        # Check if all required fields are present
        for field_name in required_vars:
            if field_name not in credentials or not credentials[field_name]:
                return None

        return credentials


class BedrockCredentialsValidator(CredentialsValidator):
    """AWS/Bedrock credentials validator"""

    @staticmethod
    def parse_credentials(credentials_text: str) -> Optional[Dict[str, str]]:
        """Parse AWS credentials text"""
        if not credentials_text:
            return None

        credentials = {}
        lines = credentials_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]  # Remove 'export '

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")  # Remove quotes if present

                # Collect all AWS-related environment variables
                if key in [
                    "ISENGARD_PRODUCTION_ACCOUNT",
                    "AWS_DEFAULT_REGION",
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_SESSION_TOKEN",
                ]:
                    credentials[key] = value

        # Check if minimum required fields are present
        # For AWS, we need at least ACCESS_KEY_ID and SECRET_ACCESS_KEY
        if "AWS_ACCESS_KEY_ID" not in credentials or not credentials["AWS_ACCESS_KEY_ID"]:
            return None
        if "AWS_SECRET_ACCESS_KEY" not in credentials or not credentials["AWS_SECRET_ACCESS_KEY"]:
            return None

        # Set default region if not provided
        if "AWS_DEFAULT_REGION" not in credentials:
            credentials["AWS_DEFAULT_REGION"] = "us-west-2"

        return credentials

    @staticmethod
    def validate_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Validate AWS credentials"""
        try:
            # Create a session with the provided credentials
            session_params = {
                "aws_access_key_id": credentials["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": credentials["AWS_SECRET_ACCESS_KEY"],
                "region_name": credentials.get("AWS_DEFAULT_REGION", "us-west-2"),
            }

            # Add session token if present (for temporary credentials)
            if "AWS_SESSION_TOKEN" in credentials:
                session_params["aws_session_token"] = credentials["AWS_SESSION_TOKEN"]

            session = boto3.Session(**session_params)

            # Try to make a simple API call to verify credentials
            sts = session.client("sts")
            caller_identity = sts.get_caller_identity()

            return True, f"Credentials valid (Account: {caller_identity['Account']})"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ExpiredToken":
                return False, "Credentials expired"
            elif error_code == "InvalidClientTokenId":
                return False, "Invalid Access Key ID"
            elif error_code == "SignatureDoesNotMatch":
                return False, "Invalid Secret Access Key"
            else:
                return False, f"Validation failed: {e.response['Error']['Message']}"
        except NoCredentialsError:
            return False, "Invalid credentials format"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"


class OpenAICredentialsValidator(CredentialsValidator):
    """OpenAI credentials validator"""

    @staticmethod
    def parse_credentials(credentials_text: str) -> Optional[Dict[str, str]]:
        """Parse OpenAI credentials text"""
        required_vars = ["OPENAI_API_KEY"]
        return CredentialsValidator.parse_credentials(credentials_text, required_vars)

    @staticmethod
    def validate_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Validate OpenAI credentials"""
        try:
            import openai
        except ImportError:
            return False, "OpenAI library not installed. Please install with: pip install openai"

        try:
            # Set the API key
            _client = openai.OpenAI(api_key=credentials["OPENAI_API_KEY"])

            # Try to list models to verify the key
            _models = _client.models.list()
            # If we can list models, the key is valid
            return True, "OpenAI API key is valid"

        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                return False, "Invalid OpenAI API key"
            elif "429" in str(e):
                return False, "Rate limit exceeded (key is valid but rate limited)"
            else:
                return False, f"Validation failed: {str(e)}"


class AnthropicCredentialsValidator(CredentialsValidator):
    """Anthropic credentials validator"""

    @staticmethod
    def parse_credentials(credentials_text: str) -> Optional[Dict[str, str]]:
        """Parse Anthropic credentials text"""
        required_vars = ["ANTHROPIC_API_KEY"]
        return CredentialsValidator.parse_credentials(credentials_text, required_vars)

    @staticmethod
    def validate_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Validate Anthropic credentials"""
        try:
            import anthropic
        except ImportError:
            return False, "Anthropic library not installed. Please install with: pip install anthropic"

        try:
            # Create client with the API key
            _client = anthropic.Anthropic(api_key=credentials["ANTHROPIC_API_KEY"])

            # Try a minimal API call to verify the key
            _response = _client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, "Anthropic API key is valid"

        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                return False, "Invalid Anthropic API key"
            elif "429" in str(e):
                return False, "Rate limit exceeded (key is valid but rate limited)"
            else:
                return False, f"Validation failed: {str(e)}"


# ==================== Config File Handler ====================
class ConfigFileHandler:
    """Handle config file operations"""

    @staticmethod
    def load_default_config() -> Dict:
        """Load the default config file"""
        try:
            with open(DEFAULT_CONFIG_PATH, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Failed to load default config: {str(e)}")
            return {}

    @staticmethod
    def extract_provider_model(config_dict: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract provider and model from config dictionary"""
        try:
            llm_config = config_dict.get("llm", {})
            provider = llm_config.get("provider")
            model = llm_config.get("model")
            return provider, model
        except Exception:
            return None, None

    @staticmethod
    def save_modified_config(base_config: Dict, overrides: Dict, save_path: Path) -> str:
        """Save config with overrides applied"""
        try:
            # Deep copy to avoid modifying the original
            import copy

            config = copy.deepcopy(base_config)

            # Apply provider and model overrides
            if overrides.get("provider") and overrides.get("model"):
                # Update the llm section
                if "llm" not in config:
                    config["llm"] = {}
                config["llm"]["provider"] = overrides["provider"]
                config["llm"]["model"] = overrides["model"]

                # Update all agent sections that have provider/model
                agent_sections = AGENTS_LIST

                for agent in agent_sections:
                    if agent in config and isinstance(config[agent], dict):
                        config[agent]["provider"] = overrides["provider"]
                        config[agent]["model"] = overrides["model"]

            # Apply template overrides
            templates = overrides.get("templates", {})
            for agent, template_config in templates.items():
                if template_config["mode"] != "use_default":
                    if agent not in config:
                        config[agent] = {}
                    config[agent]["template"] = template_config["value"]

            # Save to file
            with open(save_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            return str(save_path)
        except Exception as e:
            st.error(f"Failed to save config: {str(e)}")
            return str(DEFAULT_CONFIG_PATH)


# ==================== Session State ====================
class SessionState:
    """Session state manager"""

    @staticmethod
    def init():
        """Initialize session state"""
        defaults = {
            "user_session_id": uuid.uuid4().hex,
            "messages": [
                Message.text(
                    "Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start."
                )
            ],
            "data_src": None,
            "task_running": False,
            "run_id": None,
            "task_id": None,  # New: track task_id separately
            "queue_position": None,  # New: track queue position
            "current_task_logs": [],
            "running_config": None,
            "current_input_dir": None,
            "waiting_for_input": False,
            "input_prompt": None,
            "current_iteration": 0,
            "current_output_dir": None,
            "prev_iter_placeholder": None,  # Placeholder object
            # Centralized config overrides
            "config_overrides": {
                "provider": None,
                "model": None,
                "templates": {agent: {"mode": "use_default", "value": None} for agent in AGENTS_LIST},
            },
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def update_provider_model_from_config(provider: str, model: str):
        """Update provider and model from config file"""
        st.session_state.config_overrides["provider"] = provider
        st.session_state.config_overrides["model"] = model

    @staticmethod
    def update_templates_from_config(config_dict: Dict):
        """Update templates from uploaded config file (only for use_default items)"""
        for agent in AGENTS_LIST:
            if agent in config_dict and config_dict[agent].get("template"):
                # Only update if current mode is use_default
                if st.session_state.config_overrides["templates"][agent]["mode"] == "use_default":
                    template_value = config_dict[agent]["template"]
                    if isinstance(template_value, str):
                        if template_value.endswith(".txt"):
                            st.session_state.config_overrides["templates"][agent] = {
                                "mode": "path_specify",
                                "value": template_value,
                            }
                        else:
                            st.session_state.config_overrides["templates"][agent] = {
                                "mode": "text_edit",
                                "value": template_value,
                            }

    @staticmethod
    def start_task(task_id: str, config: TaskConfig, input_dir: str, position: int):
        """Start new task (or queue it)"""
        st.session_state.task_running = True
        st.session_state.task_id = task_id
        st.session_state.queue_position = position
        st.session_state.run_id = None  # Will be set when task actually starts
        st.session_state.current_task_logs = []
        st.session_state.running_config = config
        st.session_state.current_input_dir = input_dir
        st.session_state.waiting_for_input = False
        st.session_state.input_prompt = None
        st.session_state.current_iteration = 0
        st.session_state.current_output_dir = None

        # Clean up old log processors
        SessionState._cleanup_processors()

    @staticmethod
    def finish_task():
        """Finish task"""
        st.session_state.task_running = False
        st.session_state.task_id = None
        st.session_state.queue_position = None
        st.session_state.running_config = None
        st.session_state.current_task_logs = []
        st.session_state.current_input_dir = None
        st.session_state.waiting_for_input = False
        st.session_state.input_prompt = None
        st.session_state.current_iteration = 0
        st.session_state.current_output_dir = None

        # Clean up current task's processor
        if st.session_state.run_id:
            processor_key = f"log_processor_{st.session_state.run_id}"
            if processor_key in st.session_state:
                del st.session_state[processor_key]

    @staticmethod
    def set_waiting_for_input(waiting: bool, prompt: Optional[str] = None, iteration: Optional[int] = None):
        """Set waiting for input state"""
        st.session_state.waiting_for_input = waiting
        st.session_state.input_prompt = prompt
        if iteration is not None:
            st.session_state.current_iteration = iteration

    @staticmethod
    def add_message(message: Message):
        """Add message"""
        st.session_state.messages.append(message)

    @staticmethod
    def delete_task_from_history(run_id: str):
        """Delete task-related messages from history"""
        # First, find the task_log message to get its index
        task_log_index = None
        for i, msg in enumerate(st.session_state.messages):
            if msg.type == "task_log" and msg.content.get("run_id") == run_id:
                task_log_index = i
                break

        if task_log_index is None:
            return

        # Find the associated messages (user_summary, command, queue_status, etc.)
        start_index = task_log_index

        # Look backwards for related messages
        i = task_log_index - 1
        while i >= 0:
            msg = st.session_state.messages[i]

            # Check for various message types
            if msg.type in ["command", "user_summary", "queue_status"]:
                start_index = i
                i -= 1
                continue

            # Check for cancel-related messages
            elif msg.type == "text" and msg.role == "user" and msg.content.get("text", "").strip().lower() == "cancel":
                start_index = i
                i -= 1
                continue

            # Check for cancel confirmation message
            elif (
                msg.type == "text" and msg.role == "assistant" and "has been cancelled" in msg.content.get("text", "")
            ):
                start_index = i
                i -= 1
                continue

            # If we hit any other message type, stop looking
            else:
                break

        # Find the end index (task_results should be right after task_log)
        end_index = task_log_index
        if (
            task_log_index + 1 < len(st.session_state.messages)
            and st.session_state.messages[task_log_index + 1].type == "task_results"
            and st.session_state.messages[task_log_index + 1].content.get("run_id") == run_id
        ):
            end_index = task_log_index + 1

        # Create new message list without the task-related messages
        new_messages = []
        for i, msg in enumerate(st.session_state.messages):
            if i < start_index or i > end_index:
                new_messages.append(msg)

        st.session_state.messages = new_messages

    @staticmethod
    def _cleanup_processors():
        """Clean up old log processors"""
        keys_to_delete = [k for k in st.session_state if k.startswith("log_processor_")]
        for key in keys_to_delete:
            del st.session_state[key]


# ==================== Backend API ====================
class BackendAPI:
    """Backend API communication"""

    @staticmethod
    def start_task(data_src: str, config_path: str, user_prompt: str, config: TaskConfig) -> Tuple[str, int]:
        """Start task - returns (task_id, position)"""
        payload = {
            "data_src": data_src,
            "config_path": config_path,
            "max_iter": config.max_iter,
            "init_prompt": user_prompt or None,
            "control": config.control,
            "verbosity": VERBOSITY_MAP[config.log_verbosity],
        }

        # Add credentials based on provider
        if config.provider == "bedrock" and config.credentials:
            # For bedrock, use the AWS credentials structure
            payload["aws_credentials"] = config.credentials
        elif config.provider in ["openai", "anthropic"] and config.credentials:
            # For other providers, pass the credentials as environment variables
            payload["aws_credentials"] = config.credentials  # Reusing the field for all credentials

        response = requests.post(f"{API_URL}/run", json=payload)
        result = response.json()
        return result["task_id"], result["position"]

    @staticmethod
    def fetch_logs(run_id: str) -> List[Dict]:
        """Get logs"""
        response = requests.get(f"{API_URL}/logs", params={"run_id": run_id})
        return response.json().get("lines", [])

    @staticmethod
    def check_status(run_id: str) -> Dict:
        """Check task status"""
        response = requests.get(f"{API_URL}/status", params={"run_id": run_id})
        return response.json()

    @staticmethod
    def check_queue_status(task_id: str) -> Dict:
        """Check queue status for a task"""
        response = requests.get(f"{API_URL}/queue/status/{task_id}")
        return response.json()

    @staticmethod
    def send_user_input(run_id: str, user_input: str) -> bool:
        """Send user input to backend"""
        try:
            response = requests.post(f"{API_URL}/input", json={"run_id": run_id, "input": user_input})
            return response.json().get("success", False)
        except Exception as e:
            st.error(f"Error sending input: {str(e)}")
            return False

    @staticmethod
    def cancel_task(run_id: str = None, task_id: str = None) -> bool:
        """Cancel task (either running or queued)"""
        try:
            payload = {}
            if run_id:
                payload["run_id"] = run_id
            if task_id:
                payload["task_id"] = task_id

            response = requests.post(f"{API_URL}/cancel", json=payload)
            return response.json().get("cancelled", False)
        except:
            return False


# ==================== UI Components ====================
class UI:
    """UI components"""

    @staticmethod
    def render_template_dialog():
        """Render template setter dialog"""

        @st.dialog("Template Settings", width="large")
        def template_dialog():
            st.markdown("### 🛠️ Customize Agent Templates")
            st.caption("Configure custom templates for each agent. Changes will override config file settings.")

            # Create a temporary state for the dialog
            if "temp_template_settings" not in st.session_state:
                # Initialize with current settings
                st.session_state.temp_template_settings = {}
                for agent in AGENTS_LIST:
                    current = st.session_state.config_overrides["templates"][agent]
                    mode = current["mode"]
                    value = current["value"]

                    # Only load default template for text_edit mode when value is None
                    if mode == "text_edit" and not value:
                        value = get_default_template(agent)
                    # For path_specify mode, keep None if no value
                    elif mode == "path_specify" and not value:
                        value = None

                    st.session_state.temp_template_settings[agent] = {"mode": mode, "value": value}

            # Group agents by category
            categories = {
                "Code Generation": ["coder"],
                "Execution & Analysis": ["executer", "error_analyzer"],
                "Data Processing": ["reader"],
                "Information Retrieval": ["retriever", "reranker", "description_file_retriever"],
                "Task Management": ["task_descriptor", "tool_selector"],
            }

            for category, agents in categories.items():
                with st.expander(f"**{category}**", expanded=False):
                    for agent in agents:
                        st.markdown(f"#### {agent.replace('_', ' ').title()}")

                        # Get current settings from temp state
                        current_mode = st.session_state.temp_template_settings[agent]["mode"]

                        # Radio selection with callback
                        def on_mode_change(agent_name):
                            mode = st.session_state[f"{agent_name}_mode_radio"]
                            old_mode = st.session_state.temp_template_settings[agent_name]["mode"]
                            st.session_state.temp_template_settings[agent_name]["mode"] = mode

                            # Handle mode transitions
                            if (
                                mode == "text_edit"
                                and not st.session_state.temp_template_settings[agent_name]["value"]
                            ):
                                # If switching to text_edit and no value exists, load default template
                                st.session_state.temp_template_settings[agent_name]["value"] = get_default_template(
                                    agent_name
                                )
                            elif mode == "use_default":
                                # If switching to use_default, clear the value
                                st.session_state.temp_template_settings[agent_name]["value"] = None
                            elif mode == "path_specify":
                                # If switching to path_specify from text_edit, clear the value if it's the default template
                                if old_mode == "text_edit":
                                    current_value = st.session_state.temp_template_settings[agent_name]["value"]
                                    default_template = get_default_template(agent_name)
                                    if current_value == default_template:
                                        st.session_state.temp_template_settings[agent_name]["value"] = None

                        mode = st.radio(
                            f"Template source for {agent}",
                            ["use_default", "text_edit", "path_specify"],
                            format_func=lambda x: {
                                "use_default": "Use Default",
                                "text_edit": "Custom Text",
                                "path_specify": "File Path",
                            }[x],
                            index=["use_default", "text_edit", "path_specify"].index(current_mode),
                            key=f"{agent}_mode_radio",
                            horizontal=True,
                            label_visibility="collapsed",
                            on_change=on_mode_change,
                            args=(agent,),
                        )

                        # Show appropriate input based on mode
                        if mode == "text_edit":
                            # Get value from temp settings
                            temp_value = st.session_state.temp_template_settings[agent]["value"]
                            display_text = temp_value if temp_value else get_default_template(agent)

                            def on_text_change(agent_name):
                                st.session_state.temp_template_settings[agent_name]["value"] = st.session_state[
                                    f"{agent_name}_text"
                                ]

                            value = st.text_area(
                                f"Template for {agent}",
                                value=display_text,
                                height=250,
                                key=f"{agent}_text",
                                label_visibility="collapsed",
                                on_change=on_text_change,
                                args=(agent,),
                            )

                        elif mode == "path_specify":

                            def on_path_change(agent_name):
                                path_value = st.session_state[f"{agent_name}_path"]
                                st.session_state.temp_template_settings[agent_name]["value"] = (
                                    path_value if path_value else None
                                )

                            # For path mode, only show the actual path value from temp settings
                            temp_value = st.session_state.temp_template_settings[agent]["value"]
                            path_value = ""
                            if (
                                temp_value
                                and isinstance(temp_value, str)
                                and (temp_value.endswith(".txt") or "/" in temp_value or "\\" in temp_value)
                            ):
                                path_value = temp_value

                            value = st.text_input(
                                f"Template file path for {agent}",
                                value=path_value,
                                placeholder="Enter path to template file (e.g., /path/to/template.txt)",
                                key=f"{agent}_path",
                                label_visibility="collapsed",
                                on_change=on_path_change,
                                args=(agent,),
                            )

                        else:  # use_default
                            # Already handled in on_mode_change
                            st.info("Using default template")

                        st.markdown("---")

            # Save and Close buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel", use_container_width=True):
                    # Clear temp settings
                    if "temp_template_settings" in st.session_state:
                        del st.session_state.temp_template_settings
                    st.rerun()

            with col2:
                if st.button("Save", type="primary", use_container_width=True):
                    # Apply temp settings to actual settings
                    for agent in AGENTS_LIST:
                        st.session_state.config_overrides["templates"][agent] = {
                            "mode": st.session_state.temp_template_settings[agent]["mode"],
                            "value": st.session_state.temp_template_settings[agent]["value"],
                        }
                    # Clear temp settings
                    del st.session_state.temp_template_settings
                    st.rerun()

        # Show the dialog
        template_dialog()

    @staticmethod
    def render_sidebar() -> TaskConfig:
        """Render sidebar"""
        with st.sidebar:
            with st.expander("⚙️ Settings", expanded=False):
                # Upper section: iterations, control, verbosity
                max_iter = st.number_input("Max iterations", min_value=1, max_value=20, value=5, key="max_iterations")
                control = st.checkbox("Manual prompts between iterations", key="control_prompts")
                log_verbosity = st.select_slider(
                    "Log verbosity",
                    options=["BRIEF", "INFO", "DETAIL"],
                    value="BRIEF",
                    key="log_verbosity",
                )

                # Single divider
                st.markdown("---")

                # Lower section: config, provider, model, credentials (compact)
                # Config file uploader
                uploaded_config = st.file_uploader(
                    "Config file (optional)",
                    type=["yaml", "yml"],
                    key="config_uploader",
                    help="Upload a custom YAML config file. If not provided, default config will be used.",
                )

                # Parse config if uploaded
                config_provider = None
                config_model = None
                disable_provider_model = False

                if uploaded_config:
                    try:
                        config_content = yaml.safe_load(uploaded_config.getvalue())
                        config_provider, config_model = ConfigFileHandler.extract_provider_model(config_content)
                        if config_provider and config_model:
                            # Update centralized overrides
                            SessionState.update_provider_model_from_config(config_provider, config_model)
                            disable_provider_model = True

                            # Also update templates from config (only for use_default items)
                            SessionState.update_templates_from_config(config_content)
                    except Exception as e:
                        st.error(f"Failed to parse config file: {str(e)}")

                # Get current overrides
                overrides = st.session_state.config_overrides

                # Provider selection
                provider = st.selectbox(
                    "LLM Provider",
                    options=["bedrock", "openai", "anthropic"],
                    index=["bedrock", "openai", "anthropic"].index(overrides["provider"] or "bedrock"),
                    key="llm_provider",
                    disabled=disable_provider_model,
                )

                # Update override if not disabled
                if not disable_provider_model:
                    st.session_state.config_overrides["provider"] = provider

                # Model input (with default based on provider)
                model_default = overrides["model"] if overrides["model"] else PROVIDER_DEFAULTS.get(provider, "")
                model = st.text_input(
                    "Model Name",
                    value=model_default,
                    key="model_name",
                    disabled=disable_provider_model,
                    help="Enter the model identifier",
                )

                # Update override if not disabled
                if not disable_provider_model:
                    st.session_state.config_overrides["model"] = model

                # Credentials section (dynamic based on provider)
                credentials = None
                if provider == "bedrock":
                    use_custom_creds = st.checkbox(
                        "Use custom AWS credentials",
                        key="bedrock_custom_creds",
                        help="If unchecked, will use EC2 IAM role",
                    )

                    if use_custom_creds:
                        credentials_text = st.text_area(
                            "AWS Credentials",
                            height=120,
                            key="bedrock_credentials",
                            placeholder='export AWS_ACCESS_KEY_ID="ASIA..."\nexport AWS_SECRET_ACCESS_KEY="..."\nexport AWS_SESSION_TOKEN="..."  # For temporary credentials\nexport AWS_DEFAULT_REGION="us-west-2"  # Optional',
                            help="Paste your AWS credentials (permanent or temporary)",
                        )

                        if credentials_text:
                            parsed_creds = BedrockCredentialsValidator.parse_credentials(credentials_text)
                            if parsed_creds:
                                is_valid, message = BedrockCredentialsValidator.validate_credentials(parsed_creds)
                                if is_valid:
                                    st.success(f"✅ {message}")
                                    credentials = parsed_creds
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                st.error("❌ Invalid format. Please include all required fields.")

                elif provider == "openai":
                    credentials_text = st.text_area(
                        "OpenAI API Key",
                        height=68,
                        key="openai_credentials",
                        placeholder='export OPENAI_API_KEY="sk-..."',
                        help="Paste your OpenAI API key",
                    )

                    if credentials_text:
                        parsed_creds = OpenAICredentialsValidator.parse_credentials(credentials_text)
                        if parsed_creds:
                            is_valid, message = OpenAICredentialsValidator.validate_credentials(parsed_creds)
                            if is_valid:
                                st.success(f"✅ {message}")
                                credentials = parsed_creds
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.error('❌ Invalid format. Please use: export OPENAI_API_KEY="sk-..."')

                elif provider == "anthropic":
                    credentials_text = st.text_area(
                        "Anthropic API Key",
                        height=68,
                        key="anthropic_credentials",
                        placeholder='export ANTHROPIC_API_KEY="your-anthropic-api-key"',
                        help="Paste your Anthropic API key",
                    )

                    if credentials_text:
                        parsed_creds = AnthropicCredentialsValidator.parse_credentials(credentials_text)
                        if parsed_creds:
                            is_valid, message = AnthropicCredentialsValidator.validate_credentials(parsed_creds)
                            if is_valid:
                                st.success(f"✅ {message}")
                                credentials = parsed_creds
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.error('❌ Invalid format. Please use: export ANTHROPIC_API_KEY="..."')

                # Template setter button
                st.markdown("---")
                if st.button("🔧 Launch template setter", use_container_width=True):
                    # Clear any existing temp settings before opening dialog
                    if "temp_template_settings" in st.session_state:
                        del st.session_state.temp_template_settings
                    UI.render_template_dialog()

                # Create config object
                config = TaskConfig(
                    uploaded_config=uploaded_config,
                    max_iter=max_iter,
                    control=control,
                    log_verbosity=log_verbosity,
                    provider=provider,
                    model=model,
                    credentials=credentials,
                )

            # History management
            task_count = sum(1 for msg in st.session_state.messages if msg.type == "task_log")
            if task_count > 0:
                st.markdown(f"### 📋 Task History ({task_count} tasks)")
                if st.button("🗑️ Clear All History"):
                    st.session_state.messages = [
                        Message.text(
                            "Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start."
                        )
                    ]
                    st.rerun()

        return config

    @staticmethod
    def render_single_message(msg):
        """Render a single message as a fragment to isolate interactions"""
        if msg.type == "text":
            st.markdown(msg.content["text"])
        elif msg.type == "user_summary":
            st.markdown(msg.content["summary"])
        elif msg.type == "command":
            st.code(msg.content["command"], language="bash")
        elif msg.type == "queue_status":
            content = msg.content
            position = content.get("position", 0)
            st.info(f"⏳ Task submitted! Position in queue: {position}")
        elif msg.type == "task_log":
            content = msg.content
            st.caption(f"ID: {content['run_id'][:8]}... | Completed: {content['timestamp']}")
            render_task_logs(content["phase_states"], content["max_iter"], show_progress=False)
        elif msg.type == "task_results":
            content = msg.content
            if "output_dir" in content and content["output_dir"]:
                from autogluon.assistant.webui.result_manager import ResultManager

                manager = ResultManager(content["output_dir"], content["run_id"])
                manager.render()
        elif msg.type == "debug_config":
            # DEBUG block - easy to remove later
            with st.expander("🐛 DEBUG: Final Config Content", expanded=True):
                st.caption(f"Config saved to: {msg.content['path']}")
                st.code(msg.content["content"], language="yaml")

    @staticmethod
    def render_messages():
        """Render message history"""
        for msg in st.session_state.messages:
            with st.chat_message(msg.role):
                UI.render_single_message(msg)

    @staticmethod
    def format_user_summary(files: List[str], config: TaskConfig, prompt: str, config_file: str) -> str:
        """Format user input summary"""
        parts = [
            "📂 **Uploaded files:**",
            "\n".join(f"- {f}" for f in files) if files else "- (none)",
            "\n⚙️ **Settings:**\n",
            f"- Config file: {config_file}",
            f"- Max iterations: {config.max_iter}",
            f"- Manual prompts: {config.control}",
            f"- Log verbosity: {config.log_verbosity}",
            f"- LLM Provider: {config.provider}",
            f"- Model: {config.model}",
        ]

        if config.provider == "bedrock" and config.credentials:
            parts.append("- Using custom AWS credentials: ✅")
        elif config.provider in ["openai", "anthropic"] and config.credentials:
            parts.append(f"- Using {config.provider} API key: ✅")

        parts.extend(["\n✏️ **Initial prompt:**\n", f"> {prompt or '(none)'}"])

        return "\n".join(parts)


# ==================== Task Manager ====================
class TaskManager:
    """Task manager"""

    def __init__(self, config: TaskConfig):
        self.config = config

    def _render_previous_iteration_files(self, output_dir: str, iteration: int):
        """Render previous iteration file contents"""
        if iteration <= 0 or not output_dir:
            return

        prev_iter = iteration - 1

        # Check both possible directory names (prefer generation_iter_)
        iter_dir = Path(output_dir) / f"generation_iter_{prev_iter}"

        if not iter_dir.exists():
            # Try the alternative naming
            iter_dir = Path(output_dir) / f"iteration_{prev_iter}"

        if not iter_dir.exists():
            st.warning("Cannot find iteration directory")
            # List what's actually in the output directory
            try:
                if Path(output_dir).exists():
                    contents = list(Path(output_dir).iterdir())
                    available_dirs = [d.name for d in contents if d.is_dir()]
                    st.info(f"Available directories: {available_dirs}")
                else:
                    st.error(f"Output directory does not exist: {output_dir}")
            except Exception as e:
                st.error(f"Error: {e}")
            return

        # File paths
        exec_script_path = iter_dir / "execution_script.sh"
        gen_code_path = iter_dir / "generated_code.py"
        stderr_path = iter_dir / "states" / "stderr"

        # Create tabs for the files
        tabs = st.tabs(["🔧 Execution Script", "🐍 Generated Code", "❌ Stderr"])

        with tabs[0]:
            if exec_script_path.exists():
                with open(exec_script_path, "r") as f:
                    st.code(f.read(), language="bash")
            else:
                st.info("Execution script not found")

        with tabs[1]:
            if gen_code_path.exists():
                with open(gen_code_path, "r") as f:
                    st.code(f.read(), language="python")
            else:
                st.info("Generated code not found")

        with tabs[2]:
            if stderr_path.exists():
                with open(stderr_path, "r") as f:
                    content = f.read()
                    if content.strip():
                        # Clean markup tags from stderr content
                        cleaned_content = re.sub(r"\[/?bold\s*(green|red)\]", "", content)
                        st.code(cleaned_content, language="text")
                    else:
                        st.info("No error logs")
            else:
                st.info("Error log not found")

    def handle_submission(self, submission):
        """Handle user submission"""
        # If waiting for input, handle it as iteration input
        if st.session_state.waiting_for_input:
            self.handle_iteration_input(submission)
            return

        # When accept_file="multiple", submission has .files and .text attributes
        files = submission.files or []
        user_text = submission.text.strip() if submission.text else ""

        if not files:
            SessionState.add_message(
                Message.text("⚠️ No data files provided. Please drag and drop your data files or ZIP.")
            )
            st.rerun()
            return

        # Validate credentials if needed
        if self.config.provider in ["openai", "anthropic"] and not self.config.credentials:
            SessionState.add_message(Message.text(f"❌ Please provide {self.config.provider} API key first"))
            st.rerun()
            return

        # Process files
        data_folder = handle_uploaded_files(files)
        st.session_state.data_src = data_folder

        # Save config file
        config_path = self._save_config(data_folder)
        config_name = self.config.uploaded_config.name if self.config.uploaded_config else "default.yaml (modified)"

        # Add user summary
        summary = UI.format_user_summary([f.name for f in files], self.config, user_text, config_name)
        SessionState.add_message(Message.user_summary(summary, input_dir=data_folder))

        # Start task
        self._start_task(data_folder, config_path, user_text)

    def handle_iteration_input(self, submission):
        """Handle iteration input"""
        # When accept_file=False, submission is just a string
        if not submission:
            user_input = ""  # Empty input means skip
        else:
            user_input = submission.strip()

        # Don't add iteration prompt as a separate message - it will be shown in logs

        # Send input to backend
        if BackendAPI.send_user_input(st.session_state.run_id, user_input):
            SessionState.set_waiting_for_input(False)
            # Force update by clearing the processor's waiting state
            run_id = st.session_state.run_id
            processor_key = f"log_processor_{run_id}"
            if processor_key in st.session_state:
                processor = st.session_state[processor_key]
                processor.waiting_for_input = False
                processor.input_prompt = None

            # Clear placeholder content
            if st.session_state.prev_iter_placeholder:
                st.session_state.prev_iter_placeholder.empty()
        else:
            SessionState.add_message(Message.text("❌ Failed to send input to the process."))

        st.rerun()

    def handle_cancel_request(self):
        """Handle cancel request"""
        # Display user's cancel command
        SessionState.add_message(Message.text("cancel", role="user"))

        # Check if task is queued or running
        if st.session_state.queue_position is not None and st.session_state.queue_position > 0:
            # Task is queued, cancel from queue
            task_id = st.session_state.task_id
            if BackendAPI.cancel_task(task_id=task_id):
                SessionState.add_message(Message.text(f"🛑 Queued task {task_id[:8]}... has been cancelled."))
                SessionState.finish_task()
            else:
                SessionState.add_message(Message.text("❌ Failed to cancel the queued task."))
        else:
            # Task is running, cancel the run
            run_id = st.session_state.run_id
            if not run_id:
                return

            # Try to cancel task
            if BackendAPI.cancel_task(run_id=run_id):
                SessionState.add_message(Message.text(f"🛑 Task {run_id[:8]}... has been cancelled."))

                if st.session_state.prev_iter_placeholder:
                    st.session_state.prev_iter_placeholder.empty()
                    st.session_state.prev_iter_placeholder = None
                # Save current logs
                if st.session_state.current_task_logs:
                    processed = process_logs(
                        st.session_state.current_task_logs, st.session_state.running_config.max_iter
                    )

                    # Extract output directory if available
                    output_dir = self._extract_output_dir(processed["phase_states"])

                    SessionState.add_message(
                        Message.task_log(
                            st.session_state.run_id,
                            processed["phase_states"],
                            st.session_state.running_config.max_iter,
                            output_dir,
                            st.session_state.current_input_dir,
                        )
                    )

                    # Add task results message if output directory found
                    if output_dir:
                        SessionState.add_message(Message.task_results(st.session_state.run_id, output_dir))

                SessionState.finish_task()
            else:
                SessionState.add_message(Message.text("❌ Failed to cancel the task."))

        st.rerun()

    def handle_task_deletion(self):
        """Handle task deletion request"""
        # Check for deletion flags
        keys_to_check = [k for k in st.session_state if k.startswith("delete_task_")]

        for key in keys_to_check:
            if st.session_state.get(key):
                run_id = key.replace("delete_task_", "")

                # Find the task messages to get directories
                output_dir = None
                input_dir = None

                for msg in st.session_state.messages:
                    if msg.type == "task_log" and msg.content.get("run_id") == run_id:
                        output_dir = msg.content.get("output_dir")
                        input_dir = msg.content.get("input_dir")
                        break

                # Delete directories
                success = True
                error_msg = ""

                try:
                    # Delete output directory
                    if output_dir and Path(output_dir).exists():
                        shutil.rmtree(output_dir)

                    # Delete input directory
                    if input_dir and Path(input_dir).exists():
                        shutil.rmtree(input_dir)
                except Exception as e:
                    success = False
                    error_msg = str(e)

                # Remove from message history
                SessionState.delete_task_from_history(run_id)

                # Clear the deletion flag
                del st.session_state[key]

                # Show result message
                if success:
                    st.success(f"Task {run_id[:8]}... and all associated data have been deleted.")
                else:
                    st.error(f"Error deleting task data: {error_msg}")

                # Force a complete rerun
                st.rerun()

    def render_running_task(self):
        """Render the currently running task"""
        if not st.session_state.task_running:
            return

        task_id = st.session_state.task_id
        run_id = st.session_state.run_id
        config = st.session_state.running_config

        if not config:
            st.error("Running configuration not found!")
            return

        # If we don't have a run_id yet, check task status
        if not run_id:
            queue_status = BackendAPI.check_queue_status(task_id)

            if not queue_status:
                with st.chat_message("assistant"):
                    st.error("Failed to get task status")
                return

            position = queue_status.get("position", 0)
            st.session_state.queue_position = position

            # Update run_id if available
            if queue_status.get("run_id"):
                st.session_state.run_id = queue_status["run_id"]
                st.rerun()  # Rerun to process with run_id
                return

            # Display appropriate status based on position
            with st.chat_message("assistant"):
                if position > 0:
                    st.markdown("### Queued Task")
                    st.caption(f"ID: {task_id[:8]}... | Type 'cancel' to remove from queue")
                    st.info(f"⏳ Waiting in queue... Position: {position}")
                else:
                    st.markdown("### Starting Task")
                    st.caption(f"ID: {task_id[:8]}...")
                    st.info("🚀 Task is starting, please wait...")
            return

        # Task is running with valid run_id
        # Get new logs
        new_logs = BackendAPI.fetch_logs(run_id)
        st.session_state.current_task_logs.extend(new_logs)

        # Get status
        status = BackendAPI.check_status(run_id)

        # Display running task
        with st.chat_message("assistant"):
            st.markdown("### Current Task")
            st.caption(f"ID: {run_id[:8]}... | Type 'cancel' to stop the task")

            # Check for process errors in logs
            error_found = False
            for log_entry in st.session_state.current_task_logs:
                # log_entry is a dict with 'level', 'text', 'special' keys
                if isinstance(log_entry, dict) and log_entry.get("text", "").startswith("Process error:"):
                    st.error(f"❌ {log_entry['text']}")
                    error_found = True
                elif isinstance(log_entry, str) and log_entry.startswith("Process error:"):
                    # Fallback for string entries
                    st.error(f"❌ {log_entry}")
                    error_found = True

            # Process logs and check for input requests
            if not error_found:
                waiting_for_input, input_prompt, output_dir = messages(
                    st.session_state.current_task_logs, config.max_iter
                )

                # Update output directory in session state
                if output_dir and not st.session_state.current_output_dir:
                    st.session_state.current_output_dir = output_dir

                # Update session state if waiting for input
                if waiting_for_input and not st.session_state.waiting_for_input:
                    # Extract iteration number from logs if possible
                    iteration = self._extract_current_iteration()
                    SessionState.set_waiting_for_input(True, input_prompt, iteration)
                    # Don't rerun here - let the fragment cycle handle it

        # Check if finished
        if status.get("finished", False):
            self._complete_task()
            st.rerun()

    def monitor_running_task(self):
        """Monitor running task"""
        if st.session_state.task_running:
            # Render the running task
            self.render_running_task()

            # Create a placeholder for Previous Iteration Results
            if st.session_state.prev_iter_placeholder is None:
                st.session_state.prev_iter_placeholder = st.empty()

            # If waiting for input, show previous iteration files
            if st.session_state.waiting_for_input and self.config.control and st.session_state.current_iteration > 0:

                # Try to find output directory
                output_dir = None

                # First try session state directory
                if st.session_state.get("current_output_dir"):
                    output_dir = st.session_state.current_output_dir
                else:
                    # Extract from logs
                    for entry in reversed(st.session_state.current_task_logs[-50:]):
                        text = entry.get("text", "")
                        # New log format: match "info is stored in:"
                        if "info is stored in:" in text:
                            try:
                                import re

                                # Match path pattern, handle [/bold green] markup
                                match = re.search(r"info is stored in:\[/bold green\]\s+([^\s]+)", text)
                                if match:
                                    full_path = match.group(1)
                                    # Extract base directory, remove /initialization or /iteration_X
                                    if "/initialization" in full_path:
                                        output_dir = full_path.rsplit("/initialization", 1)[0]
                                    elif "/iteration_" in full_path:
                                        output_dir = full_path.rsplit("/iteration_", 1)[0]
                                    else:
                                        # If path is already base directory, use directly
                                        output_dir = full_path
                                    st.session_state.current_output_dir = output_dir
                                    break
                            except Exception:
                                pass

                # Display content using placeholder
                if output_dir:
                    with st.session_state.prev_iter_placeholder.container():
                        st.markdown("---")
                        st.markdown("### 📁 Previous Iteration Results")
                        self._render_previous_iteration_files(output_dir, st.session_state.current_iteration)
                        st.markdown("---")
                else:
                    # Clear placeholder to ensure no residual content
                    st.session_state.prev_iter_placeholder.empty()
            else:
                # Clear placeholder when conditions not met
                st.session_state.prev_iter_placeholder.empty()

            # Auto-refresh logic
            if st.session_state.task_running:
                time.sleep(0.5)
                st.rerun()

    def _extract_current_iteration(self) -> int:
        """Extract current iteration number from logs"""
        # Look for "Starting iteration X!" in recent logs
        for entry in reversed(st.session_state.current_task_logs[-20:]):  # Check last 20 entries
            text = entry.get("text", "")
            if "Starting iteration" in text:
                try:
                    import re

                    match = re.search(r"Starting iteration (\d+)!", text)
                    if match:
                        return int(match.group(1))
                except:
                    pass
        return 1  # Default to 1 if not found

    def _save_config(self, data_folder: str) -> str:
        """Save config file with all overrides applied"""
        overrides = st.session_state.config_overrides

        if self.config.uploaded_config:
            # Load uploaded config
            config_content = yaml.safe_load(self.config.uploaded_config.getvalue())

            # Check if uploaded config has provider/model that should be preserved
            config_provider, config_model = ConfigFileHandler.extract_provider_model(config_content)

            # If UI controls are disabled (config file has provider/model), use file as base
            # but still apply template overrides
            if config_provider and config_model and not overrides["provider"]:
                # Use uploaded config as base
                base_config = config_content
            else:
                # Apply all overrides to uploaded config
                base_config = config_content

            config_path = Path(data_folder) / self.config.uploaded_config.name
        else:
            # No uploaded config, use default
            base_config = ConfigFileHandler.load_default_config()
            config_path = Path(data_folder) / "autogluon_config.yaml"

        # Save with all overrides applied
        return ConfigFileHandler.save_modified_config(base_config, overrides, config_path)

    def _start_task(self, data_folder: str, config_path: str, user_prompt: str):
        """Start task"""

        # ===== DEBUG BLOCK START - EASY TO REMOVE =====
        # Read and display the saved config content
        try:
            with open(config_path, "r") as f:
                config_content = f.read()
            SessionState.add_message(Message.debug_config(config_path, config_content))
        except Exception as e:
            SessionState.add_message(Message.text(f"❌ DEBUG: Failed to read config: {str(e)}"))
        # ===== DEBUG BLOCK END =====

        # Submit task to queue
        task_id, position = BackendAPI.start_task(data_folder, config_path, user_prompt, self.config)

        # Build command for display
        cmd_parts = [
            "mlzero",
            "-i",
            data_folder,
            "-n",
            str(self.config.max_iter),
            "-v",
            VERBOSITY_MAP[self.config.log_verbosity],
            "-c",
            config_path,
        ]

        if user_prompt:
            cmd_parts.extend(["--initial-instruction", user_prompt])
        if self.config.control:
            cmd_parts.append("--enable-per-iteration-instruction")

        # Display command
        command_str = f"[{datetime.now().strftime('%H:%M:%S')}] Submitting AutoMLAgent task: {' '.join(cmd_parts)}"
        SessionState.add_message(Message.command(command_str))

        # Add queue status message
        SessionState.add_message(Message.queue_status(task_id, position))

        # Start task monitoring
        SessionState.start_task(task_id, self.config, data_folder, position)
        st.rerun()

    def _extract_output_dir(self, phase_states: Dict) -> Optional[str]:
        """Extract output directory from phase states"""
        output_phase = phase_states.get("Output", {})
        logs = output_phase.get("logs", [])

        for log in reversed(logs):
            import re

            # Look for "output saved in" pattern and extract the path
            match = re.search(r"output saved in\s+([^\s]+)", log)
            if match:
                output_dir = match.group(1).strip()
                # Remove any trailing punctuation
                output_dir = output_dir.rstrip(".,;:")
                return output_dir
        return None

    def _check_task_failed(self, phase_states: Dict) -> bool:
        """Check if the task failed by looking for success message in the last iteration"""
        # Find the highest iteration number
        iteration_phases = [name for name in phase_states.keys() if name.startswith("Iteration")]
        if not iteration_phases:
            return True  # No iterations found, consider it failed

        # Sort iterations by number
        iteration_phases.sort(key=lambda x: int(x.split()[1]))
        last_iteration = iteration_phases[-1]

        # Check logs in the last iteration
        last_iter_logs = phase_states.get(last_iteration, {}).get("logs", [])
        for log in last_iter_logs:
            # Check if the log contains the success marker
            if "[bold green]Code generation successful" in log or "Code generation successful" in log:
                return False  # Found success message, task succeeded

        return True  # No success message found, task failed

    def _complete_task(self):
        """Complete task"""
        # Clear placeholder
        if st.session_state.prev_iter_placeholder:
            st.session_state.prev_iter_placeholder.empty()
            st.session_state.prev_iter_placeholder = None

        # Save task logs
        if st.session_state.current_task_logs:
            processed = process_logs(st.session_state.current_task_logs, st.session_state.running_config.max_iter)

            # Extract output directory
            output_dir = self._extract_output_dir(processed["phase_states"])

            SessionState.add_message(
                Message.task_log(
                    st.session_state.run_id,
                    processed["phase_states"],
                    st.session_state.running_config.max_iter,
                    output_dir,
                    st.session_state.current_input_dir,
                )
            )

            # Check if task failed
            task_failed = self._check_task_failed(processed["phase_states"])

            # Add success or failure message
            if not task_failed:
                # Use a special message type for success to render with st.success()
                SessionState.add_message(Message.text(SUCCESS_MESSAGE))
            else:
                SessionState.add_message(Message.text("❌ Task failed. Please check the logs for details."))

            # Add task results message if output directory found
            if output_dir:
                SessionState.add_message(Message.task_results(st.session_state.run_id, output_dir))

        SessionState.finish_task()


def is_running_in_streamlit():
    """Check if running in streamlit environment"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        try:
            from streamlit.script_run_context import get_script_run_ctx

            return get_script_run_ctx() is not None
        except ImportError:
            return False


# ==================== Main App ====================
class AutoMLAgentApp:
    """Main application"""

    def __init__(self):
        SessionState.init()
        self.config = UI.render_sidebar()
        self.task_manager = TaskManager(self.config)

    def run(self):
        """Run application"""
        # Check for task deletion requests first
        self.task_manager.handle_task_deletion()

        # Render history messages
        UI.render_messages()

        # Determine chat input configuration based on state
        if st.session_state.waiting_for_input:
            # When waiting for iteration input
            placeholder = (
                st.session_state.input_prompt or "Enter your input for this iteration (press Space then Enter to skip)"
            )
            accept_file = False  # Don't accept files during iteration prompts
        elif st.session_state.task_running:
            # When task is running but not waiting for input
            placeholder = "Type 'cancel' to stop the current task"
            accept_file = False
        else:
            # Normal state - ready to accept new tasks
            placeholder = "Type optional prompt, or drag & drop your data files/ZIP here"
            accept_file = "multiple"

        # Handle user input
        submission = st.chat_input(
            placeholder=placeholder,
            accept_file=accept_file,
            key="u_input",
            max_chars=10000,
        )

        if submission:
            # If waiting for input
            if st.session_state.waiting_for_input:
                self.task_manager.handle_submission(submission)
            # If task is running
            elif st.session_state.task_running:
                # Check if it's a cancel command
                # When accept_file=False, submission is just a string
                if submission and submission.strip().lower() == "cancel":
                    self.task_manager.handle_cancel_request()
                else:
                    # Show hint message
                    SessionState.add_message(
                        Message.text(
                            "⚠️ A task is currently running. Type 'cancel' to stop it, or wait for it to complete.",
                            role="user",
                        )
                    )
                    st.rerun()
            else:
                # No task running, handle submission normally
                self.task_manager.handle_submission(submission)

        # Monitor running task
        self.task_manager.monitor_running_task()


def main():
    """Entry point"""
    st.set_page_config(
        page_title="AutoGluon Assistant",
        page_icon=LOGO_PATH,
        layout="wide",
        initial_sidebar_state="auto",
    )
    theme = st_theme()
    if theme and theme.get("base") == "dark":
        st.logo(logo_night_path, size="large", link="https://github.com/autogluon")
    else:
        st.logo(logo_day_path, size="large", link="https://github.com/autogluon")

    reload_warning = """
    <script>
        window.onbeforeunload = function () {
            return "placeholder";
        };
    </script>
    """
    components.html(reload_warning, height=0)

    app = AutoMLAgentApp()
    app.run()


def launch_streamlit():
    """Entry point for mlzero-webui command - launches streamlit server."""
    from pathlib import Path

    # Get current file path
    current_file = Path(__file__).resolve()

    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(current_file), "--server.port=8509"]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down webui...")
    except Exception as e:
        print(f"Error running webui: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if is_running_in_streamlit():
        main()
    else:
        launch_streamlit()
