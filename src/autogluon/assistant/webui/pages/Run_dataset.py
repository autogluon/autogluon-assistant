# src/autogluon/assistant/webui/pages/Run_dataset.py

import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests
import streamlit as st
import yaml
from botocore.exceptions import ClientError, NoCredentialsError

from autogluon.assistant.constants import API_URL, PROVIDER_DEFAULTS, SUCCESS_MESSAGE, VERBOSITY_MAP
from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.webui.log_processor import messages, process_logs, render_task_logs

# ==================== Constants ====================
PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"


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
            credentials["AWS_DEFAULT_REGION"] = "us-east-1"

        return credentials

    @staticmethod
    def validate_credentials(credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Validate AWS credentials"""
        try:
            # Create a session with the provided credentials
            session_params = {
                "aws_access_key_id": credentials["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": credentials["AWS_SECRET_ACCESS_KEY"],
                "region_name": credentials.get("AWS_DEFAULT_REGION", "us-east-1"),
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
    def save_modified_config(base_config: Dict, provider: str, model: str, save_path: Path) -> str:
        """Save config with modified provider and model"""
        try:
            # Deep copy to avoid modifying the original
            import copy

            config = copy.deepcopy(base_config)

            # Update the llm section
            if "llm" not in config:
                config["llm"] = {}
            config["llm"]["provider"] = provider
            config["llm"]["model"] = model

            # Update all agent sections that have provider/model
            agent_sections = [
                "coder",
                "executer",
                "reader",
                "error_analyzer",
                "retriever",
                "description_file_retriever",
                "task_descriptor",
                "tool_selector",
            ]

            for agent in agent_sections:
                if agent in config and isinstance(config[agent], dict):
                    config[agent]["provider"] = provider
                    config[agent]["model"] = model

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
            "current_task_logs": [],
            "running_config": None,
            "current_input_dir": None,
            "waiting_for_input": False,
            "input_prompt": None,
            "current_iteration": 0,
            "current_output_dir": None,
            "prev_iter_placeholder": None,  # Placeholder object
            # New provider-related states
            "config_from_file": False,
            "config_provider": None,
            "config_model": None,
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def start_task(run_id: str, config: TaskConfig, input_dir: str):
        """Start new task"""
        st.session_state.task_running = True
        st.session_state.run_id = run_id
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

        # Find the associated messages (user_summary, command, iteration_prompts, etc.)
        start_index = task_log_index

        # Look backwards for related messages
        i = task_log_index - 1
        while i >= 0:
            msg = st.session_state.messages[i]

            # Check for various message types
            if msg.type in ["command", "user_summary"]:
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
    def start_task(data_src: str, config_path: str, user_prompt: str, config: TaskConfig) -> str:
        """Start task"""
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
        return response.json()["run_id"]

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
    def send_user_input(run_id: str, user_input: str) -> bool:
        """Send user input to backend"""
        try:
            response = requests.post(f"{API_URL}/input", json={"run_id": run_id, "input": user_input})
            return response.json().get("success", False)
        except Exception as e:
            st.error(f"Error sending input: {str(e)}")
            return False

    @staticmethod
    def cancel_task(run_id: str) -> bool:
        """Cancel task"""
        try:
            response = requests.post(f"{API_URL}/cancel", json={"run_id": run_id})
            return response.json().get("cancelled", False)
        except:
            return False


# ==================== UI Components ====================
class UI:
    """UI components"""

    @staticmethod
    def setup_page():
        """Setup page"""
        st.set_page_config(page_title="AutoMLAgent Chat", layout="wide")

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
                            st.session_state.config_from_file = True
                            st.session_state.config_provider = config_provider
                            st.session_state.config_model = config_model
                            disable_provider_model = True

                            st.session_state["llm_provider"] = config_provider
                            st.session_state["model_name"] = config_model
                    except Exception as e:
                        st.error(f"Failed to parse config file: {str(e)}")
                else:
                    st.session_state.config_from_file = False
                    st.session_state.config_provider = None
                    st.session_state.config_model = None

                # Provider selection
                provider = st.selectbox(
                    "LLM Provider",
                    options=["bedrock", "openai", "anthropic"],
                    index=0 if not config_provider else ["bedrock", "openai", "anthropic"].index(config_provider),
                    key="llm_provider",
                    disabled=disable_provider_model,
                )

                # Model input (with default based on provider)
                model_default = config_model if config_model else PROVIDER_DEFAULTS.get(provider, "")
                model = st.text_input(
                    "Model Name",
                    value=model_default,
                    key="model_name",
                    disabled=disable_provider_model,
                    help="Enter the model identifier",
                )

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
                            placeholder='export AWS_ACCESS_KEY_ID="ASIA..."\nexport AWS_SECRET_ACCESS_KEY="..."\nexport AWS_SESSION_TOKEN="..."  # For temporary credentials\nexport AWS_DEFAULT_REGION="us-east-1"  # Optional',
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
            # Handle special "success" role for success messages
            if msg.role == "success":
                st.success(msg.content["text"])
            else:
                # Use markdown for text messages to properly render code blocks
                st.markdown(msg.content["text"])
        elif msg.type == "user_summary":
            st.markdown(msg.content["summary"])
        elif msg.type == "command":
            st.code(msg.content["command"], language="bash")
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
        run_id = st.session_state.run_id
        if not run_id:
            return

        # Display user's cancel command
        SessionState.add_message(Message.text("cancel", role="user"))

        # Try to cancel task
        if BackendAPI.cancel_task(run_id):
            SessionState.add_message(Message.text(f"🛑 Task {run_id[:8]}... has been cancelled."))

            if st.session_state.prev_iter_placeholder:
                st.session_state.prev_iter_placeholder.empty()
                st.session_state.prev_iter_placeholder = None
            # Save current logs
            if st.session_state.current_task_logs:
                processed = process_logs(st.session_state.current_task_logs, st.session_state.running_config.max_iter)

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
        if not st.session_state.task_running or not st.session_state.run_id:
            return

        run_id = st.session_state.run_id
        config = st.session_state.running_config

        if not config:
            st.error("Running configuration not found!")
            return

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
        """Save config file"""
        if self.config.uploaded_config:
            # If user uploaded a config and controls are disabled, use the file as-is
            if st.session_state.get("config_from_file", False):
                # Save the uploaded config as-is
                config_path = Path(data_folder) / self.config.uploaded_config.name
                with open(config_path, "wb") as f:
                    f.write(self.config.uploaded_config.getbuffer())
                return str(config_path)
            else:
                # This shouldn't happen, but handle it anyway
                config_path = Path(data_folder) / self.config.uploaded_config.name
                with open(config_path, "wb") as f:
                    f.write(self.config.uploaded_config.getbuffer())
                return str(config_path)
        else:
            # No uploaded config, create modified default config with UI selections
            default_config = ConfigFileHandler.load_default_config()
            if default_config:
                config_path = Path(data_folder) / "autogluon_config.yaml"
                return ConfigFileHandler.save_modified_config(
                    default_config, self.config.provider, self.config.model, config_path
                )
            else:
                return str(DEFAULT_CONFIG_PATH)

    def _start_task(self, data_folder: str, config_path: str, user_prompt: str):
        """Start task"""
        # First generate run_id
        run_id = BackendAPI.start_task(data_folder, config_path, user_prompt, self.config)

        # Build command
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
            cmd_parts.extend(["-u", user_prompt])
        if self.config.control:
            cmd_parts.append("--need-user-input")

        # Display command
        command_str = f"[{datetime.now().strftime('%H:%M:%S')}] Running AutoMLAgent: {' '.join(cmd_parts)}"
        SessionState.add_message(Message.command(command_str))

        # Start task
        SessionState.start_task(run_id, self.config, data_folder)
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
        """Check if the task failed by looking for failure message in the last iteration"""
        # Find the highest iteration number
        iteration_phases = [name for name in phase_states.keys() if name.startswith("Iteration")]
        if not iteration_phases:
            return False

        # Sort iterations by number
        iteration_phases.sort(key=lambda x: int(x.split()[1]))
        last_iteration = iteration_phases[-1]

        # Check logs in the last iteration
        last_iter_logs = phase_states.get(last_iteration, {}).get("logs", [])
        for log in last_iter_logs:
            # Check if the log contains the failure marker
            if "[bold red]Code generation failed in iteration" in log or "Code generation failed in iteration" in log:
                return True

        return False

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
                SessionState.add_message(Message.text(SUCCESS_MESSAGE, role="assistant"))
            else:
                SessionState.add_message(Message.text("❌ Task failed. Please check the logs for details."))

            # Add task results message if output directory found
            if output_dir:
                SessionState.add_message(Message.task_results(st.session_state.run_id, output_dir))

        SessionState.finish_task()


# ==================== Main App ====================
class AutoMLAgentApp:
    """Main application"""

    def __init__(self):
        UI.setup_page()
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
            placeholder = st.session_state.input_prompt or "Enter your input for this iteration (press Enter to skip)"
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
    app = AutoMLAgentApp()
    app.run()


if __name__ == "__main__":
    main()
