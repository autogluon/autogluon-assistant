# src/autogluon/assistant/webui/pages/Run_dataset.py

import copy
import os
import uuid
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Any
import requests

import streamlit as st

from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.webui.log_processor import messages
from autogluon.assistant.constants import INITIAL_STAGE, SUCCESS_MESSAGE, API_URL


# ==================== Constants ====================
PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

VERBOSITY_MAP = {
    "MODEL_INFO": "3",
    "DETAILED_INFO": "2", 
    "BRIEF_INFO": "1",
}

DEFAULT_SESSION_STATE = {
    "user_session_id": lambda: uuid.uuid4().hex,
    "messages": lambda: [{"role": "assistant", "text": "Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start."}],
    "data_src": None,
    "task_running": False,
    "run_id": None,
    "all_logs": lambda: [],
    "current_stage": None,
    "stage_container": lambda: copy.deepcopy(INITIAL_STAGE),
    "stage_status": lambda: {},
}


# ==================== Session State Manager ====================
class SessionStateManager:
    """Centralized session state management"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        for key, default in DEFAULT_SESSION_STATE.items():
            if key not in st.session_state:
                # Call lambda functions if they exist, otherwise use the value directly
                st.session_state[key] = default() if callable(default) else default
    
    @staticmethod
    def reset_task_state():
        """Reset task-related state when starting a new task"""
        st.session_state.task_running = True
        st.session_state.current_stage = None
        st.session_state.stage_container = copy.deepcopy(INITIAL_STAGE)
        st.session_state.stage_status = {}
        st.session_state.all_logs = []

        if "log_processor_state" in st.session_state:
            del st.session_state.log_processor_state


    @staticmethod
    def add_message(role: str, text: str):
        """Add a message to the chat history"""
        st.session_state.messages.append({"role": role, "text": text})
    
    @staticmethod
    def update_logs(new_entries: List[Dict]):
        """Update the log entries"""
        st.session_state.all_logs.extend(new_entries)


# ==================== UI Components ====================
class UIComponents:
    """Handles UI component rendering"""
    
    @staticmethod
    def render_page_config():
        """Configure the Streamlit page"""
        st.set_page_config(page_title="AutoMLAgent Chat", layout="wide")
        st.markdown(
            """
            <style>
              /* your existing CSS… */
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    @staticmethod
    def render_sidebar() -> Dict[str, Any]:
        """Render sidebar settings and return configuration"""
        with st.sidebar:
            with st.expander("⚙️ Advanced Settings", expanded=False):
                config = {
                    "out_dir": st.text_input("Output directory", value="", key="output_dir"),
                    "config_path": st.text_input(
                        "Config file",
                        value=str(DEFAULT_CONFIG_PATH),
                        help="Path to YAML config file (only default.yaml is provided)",
                        key="config_path",
                    ),
                    "max_iter": st.number_input("Max iterations", min_value=1, max_value=20, value=5, key="max_iterations"),
                    "init_prompt": st.text_area("Initial prompt (optional)", key="initial_prompt", height=80),
                    "control": st.checkbox("Manual prompts between iterations", key="control_prompts"),
                    "extract_check": st.checkbox("Extract uploaded ZIP", key="extract_check"),
                    "extract_dir": st.text_input(
                        "Extraction dir",
                        placeholder="extract_to/",
                        key="extract_dir",
                        disabled=not st.session_state.get("extract_check", False),
                    ),
                    "log_verbosity": st.select_slider(
                        "Log verbosity",
                        options=list(VERBOSITY_MAP.keys()),
                        value="BRIEF_INFO",
                        key="log_verbosity",
                    ),
                }
        return config
    
    @staticmethod
    def render_chat_history():
        """Render the chat message history"""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["text"])
                else:
                    st.write(msg["text"])
    
    @staticmethod
    def format_user_summary(file_names: List[str], config: Dict[str, Any]) -> str:
        """Format the user input summary"""
        summary_parts = ["📂 **Uploaded files:**"]
        
        if file_names:
            summary_parts.append("\n".join(f"- {n}" for n in file_names))
        else:
            summary_parts.append("- (none)")
        
        summary_parts.extend([
            "\n⚙️ **Settings:**\n",
            f"- Output directory: {config['out_dir'] or '(default runs/)'}",
            f"- Config file: {config['config_path']}",
            f"- Max iterations: {config['max_iter']}",
            f"- Manual prompts: {config['control']}",
            f"- Extract ZIP: {config['extract_check']}{' → ' + config['extract_dir'] if config['extract_check'] else ''}"
            f"- Log verbosity: {config['log_verbosity']}",
            "\n✏️ **Initial prompt:**\n",
            f"> {config['init_prompt'] or '(none)'}"
        ])
        
        return "\n".join(summary_parts)


# ==================== Task Manager ====================
class TaskManager:
    """Manages task execution and communication with backend"""
    
    @staticmethod
    def build_command(config: Dict[str, Any]) -> List[str]:
        """Build the mlzero command from configuration"""
        cmd = [
            "mlzero",
            "-i", st.session_state.data_src,
            "-n", str(config["max_iter"]),
            "-v", VERBOSITY_MAP[config["log_verbosity"]], 
            "-c", config["config_path"],
        ]
        
        if config["out_dir"]:
            cmd.extend(["-o", config["out_dir"]])
        if config["init_prompt"]:
            cmd.extend(["-u", config["init_prompt"]])
        if config["control"]:
            cmd.append("--need-user-input")
        if config["extract_check"] and config["extract_dir"]:
            cmd.extend(["-e", config["extract_dir"]])
        
        return cmd
    
    @staticmethod
    def start_task(config: Dict[str, Any]) -> str:
        """Start a new task via backend API"""
        payload = {
            "data_src": st.session_state.data_src,
            "out_dir": config["out_dir"],
            "config_path": config["config_path"],
            "max_iter": config["max_iter"],
            "init_prompt": config["init_prompt"],
            "control": config["control"],
            "extract_dir": config["extract_dir"] if config["extract_check"] else None,
            "verbosity": VERBOSITY_MAP[config["log_verbosity"]],
        }
        
        response = requests.post(f"{API_URL}/run", json=payload)
        return response.json()["run_id"]
    
    @staticmethod
    def fetch_logs(run_id: str) -> List[Dict]:
        """Fetch new log entries from backend"""
        response = requests.get(f"{API_URL}/logs", params={"run_id": run_id})
        return response.json().get("lines", [])
    
    @staticmethod
    def check_status(run_id: str) -> bool:
        """Check if task is finished"""
        response = requests.get(f"{API_URL}/status", params={"run_id": run_id})
        return response.json().get("finished", False)


# ==================== Main Application Logic ====================
class AutoMLAgentApp:
    """Main application controller"""
    
    def __init__(self):
        UIComponents.render_page_config()
        SessionStateManager.initialize()
        self.config = UIComponents.render_sidebar()
    
    def handle_user_input(self, submission):
        """Process user input (files and text)"""
        files = submission.files or []
        file_names = []
        
        # Handle file upload
        if files:
            folder = handle_uploaded_files(files)
            st.session_state.data_src = folder
            file_names = [f.name for f in files]
        
        # Validate data source
        if not st.session_state.data_src:
            self._show_error("⚠️ No data detected. Please drag & drop your folder or ZIP first.")
            return
        
        # Create and display user summary
        user_summary = UIComponents.format_user_summary(file_names, self.config)
        SessionStateManager.add_message("user", user_summary)
        with st.chat_message("user"):
            st.markdown(user_summary)
        
        # Start the task
        self._start_task()
    
    def _show_error(self, message: str):
        """Display an error message"""
        SessionStateManager.add_message("assistant", message)
        with st.chat_message("assistant"):
            st.write(message)
        st.rerun()
    
    def _start_task(self):
        """Initialize and start a new task"""
        SessionStateManager.reset_task_state()
        
        # Build and display command
        cmd = TaskManager.build_command(self.config)
        t0 = datetime.now().strftime("%H:%M:%S")
        start_msg = f"[{t0}] Running AutoMLAgent: {' '.join(cmd)}"
        
        SessionStateManager.add_message("assistant", start_msg)
        with st.chat_message("assistant"):
            st.code(start_msg, language="bash")
        
        # Start task via backend
        run_id = TaskManager.start_task(self.config)
        st.session_state.run_id = run_id
        st.session_state.task_running = True
        st.rerun()
    
    def monitor_task(self):
        """Monitor running task and display logs"""
        if not (st.session_state.task_running and st.session_state.get("run_id")):
            return
        
        run_id = st.session_state.run_id
        
        # Fetch and update logs
        new_entries = TaskManager.fetch_logs(run_id)
        SessionStateManager.update_logs(new_entries)
        
        # Display logs
        with st.chat_message("assistant"):
            messages(st.session_state.all_logs, self.config["max_iter"])
        
        # Check task status
        if TaskManager.check_status(run_id):
            st.success(SUCCESS_MESSAGE)
            st.session_state.task_running = False
        else:
            time.sleep(0.1)
            st.rerun()
    
    def run(self):
        """Main application loop"""
        # Render chat history
        UIComponents.render_chat_history()
        
        # Handle user input
        submission = st.chat_input(
            placeholder="Type optional prompt, or drag & drop your data files/ZIP here",
            accept_file="multiple",
            key="u_input",
            max_chars=10000,
        )
        
        if submission is not None:
            self.handle_user_input(submission)
        
        # Monitor running task
        self.monitor_task()


# ==================== Entry Point ====================
def main():
    """Application entry point"""
    app = AutoMLAgentApp()
    app.run()


if __name__ == "__main__":
    main()