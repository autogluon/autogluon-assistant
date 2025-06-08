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
from autogluon.assistant.webui.log_processor import messages, process_logs, render_task_logs
from autogluon.assistant.constants import INITIAL_STAGE, SUCCESS_MESSAGE, API_URL


# ==================== Constants ====================
PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

VERBOSITY_MAP = {
    "DETAIL": "3",
    "INFO": "2", 
    "BRIEF": "1",
}

# Message types
MSG_TYPE_TEXT = "text"
MSG_TYPE_USER_SUMMARY = "user_summary"
MSG_TYPE_COMMAND = "command"
MSG_TYPE_TASK_LOG = "task_log"

INITIAL_WEBPAGE_SESSION = {
    "user_session_id": lambda: uuid.uuid4().hex,
    "messages": lambda: [{
        "role": "assistant", 
        "type": MSG_TYPE_TEXT,
        "text": "Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start."
    }],
    "data_src": None,
    "task_running": False,
    "run_id": None,
    "current_task_logs": lambda: [],  # 当前任务的日志
    "current_stage": None,
    "stage_container": lambda: copy.deepcopy(INITIAL_STAGE),
    "stage_status": lambda: {},
    "running_config": None,
}


# ==================== Session State Manager ====================
class SessionStateManager:
    """Centralized session state management"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        for key, default in INITIAL_WEBPAGE_SESSION.items():
            if key not in st.session_state:
                st.session_state[key] = default() if callable(default) else default
    
    @staticmethod
    def reset_task_state():
        """Reset task-related state when starting a new task"""
        st.session_state.task_running = True
        st.session_state.current_stage = None
        st.session_state.stage_container = copy.deepcopy(INITIAL_STAGE)
        st.session_state.stage_status = {}
        st.session_state.current_task_logs = []
        
        # 清理所有旧的 log processor states
        keys_to_delete = []
        for key in st.session_state:
            if key.startswith("log_processor_state_"):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del st.session_state[key]
        
        # 兼容旧版本
        if "log_processor_state" in st.session_state:
            del st.session_state.log_processor_state

    @staticmethod
    def save_running_config(config: Dict[str, Any]):
        """Save the configuration at the time of task start"""
        st.session_state.running_config = copy.deepcopy(config)

    @staticmethod
    def get_running_config() -> Optional[Dict[str, Any]]:
        """Get the saved running configuration"""
        return st.session_state.get("running_config")

    @staticmethod
    def add_message(role: str, msg_type: str = MSG_TYPE_TEXT, **kwargs):
        """Add a message to the chat history
        
        Args:
            role: 'user' or 'assistant'
            msg_type: Type of message
            **kwargs: Additional data for the message
        """
        message = {"role": role, "type": msg_type}
        message.update(kwargs)
        st.session_state.messages.append(message)
    
    @staticmethod
    def save_completed_task():
        """Save completed task logs to messages"""
        if not st.session_state.current_task_logs:
            return
            
        running_config = st.session_state.running_config
        if not running_config:
            return
            
        # 处理日志，提取阶段信息
        processed_state = process_logs(
            st.session_state.current_task_logs, 
            running_config["max_iter"]
        )
        
        # 添加任务日志消息
        SessionStateManager.add_message(
            role="assistant",
            msg_type=MSG_TYPE_TASK_LOG,
            run_id=st.session_state.run_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            phase_states=processed_state["phase_states"],
            max_iter=running_config["max_iter"]
        )
    
    @staticmethod
    def update_logs(new_entries: List[Dict]):
        """Update the current task log entries"""
        st.session_state.current_task_logs.extend(new_entries)


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
            with st.expander("⚙️ Settings", expanded=False):
                # Config file uploader
                uploaded_config = st.file_uploader(
                    "Config file (optional)",
                    type=['yaml', 'yml'],
                    key="config_uploader",
                    help="Upload a custom YAML config file. If not provided, default config will be used."
                )
                
                config = {
                    "uploaded_config": uploaded_config,
                    "max_iter": st.number_input("Max iterations", min_value=1, max_value=20, value=5, key="max_iterations"),
                    "control": st.checkbox("Manual prompts between iterations", key="control_prompts"),
                    "log_verbosity": st.select_slider(
                        "Log verbosity",
                        options=["BRIEF", "INFO", "DETAIL"],
                        value="BRIEF",
                        key="log_verbosity",
                    ),
                }

            # 清除历史按钮
            task_count = sum(1 for msg in st.session_state.messages if msg.get("type") == MSG_TYPE_TASK_LOG)
            if task_count > 0:
                st.markdown(f"### 📋 Task History ({task_count} tasks)")
                if st.button("🗑️ Clear All History"):
                    # 只保留初始消息
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "type": MSG_TYPE_TEXT,
                        "text": "Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start."
                    }]
                    st.rerun()
        return config
    
    @staticmethod
    def render_chat_history():
        """Render the chat message history"""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                msg_type = msg.get("type", MSG_TYPE_TEXT)
                
                if msg_type == MSG_TYPE_TEXT:
                    st.write(msg["text"])
                    
                elif msg_type == MSG_TYPE_USER_SUMMARY:
                    st.markdown(msg["summary"])
                    
                elif msg_type == MSG_TYPE_COMMAND:
                    st.code(msg["command"], language="bash")
                    
                elif msg_type == MSG_TYPE_TASK_LOG:
                    st.caption(f"ID: {msg['run_id'][:8]}... | Completed: {msg['timestamp']}")
                    # 渲染任务日志
                    render_task_logs(
                        msg["phase_states"],
                        msg["max_iter"],
                        show_progress=False  # 历史任务不显示进度条
                    )
    
    @staticmethod
    def format_user_summary(file_names: List[str], config: Dict[str, Any], user_prompt: str, config_used: str) -> str:
        """Format the user input summary"""
        summary_parts = ["📂 **Uploaded files:**"]
        
        if file_names:
            summary_parts.append("\n".join(f"- {n}" for n in file_names))
        else:
            summary_parts.append("- (none)")
        
        summary_parts.extend([
            "\n⚙️ **Settings:**\n",
            f"- Config file: {config_used}",
            f"- Max iterations: {config['max_iter']}",
            f"- Manual prompts: {config['control']}",
            f"- Log verbosity: {config['log_verbosity']}",
            "\n✏️ **Initial prompt:**\n",
            f"> {user_prompt or '(none)'}"
        ])
        
        return "\n".join(summary_parts)


# ==================== Task Manager ====================
class TaskManager:
    """Manages task execution and communication with backend"""
    
    @staticmethod
    def save_config_file(uploaded_config, data_folder: str) -> str:
        """Save uploaded config file to data folder and return its path
        
        Args:
            uploaded_config: Uploaded config file from st.file_uploader
            data_folder: Path to the data folder
            
        Returns:
            Path to the saved config file
        """
        if uploaded_config:
            # 保存上传的配置文件到数据文件夹
            config_path = Path(data_folder) / uploaded_config.name
            with open(config_path, "wb") as f:
                f.write(uploaded_config.getbuffer())
            return str(config_path)
        else:
            # 使用默认配置
            return str(DEFAULT_CONFIG_PATH)
    
    @staticmethod
    def build_command(config_path: str, user_prompt: str, config: Dict[str, Any]) -> List[str]:
        """Build the mlzero command from configuration"""
        cmd = [
            "mlzero",
            "-i", st.session_state.data_src,
            "-n", str(config["max_iter"]),
            "-v", VERBOSITY_MAP[config["log_verbosity"]], 
            "-c", config_path,  # 使用实际的配置文件路径
        ]
        
        # 不指定输出目录，使用默认
        # 不指定提取目录，如果需要提取会使用默认行为
        
        if user_prompt:
            cmd.extend(["-u", user_prompt])
        if config["control"]:
            cmd.append("--need-user-input")
        
        return cmd
    
    @staticmethod
    def start_task(config_path: str, user_prompt: str, config: Dict[str, Any]) -> str:
        """Start a new task via backend API"""
        payload = {
            "data_src": st.session_state.data_src,
            "config_path": config_path,
            "max_iter": config["max_iter"],
            "init_prompt": user_prompt or None,
            "control": config["control"],
            "verbosity": VERBOSITY_MAP[config["log_verbosity"]],
            # 不发送 out_dir 和 extract_dir，让后端使用默认值
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
        user_text = submission.text.strip() if submission.text else ""
        
        # 验证是否有文件
        if not files:
            self._show_error("⚠️ No data files provided. Please drag and drop your data files or ZIP.")
            return
        
        # 处理上传的文件
        folder = handle_uploaded_files(files)
        st.session_state.data_src = folder
        file_names = [f.name for f in files]
        
        # 保存配置文件（如果有上传）
        config_path = TaskManager.save_config_file(self.config["uploaded_config"], folder)
        config_used = self.config["uploaded_config"].name if self.config["uploaded_config"] else "default.yaml"
        
        # Create and display user summary
        user_summary = UIComponents.format_user_summary(file_names, self.config, user_text, config_used)
        SessionStateManager.add_message(
            role="user",
            msg_type=MSG_TYPE_USER_SUMMARY,
            summary=user_summary
        )
        
        # Start the task with user prompt
        self._start_task(user_text, config_path)
    
    def _show_error(self, message: str):
        """Display an error message"""
        SessionStateManager.add_message(role="assistant", text=message)
        st.rerun()
    
    def _start_task(self, user_prompt: str, config_path: str):
        """Initialize and start a new task"""
        SessionStateManager.reset_task_state()
        SessionStateManager.save_running_config(self.config)
        
        # Build and display command
        cmd = TaskManager.build_command(config_path, user_prompt, self.config)
        t0 = datetime.now().strftime("%H:%M:%S")
        command_str = f"[{t0}] Running AutoMLAgent: {' '.join(cmd)}"
        
        SessionStateManager.add_message(
            role="assistant",
            msg_type=MSG_TYPE_COMMAND,
            command=command_str
        )
        
        # Start task via backend
        run_id = TaskManager.start_task(config_path, user_prompt, self.config)
        st.session_state.run_id = run_id
        st.session_state.task_running = True
        st.rerun()
    
    def monitor_task(self):
        """Monitor running task and display logs"""
        if not (st.session_state.task_running and st.session_state.get("run_id")):
            return
        
        run_id = st.session_state.run_id
        running_config = SessionStateManager.get_running_config()
        if not running_config:
            st.error("Running configuration not found!")
            return
        
        # Fetch and update logs
        new_entries = TaskManager.fetch_logs(run_id)
        SessionStateManager.update_logs(new_entries)
        
        # Display running task logs
        with st.chat_message("assistant"):
            st.markdown(f"### Current Task")
            st.caption(f"ID: {run_id[:8]}...")
            messages(st.session_state.current_task_logs, running_config["max_iter"])
        
        # Check task status
        if TaskManager.check_status(run_id):
            # 保存完成的任务到消息历史
            SessionStateManager.save_completed_task()
            
            st.success(SUCCESS_MESSAGE)
            st.session_state.task_running = False
            st.session_state.running_config = None
            st.session_state.current_task_logs = []
            
            # 清理当前任务的 log processor state
            current_state_key = f"log_processor_state_{run_id}"
            if current_state_key in st.session_state:
                del st.session_state[current_state_key]
                
            # 清理兼容的旧key
            if "log_processor_state" in st.session_state:
                del st.session_state.log_processor_state
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
            # 如果任务正在运行，不处理新的输入
            if st.session_state.get("task_running", False):
                # 输入会保留在 chat_input 中
                pass
            else:
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