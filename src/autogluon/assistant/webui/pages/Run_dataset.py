# src/autogluon/assistant/webui/pages/Run_dataset.py

import copy
import uuid
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass, field

import streamlit as st

from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.webui.log_processor import messages, process_logs, render_task_logs
from autogluon.assistant.webui.result_manager import render_task_results
from autogluon.assistant.constants import INITIAL_STAGE, SUCCESS_MESSAGE, API_URL


# ==================== Constants ====================
PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

VERBOSITY_MAP = {
    "DETAIL": "3",
    "INFO": "2", 
    "BRIEF": "1",
}


# ==================== Data Classes ====================
@dataclass
class Message:
    """聊天消息"""
    role: str
    type: str
    content: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def text(cls, text: str, role: str = "assistant") -> "Message":
        return cls(role=role, type="text", content={"text": text})
    
    @classmethod
    def user_summary(cls, summary: str) -> "Message":
        return cls(role="user", type="user_summary", content={"summary": summary})
    
    @classmethod
    def command(cls, command: str) -> "Message":
        return cls(role="assistant", type="command", content={"command": command})
    
    @classmethod
    def task_log(cls, run_id: str, phase_states: Dict, max_iter: int, output_dir: Optional[str] = None) -> "Message":
        return cls(
            role="assistant", 
            type="task_log",
            content={
                "run_id": run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "phase_states": phase_states,
                "max_iter": max_iter,
                "output_dir": output_dir  # Add output_dir to message content
            }
        )
    
    @classmethod
    def task_results(cls, run_id: str, output_dir: str) -> "Message":
        return cls(
            role="assistant",
            type="task_results",
            content={
                "run_id": run_id,
                "output_dir": output_dir
            }
        )


@dataclass 
class TaskConfig:
    """任务配置"""
    uploaded_config: Any
    max_iter: int
    log_verbosity: str


# ==================== Session State ====================
class SessionState:
    """会话状态管理器"""
    
    @staticmethod
    def init():
        """初始化会话状态"""
        defaults = {
            "user_session_id": uuid.uuid4().hex,
            "messages": [Message.text("Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start.")],
            "data_src": None,
            "task_running": False,
            "run_id": None,
            "current_task_logs": [],
            "running_config": None,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def start_task(run_id: str, config: TaskConfig):
        """开始新任务"""
        st.session_state.task_running = True
        st.session_state.run_id = run_id
        st.session_state.current_task_logs = []
        st.session_state.running_config = config
        
        # 清理旧的日志处理器
        SessionState._cleanup_processors()
    
    @staticmethod
    def finish_task():
        """结束任务"""
        st.session_state.task_running = False
        st.session_state.running_config = None
        st.session_state.current_task_logs = []
        
        # 清理当前任务的处理器
        if st.session_state.run_id:
            processor_key = f"log_processor_{st.session_state.run_id}"
            if processor_key in st.session_state:
                del st.session_state[processor_key]
    
    @staticmethod
    def add_message(message: Message):
        """添加消息"""
        st.session_state.messages.append(message)
    
    @staticmethod
    def _cleanup_processors():
        """清理旧的日志处理器"""
        keys_to_delete = [k for k in st.session_state if k.startswith("log_processor_")]
        for key in keys_to_delete:
            del st.session_state[key]


# ==================== Backend API ====================
class BackendAPI:
    """后端API通信"""
    
    @staticmethod
    def start_task(data_src: str, config_path: str, user_prompt: str, config: TaskConfig) -> str:
        """启动任务"""
        payload = {
            "data_src": data_src,
            "config_path": config_path,
            "max_iter": config.max_iter,
            "init_prompt": user_prompt or None,
            "control": False,  # Always false now
            "verbosity": VERBOSITY_MAP[config.log_verbosity],
        }
        
        response = requests.post(f"{API_URL}/run", json=payload)
        return response.json()["run_id"]
    
    @staticmethod
    def fetch_logs(run_id: str) -> List[Dict]:
        """获取日志"""
        response = requests.get(f"{API_URL}/logs", params={"run_id": run_id})
        return response.json().get("lines", [])
    
    @staticmethod
    def check_status(run_id: str) -> bool:
        """检查任务状态"""
        response = requests.get(f"{API_URL}/status", params={"run_id": run_id})
        return response.json().get("finished", False)
    
    @staticmethod
    def cancel_task(run_id: str) -> bool:
        """取消任务"""
        try:
            response = requests.post(f"{API_URL}/cancel", json={"run_id": run_id})
            return response.json().get("cancelled", False)
        except:
            return False


# ==================== UI Components ====================
class UI:
    """UI组件"""
    
    @staticmethod
    def setup_page():
        """设置页面"""
        st.set_page_config(page_title="AutoMLAgent Chat", layout="wide")
    
    @staticmethod
    def render_sidebar() -> TaskConfig:
        """渲染侧边栏"""
        with st.sidebar:
            with st.expander("⚙️ Settings", expanded=False):
                config = TaskConfig(
                    uploaded_config=st.file_uploader(
                        "Config file (optional)",
                        type=['yaml', 'yml'],
                        key="config_uploader",
                        help="Upload a custom YAML config file. If not provided, default config will be used."
                    ),
                    max_iter=st.number_input("Max iterations", min_value=1, max_value=20, value=5, key="max_iterations"),
                    log_verbosity=st.select_slider(
                        "Log verbosity",
                        options=["BRIEF", "INFO", "DETAIL"],
                        value="BRIEF",
                        key="log_verbosity",
                    )
                )
            
            # 历史管理
            task_count = sum(1 for msg in st.session_state.messages if msg.type == "task_log")
            if task_count > 0:
                st.markdown(f"### 📋 Task History ({task_count} tasks)")
                if st.button("🗑️ Clear All History"):
                    st.session_state.messages = [Message.text("Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start.")]
                    st.rerun()
                    
        return config
    
    @staticmethod
    @st.fragment
    def render_single_message(msg):
        """Render a single message as a fragment to isolate interactions"""
        if msg.type == "text":
            st.write(msg.content["text"])
        elif msg.type == "user_summary":
            st.markdown(msg.content["summary"])
        elif msg.type == "command":
            st.code(msg.content["command"], language="bash")
        elif msg.type == "task_log":
            content = msg.content
            st.caption(f"ID: {content['run_id'][:8]}... | Completed: {content['timestamp']}")
            render_task_logs(
                content["phase_states"],
                content["max_iter"],
                show_progress=False
            )
        elif msg.type == "task_results":
            # Render the result manager for completed tasks
            content = msg.content
            if "output_dir" in content and content["output_dir"]:
                from autogluon.assistant.webui.result_manager import ResultManager
                manager = ResultManager(content["output_dir"])
                manager.render()
    
    @staticmethod
    def render_messages():
        """渲染消息历史"""
        for msg in st.session_state.messages:
            with st.chat_message(msg.role):
                UI.render_single_message(msg)
    
    @staticmethod
    def format_user_summary(files: List[str], config: TaskConfig, prompt: str, config_file: str) -> str:
        """格式化用户输入摘要"""
        parts = [
            "📂 **Uploaded files:**",
            "\n".join(f"- {f}" for f in files) if files else "- (none)",
            "\n⚙️ **Settings:**\n",
            f"- Config file: {config_file}",
            f"- Max iterations: {config.max_iter}",
            f"- Log verbosity: {config.log_verbosity}",
            "\n✏️ **Initial prompt:**\n",
            f"> {prompt or '(none)'}"
        ]
        return "\n".join(parts)


# ==================== Task Manager ====================
class TaskManager:
    """任务管理器"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
    
    def handle_submission(self, submission):
        """处理用户提交"""
        files = submission.files or []
        user_text = submission.text.strip() if submission.text else ""
        
        if not files:
            SessionState.add_message(Message.text("⚠️ No data files provided. Please drag and drop your data files or ZIP."))
            st.rerun()
            return
        
        # 处理文件
        data_folder = handle_uploaded_files(files)
        st.session_state.data_src = data_folder
        
        # 保存配置文件
        config_path = self._save_config(data_folder)
        config_name = self.config.uploaded_config.name if self.config.uploaded_config else "default.yaml"
        
        # 添加用户摘要
        summary = UI.format_user_summary(
            [f.name for f in files],
            self.config,
            user_text,
            config_name
        )
        SessionState.add_message(Message.user_summary(summary))
        
        # 启动任务
        self._start_task(data_folder, config_path, user_text)
    
    def handle_cancel_request(self):
        """处理取消请求"""
        run_id = st.session_state.run_id
        if not run_id:
            return
        
        # 显示用户的取消命令
        SessionState.add_message(Message.text("cancel", role="user"))
        
        # 尝试取消任务
        if BackendAPI.cancel_task(run_id):
            SessionState.add_message(Message.text(f"🛑 Task {run_id[:8]}... has been cancelled."))
            # 保存当前已有的日志
            if st.session_state.current_task_logs:
                processed = process_logs(
                    st.session_state.current_task_logs,
                    st.session_state.running_config.max_iter
                )
                
                # Extract output directory if available
                output_dir = self._extract_output_dir(processed["phase_states"])
                
                SessionState.add_message(
                    Message.task_log(
                        st.session_state.run_id,
                        processed["phase_states"],
                        st.session_state.running_config.max_iter,
                        output_dir
                    )
                )
                
                # Add task results message if output directory found
                if output_dir:
                    SessionState.add_message(
                        Message.task_results(st.session_state.run_id, output_dir)
                    )
                    
            SessionState.finish_task()
        else:
            SessionState.add_message(Message.text("❌ Failed to cancel the task."))
        
        st.rerun()
    
    @st.fragment(run_every=0.5)
    def render_running_task(self):
        """Render the currently running task as an isolated fragment"""
        if not st.session_state.task_running or not st.session_state.run_id:
            return
        
        run_id = st.session_state.run_id
        config = st.session_state.running_config
        
        if not config:
            st.error("Running configuration not found!")
            return
        
        # 获取新日志
        new_logs = BackendAPI.fetch_logs(run_id)
        st.session_state.current_task_logs.extend(new_logs)
        
        # 显示运行中的任务
        with st.chat_message("assistant"):
            st.markdown(f"### Current Task")
            st.caption(f"ID: {run_id[:8]}... | Type 'cancel' to stop the task")
            messages(st.session_state.current_task_logs, config.max_iter)
        
        # 检查是否完成
        if BackendAPI.check_status(run_id):
            self._complete_task()
            st.rerun()  # Rerun once to update the UI after completion
    
    def monitor_running_task(self):
        """监控运行中的任务"""
        if st.session_state.task_running:
            self.render_running_task()
    
    def _save_config(self, data_folder: str) -> str:
        """保存配置文件"""
        if self.config.uploaded_config:
            config_path = Path(data_folder) / self.config.uploaded_config.name
            with open(config_path, "wb") as f:
                f.write(self.config.uploaded_config.getbuffer())
            return str(config_path)
        return str(DEFAULT_CONFIG_PATH)
    
    def _start_task(self, data_folder: str, config_path: str, user_prompt: str):
        """启动任务"""
        # 构建命令
        cmd_parts = [
            "mlzero",
            "-i", data_folder,
            "-n", str(self.config.max_iter),
            "-v", VERBOSITY_MAP[self.config.log_verbosity],
            "-c", config_path,
        ]
        
        if user_prompt:
            cmd_parts.extend(["-u", user_prompt])
        # Removed --need-user-input flag since control is always False
        
        # 显示命令
        command_str = f"[{datetime.now().strftime('%H:%M:%S')}] Running AutoMLAgent: {' '.join(cmd_parts)}"
        SessionState.add_message(Message.command(command_str))
        
        # 启动任务
        run_id = BackendAPI.start_task(data_folder, config_path, user_prompt, self.config)
        SessionState.start_task(run_id, self.config)
        st.rerun()
    
    def _extract_output_dir(self, phase_states: Dict) -> Optional[str]:
        """Extract output directory from phase states"""
        output_phase = phase_states.get("Output", {})
        logs = output_phase.get("logs", [])
        
        for log in reversed(logs):
            import re
            # Look for "output saved in" pattern and extract the path
            match = re.search(r'output saved in\s+([^\s]+)', log)
            if match:
                output_dir = match.group(1).strip()
                # Remove any trailing punctuation
                output_dir = output_dir.rstrip('.,;:')
                return output_dir
        return None
    
    def _complete_task(self):
        """完成任务"""
        # 保存任务日志
        if st.session_state.current_task_logs:
            processed = process_logs(
                st.session_state.current_task_logs,
                st.session_state.running_config.max_iter
            )
            
            # Extract output directory
            output_dir = self._extract_output_dir(processed["phase_states"])
            
            SessionState.add_message(
                Message.task_log(
                    st.session_state.run_id,
                    processed["phase_states"],
                    st.session_state.running_config.max_iter,
                    output_dir
                )
            )
            
            # Add task results message if output directory found
            if output_dir:
                SessionState.add_message(
                    Message.task_results(st.session_state.run_id, output_dir)
                )
        
        st.success(SUCCESS_MESSAGE)
        SessionState.finish_task()


# ==================== Main App ====================
class AutoMLAgentApp:
    """主应用"""
    
    def __init__(self):
        UI.setup_page()
        SessionState.init()
        self.config = UI.render_sidebar()
        self.task_manager = TaskManager(self.config)
    
    def run(self):
        """运行应用"""
        # 渲染历史消息
        UI.render_messages()
        
        # 处理用户输入
        submission = st.chat_input(
            placeholder="Type optional prompt, or drag & drop your data files/ZIP here",
            accept_file="multiple",
            key="u_input",
            max_chars=10000,
        )
        
        if submission:
            # 如果任务正在运行
            if st.session_state.task_running:
                # 检查是否是取消命令
                if submission.text and submission.text.strip().lower() == "cancel":
                    self.task_manager.handle_cancel_request()
                else:
                    # 显示提示信息
                    SessionState.add_message(
                        Message.text(
                            "⚠️ A task is currently running. Type 'cancel' to stop it, or wait for it to complete.",
                            role="user"
                        )
                    )
                    st.rerun()
            else:
                # 没有任务运行，正常处理提交
                self.task_manager.handle_submission(submission)
        
        # 监控运行中的任务
        self.task_manager.monitor_running_task()


def main():
    """入口点"""
    app = AutoMLAgentApp()
    app.run()


if __name__ == "__main__":
    main()