import copy
import os
import subprocess
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
import time

import streamlit as st

from autogluon.assistant.webui.utils.utils import get_user_data_dir, save_and_extract_zip
from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.webui.log_processor import show_log_line, messages
from autogluon.assistant.constants import INITIAL_STAGE, SUCCESS_MESSAGE, API_URL
import requests

PACKAGE_ROOT = Path(__file__).parents[2]  # pages -> webui -> assistant
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

# -------------------- page config & style --------------------
st.set_page_config(page_title="AutoMLAgent Chat", layout="wide")
st.markdown(
    """
    <style>
      /* your existing CSS… */
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- session state init --------------------
for key, default in [
    ("user_session_id", uuid.uuid4().hex),
    ("messages", [{"role":"assistant","text":"Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start."}]),
    ("data_src", None),
    ("task_running", False),
    ("task_canceled", False),
    ("process", None),
    ("return_code", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.task_canceled = False
    st.session_state.current_stage = None
    st.session_state.stage_container = copy.deepcopy(INITIAL_STAGE)
    st.session_state.stage_status = {}
    st.session_state.show_remaining_time = False
    st.session_state.start_model_train_time = None

# -------------------- sidebar settings --------------------
with st.sidebar:
    with st.expander("⚙️ Advanced Settings", expanded=False):
        out_dir = st.text_input("Output directory", value="", key="output_dir")
        config_path = st.text_input(
            "Config file",
            value=str(DEFAULT_CONFIG_PATH),
            help="Path to YAML config file (only default.yaml is provided)",
            key="config_path",
        )
        max_iter = st.number_input("Max iterations", min_value=1, max_value=20, value=5, key="max_iterations")
        init_prompt = st.text_area("Initial prompt (optional)", key="initial_prompt", height=80)
        control = st.checkbox("Manual prompts between iterations", key="control_prompts")
        extract_check = st.checkbox("Extract uploaded ZIP", key="extract_check")
        extract_dir = st.text_input(
            "Extraction dir",
            placeholder="extract_to/",
            key="extract_dir",
            disabled=not extract_check,
        )
        VERBOSITY_MAP = {
            "MODEL_INFO": 3,
            "DETAILED_INFO": 2,
            "BRIEF_INFO": 1,
        }
        log_verbosity = st.select_slider(
            "Log verbosity",
            options=list(VERBOSITY_MAP.keys()),
            value="BRIEF_INFO",
            key="log_verbosity",
        )

# -------------------- render chat history --------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])

# -------------------- unified chat_input --------------------
submission = st.chat_input(
    placeholder="Type optional prompt, or drag & drop your data files/ZIP here",
    accept_file="multiple",
    key="u_input",
    max_chars=10000,
)
if submission is not None:
    # submission.text: str, submission.files: list
    prompt_text = submission.text or ""
    files = submission.files or []

    # 1) 处理文件上传
    if files:
        folder = handle_uploaded_files(files)
        st.session_state.data_src = folder
        file_names = [f.name for f in files]
    else:
        file_names = []

    # 2) Validate we have data
    if not st.session_state.data_src:
        err = "⚠️ No data detected. Please drag & drop your folder or ZIP first."
        st.session_state.messages.append({"role": "assistant", "text": err})
        with st.chat_message("assistant"):
            st.write(err)
        st.rerun()

    user_summary = "📂 **Uploaded files:**\n"
    if file_names:
        user_summary += "\n".join(f"- {n}" for n in file_names) + "\n"
    else:
        user_summary += "- (none)\n"

    user_summary += "\n⚙️ **Settings:**\n\n"  # <-- blank line after header
    user_summary += "\n".join([
        f"- Output directory: `{out_dir or '(default runs/)'}`",
        f"- Config file: `{config_path}`",
        f"- Max iterations: `{max_iter}`",
        f"- Manual prompts: `{control}`",
        f"- Extract ZIP: `{extract_check}`{f' → `{extract_dir}`' if extract_check else ''}",
        f"- Log verbosity: `{log_verbosity}`",
    ])
    user_summary += "\n\n✏️ **Initial prompt:**\n\n"  # blank line for blockquote
    user_summary += f"> {init_prompt or '(none)'}"

    st.session_state.messages.append({"role": "user", "text": user_summary})
    with st.chat_message("user"):
        st.markdown(user_summary)

    # 4) 启动 mlzero 子进程
    toggle_running_state()
    t0 = datetime.now().strftime("%H:%M:%S")
    cmd = [
        "mlzero",
        "-i", st.session_state.data_src,
        "-n", str(max_iter),
        "-v", str(VERBOSITY_MAP[log_verbosity]), 
        "-c", config_path,
    ]
    if out_dir:     cmd += ["-o", out_dir]
    if init_prompt: cmd += ["-u", init_prompt]
    if control:     cmd += ["--need-user-input"]
    if extract_check and extract_dir:
        cmd += ["-e", extract_dir]

    start_msg = f"[{t0}] Running AutoMLAgent: `{' '.join(cmd)}`"
    st.session_state.messages.append({"role":"assistant","text":start_msg})
    with st.chat_message("assistant"):
        st.code(start_msg, language="bash")

    # === POST to backend to start run ===
    payload = {
        "data_src":    st.session_state.data_src,
        "out_dir":     out_dir,
        "config_path": config_path,
        "max_iter":    max_iter,
        "init_prompt": init_prompt,
        "control":     control,
        "extract_dir": extract_dir if extract_check else None,
        "verbosity":   VERBOSITY_MAP[log_verbosity],
    }
    resp = requests.post(f"{API_URL}/run", json=payload).json()
    st.session_state.run_id = resp["run_id"]
    st.session_state.task_running = True
    st.rerun()

# -------------------- log streaming & final state --------------------
if st.session_state.task_running and st.session_state.get("run_id"):
    run_id = st.session_state.run_id

    # 初始化累计日志列表
    if "all_logs" not in st.session_state:
        st.session_state.all_logs = []

    # 1) 拉取新日志
    new_lines = requests.get(f"{API_URL}/logs", params={"run_id": run_id}).json().get("lines", [])
    # 累加
    st.session_state.all_logs.extend(new_lines)

    # 2) 把所有日志一次性渲染到一个“assistant”气泡里
    chat = st.chat_message("assistant")
    with chat:
        for line in st.session_state.all_logs:
            show_log_line(line)

    # 3) 检查任务是否完成
    status = requests.get(f"{API_URL}/status", params={"run_id": run_id}).json()
    if status.get("finished", False):
        # 已完成：成功提示，清理状态
        st.success(SUCCESS_MESSAGE)
        st.session_state.task_running = False
    else:
        # 未完成：稍等一下再重跑，继续拉取新日志
        time.sleep(0.1)
        st.rerun()


# if st.session_state.task_running and st.session_state.process:
#     # stream the entire run into one assistant bubble
#     messages(st.session_state.process, total_iterations=max_iter)

# elif st.session_state.process:
#     # 有这个process，但没有在running，说明前面的运行完了
#     st.session_state.return_code = st.session_state.process.returncode
#     status = SUCCESS_MESSAGE if st.session_state.return_code == 0 else "❌ Task failed."
#     with st.chat_message("assistant"):
#         st.write(status)
