import copy
import os
import subprocess
import uuid
# import zipfile
from datetime import datetime
from pathlib import Path
import time
import requests

import streamlit as st

# from autogluon.assistant.webui.utils.utils import get_user_data_dir, save_and_extract_zip
from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.webui.log_processor import messages
from autogluon.assistant.constants import INITIAL_STAGE, SUCCESS_MESSAGE, API_URL

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
    ("run_id", None),
    ("all_logs", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.current_stage = None
    st.session_state.stage_container = copy.deepcopy(INITIAL_STAGE)
    st.session_state.stage_status = {}
    st.session_state.all_logs = []

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
            "MODEL_INFO": "3",
            "DETAILED_INFO": "2",
            "BRIEF_INFO": "1",
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
        f"- Output directory: {out_dir or '(default runs/)'}",
        f"- Config file: {config_path}",
        f"- Max iterations: {max_iter}",
        f"- Manual prompts: {control}",
        f"- Extract ZIP: {extract_check}{f' → {extract_dir}' if extract_check else ''}",
        f"- Log verbosity: {log_verbosity}",
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
        "-v", VERBOSITY_MAP[log_verbosity], 
        "-c", config_path,
    ]
    if out_dir:     cmd += ["-o", out_dir]
    if init_prompt: cmd += ["-u", init_prompt]
    if control:     cmd += ["--need-user-input"]
    if extract_check and extract_dir:
        cmd += ["-e", extract_dir]

    start_msg = f"[{t0}] Running AutoMLAgent: {' '.join(cmd)}"
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

    # 1) 向后端 /api/logs?run_id=xxx 拉取“新日志条目” —— 它们已经是 {"level": ..., "text": ...} 形式
    resp = requests.get(f"{API_URL}/logs", params={"run_id": run_id})
    new_entries = resp.json().get("lines", [])

    # 2) 累加这些“字典条目”到 all_logs 中
    st.session_state.all_logs.extend(new_entries)

    # 3) 渲染出一个“assistant”气泡，把所有累计到的日志一次性送给 messages() 去绘制
    chat = st.chat_message("assistant")
    with chat:
        # 传入的是 list[dict]，以及用户选的 max_iter
        messages(st.session_state.all_logs, max_iter)

    # 4) 再询问 /api/status?run_id=xxx，看是否已经结束
    status = requests.get(f"{API_URL}/status", params={"run_id": run_id}).json()
    if status.get("finished", False):
        st.success(SUCCESS_MESSAGE)
        st.session_state.task_running = False
    else:
        # 任务还没结束 → 稍微等待一下，再刷新页面（再次跑到顶部重新执行流）
        time.sleep(0.1)
        st.rerun()