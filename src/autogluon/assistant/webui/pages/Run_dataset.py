import copy
import uuid
import time
import requests
from datetime import datetime
from pathlib import Path
import re

import streamlit as st
from autogluon.assistant.webui.file_uploader import handle_uploaded_files
from autogluon.assistant.constants import INITIAL_STAGE, SUCCESS_MESSAGE, API_URL

PACKAGE_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"

# ── Streamlit 页面布局 & 样式 ────────────────────────────────────────────────
st.set_page_config(page_title="AutoMLAgent Chat", layout="wide")
st.markdown("<style>/* 你的自定义 CSS… */</style>", unsafe_allow_html=True)

# ── 初始化 Session State ───────────────────────────────────────────────────
for key, default in [
    ("user_session_id", uuid.uuid4().hex),
    ("messages", [{"role": "assistant", "text": "Hello! Drag your data (folder or ZIP) into the chat box below, then press ENTER to start."}]),
    ("data_src", None),
    ("task_running", False),
    ("run_id", None),
    ("all_logs", []),
    ("current_stage_index", None),      # 0=Reading, 1..N=Iteration1..N, N+1=Output
    ("stage_containers", {}),           # 动态存放已创建的 st.status 容器
    ("progress_bar", None),             # 整个流程的大进度条
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── “任务开始” 时的初始化函数 ────────────────────────────────────────────────
def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.current_stage_index = None
    st.session_state.all_logs = []
    # 清空此前可能创建过的阶段容器
    st.session_state.stage_containers = {}  
    st.session_state.progress_bar = None      # 进度条也置空，下次会重新创建

# ── Sidebar 设置 ─────────────────────────────────────────────────────────────
with st.sidebar:
    with st.expander("⚙️ Advanced Settings", expanded=False):
        out_dir = st.text_input("Output directory", value="", key="output_dir")
        config_path = st.text_input("Config file", value=str(DEFAULT_CONFIG_PATH), key="config_path")
        max_iter = st.number_input("Max iterations", min_value=1, max_value=20, value=5, key="max_iterations")
        init_prompt = st.text_area("Initial prompt (optional)", key="initial_prompt", height=80)
        control = st.checkbox("Manual prompts between iterations", key="control_prompts")
        extract_check = st.checkbox("Extract uploaded ZIP", key="extract_check")
        extract_dir = st.text_input(
            "Extraction dir", placeholder="extract_to/", key="extract_dir", disabled=not extract_check
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

# ── 渲染历史聊天记录 ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])

# ── 统一的 Chat Input ─────────────────────────────────────────────────────────
submission = st.chat_input(
    placeholder="Type optional prompt, or drag & drop your data/ZIP here",
    accept_file="multiple",
    key="u_input",
    max_chars=10000,
)

# 只有在“新的 submission”且“当前没有任务在跑”时，才发 /api/run
if submission is not None and not st.session_state.task_running:
    prompt_text = submission.text or ""
    files = submission.files or []

    # 1) 处理文件上传（若有）
    if files:
        folder = handle_uploaded_files(files)
        st.session_state.data_src = folder
        file_names = [f.name for f in files]
    else:
        file_names = []

    # 2) 验证是否有数据
    if not st.session_state.data_src:
        err = "⚠️ No data detected. Please drag & drop your folder or ZIP first."
        st.session_state.messages.append({"role": "assistant", "text": err})
        with st.chat_message("assistant"):
            st.write(err)
        st.rerun()

    # 3) 拼接用户的 summary 并展示
    user_summary = "📂 **Uploaded files:**\n"
    if file_names:
        user_summary += "\n".join(f"- {n}" for n in file_names) + "\n"
    else:
        user_summary += "- (none)\n"

    user_summary += "\n⚙️ **Settings:**\n\n"
    user_summary += "\n".join([
        f"- Output directory: `{out_dir or '(default runs/)'}`",
        f"- Config file: `{config_path}`",
        f"- Max iterations: `{max_iter}`",
        f"- Manual prompts: `{control}`",
        f"- Extract ZIP: `{extract_check}`{(f' → `{extract_dir}`') if extract_check else ''}",
        f"- Log verbosity: `{log_verbosity}`",
    ])
    user_summary += "\n\n✏️ **Initial prompt:**\n\n"
    user_summary += f"> {init_prompt or '(none)'}"

    st.session_state.messages.append({"role": "user", "text": user_summary})
    with st.chat_message("user"):
        st.markdown(user_summary)

    # —— 真正启动 mlzero 子进程，并记录 run_id、标记 task_running=True
    toggle_running_state()
    t0 = datetime.now().strftime("%H:%M:%S")
    cmd = [
        "mlzero",
        "-i", st.session_state.data_src,
        "-n", str(max_iter),
        "-v", VERBOSITY_MAP[log_verbosity],
        "-c", config_path,
    ]
    if out_dir:
        cmd += ["-o", out_dir]
    if init_prompt:
        cmd += ["-u", init_prompt]
    if control:
        cmd += ["--need-user-input"]
    if extract_check and extract_dir:
        cmd += ["-e", extract_dir]

    start_msg = f"[{t0}] Running AutoMLAgent: {' '.join(cmd)}"
    st.session_state.messages.append({"role": "assistant", "text": start_msg})
    with st.chat_message("assistant"):
        st.code(start_msg, language="bash")

    payload = {
        "data_src": st.session_state.data_src,
        "out_dir": out_dir,
        "config_path": config_path,
        "max_iter": max_iter,
        "init_prompt": init_prompt,
        "control": control,
        "extract_dir": (extract_dir if extract_check else None),
        "verbosity": VERBOSITY_MAP[log_verbosity],
    }
    resp = requests.post(f"{API_URL}/run", json=payload).json()
    st.session_state.run_id = resp["run_id"]
    st.session_state.task_running = True

    # 发完后立刻 rerun，让下面的 “log streaming“ 分支跑起来
    st.rerun()

# ── log streaming & 渲染 ────────────────────────────────────────────────────────
if st.session_state.task_running and st.session_state.run_id:
    run_id = st.session_state.run_id

    # 如果还没创建过大进度条，就先创建一次
    if st.session_state.progress_bar is None:
        total_stages = max_iter + 2
        st.session_state.progress_bar = st.progress(0.0, text="Starting…")

    # 1) 拉取「新增的」日志条目（已由后端 parse 为 {"level":..., "text":...} 形式）
    resp = requests.get(f"{API_URL}/logs", params={"run_id": run_id})
    new_entries = resp.json().get("lines", [])

    # 2) 遍历这些新增条目，按阶段写入对应的 st.status 容器
    total_stages = max_iter + 2
    for entry in new_entries:
        text = entry.get("text", "")

        # ——— Reading 阶段开始 —————————————————————————————————————
        if "DataPerceptionAgent: beginning to scan data folder and group similar files." in text:
            # 如果此时当前阶段并非 Reading，就切换到 Reading
            if st.session_state.current_stage_index != 0:
                st.session_state.current_stage_index = 0
                st.session_state.progress_bar.progress(0 / total_stages, text="Reading")
                # 第一次进入 Reading 时再创建它的 StatusContainer
                st.session_state.stage_containers["Reading"] = st.status("Reading", expanded=True)
            # 一旦进入 Reading，就把行写入 Reading 容器
            st.session_state.stage_containers["Reading"].write(text)
            continue

        # ——— Reading 阶段结束 信号（新版日志里以 ToolSelectorAgent: selected… 作为结束） —————————————————
        if "ToolSelectorAgent: selected" in text:
            # 如果 current_stage_index 确实是 0（Reading），先把结束行写进 Reading，然后 mark complete
            if st.session_state.current_stage_index == 0:
                st.session_state.stage_containers["Reading"].write(text)
                st.session_state.stage_containers["Reading"].update(state="complete")
                st.session_state.current_stage_index = None
            else:
                # 若之前没记录 Reading 开始，也兜底创建一次再 complete
                if "Reading" not in st.session_state.stage_containers:
                    st.session_state.stage_containers["Reading"] = st.status("Reading", expanded=True)
                st.session_state.stage_containers["Reading"].write(text)
                st.session_state.stage_containers["Reading"].update(state="complete")
            continue

        # ——— Iteration 阶段开始（匹配 “Starting iteration X!”） —————————————————————————
        m_iter = re.search(r"Starting iteration (\d+)!", text)
        if m_iter:
            idx = int(m_iter.group(1)) + 1  # “0”→第1次
            if st.session_state.current_stage_index != idx:
                st.session_state.current_stage_index = idx
                phase_name = f"Iteration {idx}"
                st.session_state.progress_bar.progress(idx / total_stages, text=phase_name)
                st.session_state.stage_containers[phase_name] = st.status(phase_name, expanded=True)
            st.session_state.stage_containers[f"Iteration {idx}"].write(text)
            continue

        # ——— Iteration 阶段结束（匹配 “Code generation failed…” 或 “Code generation successful…”） —————————
        if re.search(r"Code generation (failed|successful)", text):
            idx = st.session_state.current_stage_index
            if idx and 1 <= idx <= max_iter:
                phase_name = f"Iteration {idx}"
                st.session_state.stage_containers[phase_name].write(text)
                st.session_state.stage_containers[phase_name].update(state="complete")
                st.session_state.current_stage_index = None
            else:
                # 若 current_stage_index 不在 1..max_iter 范围之内，换个“Iteration (Unknown)”容器来写
                fallback = "Iteration (Unknown)"
                if fallback not in st.session_state.stage_containers:
                    st.session_state.stage_containers[fallback] = st.status(fallback, expanded=True)
                st.session_state.stage_containers[fallback].write(text)
                st.session_state.stage_containers[fallback].update(state="complete")
            continue

        # ——— Output 阶段开始（匹配 “Total tokens”） —————————————————————————————————
        if "Total tokens" in text:
            if st.session_state.current_stage_index != max_iter + 1:
                st.session_state.current_stage_index = max_iter + 1
                st.session_state.progress_bar.progress((max_iter + 1) / total_stages, text="Output")
                st.session_state.stage_containers["Output"] = st.status("Output", expanded=True)
            st.session_state.stage_containers["Output"].write(text)
            continue

        # ——— Output 阶段结束（匹配 “output saved in”） ————————————————————————————————
        if "output saved in" in text:
            if st.session_state.current_stage_index == max_iter + 1:
                st.session_state.stage_containers["Output"].write(text)
                st.session_state.stage_containers["Output"].update(state="complete")
                st.session_state.progress_bar.progress(1.0, text="Complete")
                st.session_state.current_stage_index = None
            else:
                # 如果 current_stage_index 不是 max_iter+1，也兜底
                if "Output" not in st.session_state.stage_containers:
                    st.session_state.stage_containers["Output"] = st.status("Output", expanded=True)
                st.session_state.stage_containers["Output"].write(text)
                st.session_state.stage_containers["Output"].update(state="complete")
                st.session_state.progress_bar.progress(1.0, text="Complete")
            continue

        # ——— 如果当前正处于某阶段内部（current_stage_index is not None），将这一行追加进去 ——————————
        if st.session_state.current_stage_index is not None:
            idx = st.session_state.current_stage_index
            if idx == 0:
                st.session_state.stage_containers["Reading"].write(text)
            elif 1 <= idx <= max_iter:
                st.session_state.stage_containers[f"Iteration {idx}"].write(text)
            else:
                st.session_state.stage_containers["Output"].write(text)
        else:
            # 否则视作“孤立日志”，直接输出
            st.write(text)

    # 3) 查询是否结束
    status = requests.get(f"{API_URL}/status", params={"run_id": run_id}).json()
    if status.get("finished", False):
        st.success(SUCCESS_MESSAGE)
        st.session_state.task_running = False
    else:
        # 若没结束，等 0.5s 再拉一次
        time.sleep(0.5)
        st.rerun()
