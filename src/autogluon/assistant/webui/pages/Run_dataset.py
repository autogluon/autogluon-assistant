import streamlit as st
from pathlib import Path
from datetime import datetime

# -------------------- 页面配置 --------------------
st.set_page_config(
    page_title="AutoMLAgent Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- 样式定制 --------------------
st.markdown(
    """
    <style>
    /* 隐藏默认 header/footer */
    #MainMenu, header, footer {visibility: hidden;}

    /* 侧边栏标题 */
    .sidebar .css-1d391kg {padding-top: 1rem;}
    .sidebar .css-1d391kg h2 {font-size: 1.3rem; font-weight: bold;}

    /* 聊天气泡 */
    .userBubble {
        background-color: #007bff;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 1rem 1rem 0.5rem 1rem;
        max-width: 60%;
        margin-left: auto;
        margin-bottom: 0.5rem;
    }
    .botBubble {
        background-color: #f1f0f0;
        color: #111;
        padding: 0.6rem 1rem;
        border-radius: 1rem 1rem 1rem 0.5rem;
        max-width: 60%;
        margin-bottom: 0.5rem;
    }

    /* 聊天区域容器 */
    .chat-container {
        padding: 1rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- 侧边栏上传区 --------------------
with st.sidebar:
    st.markdown("## 📁 Upload Data")
    # 1) 本地路径输入
    data_path = st.text_input("Folder path", placeholder="/path/to/data", key="input_path")
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#888'>OR</div>", unsafe_allow_html=True)
    st.markdown("---")
    # 2) ZIP 上传
    uploaded_zip = st.file_uploader("Upload ZIP", type="zip", key="uploaded_zip")

    st.markdown("---")
    with st.expander("⚙️ Advanced Settings", expanded=False):
        out_dir = st.text_input("Output directory", value="runs/", key="output_dir")
        config_options = ["configs/default.yaml", "configs/custom.yaml"]
        config_path = st.selectbox("Config file", options=config_options, key="config_path")
        max_iter = st.number_input("Max iterations", min_value=1, max_value=20, value=5, step=1, key="max_iterations")
        init_prompt = st.text_area("Initial prompt", placeholder="（optional）", key="initial_prompt", height=80)
        control = st.checkbox("Manual prompts between iterations", key="control_prompts")
        extract_zip = st.checkbox("Extract ZIP to separate dir", key="extract_check")
        extract_dir = st.text_input("Extraction dir", placeholder="extract_to/", key="extract_dir", disabled=not extract_zip)
        log_level = st.select_slider(
            "Log verbosity",
            options=["DEBUG", "MODEL_INFO", "DETAILED_INFO", "BRIEF_INFO"],
            value="DETAILED_INFO",
            key="log_verbosity",
        )
    st.markdown("---")
    run_button = st.button("▶️ Run Agent", use_container_width=True)

# -------------------- 聊天主区 --------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("### 💬 AutoMLAgent Chat")
# 消息存储
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "Hello! upload your data on the left, then chat to start AutoML."}
    ]

# 渲染历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='userBubble'>{msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='botBubble'>{msg['text']}</div>", unsafe_allow_html=True)

# 用户输入框
user_input = st.chat_input("Type a message…")
if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})
    st.rerun()

# 处理 Run 按钮或用户输入后的逻辑
if run_button or (st.session_state.messages and st.session_state.messages[-1]["role"] == "user"):
    # 校验至少提供了一种数据源
    if not data_path and not uploaded_zip:
        error = "⚠️ Please provide folder path or upload a ZIP."
        st.session_state.messages.append({"role": "bot", "text": error})
        st.rerun()
    # 启动后端 run_agent（这里演示，实际请替换为你的调用）
    time_stamp = datetime.now().strftime("%H:%M:%S")
    reply = f"[{time_stamp}] Running AutoML on “{data_path or uploaded_zip.name}” with {max_iter} iters…"
    st.session_state.messages.append({"role": "bot", "text": reply})
    # TODO: 在这里真正调用 run_agent(...) 并收集日志逐行追加到 messages
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
