import re
import streamlit as st

# ─── 阶段匹配关键字 ────────────────────────────────────────────
READING_START = "DataPerceptionAgent: beginning to scan data folder and group similar files."
READING_END   = "ToolSelectorAgent: selected"
ITER_START_RE = re.compile(r"Starting iteration (\d+)!")
ITER_END_RE   = re.compile(r"Code generation (failed|successful)")
OUTPUT_START  = "Total tokens"
OUTPUT_END    = "output saved in"


def messages(log_entries, max_iter):
    """
    把日志分成 Reading / Iteration1…IterationN / Output 几个阶段渲染，
    并且把“当前阶段名称”保存在 session_state，避免组件重置时闪动。

    参数：
      - log_entries: 后端 `/api/logs` 返回的列表，每项是 {"level": ..., "text": ...}
      - max_iter: 用户在前端设置的最大迭代次数（整数，比如 5）
    """

    total_stages = max_iter + 2  # Reading + max_iter 次迭代 + Output

    # ─── 1) 在 session_state 里记下“当前阶段名称”（第一次调用时设为 None）
    if "current_phase_name" not in st.session_state:
        st.session_state.current_phase_name = None

    # ─── 2) 在函数开头，先创建一个“阶段标题”的占位和一个“纯进度条”的组件 ────────────────────────
    stage_text   = st.empty()     # ← 用来在进度条上方显示“Reading”、“Iteration X”、“Output”等标题
    progress_bar = st.progress(0) # ← 仅显示进度数值，不带文字

    # 如果 session_state 里已有阶段名称，就先把它写上去（保证在 rerun 时不闪回空白）
    if st.session_state.current_phase_name:
        stage_text.markdown(f"### {st.session_state.current_phase_name}")

    # ─── 3) 用来存储本次调用中各阶段对应的 StatusContainer （本地字典即可）
    stage_containers = {}  # key: 阶段名（"Reading"/"Iteration 1"/…/"Output"）→ value: 对应 st.status
    current_phase = st.session_state.current_phase_name

    # ─── 4) 依次处理每一行日志 ─────────────────────────────────────────────────
    for entry in log_entries:
        text = entry.get("text", "")  # 每条 entry 都是一个 dict，包含 {"level": "...", "text": "..."}，我们只关心 text 部分

        # —— 1) 检测 Reading 阶段 “开始” ────────────────────────────────────
        if READING_START in text:
            current_phase = "Reading"
            st.session_state.current_phase_name = current_phase

            # 只在第一次进入 Reading 时创建一个 st.status("Reading")
            if "Reading" not in stage_containers:
                stage_containers["Reading"] = st.status("Reading", expanded=True)

            # 更新阶段标题和进度条
            stage_text.markdown("### Reading")
            progress_bar.progress(0 / total_stages)
            # 把这条日志写到 Reading 容器里
            stage_containers["Reading"].write(text)
            continue

        # —— 2) 检测 Reading 阶段 “结束” ────────────────────────────────────
        if READING_END in text:
            # 如果当前确实处于 Reading 阶段，就标记完成并清空 current_phase
            if current_phase == "Reading":
                stage_containers["Reading"].write(text)
                stage_containers["Reading"].update(state="complete")
                current_phase = None
                st.session_state.current_phase_name = None
            else:
                # 否则，说明我们之前已经标记过 Reading 完成，这里只需追加 `.write(...)`，不再 `.update(...)`
                if "Reading" in stage_containers:
                    stage_containers["Reading"].write(text)
                else:
                    # 如果前面没检测到“开始”，就先建一个容器再标 complete
                    stage_containers["Reading"] = st.status("Reading", expanded=True)
                    stage_containers["Reading"].write(text)
                    stage_containers["Reading"].update(state="complete")
                current_phase = None
                st.session_state.current_phase_name = None
            continue

        # —— 3) 检测某次 Iteration 的“开始”，如 “Starting iteration 0!” ────────────
        m_start = ITER_START_RE.search(text)
        if m_start:
            idx = int(m_start.group(1)) + 1   # “0” → 第 1 次迭代
            phase_name = f"Iteration {idx}"
            current_phase = phase_name
            st.session_state.current_phase_name = current_phase

            if phase_name not in stage_containers:
                stage_containers[phase_name] = st.status(phase_name, expanded=True)

            # 更新阶段标题和进度条
            stage_text.markdown(f"### {phase_name}")
            progress_bar.progress(idx / total_stages)
            stage_containers[phase_name].write(text)
            continue

        # —— 4) 检测某次 Iteration 的“结束”（失败或成功都算）──────────────────
        if ITER_END_RE.search(text):
            if current_phase and current_phase.startswith("Iteration"):
                stage_containers[current_phase].write(text)
                stage_containers[current_phase].update(state="complete")
                current_phase = None
                st.session_state.current_phase_name = None
            else:
                # fallback：如果 current_phase 不是某个 iteration，就把它写到 “Iteration (Unknown)” 里
                fallback = "Iteration (Unknown)"
                if fallback not in stage_containers:
                    stage_containers[fallback] = st.status(fallback, expanded=True)
                stage_containers[fallback].write(text)
                stage_containers[fallback].update(state="complete")
                current_phase = None
                st.session_state.current_phase_name = None
            continue

        # —— 5) 检测 Output 阶段 “开始” ────────────────────────────────────
        if OUTPUT_START in text:
            current_phase = "Output"
            st.session_state.current_phase_name = current_phase

            if "Output" not in stage_containers:
                stage_containers["Output"] = st.status("Output", expanded=True)

            stage_text.markdown("### Output")
            progress_bar.progress((max_iter + 1) / total_stages)
            stage_containers["Output"].write(text)
            continue

        # —— 6) 检测 Output 阶段 “结束” ────────────────────────────────────
        if OUTPUT_END in text:
            if current_phase == "Output":
                stage_containers["Output"].write(text)
                stage_containers["Output"].update(state="complete")
                progress_bar.progress(1.0)
                current_phase = None
                st.session_state.current_phase_name = None
            else:
                # fallback：如果之前没检测到 Output 开始，只做一次 create + complete
                if "Output" in stage_containers:
                    stage_containers["Output"].write(text)
                else:
                    stage_containers["Output"] = st.status("Output", expanded=True)
                    stage_containers["Output"].write(text)
                    stage_containers["Output"].update(state="complete")
                progress_bar.progress(1.0)
                current_phase = None
                st.session_state.current_phase_name = None
            continue

        # —— 7) 如果这个日志行并不触发任何“开始/结束”标志，那么就视为“当前阶段内部”的普通日志
        if current_phase:
            stage_containers[current_phase].write(text)
        else:
            # 不属于任何阶段时，就当作“孤立日志”直接输出
            st.write(text)
