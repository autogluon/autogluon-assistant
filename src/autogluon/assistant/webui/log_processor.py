import re
import streamlit as st

import re
import time

READING_START = "DataPerceptionAgent: beginning to scan data folder and group similar files."
READING_END   = "DataPerceptionAgent: completed folder scan and assembled data prompt."
ITER_START_RE = re.compile(r"Starting iteration (\d+)!")
ITER_END_RE   = re.compile(r"Code generation (failed|successful)")
OUTPUT_START  = "Total tokens"
OUTPUT_END    = "output saved in"



def messages(log_entries: list[dict], max_iter: int):
    """
    对后端返回的多条日志条目（每条是 {"level": ..., "text": ...}），
    按照“Reading → Iteration1…IterationN → Output”几个阶段来渲染。

    关键步骤：
      1. 用 st.progress() 显示一个总进度条，阶段数量 = max_iter + 2
         （index0=Reading, index1~indexN=各Iteration, indexN+1=Output）；
      2. 每当遇到“阶段开始”的 text 时，就创建一个 st.status(stage_name) 容器，
         并把阶段内的所有 text 追加到该容器里；
      3. 当遇到“阶段结束”的 text 时，就把该容器更新为 state="complete" 并推进进度条；
      4. 剩余无法匹配入任何阶段边界/内部的日志，就直接用 st.write() 输出。

    参数：
      - log_entries: 后端 `/api/logs` 返回的列表，每项是 {"level": ..., "text": ...}；
      - max_iter: 用户在前端设置的最大迭代次数（整数，比如 5）。
    """

    total_stages = max_iter + 2  # 阶段总数：Reading + (Iteration x max_iter) + Output
    prog = st.progress(0.0, text="Starting…")

    # 存储各阶段对应的 StatusContainer：{ phase_name: st.status(...) }
    stage_containers: dict[str, st.delta_generator.StatusContainer] = {}
    current_phase: str | None = None  # 记录当前正在写入哪个阶段

    for entry in log_entries:
        text = entry.get("text", "")  # 先取出纯文本
        # —— 1) 检测 Reading 阶段的“开始”
        if READING_START in text:
            current_phase = "Reading"
            if current_phase not in stage_containers:
                stage_containers[current_phase] = st.status(current_phase, expanded=False)
            prog.progress(0 / total_stages, text=current_phase)
            stage_containers[current_phase].write(text)
            continue

        # —— 2) 检测 Reading 阶段的“结束”
        if READING_END in text:
            if current_phase == "Reading":
                stage_containers["Reading"].write(text)
                stage_containers["Reading"].update(state="complete")
                current_phase = None
            else:
                # 如果没记录当前phase，也强制创建一次并标记完成
                if "Reading" not in stage_containers:
                    stage_containers["Reading"] = st.status("Reading", expanded=False)
                stage_containers["Reading"].write(text)
                stage_containers["Reading"].update(state="complete")
            continue

        # —— 3) 检测某次 Iteration 的“开始”，如 “Starting iteration 0!”
        m_start = ITER_START_RE.search(text)
        if m_start:
            idx = int(m_start.group(1)) + 1  # “0”→第1次，显示字符串为 f"Iteration 1"
            phase_name = f"Iteration {idx}"
            current_phase = phase_name
            if phase_name not in stage_containers:
                stage_containers[phase_name] = st.status(phase_name, expanded=False)
            prog.progress(idx / total_stages, text=phase_name)
            stage_containers[phase_name].write(text)
            continue

        # —— 4) 检测某次 Iteration 的“结束”（失败或成功都算）
        if ITER_END_RE.search(text):
            if current_phase and current_phase.startswith("Iteration"):
                stage_containers[current_phase].write(text)
                stage_containers[current_phase].update(state="complete")
                current_phase = None
            else:
                # 如果没有 current_phase，或者格式意外，就写到“Iteration (Unknown)”里
                fallback = "Iteration (Unknown)"
                if fallback not in stage_containers:
                    stage_containers[fallback] = st.status(fallback, expanded=False)
                stage_containers[fallback].write(text)
                stage_containers[fallback].update(state="complete")
            continue

        # —— 5) 检测 Output 阶段的“开始”
        if OUTPUT_START in text:
            current_phase = "Output"
            if current_phase not in stage_containers:
                stage_containers[current_phase] = st.status(current_phase, expanded=False)
            prog.progress((max_iter + 1) / total_stages, text=current_phase)
            stage_containers[current_phase].write(text)
            continue

        # —— 6) 检测 Output 阶段的“结束”
        if OUTPUT_END in text:
            if current_phase == "Output":
                stage_containers["Output"].write(text)
                stage_containers["Output"].update(state="complete")
                prog.progress(1.0, text="Complete")
                current_phase = None
            else:
                if "Output" not in stage_containers:
                    stage_containers["Output"] = st.status("Output", expanded=False)
                stage_containers["Output"].write(text)
                stage_containers["Output"].update(state="complete")
                prog.progress(1.0, text="Complete")
            continue

        # —— 7) 如果正在某阶段内部，把这条直接追加进去
        if current_phase:
            stage_containers[current_phase].write(text)
        else:
            # 不属于任何阶段内部或开始/结束，把它当“孤立”日志直接写
            st.write(text)
