import re
import time

import streamlit as st
from autogluon.assistant.constants import (
    IGNORED_MESSAGES,
    STAGE_COMPLETE_SIGNAL,
    STAGE_MESSAGES,
    STATUS_BAR_STAGE,
    SUCCESS_MESSAGE,
    TIME_LIMIT_MAPPING,
    BRIEF_STAGE_MESSAGES,
)


def show_log_line(entry):
    """
    entry: {"level": "<LEVEL>", "text": "<MESSAGE>"}
    Prints "<LEVEL>\t<MESSAGE>" with simple styling.
    """
    # 新格式：dict 包含 level/text
    if isinstance(entry, dict):
        level = entry.get("level", "").upper()
        text  = entry.get("text", "")
    else:
        # 若不符合新格式，兜底当 INFO
        st.write("error: missing level (eg. INFO)")

    # 将 level 和 text 以制表符分隔打印
    # 你可以根据 level 再加不同的颜色或组件
    st.write(f"{text}")

def show_logs():
    st.write("hello world")
    # """
    # Display logs and task status when task is finished.
    # """
    # if st.session_state.logs:
    #     status_container = st.empty()
    #     if st.session_state.return_code == 0:
    #         status_container.success(SUCCESS_MESSAGE)
    #     else:
    #         status_container.error("Error detected in the process...Check the logs for more details")
    #     tab1, tab2 = st.tabs(["Messages", "Logs"])
    #     with tab1:
    #         for stage, logs in st.session_state.stage_container.items():
    #             if logs:
    #                 with st.status(stage, expanded=False, state="complete"):
    #                     for log in logs:
    #                         show_log_line(log)
    #     with tab2:
    #         log_container = st.empty()
    #         log_container.text_area("Real-Time Logs", st.session_state.logs, height=400)


def format_log_line(line):
    """
    Format log lines by removing ANSI escape codes, formatting markdown syntax,
    and cleaning up process-related information.

    Args:
        line (str): Raw log line to be formatted.

    Returns:
        str: Formatted log line with:
    """
    line = re.sub(r"\x1B\[1m(.*?)\x1B\[0m", r"**\1**", line)
    line = re.sub(r"^#", r"\\#", line)
    line = re.sub(r"\033\[\d+m", "", line)
    line = re.sub(r"^\s*\(\w+ pid=\d+\)\s*", "", line)
    return line


def process_realtime_logs(line):
    """
    Handles the real-time processing of log lines, updating the UI state,
    managing progress bars, and displaying status updates in the  interface.

    Args:  line (str): A single line from the log stream to process.
    """
    if any(ignored_msg in line for ignored_msg in IGNORED_MESSAGES):
        return
    stage = get_stage_from_log(line)
    if stage:
        if stage != st.session_state.current_stage:
            if st.session_state.current_stage:
                st.session_state.stage_status[st.session_state.current_stage].update(
                    state="complete",
                )
            st.session_state.current_stage = stage
        if stage not in st.session_state.stage_status:
            st.session_state.stage_status[stage] = st.status(stage, expanded=False)

    if st.session_state.current_stage:
        if "AutoGluon training complete" in line:
            st.session_state.show_remaining_time = False
        with st.session_state.stage_status[st.session_state.current_stage]:
            if "Fitting model" in line and not st.session_state.show_remaining_time:
                st.session_state.progress_bar = st.progress(0, text="Elapsed Time for Fitting models:")
                st.session_state.show_remaining_time = True
                st.session_state.elapsed_time = time.time() - st.session_state.start_time
                st.session_state.remaining_time = (
                    TIME_LIMIT_MAPPING[st.session_state.time_limit] - st.session_state.elapsed_time
                )
                st.session_state.start_model_train_time = time.time()
            if st.session_state.show_remaining_time:
                st.session_state.elapsed_time = time.time() - st.session_state.start_model_train_time
                progress_bar = st.session_state.progress_bar
                current_time = min(st.session_state.elapsed_time, st.session_state.remaining_time)
                progress = current_time / st.session_state.remaining_time
                time_ratio = f"Elapsed Time for Fitting models: | ({progress:.1%})"
                progress_bar.progress(progress, text=time_ratio)
            if not st.session_state.show_remaining_time:
                st.session_state.stage_container[st.session_state.current_stage].append(line)
                show_log_line(line)


import re
import streamlit as st

def messages(process, total_iterations):
    """
    Parse the given process.stdout line-by-line, grouping BRIEF logs into
    stages and updating the Streamlit UI. All output for this run appears
    in one assistant bubble, with a top-level progress bar and per-stage
    collapsible sections.
    """
    chat_bubble = st.chat_message("assistant")
    with chat_bubble:
        # overall progress bar (0→1)
        progress = st.progress(0.0)
        current_status = None

        # e.g. "2025-05-29 00:00:24 BRIEF    [module] Message..."
        log_pattern = re.compile(r'^\S+\s+\S+\s+(\w+)\s+\[[^\]]+\]\s+(.*)$')

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            m = log_pattern.match(line)
            if not m:
                continue
            level, message = m.group(1), m.group(2)

            # on each BRIEF log, start a new stage
            if level == "BRIEF":
                if current_status is not None:
                    current_status.update(state="complete")
                key = message.split(":", 1)[0].strip()
                # use the text before the first colon as the section title
                current_status = st.status(key, expanded=False)

            # emit this line into the current stage
            if current_status is not None:
                if level == "BRIEF":
                    current_status.markdown(f":green[{message}]")
                elif level == "INFO":
                    current_status.markdown(f":blue[{message}]")
                elif level.upper().startswith("WARN"):
                    current_status.markdown(f":orange[{message}]")
                elif level.upper().startswith("ERROR") or level.upper().startswith("EXCEPTION"):
                    current_status.markdown(f":red[{message}]")
                else:
                    current_status.write(message)

        # mark last stage done
        if current_status is not None:
            current_status.update(state="complete")

        # finish overall progress
        progress.progress(1.0)
