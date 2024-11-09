import re
import time

import streamlit as st
from constants import STAGE_COMPLETE_SIGNAL, STAGE_MESSAGES, STATUS_BAR_STAGE, TIME_LIMIT_MAPPING
from stqdm import stqdm


def parse_model_path(log):
    """
    Extract the AutogluonModels path from the log text.

    Args:
        log (str): The log text containing the model path

    Returns:
        str or None: The extracted model path or None if not found
    """
    pattern = r'"([^"]*AutogluonModels[^"]*)"'
    match = re.search(pattern, log)
    if match:
        return match.group(1)
    return None


def show_log_line(line):
    """
    Show log line based on prefix and Rich syntax
    - Lines starting with WARNING: → st.warning
    - Other lines → st.markdown
    """
    if "INFO:" in line:
        line = line.split(":", 1)[1].split(":", 1)[1]
    if any(message in line for message in STAGE_COMPLETE_SIGNAL):
        return st.success(line)
    elif line.startswith("WARNING:"):
        return st.warning(line)
    return st.markdown(line)


def get_stage_from_log(log_line):
    for message, stage in STAGE_MESSAGES.items():
        if message in log_line:
            return stage
    return None


def show_logs():
    """
    Display logs and task status when task is finished.
    """
    if st.session_state.logs:
        status_container = st.empty()
        if st.session_state.return_code == 0:
            status_container.success("Task completed successfully!")
        else:
            status_container.error("Error detected in the process...Check the logs for more details")
        tab1, tab2 = st.tabs(["Messages", "Logs"])
        with tab1:
            for stage, logs in st.session_state.stage_container.items():
                if logs:
                    with st.status(stage, expanded=False, state="complete"):
                        for log in logs:
                            show_log_line(log)
        with tab2:
            log_container = st.empty()
            log_container.text_area("Real-Time Logs", st.session_state.logs, height=400)


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
    stage = get_stage_from_log(line)
    if stage:
        if stage != st.session_state.current_stage:
            st.session_state.current_stage = stage
        if stage not in st.session_state.stage_status:
            st.session_state.stage_status[stage] = st.status(stage, expanded=False)

    if st.session_state.current_stage:
        st.session_state.stage_status[st.session_state.current_stage].update(
            state="running",
        )
        if "AutoGluon training complete" in line:
            st.session_state.show_remaining_time = False
        with st.session_state.stage_status[st.session_state.current_stage]:
            time.sleep(1)
            if "Fitting model" in line and not st.session_state.show_remaining_time:
                st.session_state.progress_bar = stqdm(
                    desc="Elapsed Time for Fitting models: ", total=TIME_LIMIT_MAPPING[st.session_state.time_limit]
                )
                st.session_state.show_remaining_time = True
                st.session_state.increment_time = 0
            if st.session_state.show_remaining_time:
                st.session_state.increment_time += 1
                if st.session_state.increment_time <= TIME_LIMIT_MAPPING[st.session_state.time_limit]:
                    st.session_state.progress_bar.update(1)
            if not st.session_state.show_remaining_time:
                st.session_state.stage_container[st.session_state.current_stage].append(line)
                show_log_line(line)


def messages():
    """
    Handles the streaming of log messages from a subprocess, updates the UI with progress
    indicators, and manages the display of different training stages.
    processes ANSI escape codes, formats markdown, and updates various progress indicators.

    """
    if st.session_state.process is not None:
        process = st.session_state.process
        st.session_state.logs = ""
        progress = st.progress(0)
        status_container = st.empty()
        status_container.info("Running Tasks...")
        for line in process.stdout:
            print(line, end="")
            line = format_log_line(line)
            st.session_state.logs += line
            if "TabularPredictor saved" in line:
                model_path = parse_model_path(line)
                if model_path:
                    st.session_state.model_path = model_path
            if "Prediction complete" in line:
                status_container.success("Task completed successfully!")
                progress.progress(100)
                process_realtime_logs(line)
                break
            else:
                for stage, progress_value in STATUS_BAR_STAGE.items():
                    if stage.lower() in line.lower():
                        progress.progress(progress_value / 100)
                        status_container.info(stage)
                        break
            process_realtime_logs(line)
