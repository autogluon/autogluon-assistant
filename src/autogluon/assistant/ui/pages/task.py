import copy
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import psutil
import requests
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from autogluon.assistant.ui.constants import (
    BASE_DATA_DIR,
    CAPTIONS,
    DATASET_OPTIONS,
    EXTRACT_DIR,
    INITIAL_STAGE,
    LLM_MAPPING,
    LLM_OPTIONS,
    LOCAL_ZIP_PATH,
    PRESET_DEFAULT_CONFIG,
    PRESET_MAPPING,
    PRESET_OPTIONS,
    PROVIDER_MAPPING,
    S3_URL,
    TIME_LIMIT_MAPPING,
    TIME_LIMIT_OPTIONS,
)
from autogluon.assistant.ui.file_uploader import description_file_uploader, file_uploader, save_description_file
from autogluon.assistant.ui.log_processor import messages, show_logs
from autogluon.assistant.ui.utils import (
    generate_model_file,
    generate_output_file,
    generate_output_filename,
    get_user_data_dir,
    get_user_session_id,
    save_all_files,
)

os.makedirs(BASE_DATA_DIR, exist_ok=True)


def update_config_overrides():
    """
    Update the configuration overrides based on the current session state.
    """
    config_overrides = []
    if st.session_state.time_limit:
        config_overrides.append(f"time_limit={TIME_LIMIT_MAPPING[st.session_state.time_limit]}")
    if st.session_state.llm:
        config_overrides.append(f"llm.model={LLM_MAPPING[st.session_state.llm]}")
        config_overrides.append(f"llm.provider={PROVIDER_MAPPING[st.session_state.llm]}")

    if st.session_state.feature_generation:
        config_overrides.append("feature_transformers.enabled_models=[CAAFE, OpenFE, PretrainedEmbedding]")
    else:
        config_overrides.append("feature_transformers.enabled_models=None")

    st.session_state.config_overrides = config_overrides


def store_value(key):
    """
    Store a value in the session state.

    Args:
        key (str): The key to store the value under in the session state.
    """
    st.session_state[key] = st.session_state["_" + key]
    if key == "preset":
        preset_config = PRESET_DEFAULT_CONFIG.get(st.session_state.preset)
        st.session_state["time_limit"] = preset_config.get("time_limit")
        st.session_state["feature_generation"] = preset_config.get("feature_generation")


def load_value(key):
    """
    Load a value from the session state into a temporary key.

    Args:
        key (str): The key to load the value from in the session state.
    """
    st.session_state["_" + key] = st.session_state[key]


def config_autogluon_preset():
    load_value("preset")
    st.selectbox(
        "Autogluon Preset",
        placeholder="Autogluon Preset",
        options=PRESET_OPTIONS,
        key="_preset",
        on_change=store_value,
        args=["preset"],
        label_visibility="collapsed",
    )


@st.fragment
def config_time_limit():
    load_value("time_limit")
    st.selectbox(
        "Time Limit",
        placeholder="Time Limit",
        options=TIME_LIMIT_OPTIONS,
        key="_time_limit",
        on_change=store_value,
        args=["time_limit"],
        label_visibility="collapsed",
    )


@st.fragment
def config_llm():
    st.selectbox(
        "Choose an LLM model",
        placeholder="Choose an LLM model",
        options=LLM_OPTIONS,
        key="_llm",
        on_change=store_value,
        args=["llm"],
        label_visibility="collapsed",
    )


@st.fragment
def config_feature_generation():
    load_value("feature_generation")
    checkbox = st.checkbox(
        "Feature Generation (Beta Feature)",
        key="_feature_generation",
        on_change=store_value,
        args=["feature_generation"],
    )
    # Show warning if checkbox is checked
    if checkbox:
        st.warning(
            "Feature Generation is an experimental feature which requires optional packages. Please install [requirements](https://github.com/autogluon/autogluon-assistant/blob/main/requirements.txt) with: "
            "'pip install -r requirements.txt'",
            icon="⚠️",
        )


def store_value_and_save_file(key):
    store_value(key)
    save_description_file(st.session_state.task_description)


@st.fragment
def display_description():
    load_value("task_description")
    st.text_area(
        label="Dataset Description",
        placeholder="Enter your task description : ",
        key="_task_description",
        on_change=store_value_and_save_file,
        args=["task_description"],
        height=300,
    )


@st.fragment
def show_output_download_button(data, file_name):
    st.download_button(label="💾&nbsp;&nbsp;Download Predictions", data=data, file_name=file_name, mime="text/csv")


def show_cancel_task_button():
    try:
        if st.button(
            "⏹️&nbsp;&nbsp;Stop Task",
            on_click=toggle_cancel_state,
            key="cancel_task",
        ):
            p = st.session_state.process
            print("Stopping the task ...")
            p.terminate()
            p.wait()
            st.session_state.process = None
            st.session_state.pid = None
            print("The Task has stopped")
            st.session_state.task_running = False
            st.session_state.show_remaining_time = False
            st.session_state.output_filename = None
            st.session_state.zip_path = None
            st.rerun()
    except psutil.NoSuchProcess:
        st.session_state.task_running = False
        st.session_state.process = None
        st.session_state.pid = None
        st.error("No running task is found")
    except Exception as e:
        st.session_state.task_running = False
        st.session_state.process = None
        st.session_state.pid = None
        st.error(f"An error occurred: {e}")


def run_autogluon_assistant(data_dir):
    """
    Run the AutoGluon assistant with the specified configuration and data directories.

    This function constructs the command to run the AutoGluon assistant, including
    any config overrides, and starts the process. The process object and output
    filename are stored in the session state.

    Args:
        config_dir (str): The path to the configuration directory.
        data_dir (str): The path to the data directory.
    """
    output_filename = generate_output_filename()
    command = ["aga", "run", data_dir]
    if st.session_state.preset:
        command.extend(["--presets", PRESET_MAPPING[st.session_state.preset]])
    if st.session_state.config_overrides:
        command.extend(["--config_overrides", ",".join(st.session_state.config_overrides)])
    command.extend(["--output-filename", output_filename])
    st.session_state.output_file = None
    st.session_state.output_filename = output_filename
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        st.session_state.process = process
        st.session_state.pid = process.pid
    except Exception as e:
        print(f"An error occurred: {e}")


def download_model_button():
    """
    Create and display a download button for the log file.
    """
    if st.session_state.zip_path and st.session_state.task_running is False:
        zip_path = st.session_state.zip_path
        if zip_path and os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️&nbsp;&nbsp;Download Model",
                    data=f,
                    file_name=os.path.basename(zip_path),
                    mime="application/zip",
                )


def download_log_button():
    """
    Create and display a download button for the log file.
    """
    if st.session_state.logs and st.session_state.task_running is False:
        st.download_button(
            label="📥&nbsp;&nbsp;Download Logs",
            data=st.session_state.logs,
            file_name="aga_logs.txt",
            mime="text/plain",
        )


def download_output_button():
    """
    Create and display a download button for the output file.
    """
    if st.session_state.output_file is not None and st.session_state.task_running is False:
        output_file = st.session_state.output_file
        output_filename = st.session_state.output_filename
        final_name = os.path.basename(output_filename)
        show_output_download_button(output_file.to_csv(index=False), final_name)


def toggle_running_state():
    st.session_state.task_running = True
    st.session_state.task_canceled = False
    st.session_state.current_stage = None
    st.session_state.stage_container = copy.deepcopy(INITIAL_STAGE)
    st.session_state.stage_status = {}
    st.session_state.increment_time = 0


def toggle_cancel_state():
    st.session_state.task_canceled = True


def run_button():
    """
    Create and handle the "Run" button for starting the AutoGluon task.
    """
    if st.button(
        label="🔘&nbsp;&nbsp;Run!",
        key="run_task",
        on_click=toggle_running_state,
        disabled=st.session_state.task_running,
    ):
        if st.session_state.selected_dataset == "Sample Dataset":
            if st.session_state.sample_dataset_dir is not None:
                sample_data_dir = st.session_state.sample_dataset_dir
                run_autogluon_assistant(sample_data_dir)
            else:
                st.warning("Please choose the sample dataset you want to run")
                st.session_state.task_running = False
        elif st.session_state.selected_dataset == "Upload Dataset":
            user_data_dir = get_user_data_dir()
            if st.session_state.uploaded_files:
                save_all_files(user_data_dir)
                run_autogluon_assistant(user_data_dir)
            else:
                st.warning("Please upload files before running the task.")
                st.session_state.task_running = False


def show_cancel_container():
    """
    Display a cancellation message when the task is cancelled.
    """
    status_container = st.empty()
    status_container.info("Task has been cancelled")


def set_description():
    """
    Set up the description section of the user interface.

    This function calls helper functions to display a description and set up
    a file uploader for the description
    """
    if st.session_state.selected_dataset == "Upload Dataset":
        display_description()
        description_file_uploader()
    elif st.session_state.selected_dataset == "Sample Dataset":
        load_value("sample_description")
        st.text_area(
            label="Sample Dataset Description",
            placeholder="Enter your task description : ",
            key="_sample_description",
            on_change=store_value,
            args=["sample_description"],
            height=300,
        )
    add_vertical_space(4)


def wait_for_process():
    """
    Wait for the process to complete and update session state based on return code.
    """
    if st.session_state.process is not None:
        process = st.session_state.process
        process.wait()
        st.session_state.task_running = False
        st.session_state.return_code = process.returncode
        st.session_state.process = None
        st.session_state.pid = None
        generate_output_file()
        generate_model_file()
        st.rerun()


def run_section():
    """
    Set up and display the 'Run AutoGluon' section of the user interface.
    """
    st.markdown(
        """
           <h1 style='
               font-weight: light;
               padding-left: 20px;
               padding-right: 20px;
               margin-left:60px;
               font-size: 2em;
           '>
               Run Autogluon
           </h1>
       """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4, col5 = st.columns([1, 10.9, 0.2, 10.9, 1], gap="large")
    with col4:
        dataset_selector()
    with col2:
        col11, col12 = st.columns(2)
        with col11:
            config_autogluon_preset()
            config_time_limit()
        with col12:
            config_llm()
            config_feature_generation()
        set_description()
    with col3:
        st.html(
            """
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid rgba(49, 51, 63, 0.2);
                        height: 590px;
                        margin: auto;
                    }
                </style>
            """
        )
    update_config_overrides()
    _, mid_pos, _ = st.columns([1, 22, 1], gap="large")
    with mid_pos:
        run_button()
        if st.session_state.task_running:
            show_cancel_task_button()
    _, mid_pos, _ = st.columns([1, 22, 1], gap="large")
    with mid_pos:
        if st.session_state.task_running:
            messages()
        elif not st.session_state.task_running and not st.session_state.task_canceled:
            show_logs()
        elif st.session_state.task_canceled:
            show_cancel_container()
    wait_for_process()
    _, download_pos, _ = st.columns([4, 2, 4])
    with download_pos:
        if not st.session_state.task_running:
            download_log_button()
            download_output_button()
            download_model_button()
    st.markdown("---", unsafe_allow_html=True)


def setup_local_dataset():
    """Download dataset from S3 to local directory"""
    try:
        if not os.path.exists(EXTRACT_DIR):
            response = requests.get(S3_URL, stream=True)
            if response.status_code == 200:
                with open(LOCAL_ZIP_PATH, "wb") as f:
                    f.write(response.content)
                try:
                    with zipfile.ZipFile(LOCAL_ZIP_PATH, "r") as zip_ref:
                        zip_ref.extractall(EXTRACT_DIR)
                except zipfile.BadZipFile:
                    st.error("Failed to extract the zip file.")
                    return
                finally:
                    os.remove(LOCAL_ZIP_PATH)
            else:
                st.error(f"Failed to download the file. HTTP status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while downloading the sample dataset: {e}")
    except OSError as e:
        st.error(f"An error occurred while handling files or directories: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


def get_sample_dataset_files(dataset_dir):
    """
    Get all files from the given sample dataset directory
    """
    st.session_state.sample_files = {}
    for file in dataset_dir.glob("*.csv"):
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        st.session_state.sample_files[file.name] = {"file": file, "df": df}


def get_available_datasets(sample_dataset_dir):
    """Get all subdirectories in the sample_dataset directory."""
    return [d.name for d in Path(sample_dataset_dir).iterdir() if d.is_dir()]


def sample_dataset_selector():
    sample_dataset_dir = "sample_dataset"
    sample_datasets = get_available_datasets(sample_dataset_dir)
    load_value("sample_dataset_selector")
    selected_dataset = st.selectbox(
        "Select a dataset:",
        options=sample_datasets,
        key="_sample_dataset_selector",
        index=None,
        on_change=store_value,
        args=["sample_dataset_selector"],
    )
    if selected_dataset:
        selected_dataset_dir = Path(sample_dataset_dir) / selected_dataset
        st.session_state.sample_dataset_dir = selected_dataset_dir
        get_sample_dataset_files(selected_dataset_dir)
        description_file = selected_dataset_dir / "descriptions.txt"
        if description_file.exists():
            try:
                with open(description_file, "r", encoding="utf-8") as f:
                    st.session_state.sample_description = f.read()
            except Exception as e:
                st.error(f"Error reading description.txt: {e}")
        st.success(
            f"Selected sample dataset: ****{selected_dataset}****, preview the dataset in the ****Preview**** Section below"
        )
        st.success("Click the 🔘&nbsp;&nbsp;****Run**** button to get started")


def dataset_selector():
    dataset_choice = st.radio(
        "Choose dataset option",
        key="dataset_choose",
        options=DATASET_OPTIONS,
        horizontal=True,
        captions=CAPTIONS,
        label_visibility="collapsed",
    )
    if dataset_choice == "Sample Dataset":
        st.session_state.selected_dataset = "Sample Dataset"
        sample_dataset_selector()
    elif dataset_choice == "Upload Dataset":
        st.session_state.selected_dataset = "Upload Dataset"
        file_uploader()


def main():
    setup_local_dataset()
    get_user_session_id()
    run_section()


if __name__ == "__main__":
    main()
