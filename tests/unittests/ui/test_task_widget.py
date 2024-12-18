from unittest.mock import Mock, mock_open, patch

import pandas as pd
import psutil
import pytest
from streamlit.testing.v1 import AppTest

from autogluon.assistant.ui.constants import LLM_MAPPING, PRESET_MAPPING, PROVIDER_MAPPING, TIME_LIMIT_MAPPING


@pytest.fixture()
def at():
    """Fixture that prepares the Streamlit app tests"""
    yield AppTest.from_file("src/autogluon/assistant/ui/app.py").run()


def test_app_starts(at):
    """Verify the app starts without errors"""
    assert not at.exception


def test_default_config(at):
    """Test the default configuration of the application"""
    assert at.selectbox(key="_preset").value == "Medium Quality"
    assert at.selectbox(key="_time_limit").value == "10 mins"
    assert at.selectbox(key="_llm").value == "Claude 3.5 with Amazon Bedrock"
    assert not at.checkbox(key="_feature_generation").value
    assert at.session_state.preset == "Medium Quality"
    assert at.session_state.time_limit == "10 mins"
    assert not at.session_state.feature_generation


def test_best_quality(at):
    """Test the 'Best Quality' preset configuration."""
    at.selectbox(key="_preset").set_value("Best Quality").run()
    assert at.selectbox(key="_time_limit").value == "4 hrs"
    assert not at.checkbox(key="_feature_generation").value
    assert at.session_state.preset == "Best Quality"
    assert at.session_state.time_limit == "4 hrs"
    assert not at.session_state.feature_generation


def test_high_quality(at):
    """Test the 'High Quality' preset configuration."""
    at.selectbox(key="_preset").set_value("High Quality").run()
    assert at.selectbox(key="_time_limit").value == "1 hr"
    assert not at.checkbox(key="_feature_generation").value
    assert at.session_state.preset == "High Quality"
    assert at.session_state.time_limit == "1 hr"
    assert not at.session_state.feature_generation


def test_enable_feature_generation(at):
    """Test enabling the feature generation option."""
    at.checkbox(key="_feature_generation").check().run()
    assert at.session_state.feature_generation
    assert len(at.warning) == 1


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.Popen"""
    with patch("subprocess.Popen") as mock_popen:
        process_mock = Mock()
        process_mock.pid = 12345
        mock_popen.return_value = process_mock
        yield mock_popen


def test_run_sample_dataset_no_dir(at):
    """Test sample dataset with no directory selected"""
    at.session_state.selected_dataset = "Sample Dataset"
    at.session_state.sample_dataset_dir = None
    at.run()
    button = at.button(key="run_task")
    button.click().run()
    assert len(at.warning) == 1
    assert "Please choose the sample dataset you want to run" in at.warning[0].value


def test_run_sample_dataset_success(mock_subprocess, at):
    """Test successful run with sample dataset"""
    at.session_state.sample_dataset_dir = "/test/sample/path"
    at.run()
    at.button(key="run_task").click().run()
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert args[0] == "aga"
    assert args[1] == "run"
    assert args[2] == "/test/sample/path"


def test_run_no_upload_dataset(at):
    """Test Run task with no files uploaded"""
    at.radio(key="dataset_choose").set_value("Upload Dataset")
    at.button(key="run_task").click().run()
    assert len(at.warning) == 1
    assert "Please upload files before running the task." in at.warning[0].value


def test_run_with_config_overrides(mock_subprocess, at):
    """Test Run Task with config overrides"""
    at.session_state.sample_dataset_dir = "/test/sample/path"
    at.run()
    at.selectbox(key="_time_limit").set_value("3 mins")
    at.selectbox(key="_preset").set_value("Best Quality")
    at.selectbox(key="_llm").set_value("GPT 4o")
    at.button(key="run_task").click()
    at.checkbox(key="_feature_generation").check()
    at.run()
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert args[0] == "aga"
    assert args[1] == "run"
    assert args[2] == "/test/sample/path"
    assert args[3] == "--presets"
    assert args[4] == PRESET_MAPPING["Best Quality"]
    assert args[5] == "--config_overrides"
    assert (
        args[6]
        == f"time_limit={TIME_LIMIT_MAPPING['3 mins']},llm.model={LLM_MAPPING['GPT 4o']},llm.provider={PROVIDER_MAPPING['GPT 4o']},feature_transformers.enabled_models=[CAAFE, OpenFE, PretrainedEmbedding]"
    )
    assert at.session_state.pid == mock_subprocess.return_value.pid


def test_show_cancel_button(at):
    """Test the cancel task button is shown when the task is running"""
    at.session_state.task_running = True
    at.run()
    assert len(at.button) == 2
    assert at.button[1].key == "cancel_task"


def test_cancel_button_terminates_process(mock_subprocess, at):
    """Test click the cancel task button will call the appropriate method to clean up the process"""
    process_mock = mock_subprocess.return_value
    at.session_state.process = process_mock
    at.session_state.task_running = True
    at.session_state.pid = process_mock.pid
    at.run()
    cancel_button = at.button(key="cancel_task")
    cancel_button.click().run()
    process_mock.terminate.assert_called_once()
    process_mock.wait.assert_called_once()
    assert not at.session_state.task_running
    assert at.session_state.process is None
    assert at.session_state.pid is None


def test_cancel_button_handles_no_process(mock_subprocess, at):
    """Test click the cancel task button when NoSuchProcess exception is thrown"""
    at.session_state.task_running = True
    process_mock = mock_subprocess.return_value
    process_mock.terminate.side_effect = psutil.NoSuchProcess(pid=1234)
    at.session_state.process = process_mock
    at.run()
    cancel_button = at.button(key="cancel_task")
    cancel_button.click().run()
    assert "No running task is found" in at.error[0].value


def test_cancel_button_handles_termination_error(mock_subprocess, at):
    """Test click the cancel task button when other exception is thrown"""
    process_mock = mock_subprocess.return_value
    process_mock.terminate.side_effect = Exception("Termination failed")
    at.session_state.process = process_mock
    at.session_state.task_running = True
    at.run()
    cancel_button = at.button(key="cancel_task")
    cancel_button.click().run()
    assert "An error occurred: Termination failed" in at.error[0].value


def test_download_model_button(at, tmp_path):
    """Test download model button successfully appears when the task finished"""
    zip_path = tmp_path / "model.zip"
    zip_path.write_bytes(b"test data")
    at.session_state.zip_path = str(zip_path)
    at.session_state.task_running = False

    with patch("builtins.open", mock_open(read_data=b"test data")) as mock_file, patch(
        "streamlit.download_button"
    ) as mock_download_button:
        at.run()
        mock_download_button.assert_called_once_with(
            label="⬇️&nbsp;&nbsp;Download Model",
            data=mock_file.return_value,
            file_name="model.zip",
            mime="application/zip",
        )


def test_download_log_button(at):
    """Test download log button appears when logs exist and task is finished"""
    at.session_state.logs = "test log data"
    at.session_state.task_running = False
    with patch("streamlit.download_button") as mock_download_button:
        at.run()
        mock_download_button.assert_called_once_with(
            label="📥&nbsp;&nbsp;Download Logs",
            data="test log data",
            file_name="aga_logs.txt",
            mime="text/plain",
        )


def test_download_log_button_no_logs(at):
    """Test download log button doesn't appear when no logs exist"""
    at.session_state.logs = None
    at.session_state.task_running = False
    with patch("streamlit.download_button") as mock_download_button:
        at.run()
        mock_download_button.assert_not_called()


def test_download_output_button(at):
    """Test download output button appears when task finished successfully"""
    df = pd.DataFrame({"col1": ["value1", "value2"], "col2": ["value3", "value4"]})
    at.session_state.output_file = df
    at.session_state.output_filename = "/path/to/predictions.csv"
    at.session_state.task_running = False
    with patch("streamlit.download_button") as mock_show_button:
        at.run()
        mock_show_button.assert_called_once_with(
            label="💾&nbsp;&nbsp;Download Predictions",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )


def test_sample_dataset_selection(at):
    """Test sample dataset selection workflow"""
    at.radio(key="dataset_choose").set_value("Sample Dataset")
    assert at.session_state.selected_dataset == "Sample Dataset"
    sample_selector = at.selectbox(key="_sample_dataset_selector")
    assert sample_selector is not None


def test_upload_dataset_selection(at):
    """Test Upload Dataset selection workflow"""
    file_uploader = at.get("file_uploader")
    assert (len(file_uploader)) == 0
    at.radio(key="dataset_choose").set_value("Upload Dataset")
    at.run()
    file_uploader = at.get("file_uploader")
    assert at.session_state.selected_dataset == "Upload Dataset"
    assert (len(file_uploader)) == 2
