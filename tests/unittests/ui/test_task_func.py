import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest

from autogluon.assistant.ui.constants import PRESET_MAPPING
from autogluon.assistant.ui.pages.task import run_autogluon_assistant, setup_local_dataset


@pytest.fixture
def mock_session_state():
    """Fixture to mock Streamlit session state"""
    with patch("streamlit.session_state") as mock_state:
        mock_state.preset = None
        mock_state.config_overrides = None
        mock_state.process = None
        mock_state.pid = None
        mock_state.output_file = None
        mock_state.output_filename = None
        mock_state.task_running = False
        yield mock_state


@pytest.fixture
def mock_subprocess():
    """Fixture to mock subprocess.Popen"""
    with patch("subprocess.Popen") as mock_popen:
        process_mock = MagicMock()
        process_mock.pid = 12345
        process_mock.returncode = 0
        mock_popen.return_value = process_mock
        yield mock_popen


@pytest.fixture
def mock_generate_output():
    """Fixture to mock generate_output_filename"""
    with patch("autogluon.assistant.ui.pages.task.generate_output_filename") as mock_gen:
        mock_gen.return_value = "test_output.csv"
        yield mock_gen


@pytest.fixture
def sample_csv_content():
    """Fixture to provide sample CSV content"""
    return """model,accuracy,training_time
        model1,0.95,120
        model2,0.92,90
        model3,0.88,60"""


def test_setup_local_dataset_success():
    """Test successful download and extraction of dataset"""
    with patch("requests.get") as mock_get, patch("zipfile.ZipFile") as mock_zip, patch(
        "os.path.exists", return_value=False
    ), patch("os.remove"):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_zip_content"
        mock_get.return_value = mock_response
        setup_local_dataset()
        mock_get.assert_called_once()
        mock_zip.assert_called_once()
        mock_zip.return_value.__enter__().extractall.assert_called_once()


def test_basic_run(mock_session_state, mock_subprocess, mock_generate_output):
    """Test basic run with minimal parameters"""
    data_dir = "/path/to/data"
    run_autogluon_assistant(data_dir)
    mock_subprocess.assert_called_once_with(
        ["aga", "run", data_dir, "--output-filename", "test_output.csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert mock_session_state.output_filename == "test_output.csv"
    assert mock_session_state.process == mock_subprocess.return_value
    assert mock_session_state.pid == 12345


def test_run_with_preset(mock_session_state, mock_subprocess, mock_generate_output):
    """Test run with preset configuration"""
    data_dir = "/path/to/data"
    mock_session_state.preset = "Best Quality"
    run_autogluon_assistant(data_dir)
    mock_subprocess.assert_called_once_with(
        ["aga", "run", data_dir, "--presets", PRESET_MAPPING["Best Quality"], "--output-filename", "test_output.csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert mock_session_state.output_filename == "test_output.csv"
    assert mock_session_state.process == mock_subprocess.return_value
    assert mock_session_state.pid == 12345


def test_run_with_config_overrides(mock_session_state, mock_subprocess, mock_generate_output):
    """Test run with config overrides"""
    data_dir = "/path/to/data"
    mock_session_state.config_overrides = ["time_limit=value1", "llm.model=value2", "llm.provider=value3"]
    run_autogluon_assistant(data_dir)
    mock_subprocess.assert_called_once_with(
        [
            "aga",
            "run",
            data_dir,
            "--config_overrides",
            "time_limit=value1,llm.model=value2,llm.provider=value3",
            "--output-filename",
            "test_output.csv",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert mock_session_state.output_filename == "test_output.csv"
    assert mock_session_state.process == mock_subprocess.return_value
    assert mock_session_state.pid == 12345


def test_subprocess_error(mock_session_state, mock_subprocess, mock_generate_output):
    """Test handling of subprocess error"""
    data_dir = "/path/to/data"
    mock_subprocess.side_effect = Exception("Command failed")

    with patch("builtins.print") as mock_print:
        run_autogluon_assistant(data_dir)
        mock_print.assert_called_once_with("An error occurred: Command failed")
    assert mock_session_state.process is None
    assert mock_session_state.pid is None


def test_invalid_data_dir(mock_session_state, mock_subprocess, mock_generate_output):
    """Test with invalid data directory"""
    data_dir = "/nonexistent/path"
    mock_subprocess.side_effect = FileNotFoundError("No such directory")

    with patch("builtins.print") as mock_print:
        run_autogluon_assistant(data_dir)
        mock_print.assert_called_once_with("An error occurred: No such directory")
    assert mock_session_state.process is None
    assert mock_session_state.pid is None
