import os
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from autogluon.assistant.ui.file_uploader import description_file_uploader, file_uploader, save_description_file


@pytest.fixture
def mock_session_state():
    """Fixture to mock streamlit session state"""
    with patch("streamlit.session_state") as mock_state:
        mock_state.description_uploader_key = 0
        mock_state.task_description = ""
        mock_state.uploaded_files = {}
        yield mock_state


@pytest.fixture
def mock_user_data_dir(tmp_path):
    """Fixture to create a temporary user data directory"""
    with patch("autogluon.assistant.ui.file_uploader.get_user_data_dir", return_value=str(tmp_path)):
        yield tmp_path


def test_save_description_file(mock_user_data_dir):
    """Test saving description file"""
    test_description = "Test task description"
    save_description_file(test_description)

    description_file = os.path.join(mock_user_data_dir, "description.txt")
    assert os.path.exists(description_file)

    with open(description_file, "r") as f:
        saved_content = f.read()
    assert saved_content == test_description


@patch("streamlit.file_uploader")
def test_description_file_uploader(mock_uploader, mock_session_state):
    """Test description file uploader"""
    mock_file = Mock()
    mock_file.read.return_value = b"Test description"
    mock_uploader.return_value = mock_file

    with patch("autogluon.assistant.ui.file_uploader.save_description_file") as mock_save:
        description_file_uploader()
        assert mock_session_state.task_description == "Test description"
        mock_save.assert_called_once_with("Test description")
        assert mock_session_state.description_uploader_key == 1


@patch("streamlit.file_uploader")
def test_file_uploader_csv(mock_uploader, mock_session_state):
    """Test CSV file upload handling"""
    csv_content = "col1,col2\n1,2\n3,4"
    mock_file = Mock()
    mock_file.name = "test.csv"
    mock_file.read.return_value = csv_content.encode()
    df = pd.read_csv(StringIO(csv_content))
    with patch("pandas.read_csv", return_value=df):
        mock_uploader.return_value = [mock_file]
        file_uploader()
        assert "test.csv" in mock_session_state.uploaded_files
        assert mock_session_state.uploaded_files["test.csv"]["file"] == mock_file
        pd.testing.assert_frame_equal(mock_session_state.uploaded_files["test.csv"]["df"], df)


@patch("streamlit.file_uploader")
def test_file_uploader_excel(mock_uploader, mock_session_state):
    """Test Excel file upload handling"""
    mock_file = Mock()
    mock_file.name = "test.xlsx"
    df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
    with patch("pandas.read_excel", return_value=df):
        mock_uploader.return_value = [mock_file]
        file_uploader()
        assert "test.xlsx" in mock_session_state.uploaded_files
        assert mock_session_state.uploaded_files["test.xlsx"]["file"] == mock_file
        pd.testing.assert_frame_equal(mock_session_state.uploaded_files["test.xlsx"]["df"], df)


def test_file_uploader_no_files(mock_session_state):
    """Test file uploader with no files"""
    with patch("streamlit.file_uploader", return_value=[]):
        file_uploader()
        assert mock_session_state.uploaded_files == {}
