# /opt/dlami/nvme/autogluon-assistant/src/autogluon/assistant/webui/utils/utils.py
import os
import uuid
import zipfile
from pathlib import Path
import streamlit as st

def get_user_data_dir() -> Path:
    """
    Returns a per-session folder in the user's home directory, creating it if needed.
    """
    base = Path.home() / ".autogluon_assistant" / st.session_state.get("user_session_id", "default")
    base.mkdir(parents=True, exist_ok=True)
    return base

def save_and_extract_zip(uploaded_zip) -> str:
    """
    Saves the uploaded ZipFile to a unique temp folder under the user_data_dir,
    then extracts it there. Returns the extraction directory path.
    """
    data_dir = get_user_data_dir()
    run_id = uuid.uuid4().hex[:8]
    extract_dir = data_dir / f"upload_{run_id}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_path = extract_dir / uploaded_zip.name
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    return str(extract_dir)
