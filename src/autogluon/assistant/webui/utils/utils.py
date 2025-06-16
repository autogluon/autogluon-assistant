# src/autogluon/assistant/webui/utils/utils.py
import json
import uuid
import zipfile
from pathlib import Path

import streamlit as st


def _path(session_id: str) -> Path:
    """Get path for session chat history file"""
    base = Path.home() / ".autogluon_assistant" / session_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "chat.json"


def load_messages(session_id: str) -> list[dict]:
    """Load messages from session file"""
    p = _path(session_id)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def save_messages(session_id: str, messages: list[dict]) -> None:
    """Save messages to session file"""
    p = _path(session_id)
    p.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


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
