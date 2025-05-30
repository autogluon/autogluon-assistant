import os
import uuid
from pathlib import Path
import zipfile
import streamlit as st

from autogluon.assistant.webui.utils.utils import get_user_data_dir

def handle_uploaded_files(uploaded_files) -> str:
    """
    统一处理用户通过 st.chat_input 上传的文件列表：
    - 如果只有一个 ZIP，解压到独立子目录，返回目录路径
    - 否则，将所有文件原样写入独立子目录，返回目录路径
    """
    user_dir = get_user_data_dir()
    run_id = uuid.uuid4().hex[:8]
    target_dir = user_dir / f"upload_{run_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 只有一个 ZIP 的情况
    if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".zip"):
        zip_file = uploaded_files[0]
        zip_path = target_dir / zip_file.name
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        # 删除原 zip 文件（可选）
        # zip_path.unlink()
        return str(target_dir)

    # 其他情况：逐个写出文件
    for up in uploaded_files:
        with open(target_dir / up.name, "wb") as f:
            f.write(up.getbuffer())
    return str(target_dir)
