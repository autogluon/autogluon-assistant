import uuid
import zipfile

from autogluon.assistant.webui.utils.utils import get_user_data_dir


def handle_uploaded_files(uploaded_files) -> str:
    """
    Handle files uploaded by user through st.chat_input:
    - If only one ZIP file, extract it to an independent subdirectory and return the directory path
    - Otherwise, write all files as-is to an independent subdirectory and return the directory path
    """
    user_dir = get_user_data_dir()
    run_id = uuid.uuid4().hex[:8]
    target_dir = user_dir / f"upload_{run_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Handle single ZIP file case
    if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".zip"):
        zip_file = uploaded_files[0]
        zip_path = target_dir / zip_file.name
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        # Optionally delete the original zip file
        # zip_path.unlink()
        return str(target_dir)

    # Other cases: write out files individually
    for up in uploaded_files:
        with open(target_dir / up.name, "wb") as f:
            f.write(up.getbuffer())
    return str(target_dir)
