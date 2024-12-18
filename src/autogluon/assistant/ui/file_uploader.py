import os

import pandas as pd
import streamlit as st

from autogluon.assistant.ui.utils import get_user_data_dir


def save_description_file(description):
    """
    Save the task description to a file in the user's data directory.

    Args:
        description (str): The task description to save.
    """
    try:
        user_data_dir = get_user_data_dir()
        description_file = os.path.join(user_data_dir, "description.txt")
        with open(description_file, "w") as f:
            f.write(description)
    except Exception as e:
        print(f"Error saving file: {str(e)}")


def description_file_uploader():
    """
    Handle Description file uploads
    """
    uploaded_file = st.file_uploader(
        "Upload task description file",
        type="txt",
        key=st.session_state.description_uploader_key,
        help="Accepted file format: .txt",
        label_visibility="collapsed",
    )
    if uploaded_file:
        task_description = uploaded_file.read().decode("utf-8")
        st.session_state.task_description = task_description
        save_description_file(st.session_state.task_description)
        st.session_state.description_uploader_key += 1
        st.rerun()


def file_uploader():
    """
    Handle file uploads
    """
    header_html = """
     <div class="header-hover">
         <h4>Upload Dataset</h4>
         <div class="tooltip">
             Recommended file naming:<br>
             • train.csv/xlsx<br>
             • test.csv/xlsx<br>
             • sample_dataset.csv/xlsx
         </div>
     </div>
     """
    st.markdown(header_html, unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Select the dataset", accept_multiple_files=True, label_visibility="collapsed", type=["csv", "xlsx"]
    )
    st.session_state.uploaded_files = {}
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        st.session_state.uploaded_files[file.name] = {"file": file, "df": df}
