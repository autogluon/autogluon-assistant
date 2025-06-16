import os
from copy import deepcopy

import streamlit as st
import streamlit.components.v1 as components

from autogluon.assistant.constants import LOGO_PATH
from autogluon.assistant.webui.start_page import main as start_page

st.set_page_config(
    page_title="AutoGluon Assistant",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# fontawesome
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

# Bootstrap 4.1.3
st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.1.3/dist/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
    """,
    unsafe_allow_html=True,
)
current_dir = os.path.dirname(os.path.abspath(__file__))

css_file_path = os.path.join(current_dir, "style.css")

with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


reload_warning = """
<script>
  window.onbeforeunload = function () {

    return  "Are you sure want to LOGOUT the session ?";
};
</script>
"""

components.html(reload_warning, height=0)


def main():
    start_page()


if __name__ == "__main__":
    main()
