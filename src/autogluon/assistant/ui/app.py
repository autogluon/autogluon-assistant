import os
from copy import deepcopy

import streamlit as st
import streamlit.components.v1 as components

from autogluon.assistant.ui.constants import DEFAULT_SESSION_VALUES, LOGO_PATH
from autogluon.assistant.ui.pages.demo import main as demo
from autogluon.assistant.ui.pages.feature import main as feature
from autogluon.assistant.ui.pages.nav_bar import nav_bar
from autogluon.assistant.ui.pages.preview import main as preview
from autogluon.assistant.ui.pages.task import main as run
from autogluon.assistant.ui.pages.tutorial import main as tutorial

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


def initial_session_state():
    """
    Initial Session State
    """
    for key, default_value in DEFAULT_SESSION_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = (
                deepcopy(default_value) if isinstance(default_value, (dict, list)) else default_value
            )


def main():
    initial_session_state()
    nav_bar()
    tutorial()
    demo()
    feature()
    run()
    preview()

    st.markdown(
        """
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.14.3/dist/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.1.3/dist/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
