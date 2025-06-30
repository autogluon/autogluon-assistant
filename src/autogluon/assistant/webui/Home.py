import os
import sys

import streamlit as st
import streamlit.components.v1 as components

from autogluon.assistant.constants import LOGO_PATH
from autogluon.assistant.webui.start_page import main as start_page

# Check if running in streamlit environment
def is_running_in_streamlit():
    """Check if running in streamlit environment"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        try:
            from streamlit.script_run_context import get_script_run_ctx
            return get_script_run_ctx() is not None
        except ImportError:
            return False


# Only execute page configuration and rendering in streamlit environment
if is_running_in_streamlit():
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
        return "Are you sure you want to leave?";
      };
    </script>
    """

    components.html(reload_warning, height=0)

    # Execute main application logic
    start_page()


def main():
    """Entry point for mlzero-webui command - launches streamlit server."""
    import subprocess
    from pathlib import Path
    
    # Get current file path
    current_file = Path(__file__).resolve()
    
    # Run streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(current_file),
        "--server.port=8509"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down webui...")
    except Exception as e:
        print(f"Error running webui: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # If running this file directly and not in streamlit environment, launch streamlit server
    if not is_running_in_streamlit():
        main()
