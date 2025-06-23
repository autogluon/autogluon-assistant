import os
import sys

import streamlit as st
import streamlit.components.v1 as components

from autogluon.assistant.constants import LOGO_PATH
from autogluon.assistant.webui.start_page import main as start_page

# 判断是否在 streamlit 运行环境中
def is_running_in_streamlit():
    """检查是否在 streamlit 环境中运行"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        try:
            from streamlit.script_run_context import get_script_run_ctx
            return get_script_run_ctx() is not None
        except ImportError:
            return False


# 只有在 streamlit 环境中才执行页面配置和渲染
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

        return  "Are you sure you want to LOGOUT this session?";
    };
    </script>
    """

    components.html(reload_warning, height=0)

    # 执行主应用逻辑
    start_page()


def main():
    """Entry point for mlzero-webui command - launches streamlit server."""
    import subprocess
    from pathlib import Path
    
    # 获取当前文件路径
    current_file = Path(__file__).resolve()
    
    # 运行 streamlit
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
    # 如果直接运行此文件且不在 streamlit 环境中，启动 streamlit 服务器
    if not is_running_in_streamlit():
        main()