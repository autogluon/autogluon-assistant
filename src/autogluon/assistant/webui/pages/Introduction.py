import base64
import os

import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.add_vertical_space import add_vertical_space

from autogluon.assistant.constants import DEMO_URL

# Get current directory and static files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
static_dir = os.path.join(parent_dir, "static")
background_file = os.path.join(static_dir, "background.png")


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


def get_base64_of_bin_file(bin_file):
    """Convert binary file to base64 string"""
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background_image():
    """Set the background image for the page"""
    bin_str = get_base64_of_bin_file(background_file)
    page_bg_img = f"""
    <style>
    @media (max-width: 800px) {{
        .left-section {{
            font-size: 0.9rem;
            width: 100vw !important;
            background-color: white !important;
            justify-content: center;
            background-size: 120vw !important;
            min-height: 20vh !important;
        }}
    }}
    .left-section {{
        width: 47vw;
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: 45vw;
        background-repeat: no-repeat;
        background-position: left;
        display: flex;
        background-color: #ececec;
        flex-direction: column;
        min-height: 70vh;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def render_demo_section():
    """Render the demo section with video and get started button"""
    # Button styling
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #007bff !important;
            color: white !important;
            border: none !important;
            border-radius: 0px !important;
            width: 160px !important;
            height: 48px !important;
            font-size: 1rem !important;
            font-weight: bold !important;
            margin: 0 auto !important;
        }
        div.stButton > button:hover {
            background-color: #0056b3 !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([1, 6, 10, 1])

    with col2:
        st.markdown(
            """
            <h1 style='font-size:2.5rem; line-height:1.2;'>Quick Demo!</h1>
            <h2 style='font-size:2.5rem; line-height:1.2; margin-top:0;'>Learn about AG-A</h2>
        """,
            unsafe_allow_html=True,
        )

        st.write("")  # spacer

        # Center the button
        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        if st.button("Get Started", key="launch_mlzero"):
            st.switch_page("Launch_MLZero.py")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.video(DEMO_URL, muted=True, autoplay=True, end_time=248, start_time=0, loop=True)


def render_features():
    """Render the features section"""
    st.markdown(
        """
        <h1 style='
            font-weight: light;
            padding-left: 20px;
            padding-right: 20px;
            margin-left:60px;
            font-size: 2em;
        '>
            Features of AutoGluon Assistant
        </h1>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([1, 10, 10, 1])

    # Feature 1
    with col2:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">LLM based Task Understanding</div>
            <div class="feature-description">
                Leverage the power of Large Language Models to automatically interpret and understand data science tasks. 
                Autogluon Assistant analyses user's task description and dataset files, translating them into actionable machine learning objectives without manual intervention.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Feature 2
    with col3:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">Automated Feature Engineering</div>
            <div class="feature-description">
                Streamline your data preparation process with our advanced automated feature engineering.
                Our AI identifies relevant features, handles transformations, and creates new meaningful variables,
                significantly reducing time spent on data preprocessing.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Feature 3
    with col2:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">Powered by Multi-Model Integration</div>
            <div class="feature-description">
            Leverage a unified platform that brings together multiple modeling capabilities. 
            AutoGluon Assistant now supports AutoGluon Time Series, Multi-Model Pipelines, and Tabular Modeling, enabling end-to-end automation across diverse ML tasks. 
            Integration with more Hugging Face models is coming soon, 
            expanding support for advanced NLP and multimodal applications—all without requiring deep ML expertise.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">Coming Soon</div>
            <div class="feature-description">
                Exciting new features are on the horizon! Our team is working on innovative capabilities 
                to enhance your AutoML experience. Stay tuned for updates that will further simplify 
                and improve your machine learning workflow.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_main_page():
    """Render the main landing page"""
    set_background_image()

    st.markdown(
        """
    <div class="main-container" id="get-started">
        <div class="left-section">
            <div class="titleWithLogo">
                <div class="title">AutoGluon<br>Assistant</div>
                <div class="logo">
                    <img src="https://auto.gluon.ai/stable/_images/autogluon-s.png" alt="AutoGluon Logo">
                </div>
            </div>
            <div class="subtitle">Fast and Accurate ML in 0 Lines of Code</div>
        </div>
        <div class="right-section">
            <div class="get-started-title">Get Started</div>
            <div class="description">AutoGluon Assistant (aka MLZero) is a multi-agent system that automates end-to-end multimodal machine learning or deep learning workflows by transforming raw multimodal data into high-quality ML solutions with zero human intervention. Leveraging specialized perception agents, dual-memory modules, and iterative code generation, it handles diverse data formats while maintaining high success rates across complex ML tasks.</div>
            <div class="steps">
                <ol>
                    <li>Upload a folder</li>
                    <li>Launch AutoGluon Assistant</li>
                    <li>Get accurate predictions</li>
                </ol>
            </div>    
        </div> 
    </div>
    """,
        unsafe_allow_html=True,
    )

    add_vertical_space(5)
    render_demo_section()
    add_vertical_space(5)
    st.markdown("---", unsafe_allow_html=True)
    render_features()
    st.markdown("---", unsafe_allow_html=True)


# Only execute page configuration and rendering in streamlit environment
if is_running_in_streamlit():
    # st.set_page_config(
    #     page_title="AutoGluon Assistant",
    #     page_icon=LOGO_PATH,
    #     layout="wide",
    #     initial_sidebar_state="auto",
    # )

    # Load CSS
    css_file_path = os.path.join(parent_dir, "style.css")
    with open(css_file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Reload warning
    reload_warning = """
    <script>
      window.onbeforeunload = function () {
        return "Are you sure you want to leave?";
      };
    </script>
    """
    components.html(reload_warning, height=0)

    # Execute main application logic
    render_main_page()
