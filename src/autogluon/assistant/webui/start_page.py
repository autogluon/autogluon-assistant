import base64
import os

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from autogluon.assistant.constants import DEMO_URL

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
bin_file = os.path.join(static_dir, "background.png")


def video():
    """
    Display Demo video
    """
    st.video(DEMO_URL, muted=True, autoplay=True, end_time=248, start_time=0, loop=True)


def demo():
    """
    The demo section to show a video and a centered Get Started button.
    """
    # 1) Inject our button CSS
    st.markdown(
        """
        <style>
        /* style the primary Streamlit button */
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
        # Your headings
        st.markdown(
            """
            <h1 style='font-size:2.5rem; line-height:1.2;'>Quick Demo!</h1>
            <h2 style='font-size:2.5rem; line-height:1.2; margin-top:0;'>Learn about AG-A</h2>
            """,
            unsafe_allow_html=True,
        )

        st.write("")  # small spacer

        # 2) Center the fixed-size button in a flex wrapper
        st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        if st.button("Get Started", key="get_started"):
            st.switch_page("pages/Run_dataset.py")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        video()


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = (
        """
    <style>
    @media (max-width: 800px) {
        .left-section {
            font-size: 0.9rem;
            width: 100vw !important;
            background-color: white !important;
            justify-content: center;
            background-size: 120vw !important;
            min-height: 20vh !important;
        }
    }
    .left-section {
        width: 47vw;
        background-image: url("data:image/png;base64,%s");
        background-size: 45vw;
        background-repeat: no-repeat;
        background-position: left;
        display: flex;
        background-color: #ececec;
        flex-direction: column;
        min-height: 70vh;
    }
    </style>
    """
        % bin_str
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)


def features():
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
                Autogluon Assistant analyses user’s task description and dataset files, translating them into actionable machine learning objectives without manual intervention.
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


def main():
    set_png_as_page_bg(bin_file)
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
    demo()
    add_vertical_space(5)
    st.markdown("---", unsafe_allow_html=True)
    features()
    st.markdown("---", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
