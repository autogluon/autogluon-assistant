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
    st.video(DEMO_URL, muted=True, autoplay=True)


def demo():
    """
    The demo section to show a video
    """
    col1, col2, col3, col4 = st.columns([1, 6, 10, 1])
    with col2:
        st.markdown(
            """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap")
        </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        <div style='padding: 0px; background-color: white;'>
            <h1 style='font-size: 2.5rem; font-weight: normal; margin-bottom: 0; padding-top: 0;line-height: 1.2;'>Quick Demo!</h1>
            <h2 style='font-size: 2.5rem; font-weight: normal; margin-top: 0; line-height: 1.2;'>Learn about AG-A</h2>
        </div>
            """,
            unsafe_allow_html=True,
        )
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
            <div class="description">AutoGluon Assistant (AG-A) provides users a simple interface where they can upload their data, describe their problem, and receive a highly accurate and competitive ML solution — without writing any code. By leveraging the state-of-the-art AutoML capabilities of AutoGluon and integrating them with a Large Language Model (LLM), AG-A automates the entire data science pipeline.</div>
            <div class="steps">
                <ol>
                    <li>Upload dataset files (CSV,XLSX)</li>
                    <li>Define your machine learning task</li>
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


if __name__ == "__main__":
    main()
