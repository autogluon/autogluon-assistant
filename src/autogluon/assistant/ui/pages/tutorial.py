import base64
import os

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
static_dir = os.path.join(parent_dir, "static")
bin_file = os.path.join(static_dir, "background.png")


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


if __name__ == "__main__":
    main()
