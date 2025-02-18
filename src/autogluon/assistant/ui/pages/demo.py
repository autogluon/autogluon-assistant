import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from autogluon.assistant.constants import DEMO_URL


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


def main():
    add_vertical_space(5)
    demo()
    add_vertical_space(5)
    st.markdown("---", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
