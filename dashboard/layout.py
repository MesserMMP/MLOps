import streamlit as st
from config import PRIMARY_COLOR, BG_LIGHT, BORDER_COLOR


def setup_page():
    st.set_page_config(
        page_title="ML MLOps Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .stSidebar {{
            background-color: #F3F4F6;
        }}
        h1, h2, h3 {{
            color: {PRIMARY_COLOR};
        }}
        .card {{
            padding: 1.5rem 1.75rem;
            background-color: {BG_LIGHT};
            border-radius: 0.75rem;
            border: 1px solid {BORDER_COLOR};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
