"""
OptiCrop — Smart Agricultural Production Optimization Engine
Main Streamlit application entry point.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pages import About, Dashboard, Dataset, Home, Prediction, Suitability
from utils.helper import apply_custom_css, init_session_state

st.set_page_config(
    page_title="OptiCrop — Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "home": ("🏠 Home", Home),
    "predict": ("🌱 Predict Crop", Prediction),
    "suitability": ("📊 Crop Suitability", Suitability),
    "dashboard": ("📈 Research Dashboard", Dashboard),
    "dataset": ("🗂️ Dataset Explorer", Dataset),
    "about": ("ℹ️ About", About),
}


def render_sidebar() -> str:
    init_session_state()

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <span style="font-size: 2.5rem;">🌾</span>
                <h2 style="margin: 0; color: white;">OptiCrop</h2>
                <p style="color: #b7e4c7; font-size: 0.85rem;">Smart Agri Optimization</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        labels = {k: v[0] for k, v in PAGES.items()}
        if "page" not in st.session_state:
            st.session_state.page = "home"

        selected_label = st.radio(
            "Navigation",
            options=list(labels.values()),
            index=list(labels.keys()).index(st.session_state.page),
            label_visibility="collapsed",
        )
        page_key = [k for k, v in labels.items() if v == selected_label][0]
        st.session_state.page = page_key

        st.divider()
        dark = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False))
        st.session_state.dark_mode = dark

        st.divider()
        st.markdown("##### ⚡ Quick Stats")
        st.caption("22 Crops • 7 Features • 6 ML Models")

        st.divider()
        st.caption("OptiCrop v1.0 — Sustainable Agriculture")

    return page_key


def main() -> None:
    page_key = render_sidebar()
    apply_custom_css(st.session_state.get("dark_mode", False))

    module = PAGES[page_key][1]
    module.render()


if __name__ == "__main__":
    main()
