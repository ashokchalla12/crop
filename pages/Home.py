"""OptiCrop — Home Page."""

import base64
import streamlit as st

from utils.helper import apply_custom_css, get_base_dir, init_session_state


def _image_to_base64(path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render() -> None:
    init_session_state()
    apply_custom_css(st.session_state.get("dark_mode", False))

    assets = get_base_dir() / "assets"
    bg_path = assets / "background.jpg"
    bg_b64 = _image_to_base64(bg_path)

    if bg_b64:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(rgba(27,67,50,0.85), rgba(45,106,79,0.75)),
                            url('data:image/jpeg;base64,{bg_b64}');
                background-size: cover; background-position: center;
                padding: 3rem 2rem; border-radius: 16px; margin-bottom: 2rem; text-align: center;
            ">
                <h1 class="hero-title" style="color: white;">🌾 OptiCrop</h1>
                <p class="hero-subtitle" style="color: #d8f3dc; font-size: 1.3rem;">
                    Smart Agricultural Production Optimization Engine
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #1b4332, #40916c);
                        padding: 3rem 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem;">
                <h1 class="hero-title" style="color: white;">🌾 OptiCrop</h1>
                <p style="color: #d8f3dc; font-size: 1.3rem;">
                    Smart Agricultural Production Optimization Engine
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        ### 🌍 Project Description
        **OptiCrop** is an intelligent agricultural decision support system that leverages machine learning
        to recommend the best crop based on soil nutrients (N, P, K) and environmental conditions
        (temperature, humidity, pH, rainfall). Empowering farmers with data-driven insights for
        sustainable and profitable agriculture.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🎯 Objectives</h4>
                <ul>
                    <li>Maximize crop yield through smart recommendations</li>
                    <li>Improve resource utilization (water, fertilizer)</li>
                    <li>Support sustainable farming practices</li>
                    <li>Reduce crop failure risk with data-driven decisions</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>✨ Key Features</h4>
                <ul>
                    <li>AI-powered crop recommendation</li>
                    <li>Crop suitability assessment</li>
                    <li>Interactive research dashboard</li>
                    <li>Dataset explorer & ML model training</li>
                    <li>PDF report export & prediction history</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation")
    st.info("Use the **sidebar** to navigate between modules, or click the buttons below.")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("🌱 Predict Crop", use_container_width=True, type="primary"):
            st.session_state["page"] = "predict"
            st.rerun()
    with b2:
        if st.button("📊 Crop Suitability", use_container_width=True):
            st.session_state["page"] = "suitability"
            st.rerun()
    with b3:
        if st.button("📈 Research Dashboard", use_container_width=True):
            st.session_state["page"] = "dashboard"
            st.rerun()
    with b4:
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state["page"] = "about"
            st.rerun()

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Supported Crops", "22", help="Diverse crop types from cereals to horticulture")
    m2.metric("ML Models", "6", help="Random Forest, SVM, KNN, and more")
    m3.metric("Features", "7", help="N, P, K, Temperature, Humidity, pH, Rainfall")
    m4.metric("Dataset Records", "2,200", help="Balanced crop recommendation dataset")
