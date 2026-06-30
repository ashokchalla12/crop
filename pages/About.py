"""OptiCrop — About Page."""

import streamlit as st

from utils.helper import apply_custom_css, init_session_state, render_page_header


def render() -> None:
    init_session_state()
    apply_custom_css(st.session_state.get("dark_mode", False))
    render_page_header("About OptiCrop", "Learn about the project, technology, and vision.", "ℹ️")

    st.markdown(
        """
        ### 🌾 Problem Statement
        Farmers worldwide face critical decisions about which crops to plant based on soil composition
        and environmental factors. Wrong crop selection leads to reduced yields, wasted resources,
        and economic losses. Traditional decision-making relies on experience alone, which may not
        account for the complex interplay of nitrogen, phosphorous, potassium, temperature, humidity,
        pH, and rainfall.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ### 🎯 Objectives
            - Provide AI-driven crop recommendations
            - Assess crop suitability for specific field conditions
            - Visualize agricultural data patterns
            - Train and compare multiple ML algorithms
            - Support sustainable, data-informed farming
            """
        )
    with col2:
        st.markdown(
            """
            ### 🛠️ Technology Stack
            | Layer | Technology |
            |-------|-----------|
            | Language | Python 3.10+ |
            | Frontend | Streamlit |
            | ML | Scikit-learn |
            | Data | Pandas, NumPy |
            | Viz | Plotly, Matplotlib |
            | Persistence | Joblib, Pickle |
            """
        )

    st.markdown(
        """
        ### 🔄 Workflow
        ```
        Data Collection → Preprocessing → Feature Scaling → Model Training
              → Evaluation → Best Model Selection → Deployment → Prediction
        ```
        """
    )

    with st.expander("📂 Dataset Details", expanded=True):
        st.markdown(
            """
            **Source:** Crop Recommendation Dataset (2,200 records, 22 crop classes)

            **Input Features:**
            - **N** — Nitrogen content in soil
            - **P** — Phosphorous content in soil
            - **K** — Potassium content in soil
            - **Temperature** — Average temperature (°C)
            - **Humidity** — Relative humidity (%)
            - **pH** — Soil acidity/alkalinity
            - **Rainfall** — Annual rainfall (mm)

            **Target:** Crop label (22 supported crops)
            """
        )

    with st.expander("🚀 Future Scope"):
        st.markdown(
            """
            - Integration with IoT soil sensors for real-time data
            - Weather API integration for live forecasts
            - Mobile application for field use
            - Multi-language support for regional farmers
            - Deep learning models (CNN, LSTM) for time-series prediction
            - Government subsidy and MSP price integration
            - Geospatial mapping with satellite imagery
            """
        )

    st.markdown("---")
    st.markdown(
        """
        ### 👨‍💻 Developer Information
        **Project:** Smart Agricultural Production Optimization Engine (OptiCrop)

        **Version:** 1.0.0

        **Built with:** Python • Streamlit • Scikit-learn

        *Empowering farmers with intelligent, data-driven agricultural decisions.*
        """
    )

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1b4332, #40916c);
                    color: white; padding: 2rem; border-radius: 12px; text-align: center;">
            <h3>🌱 Grow Smart. Farm Better. Choose OptiCrop.</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
