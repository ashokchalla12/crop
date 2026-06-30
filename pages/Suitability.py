"""OptiCrop — Crop Suitability Assessment Page."""

import streamlit as st

from utils.charts import ideal_vs_current_bar, ideal_vs_current_radar
from utils.helper import (
    SUPPORTED_CROPS,
    apply_custom_css,
    compute_suitability_score,
    get_ideal_conditions,
    get_recommendations,
    init_session_state,
    render_page_header,
    suitability_status,
)
from utils.preprocessing import FEATURE_COLUMNS, INPUT_RANGES, validate_inputs


def render() -> None:
    init_session_state()
    apply_custom_css(st.session_state.get("dark_mode", False))
    render_page_header(
        "Crop Suitability Assessment",
        "Select a crop and compare your field conditions against ideal requirements.",
        "📊",
    )

    crop = st.selectbox(
        "🌾 Select Crop",
        options=SUPPORTED_CROPS,
        format_func=lambda x: x.title(),
        help="Choose the crop you want to evaluate for your field conditions.",
    )

    ideal = get_ideal_conditions(crop)

    st.markdown("#### 🌡️ Enter Your Field Conditions")
    c1, c2, c3 = st.columns(3)
    labels = {
        "N": "Nitrogen (N)", "P": "Phosphorous (P)", "K": "Potassium (K)",
        "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
        "ph": "pH", "rainfall": "Rainfall (mm)",
    }
    values = {}
    cols = [c1, c2, c3]
    for i, col in enumerate(FEATURE_COLUMNS):
        lo, hi = INPUT_RANGES[col]
        with cols[i % 3]:
            values[col] = st.number_input(
                labels[col],
                min_value=float(lo), max_value=float(hi),
                value=float(ideal[col]),
                step=0.1 if col in ("temperature", "humidity", "ph", "rainfall") else 1.0,
                key=f"suit_{col}",
            )

    if st.button("📈 Assess Suitability", type="primary", use_container_width=True):
        is_valid, errors = validate_inputs(values)
        if not is_valid:
            for err in errors:
                st.error(err)
            return

        score = compute_suitability_score(values, ideal)
        status, color = suitability_status(score)
        recs = get_recommendations(values, ideal)

        st.markdown("---")
        st.markdown("### 📋 Assessment Results")

        s1, s2, s3 = st.columns(3)
        s1.metric("Suitability Score", f"{score}%")
        s2.markdown(
            f"""
            <div style="background:{color}; color:white; padding:12px; border-radius:10px; text-align:center;">
                <strong>Status: {status}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s3.metric("Crop Selected", crop.title())

        if score >= 85:
            st.success(f"🌟 Excellent! Your conditions are highly suitable for {crop.title()}.")
        elif score >= 70:
            st.info(f"✅ Good suitability for {crop.title()} with minor adjustments possible.")
        elif score >= 50:
            st.warning(f"⚠️ Moderate suitability — consider improvements before planting {crop.title()}.")
        else:
            st.error(f"❌ Poor suitability — significant changes needed or consider another crop.")

        tab1, tab2 = st.tabs(["📊 Bar Chart", "🎯 Radar Chart"])
        with tab1:
            st.plotly_chart(ideal_vs_current_bar(values, ideal, crop), use_container_width=True)
        with tab2:
            st.plotly_chart(ideal_vs_current_radar(values, ideal, crop), use_container_width=True)

        st.markdown("#### 💡 Recommendations")
        for rec in recs:
            st.markdown(f"- {rec}")

        with st.expander("📌 Ideal Conditions Reference"):
            ref_cols = st.columns(4)
            display_labels = list(labels.values())
            for i, col in enumerate(FEATURE_COLUMNS):
                with ref_cols[i % 4]:
                    st.metric(display_labels[i], f"{ideal[col]:.1f}")
