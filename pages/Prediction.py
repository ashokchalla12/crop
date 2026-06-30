"""OptiCrop — Smart Crop Recommendation Page."""

from datetime import datetime

import streamlit as st

from utils.helper import (
    apply_custom_css,
    generate_prediction_pdf,
    get_crop_image_path,
    get_crop_info,
    init_session_state,
    predict_crop,
    render_page_header,
)
from utils.preprocessing import FEATURE_COLUMNS, INPUT_RANGES, validate_inputs
from utils.charts import prediction_confidence_chart


def _default_values() -> dict[str, float]:
    return {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0}


def render() -> None:
    init_session_state()
    apply_custom_css(st.session_state.get("dark_mode", False))
    render_page_header(
        "Smart Crop Recommendation",
        "Enter soil and environmental parameters to get AI-powered crop suggestions.",
        "🌱",
    )

    if "form_values" not in st.session_state:
        st.session_state.form_values = _default_values()

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("#### 🧪 Soil & Environmental Parameters")
        c1, c2, c3 = st.columns(3)

        values = {}
        labels = {
            "N": ("Nitrogen (N)", "Ratio of nitrogen content in soil"),
            "P": ("Phosphorous (P)", "Ratio of phosphorous content in soil"),
            "K": ("Potassium (K)", "Ratio of potassium content in soil"),
            "temperature": ("Temperature (°C)", "Average temperature in Celsius"),
            "humidity": ("Humidity (%)", "Relative humidity percentage"),
            "ph": ("pH", "Soil pH value"),
            "rainfall": ("Rainfall (mm)", "Annual rainfall in millimeters"),
        }

        cols = [c1, c2, c3]
        for i, col_name in enumerate(FEATURE_COLUMNS):
            lo, hi = INPUT_RANGES[col_name]
            label, help_text = labels[col_name]
            with cols[i % 3]:
                values[col_name] = st.number_input(
                    label, min_value=float(lo), max_value=float(hi),
                    value=float(st.session_state.form_values.get(col_name, _default_values()[col_name])),
                    step=0.1 if col_name in ("temperature", "humidity", "ph", "rainfall") else 1.0,
                    help=help_text,
                )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
        with btn_col1:
            submitted = st.form_submit_button("🔮 Predict Crop", type="primary", use_container_width=True)
        with btn_col2:
            reset = st.form_submit_button("🔄 Reset Form", use_container_width=True)
        with btn_col3:
            pass

    if reset:
        st.session_state.form_values = _default_values()
        st.toast("Form reset to default values!", icon="🔄")
        st.rerun()

    if submitted:
        is_valid, errors = validate_inputs(values)
        if not is_valid:
            for err in errors:
                st.error(err)
            return

        st.session_state.form_values = values

        with st.spinner("Analyzing soil conditions and running ML model..."):
            progress = st.progress(0)
            progress.progress(30)
            try:
                crop, confidence, all_probs = predict_crop(values)
                progress.progress(100)
            except FileNotFoundError as e:
                st.error(str(e))
                st.warning("Run `python train_model.py` to train and save the model first.")
                return
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        st.session_state.last_prediction = {
            "crop": crop,
            "confidence": confidence,
            "inputs": values.copy(),
            "probabilities": all_probs,
            "timestamp": datetime.now().isoformat(),
        }
        info = get_crop_info(crop)
        st.session_state.last_prediction["info"] = info

        history = st.session_state.prediction_history
        history.insert(0, st.session_state.last_prediction.copy())
        st.session_state.prediction_history = history[:20]

        st.toast(f"Recommended: {crop.title()}!", icon="✅")

    if st.session_state.get("last_prediction"):
        pred = st.session_state.last_prediction
        crop = pred["crop"]
        info = get_crop_info(crop)

        st.success(f"✅ Analysis complete — best match found!")
        st.markdown(f'<p class="crop-result">🌾 Recommended Crop: {crop.title()}</p>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction Confidence", f"{pred['confidence']:.1f}%")
        m2.metric("Crop Category", "Cereal" if crop in ("rice", "maize") else "Specialty")
        m3.metric("Analysis Time", pred.get("timestamp", "N/A")[:19].replace("T", " "))

        left, right = st.columns([1, 1])
        with left:
            img_path = get_crop_image_path(crop)
            if img_path:
                st.image(str(img_path), caption=f"{crop.title()} — Recommended Crop", use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #d8f3dc, #52b788);
                                border-radius: 16px; padding: 3rem; text-align: center; font-size: 5rem;">
                        {info.get('emoji', '🌱')}
                    </div>
                    <p style="text-align:center; font-weight:600;">{crop.title()}</p>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("📋 Detailed Crop Information", expanded=True):
                st.markdown(f"**Suitable Soil Condition:** {info['soil']}")
                st.markdown(f"**Growing Tips:** {info['tips']}")
                st.markdown(f"**Expected Yield:** {info['yield']}")
                st.markdown(f"**Advantages:** {info['advantages']}")
                st.markdown(f"**Market Demand:** {info['market']}")

        with right:
            if pred.get("probabilities"):
                st.plotly_chart(
                    prediction_confidence_chart(pred["probabilities"]),
                    use_container_width=True,
                )

        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            pdf_bytes = generate_prediction_pdf(pred)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"OptiCrop_{crop}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with d2:
            import json
            report_json = json.dumps(
                {k: v for k, v in pred.items() if k != "probabilities"},
                indent=2, default=str,
            )
            st.download_button(
                "📥 Download JSON Report",
                data=report_json,
                file_name=f"OptiCrop_{crop}_report.json",
                mime="application/json",
                use_container_width=True,
            )

    if st.session_state.prediction_history:
        with st.expander("📜 Prediction History (last 20)", expanded=False):
            for i, h in enumerate(st.session_state.prediction_history):
                st.markdown(
                    f"**{i+1}.** {h['crop'].title()} — "
                    f"{h['confidence']:.1f}% — {h.get('timestamp', '')[:19]}"
                )
