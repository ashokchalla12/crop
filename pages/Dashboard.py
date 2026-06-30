"""OptiCrop — Research Dashboard & ML Training Page."""

import json
from pathlib import Path

import streamlit as st

from utils.charts import (
    confusion_matrix_heatmap,
    correlation_heatmap,
    crop_distribution_pie,
    crop_frequency_bar,
    humidity_vs_temp_scatter,
    model_comparison_bar,
    npk_distribution,
    rainfall_analysis,
    scatter_npk,
    temperature_histogram,
)
from utils.helper import (
    apply_custom_css,
    get_base_dir,
    init_session_state,
    load_dataset,
    load_metrics,
    load_ml_artifacts,
    render_page_header,
)
from utils.charts import feature_importance_chart
from utils.preprocessing import FEATURE_COLUMNS


def render() -> None:
    init_session_state()
    apply_custom_css(st.session_state.get("dark_mode", False))
    render_page_header(
        "Research Dashboard",
        "Interactive analytics, visualizations, and machine learning model performance.",
        "📈",
    )

    df = load_dataset()
    metrics = load_metrics()

    tab_analytics, tab_ml = st.tabs(["📊 Data Analytics", "🤖 ML Model Training"])

    with tab_analytics:
        st.markdown("### 🌍 Agricultural Data Analytics")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(crop_distribution_pie(df), use_container_width=True)
        with r1c2:
            st.plotly_chart(temperature_histogram(df), use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.plotly_chart(npk_distribution(df), use_container_width=True)
        with r2c2:
            st.plotly_chart(correlation_heatmap(df), use_container_width=True)

        st.plotly_chart(crop_frequency_bar(df), use_container_width=True)
        st.plotly_chart(rainfall_analysis(df), use_container_width=True)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            st.plotly_chart(scatter_npk(df), use_container_width=True)
        with r3c2:
            st.plotly_chart(humidity_vs_temp_scatter(df), use_container_width=True)

        with st.expander("📋 Dataset Summary Statistics"):
            st.dataframe(df.describe(), use_container_width=True)

    with tab_ml:
        st.markdown("### 🤖 Machine Learning Model Training & Evaluation")

        col_train, col_info = st.columns([1, 2])
        with col_train:
            if st.button("🔄 Train / Retrain All Models", type="primary", use_container_width=True):
                with st.spinner("Training 6 ML algorithms — this may take a minute..."):
                    progress = st.progress(0)
                    try:
                        from train_model import train_all_models
                        progress.progress(50)
                        result = train_all_models()
                        progress.progress(100)
                        st.session_state["train_result"] = result
                        st.success(f"✅ Best model: **{result['best_model']}** (Accuracy: {result['best_accuracy']:.2%})")
                        st.toast("Models trained and saved!", icon="🎉")
                        st.cache_resource.clear()
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

            st.info(
                "Trains: Random Forest, Decision Tree, SVM, KNN, "
                "Naive Bayes, Logistic Regression. Best model auto-saved."
            )

        metrics = load_metrics()
        if not metrics:
            st.warning("No trained models found. Click **Train / Retrain All Models** to begin.")
            return

        with col_info:
            st.metric("Best Model", metrics.get("best_model", "N/A"))
            st.metric("Best Accuracy", f"{metrics.get('best_accuracy', 0):.2%}")

        comp_fig = model_comparison_bar(metrics)
        if comp_fig:
            st.plotly_chart(comp_fig, use_container_width=True)

        st.markdown("#### 📊 Model Metrics Comparison")
        rows = []
        for name, m in metrics.get("models", {}).items():
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
            })
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).style.highlight_max(axis=0, subset=["Accuracy", "F1 Score"]),
                         use_container_width=True)

        best_name = metrics.get("best_model", "")
        best_metrics = metrics.get("models", {}).get(best_name, {})
        classes = metrics.get("classes", [])

        if best_metrics.get("confusion_matrix") and classes:
            st.plotly_chart(
                confusion_matrix_heatmap(
                    best_metrics["confusion_matrix"], classes,
                    title=f"Confusion Matrix — {best_name}",
                ),
                use_container_width=True,
            )

        model, _, _ = load_ml_artifacts()
        if model is not None:
            fi_fig = feature_importance_chart(model, FEATURE_COLUMNS)
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            elif hasattr(model, "coef_"):
                import pandas as pd
                import numpy as np
                coef = np.abs(model.coef_).mean(axis=0)
                imp_df = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": coef})
                st.bar_chart(imp_df.set_index("Feature"))

        with st.expander("📄 Raw Metrics JSON"):
            st.code(json.dumps(metrics, indent=2), language="json")

        metrics_path = get_base_dir() / "model" / "metrics.json"
        if metrics_path.exists():
            st.download_button(
                "📥 Download Metrics JSON",
                data=metrics_path.read_text(encoding="utf-8"),
                file_name="model_metrics.json",
                mime="application/json",
            )
