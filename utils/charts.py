"""
Plotly and Matplotlib chart builders for OptiCrop dashboards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# Agriculture green palette
AGRI_COLORS = [
    "#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2",
    "#b7e4c7", "#1b4332", "#081c15", "#d8f3dc", "#95d5b2",
]

PLOTLY_TEMPLATE = "plotly_white"


def crop_distribution_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["crop", "count"]
    fig = px.pie(
        counts, values="count", names="crop", title="Crop Distribution",
        color_discrete_sequence=AGRI_COLORS, hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
    return fig


def crop_frequency_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["label"].value_counts().sort_values(ascending=True).reset_index()
    counts.columns = ["crop", "count"]
    fig = px.bar(
        counts, x="count", y="crop", orientation="h",
        title="Crop Frequency in Dataset", color="count",
        color_continuous_scale=["#b7e4c7", "#2d6a4f"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=600, showlegend=False)
    return fig


def temperature_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="temperature", nbins=30, title="Temperature Distribution",
        color_discrete_sequence=["#40916c"],
        labels={"temperature": "Temperature (°C)", "count": "Frequency"},
    )
    fig.add_vline(x=df["temperature"].mean(), line_dash="dash", line_color="#e76f51",
                  annotation_text=f"Mean: {df['temperature'].mean():.1f}°C")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
    return fig


def rainfall_analysis(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="label", y="rainfall", title="Rainfall Analysis by Crop",
        color="label", color_discrete_sequence=AGRI_COLORS,
        labels={"rainfall": "Rainfall (mm)", "label": "Crop"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=500, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig


def npk_distribution(df: pd.DataFrame) -> go.Figure:
    melted = df.melt(id_vars=["label"], value_vars=["N", "P", "K"], var_name="nutrient", value_name="value")
    fig = px.violin(
        melted, x="nutrient", y="value", color="nutrient",
        title="NPK Nutrient Distribution", box=True,
        color_discrete_sequence=["#2d6a4f", "#52b788", "#95d5b2"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    corr = numeric.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
    return fig


def scatter_npk(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="N", y="P", color="label", size="K",
        title="N vs P ( sized by K )", opacity=0.7,
        color_discrete_sequence=AGRI_COLORS,
        labels={"N": "Nitrogen", "P": "Phosphorous", "K": "Potassium"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=500)
    return fig


def humidity_vs_temp_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="temperature", y="humidity", color="label",
        title="Temperature vs Humidity by Crop",
        color_discrete_sequence=AGRI_COLORS, opacity=0.65,
        labels={"temperature": "Temperature (°C)", "humidity": "Humidity (%)"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
    return fig


def ideal_vs_current_radar(current: dict[str, float], ideal: dict[str, float], crop: str) -> go.Figure:
    """Radar chart comparing ideal vs current conditions (normalized)."""
    features = list(current.keys())
    ranges = {"N": 140, "P": 145, "K": 205, "temperature": 55, "humidity": 100, "ph": 14, "rainfall": 500}

    curr_norm = [min(current[f] / ranges[f], 1.0) for f in features]
    ideal_norm = [min(ideal[f] / ranges[f], 1.0) for f in features]

    labels_display = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=curr_norm + [curr_norm[0]], theta=labels_display + [labels_display[0]],
        fill="toself", name="Current", line_color="#e76f51",
    ))
    fig.add_trace(go.Scatterpolar(
        r=ideal_norm + [ideal_norm[0]], theta=labels_display + [labels_display[0]],
        fill="toself", name=f"Ideal ({crop.title()})", line_color="#2d6a4f",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Ideal vs Current — {crop.title()}",
        template=PLOTLY_TEMPLATE, height=450,
    )
    return fig


def ideal_vs_current_bar(current: dict[str, float], ideal: dict[str, float], crop: str) -> go.Figure:
    features = list(current.keys())
    labels_display = ["Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=labels_display, y=[current[f] for f in features], marker_color="#e76f51"))
    fig.add_trace(go.Bar(name="Ideal", x=labels_display, y=[ideal[f] for f in features], marker_color="#2d6a4f"))
    fig.update_layout(
        barmode="group", title=f"Ideal vs Current Values — {crop.title()}",
        template=PLOTLY_TEMPLATE, height=420,
    )
    return fig


def model_comparison_bar(metrics: dict) -> go.Figure | None:
    if not metrics or "models" not in metrics:
        return None
    rows = []
    for name, m in metrics["models"].items():
        rows.append({"Model": name, "Metric": "Accuracy", "Value": m["accuracy"]})
        rows.append({"Model": name, "Metric": "F1 Score", "Value": m["f1_score"]})
    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Model", y="Value", color="Metric", barmode="group",
        title="Model Performance Comparison",
        color_discrete_sequence=["#2d6a4f", "#52b788"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420)
    return fig


def confusion_matrix_heatmap(cm: list, classes: list[str], title: str = "Confusion Matrix") -> go.Figure:
    fig = px.imshow(
        cm, text_auto=True, aspect="auto", title=title,
        labels=dict(x="Predicted", y="Actual"),
        x=classes, y=classes,
        color_continuous_scale="Greens",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=600)
    return fig


def feature_importance_chart(model: Any, feature_names: list[str]) -> go.Figure | None:
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance")
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title="Feature Importance (Random Forest / Tree-based)",
        color="importance", color_continuous_scale=["#b7e4c7", "#1b4332"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=350, showlegend=False)
    return fig


def multiclass_roc_data(y_test, y_score, classes: list[str]) -> go.Figure | None:
    """Build multi-class ROC curve figure."""
    try:
        y_bin = label_binarize(y_test, classes=range(len(classes)))
        n_classes = y_bin.shape[1]
        fig = go.Figure()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{classes[i]} (AUC={roc_auc:.2f})",
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(dash="dash", color="gray")))
        fig.update_layout(
            title="Multi-class ROC Curves (One-vs-Rest)",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            template=PLOTLY_TEMPLATE, height=500,
        )
        return fig
    except Exception:
        return None


def prediction_confidence_chart(probabilities: dict[str, float], top_n: int = 8) -> go.Figure:
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    crops = [c.title() for c, _ in sorted_probs]
    values = [v for _, v in sorted_probs]
    fig = px.bar(
        x=values, y=crops, orientation="h",
        title="Top Crop Prediction Confidence",
        labels={"x": "Confidence (%)", "y": "Crop"},
        color=values, color_continuous_scale=["#b7e4c7", "#1b4332"],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False)
    return fig
