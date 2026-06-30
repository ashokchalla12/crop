"""
Shared utilities: paths, styling, crop metadata, model loading, PDF export.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.preprocessing import FEATURE_COLUMNS, load_and_clean_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "model" / "label_encoder.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
ASSETS_DIR = BASE_DIR / "assets"
CROP_IMAGES_DIR = ASSETS_DIR / "crop_images"

SUPPORTED_CROPS = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee",
]

CROP_DISPLAY = {c: c.replace("_", " ").title() for c in SUPPORTED_CROPS}

# Ideal conditions derived from dataset medians (populated at runtime)
_IDEAL_CACHE: dict[str, dict[str, float]] = {}


def get_base_dir() -> Path:
    return BASE_DIR


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return load_and_clean_dataset(DATA_PATH)


@st.cache_resource
def load_ml_artifacts() -> tuple[Any, Any, Any]:
    """Load model, scaler, and label encoder."""
    if not MODEL_PATH.exists():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_ideal_conditions(crop: str) -> dict[str, float]:
    """Return ideal NPK and environmental values for a crop."""
    crop = crop.lower().strip()
    if crop in _IDEAL_CACHE:
        return _IDEAL_CACHE[crop]

    df = load_dataset()
    subset = df[df["label"] == crop]
    if subset.empty:
        return {col: 0.0 for col in FEATURE_COLUMNS}

    ideals = {col: float(subset[col].median()) for col in FEATURE_COLUMNS}
    _IDEAL_CACHE[crop] = ideals
    return ideals


def compute_suitability_score(current: dict[str, float], ideal: dict[str, float]) -> float:
    """Compute suitability score (0-100) by comparing current vs ideal."""
    weights = {"N": 1.2, "P": 1.2, "K": 1.2, "temperature": 1.0, "humidity": 1.0, "ph": 1.5, "rainfall": 1.0}
    total_weight = sum(weights.values())
    score_sum = 0.0

    for col in FEATURE_COLUMNS:
        ideal_val = ideal.get(col, 1.0) or 1.0
        curr_val = current.get(col, 0.0)
        # Normalized absolute difference; scale by feature range
        ranges = {"N": 140, "P": 145, "K": 205, "temperature": 55, "humidity": 100, "ph": 14, "rainfall": 500}
        diff = abs(curr_val - ideal_val) / ranges[col]
        feature_score = max(0.0, 1.0 - diff) * 100
        score_sum += feature_score * weights[col]

    return round(score_sum / total_weight, 1)


def suitability_status(score: float) -> tuple[str, str]:
    """Return (status label, color hex) for suitability score."""
    if score >= 85:
        return "Excellent", "#2ecc71"
    if score >= 70:
        return "Good", "#27ae60"
    if score >= 50:
        return "Moderate", "#f39c12"
    return "Poor", "#e74c3c"


def get_recommendations(current: dict[str, float], ideal: dict[str, float]) -> list[str]:
    """Generate actionable recommendations based on ideal vs current."""
    recs: list[str] = []
    thresholds = {"N": 10, "P": 8, "K": 8, "temperature": 3, "humidity": 8, "ph": 0.5, "rainfall": 30}

    labels = {
        "N": "Nitrogen", "P": "Phosphorous", "K": "Potassium",
        "temperature": "Temperature", "humidity": "Humidity",
        "ph": "pH", "rainfall": "Rainfall",
    }

    for col in FEATURE_COLUMNS:
        diff = current[col] - ideal[col]
        if abs(diff) <= thresholds[col]:
            continue
        name = labels[col]
        if col == "N" and diff < 0:
            recs.append(f"Increase Nitrogen — current {current[col]:.1f}, ideal ~{ideal[col]:.1f}")
        elif col == "P" and diff < 0:
            recs.append(f"Add Phosphorous fertilizer — boost from {current[col]:.1f} to ~{ideal[col]:.1f}")
        elif col == "K" and diff < 0:
            recs.append(f"Increase Potassium levels via potash application")
        elif col == "ph" and diff > 0:
            recs.append(f"Reduce pH — apply sulfur or organic matter (target ~{ideal[col]:.1f})")
        elif col == "ph" and diff < 0:
            recs.append(f"Raise pH — apply agricultural lime (target ~{ideal[col]:.1f})")
        elif col == "rainfall" and diff < 0:
            recs.append("Improve irrigation — supplemental watering recommended")
        elif col == "rainfall" and diff > 0:
            recs.append("Improve drainage — excess rainfall may cause waterlogging")
        elif col == "humidity" and diff < 0:
            recs.append(f"Increase humidity through mulching or micro-sprinklers")
        elif col == "temperature" and diff < 0:
            recs.append(f"Temperature is below optimal — consider greenhouse or delayed planting")
        elif col == "temperature" and diff > 0:
            recs.append(f"Temperature is above optimal — provide shade nets or adjust season")
        else:
            recs.append(f"Adjust {name}: current {current[col]:.1f}, ideal ~{ideal[col]:.1f}")

    if not recs:
        recs.append("Conditions are well aligned — maintain current soil and irrigation practices")
    if any(r for r in recs if "Nitrogen" in r or "Phosphorous" in r or "Potassium" in r):
        recs.append("Add organic fertilizer to improve long-term soil health")

    return recs[:8]


CROP_INFO: dict[str, dict[str, str]] = {
    "rice": {
        "soil": "Clay loam, pH 5.5–7.0, high water retention",
        "tips": "Maintain flooded conditions during vegetative stage. Apply N in splits.",
        "yield": "25–40 quintals/acre under optimal conditions",
        "advantages": "Staple food crop, high domestic demand, government MSP support",
        "market": "Very High — consistent year-round demand",
        "emoji": "🌾",
    },
    "maize": {
        "soil": "Well-drained loamy soil, pH 6.0–7.5",
        "tips": "Ensure adequate spacing. Apply balanced NPK at sowing and knee-high stage.",
        "yield": "20–35 quintals/acre",
        "advantages": "Dual use (food & feed), short duration, drought tolerant varieties available",
        "market": "High — poultry and industrial demand",
        "emoji": "🌽",
    },
    "chickpea": {
        "soil": "Sandy loam to clay loam, pH 6.0–7.5",
        "tips": "Inoculate seeds with Rhizobium. Avoid waterlogging.",
        "yield": "8–15 quintals/acre",
        "advantages": "Nitrogen-fixing legume, low input cost, protein-rich",
        "market": "High — export and domestic pulse demand",
        "emoji": "🫘",
    },
    "kidneybeans": {
        "soil": "Well-drained fertile soil, pH 6.0–7.0",
        "tips": "Provide trellis support. Harvest when pods are dry.",
        "yield": "10–18 quintals/acre",
        "advantages": "High protein, suitable for intercropping",
        "market": "Moderate to High",
        "emoji": "🫘",
    },
    "pigeonpeas": {
        "soil": "Light to medium black soil, pH 6.5–7.5",
        "tips": "Deep ploughing before sowing. Control pod borer with IPM.",
        "yield": "8–12 quintals/acre",
        "advantages": "Drought tolerant, improves soil fertility",
        "market": "Moderate — regional pulse demand",
        "emoji": "🫛",
    },
    "mothbeans": {
        "soil": "Sandy loam, pH 6.5–8.0",
        "tips": "Ideal for arid regions. Minimal irrigation needed.",
        "yield": "5–10 quintals/acre",
        "advantages": "Extremely drought hardy, low water requirement",
        "market": "Moderate — niche pulse market",
        "emoji": "🫘",
    },
    "mungbean": {
        "soil": "Loamy soil, pH 6.2–7.2",
        "tips": "Short duration crop — 60–70 days. Ensure good drainage.",
        "yield": "6–12 quintals/acre",
        "advantages": "Quick returns, soil enrichment, summer fit",
        "market": "High — dal and sprout industry",
        "emoji": "🫘",
    },
    "blackgram": {
        "soil": "Black cotton or loamy soil, pH 6.0–7.5",
        "tips": "Sow after rains. Apply phosphorus at sowing.",
        "yield": "6–10 quintals/acre",
        "advantages": "Kharif fit, nitrogen fixer, low investment",
        "market": "Moderate to High",
        "emoji": "🫘",
    },
    "lentil": {
        "soil": "Well-drained loam, pH 6.0–7.0",
        "tips": "Cool season crop. Seed treatment recommended.",
        "yield": "8–14 quintals/acre",
        "advantages": "Rabi season option, high nutritional value",
        "market": "High — staple pulse",
        "emoji": "🫘",
    },
    "pomegranate": {
        "soil": "Deep loamy soil, pH 6.5–7.5",
        "tips": "Prune annually. Drip irrigation improves fruit quality.",
        "yield": "80–120 kg/tree/year",
        "advantages": "High value horticulture, export potential",
        "market": "Very High — premium fruit market",
        "emoji": "🍎",
    },
    "banana": {
        "soil": "Rich loamy soil, pH 6.0–7.5",
        "tips": "Regular irrigation and de-suckering. Wind protection needed.",
        "yield": "300–500 quintals/acre",
        "advantages": "Year-round harvest potential, high productivity",
        "market": "Very High — universal demand",
        "emoji": "🍌",
    },
    "mango": {
        "soil": "Deep well-drained soil, pH 5.5–7.5",
        "tips": "Flowering stage needs water stress management. Spray for pests.",
        "yield": "8–16 tonnes/acre ( mature orchard )",
        "advantages": "King of fruits, export and processing demand",
        "market": "Very High",
        "emoji": "🥭",
    },
    "grapes": {
        "soil": "Sandy loam, pH 6.5–7.5",
        "tips": "Training and pruning critical. Monitor powdery mildew.",
        "yield": "15–25 tonnes/acre",
        "advantages": "Wine and table grape markets, high ROI",
        "market": "High — domestic and export",
        "emoji": "🍇",
    },
    "watermelon": {
        "soil": "Sandy loam, pH 6.0–7.0",
        "tips": "Warm season crop. Mulch to retain moisture.",
        "yield": "200–400 quintals/acre",
        "advantages": "Short duration, high summer demand",
        "market": "High — seasonal peak pricing",
        "emoji": "🍉",
    },
    "muskmelon": {
        "soil": "Well-drained sandy loam, pH 6.0–7.0",
        "tips": "Harvest at half-slip stage. Avoid overhead irrigation near harvest.",
        "yield": "150–250 quintals/acre",
        "advantages": "Early maturity, good market price in summer",
        "market": "Moderate to High",
        "emoji": "🍈",
    },
    "apple": {
        "soil": "Loamy soil, pH 6.0–7.0, cool climate",
        "tips": "Requires chilling hours. Thin fruits for size quality.",
        "yield": "100–200 quintals/acre",
        "advantages": "Premium horticulture crop, storage value",
        "market": "Very High",
        "emoji": "🍎",
    },
    "orange": {
        "soil": "Well-drained loam, pH 6.0–7.5",
        "tips": "Regular micronutrient sprays. Protect from citrus psylla.",
        "yield": "150–250 quintals/acre",
        "advantages": "Vitamin C rich, juice industry demand",
        "market": "High",
        "emoji": "🍊",
    },
    "papaya": {
        "soil": "Deep fertile soil, pH 6.0–7.0",
        "tips": "Protect from waterlogging. Apply FYM at planting.",
        "yield": "200–350 quintals/acre",
        "advantages": "Fast bearing — fruit in 9–12 months",
        "market": "High — fresh and papain industry",
        "emoji": "🥭",
    },
    "coconut": {
        "soil": "Coastal sandy loam, pH 5.5–8.0",
        "tips": "Regular irrigation in summer. Apply 50% inorganic + 50% organic.",
        "yield": "80–120 nuts/palm/year",
        "advantages": "Multi-product tree — oil, coir, tender coconut",
        "market": "Very High — diverse products",
        "emoji": "🥥",
    },
    "cotton": {
        "soil": "Black cotton soil, pH 6.0–8.0",
        "tips": "Monitor bollworm. Avoid excess nitrogen late season.",
        "yield": "10–20 quintals/acre",
        "advantages": "Cash crop, textile industry backbone",
        "market": "Very High — industrial demand",
        "emoji": "☁️",
    },
    "jute": {
        "soil": "Alluvial loam, pH 6.0–7.5, high rainfall",
        "tips": "Sown at onset of monsoon. Retting process affects fiber quality.",
        "yield": "20–30 quintals/acre ( raw jute )",
        "advantages": "Eco-friendly fiber, biodegradable packaging demand rising",
        "market": "Moderate — export oriented",
        "emoji": "🌿",
    },
    "coffee": {
        "soil": "Volcanic/red loam, pH 5.5–6.5, shaded",
        "tips": "Shade management essential. Process within 24h of harvest.",
        "yield": "800–1200 kg/acre ( green bean )",
        "advantages": "Premium export crop, specialty coffee markets",
        "market": "Very High — global demand",
        "emoji": "☕",
    },
}


def get_crop_info(crop: str) -> dict[str, str]:
    crop = crop.lower().strip()
    default = {
        "soil": "Well-drained fertile soil with balanced NPK",
        "tips": "Monitor soil moisture and apply balanced fertilization.",
        "yield": "Varies by region and management practices",
        "advantages": "Suitable for local agro-climatic conditions",
        "market": "Moderate demand",
        "emoji": "🌱",
    }
    return {**default, **CROP_INFO.get(crop, {})}


def get_crop_image_path(crop: str) -> Path | None:
    """Return path to crop image if available."""
    crop = crop.lower().strip()
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = CROP_IMAGES_DIR / f"{crop}{ext}"
        if path.exists():
            return path
    return None


def predict_crop(values: dict[str, float]) -> tuple[str, float, dict[str, float]]:
    """Run ML prediction; returns (crop_name, confidence, all_probabilities)."""
    model, scaler, encoder = load_ml_artifacts()
    if model is None:
        raise FileNotFoundError("Model not found. Run train_model.py first.")

    from utils.preprocessing import inputs_to_dataframe

    X = inputs_to_dataframe(values)
    X_scaled = scaler.transform(X) if scaler else X.values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx]) * 100
        all_probs = {encoder.classes_[i]: float(proba[i]) * 100 for i in range(len(proba))}
    else:
        idx = int(model.predict(X_scaled)[0])
        confidence = 100.0
        all_probs = {encoder.classes_[idx]: 100.0}

    crop = encoder.classes_[idx]
    return crop, confidence, all_probs


def generate_prediction_pdf(prediction: dict) -> bytes:
    """Generate a simple PDF report for a prediction."""
    from fpdf import FPDF

    def _safe(text: str) -> str:
        return str(text).encode("ascii", errors="replace").decode("ascii")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "OptiCrop - Crop Recommendation Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _safe(f"Recommended Crop: {prediction.get('crop', 'N/A').title()}"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Confidence: {prediction.get('confidence', 0):.1f}%", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Input Parameters", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for key in FEATURE_COLUMNS:
        pdf.cell(0, 6, f"  {key}: {prediction.get('inputs', {}).get(key, 'N/A')}", ln=True)

    pdf.ln(3)
    info = prediction.get("info", {})
    for label, key in [
        ("Soil Condition", "soil"),
        ("Growing Tips", "tips"),
        ("Expected Yield", "yield"),
        ("Market Demand", "market"),
    ]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, label, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _safe(info.get(key, "")))

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def init_session_state() -> None:
    """Initialize shared session state keys."""
    defaults = {
        "dark_mode": False,
        "prediction_history": [],
        "last_prediction": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def apply_custom_css(dark_mode: bool = False) -> None:
    """Inject agriculture-themed CSS."""
    bg = "#0d1f0d" if dark_mode else "#f4faf4"
    card = "#1a2e1a" if dark_mode else "#ffffff"
    text = "#e8f5e9" if dark_mode else "#1b4332"
    accent = "#2d6a4f"
    accent_light = "#52b788"
    sidebar = "#1b4332" if not dark_mode else "#0a1a0a"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{
            background: linear-gradient(135deg, {bg} 0%, {'#142814' if dark_mode else '#e8f5e9'} 100%);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {sidebar} 0%, #2d6a4f 100%);
        }}
        [data-testid="stSidebar"] * {{ color: #ffffff !important; }}
        .hero-title {{
            font-size: 2.8rem; font-weight: 700; color: {text};
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }}
        .hero-subtitle {{ font-size: 1.2rem; color: {'#a8dadc' if dark_mode else '#40916c'}; }}
        .feature-card {{
            background: {card}; border-radius: 16px; padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(45,106,79,0.12);
            border-left: 5px solid {accent_light}; margin-bottom: 1rem;
            transition: transform 0.2s ease;
        }}
        .feature-card:hover {{ transform: translateY(-3px); }}
        .metric-card {{
            background: linear-gradient(135deg, {accent} 0%, {accent_light} 100%);
            color: white; border-radius: 12px; padding: 1rem; text-align: center;
        }}
        .crop-result {{
            font-size: 2rem; font-weight: 700; color: {accent};
            animation: fadeIn 0.6s ease-in;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: scale(0.95); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        div[data-testid="stMetric"] {{
            background: {card}; padding: 12px; border-radius: 10px;
            border: 1px solid {'#2d6a4f' if dark_mode else '#b7e4c7'};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = "", icon: str = "🌾") -> None:
    st.markdown(f"## {icon} {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.divider()
