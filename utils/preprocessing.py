"""
Data loading, cleaning, validation, and prediction preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

COLUMN_ALIASES = {
    "n": "N",
    "nitrogen": "N",
    "p": "P",
    "phosphorous": "P",
    "phosphorus": "P",
    "k": "K",
    "potassium": "K",
    "temperature": "temperature",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall",
    "label": "label",
    "crop": "label",
}

INPUT_RANGES = {
    "N": (0, 140),
    "P": (0, 145),
    "K": (0, 205),
    "temperature": (0, 55),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 500),
}


def load_and_clean_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV, normalize column names, and handle missing values."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    df = df[[c for c in FEATURE_COLUMNS + ["label"] if c in df.columns]]

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def validate_inputs(values: dict[str, float]) -> tuple[bool, list[str]]:
    """Validate user input ranges; return (is_valid, error_messages)."""
    errors: list[str] = []
    for key, (lo, hi) in INPUT_RANGES.items():
        if key not in values:
            errors.append(f"Missing field: {key}")
            continue
        val = values[key]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            errors.append(f"{key} is required.")
        elif not lo <= float(val) <= hi:
            errors.append(f"{key} must be between {lo} and {hi}.")

    return len(errors) == 0, errors


def inputs_to_dataframe(values: dict[str, float]) -> pd.DataFrame:
    """Convert input dictionary to a single-row feature DataFrame."""
    row = {col: float(values[col]) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def get_dataset_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Return summary statistics for the dataset explorer."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe(include="all").transpose(),
        "crop_counts": df["label"].value_counts().to_dict(),
    }
