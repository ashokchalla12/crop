"""
OptiCrop — Machine Learning Model Training Script

Trains multiple classification algorithms, compares metrics,
and saves the best-performing model with scaler and label encoder.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils.preprocessing import FEATURE_COLUMNS, load_and_clean_dataset

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train_all_models(test_size: float = 0.2, random_state: int = 42) -> dict:
    """Train multiple models and return comparison metrics."""
    df = load_and_clean_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=random_state
        ),
    }

    results: dict = {}
    best_name = ""
    best_score = -1.0
    best_model = None

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
            ),
        }

        if acc > best_score:
            best_score = acc
            best_name = name
            best_model = model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "crop_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    summary = {
        "best_model": best_name,
        "best_accuracy": round(float(best_score), 4),
        "models": results,
        "feature_columns": FEATURE_COLUMNS,
        "classes": label_encoder.classes_.tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best model: {best_name} (Accuracy: {best_score:.4f})")
    print(f"Artifacts saved to {MODEL_DIR}")
    return summary


if __name__ == "__main__":
    train_all_models()
