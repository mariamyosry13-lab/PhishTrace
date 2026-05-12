import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "phishtrace_features.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "shap_report.txt"

logger = logging.getLogger(__name__)


def _shap_values_matrix(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        class_idx = 1 if arr.shape[-1] >= 2 else 0
        return arr[:, :, class_idx]
    return arr


def _safe_expected_value(expected_value):
    arr = np.asarray(expected_value)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    idx = 1 if flat.size >= 2 else 0
    return float(flat[idx])


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data and model...")
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing features CSV: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns in {FEATURES_CSV}: {missing_cols}"
        )
    if "label" not in df.columns:
        raise ValueError(f"Missing required column 'label' in {FEATURES_CSV}")

    if df.empty:
        raise ValueError(f"Features CSV has no rows: {FEATURES_CSV}")
    X = df[FEATURE_COLS]

    scaler_path = MODELS_DIR / "scaler.pkl"
    model_path = MODELS_DIR / "best_model.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler model: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    sample_size = min(500, len(X))
    sample_idx = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
    X_sample = pd.DataFrame(scaler.transform(X.iloc[sample_idx]), columns=FEATURE_COLS)

    logger.info("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    sv = _shap_values_matrix(shap_values)

    logger.info("Saving shap_bar.png...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, plot_type="bar", feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150)
    plt.close()

    logger.info("Saving shap_summary.png...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()

    logger.info("Saving shap_waterfall_sample.png...")
    base_val = _safe_expected_value(explainer.expected_value)

    sv0 = sv[0]
    if hasattr(sv0, "ndim") and sv0.ndim == 2:
        class_idx = 1 if sv0.shape[-1] >= 2 else 0
        sv0 = sv0[:, class_idx]

    explanation = shap.Explanation(
        values=sv0,
        base_values=base_val,
        data=X_sample.iloc[0].values,
        feature_names=FEATURE_COLS,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_sample.png", dpi=150)
    plt.close()

    mean_shap = np.abs(sv).mean(axis=0)
    top_features = sorted(zip(FEATURE_COLS, mean_shap), key=lambda x: x[1], reverse=True)

    report = [
        "=" * 50,
        "PhishTrace — SHAP Feature Importance Report",
        "=" * 50,
        "",
    ]
    for rank, (feat, val) in enumerate(top_features, 1):
        report.append(f"{rank:2}. {feat:<25} mean|SHAP| = {val:.4f}")

    report += [
        "",
        "Figures saved:",
        f"  - {FIGURES_DIR / 'shap_bar.png'}",
        f"  - {FIGURES_DIR / 'shap_summary.png'}",
        f"  - {FIGURES_DIR / 'shap_waterfall_sample.png'}",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    logger.info("\n%s", "\n".join(report))
    logger.info("Report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()