"""
PhishTrace — Model Evaluation Script
======================================
Produces ALL evaluation artefacts in one run:

  reports/figures/
    ├── confusion_matrix_<model>.png
    ├── roc_curves.png
    ├── threshold_analysis.png
    ├── threshold_cost.png
    └── feature_importance_best.png

  reports/
    ├── error_analysis.csv
    └── evaluation_report.txt

  models/
    └── evaluation_results.json

Usage
-----
  cd <project_root>
  python src/models/evaluate.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "phishtrace_features.csv"
MODELS_DIR   = ROOT / "models"
FIGURES_DIR  = ROOT / "reports" / "figures"
REPORTS_DIR  = ROOT / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ✅ FIX: 18 features — شيلنا "has_suspicious_word" (كانت redundant)
FEATURE_COLS = [
    "url_length", "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_at", "num_question", "num_equals", "num_percent", "num_digits",
    "has_ip", "has_https", "num_subdomains",
    "hostname_length", "path_length", "double_slash", "has_at_in_url",
    "num_suspicious_words"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest"       : "random_forest.pkl",
    "XGBoost"             : "xgboost.pkl",
}

THRESHOLD_DANGEROUS  = 0.75
THRESHOLD_SUSPICIOUS = 0.45

# ── Load data & scaler ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(FEATURES_CSV)

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nRun extract.py then train.py first.")

X = df[FEATURE_COLS].values
y = df["label"].values

scaler = joblib.load(MODELS_DIR / "scaler.pkl")

from sklearn.model_selection import train_test_split
_, X_test_raw, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test = scaler.transform(X_test_raw)
print(f"Test set: {len(y_test)} | Phishing: {y_test.sum()} | Legit: {(y_test==0).sum()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Per-model metrics + confusion matrices
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Per-model evaluation ─────────────────────────────────────────────────")
all_results = {}
all_fpr_tpr = {}

for model_name, model_file in MODEL_FILES.items():
    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        print(f"  [skip] {model_file} not found")
        continue

    clf    = joblib.load(model_path)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test,    y_pred, zero_division=0)
    f1   = f1_score(y_test,        y_pred, zero_division=0)

    fp   = int(((y_pred == 1) & (y_test == 0)).sum())
    fn_  = int(((y_pred == 0) & (y_test == 1)).sum())
    fpr_ = fp / max((y_test == 0).sum(), 1)

    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
    roc_auc              = auc(fpr_arr, tpr_arr)

    all_results[model_name] = {
        "accuracy" : round(acc,  4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "f1"       : round(f1,   4),
        "fpr"      : round(fpr_, 4),
        "auc"      : round(roc_auc, 4),
        "fp"       : fp,
        "fn"       : fn_,
    }
    all_fpr_tpr[model_name] = (fpr_arr, tpr_arr, roc_auc)

    print(f"\n  {model_name}")
    print(f"    Accuracy : {acc:.4f}  |  Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}  |  F1:        {f1:.4f}")
    print(f"    AUC      : {roc_auc:.4f}  |  FPR:       {fpr_:.4f}")

    cm        = confusion_matrix(y_test, y_pred)
    safe_name = model_name.lower().replace(" ", "_")
    fig, ax   = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Phishing"],
                yticklabels=["Legit", "Phishing"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)
    plt.close(fig)
    print(f"    Saved confusion_matrix_{safe_name}.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ROC curves
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ROC curves ───────────────────────────────────────────────────────────")
fig, ax = plt.subplots(figsize=(7, 5))
colors  = ["#185FA5", "#3B6D11", "#BA7517"]

for (name, (fpr_arr, tpr_arr, roc_auc)), color in zip(all_fpr_tpr.items(), colors):
    ax.plot(fpr_arr, tpr_arr, color=color, lw=1.8,
            label=f"{name}  (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate",  fontsize=11)
ax.set_title("ROC Curves — All Models", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
plt.close(fig)
print("  Saved roc_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Threshold Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Threshold analysis ───────────────────────────────────────────────────")
best_model  = joblib.load(MODELS_DIR / "best_model.pkl")
y_prob_best = best_model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.05, 0.95, 50)
precs, recs, f1s, fprs = [], [], [], []

for t in thresholds:
    y_p = (y_prob_best >= t).astype(int)
    precs.append(precision_score(y_test, y_p, zero_division=0))
    recs.append(recall_score(y_test,    y_p, zero_division=0))
    f1s.append(f1_score(y_test,         y_p, zero_division=0))
    fp_t = int(((y_p == 1) & (y_test == 0)).sum())
    fprs.append(fp_t / max((y_test == 0).sum(), 1))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, precs, label="Precision", color="#185FA5", lw=1.8)
ax.plot(thresholds, recs,  label="Recall",    color="#3B6D11", lw=1.8)
ax.plot(thresholds, f1s,   label="F1",        color="#BA7517", lw=2.2)
ax.plot(thresholds, fprs,  label="FPR",       color="#A32D2D", lw=1.4, linestyle="--")

ax.axvline(THRESHOLD_DANGEROUS,  color="#A32D2D", lw=1, linestyle=":", alpha=0.8)
ax.axvline(THRESHOLD_SUSPICIOUS, color="#BA7517", lw=1, linestyle=":", alpha=0.8)
ax.text(THRESHOLD_DANGEROUS  + 0.01, 0.05, "Dangerous\nthreshold", fontsize=8, color="#A32D2D")
ax.text(THRESHOLD_SUSPICIOUS + 0.01, 0.05, "Suspicious\nthreshold", fontsize=8, color="#BA7517")

ax.set_xlabel("Threshold", fontsize=11)
ax.set_ylabel("Score",     fontsize=11)
ax.set_title("Threshold Analysis — Best Model", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "threshold_analysis.png", dpi=150)
plt.close(fig)
print("  Saved threshold_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Cost-based threshold
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Cost-based threshold ─────────────────────────────────────────────────")
FN_COST = 100
FP_COST = 1

costs       = []
best_t_cost = None
best_cost   = float("inf")

for t, f1_val in zip(thresholds, f1s):
    y_p  = (y_prob_best >= t).astype(int)
    fn_c = int(((y_p == 0) & (y_test == 1)).sum())
    fp_c = int(((y_p == 1) & (y_test == 0)).sum())
    cost = FN_COST * fn_c + FP_COST * fp_c
    costs.append(cost)
    if cost < best_cost:
        best_cost   = cost
        best_t_cost = t

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, costs, color="#A32D2D", lw=2)
ax.axvline(best_t_cost, color="#3B6D11", lw=1.5, linestyle="--",
           label=f"Optimal = {best_t_cost:.2f}  (cost={best_cost:,})")
ax.axvline(THRESHOLD_DANGEROUS, color="#BA7517", lw=1, linestyle=":",
           label=f"Chosen = {THRESHOLD_DANGEROUS}")
ax.set_xlabel("Threshold",  fontsize=11)
ax.set_ylabel("Total cost", fontsize=11)
ax.set_title(f"Cost-based Threshold  (FN={FN_COST}, FP={FP_COST})", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "threshold_cost.png", dpi=150)
plt.close(fig)
print(f"  Optimal threshold: {best_t_cost:.2f}  |  Cost: {best_cost:,}")
print("  Saved threshold_cost.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Feature importance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Feature importance ───────────────────────────────────────────────────")
try:
    importances = best_model.feature_importances_
    sorted_idx  = np.argsort(importances)
    feat_names  = [FEATURE_COLS[i] for i in sorted_idx]
    feat_vals   = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(feat_names, feat_vals, color="#185FA5", alpha=0.8)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Feature Importance — Best Model", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_best.png", dpi=150)
    plt.close(fig)
    print("  Saved feature_importance_best.png")
except AttributeError:
    print("  [skip] Best model has no feature_importances_")


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Error analysis ───────────────────────────────────────────────────────")
y_pred_best = best_model.predict(X_test)

fp_idx    = np.where((y_pred_best == 1) & (y_test == 0))[0][:20]
fn_idx    = np.where((y_pred_best == 0) & (y_test == 1))[0][:20]
error_idx = np.concatenate([fp_idx, fn_idx])
error_type = (["False Positive"] * len(fp_idx) +
              ["False Negative"] * len(fn_idx))

print(f"  FP: {len(fp_idx)}  |  FN: {len(fn_idx)}")

shap_explainer = shap.TreeExplainer(best_model)
X_err          = X_test[error_idx]
shap_vals      = shap_explainer.shap_values(X_err)

if isinstance(shap_vals, list):
    sv = shap_vals[1]
elif shap_vals.ndim == 3:
    sv = shap_vals[:, :, 1]
else:
    sv = shap_vals

rows = []
for i, (idx, err_type) in enumerate(zip(error_idx, error_type)):
    top2   = np.argsort(np.abs(sv[i]))[::-1][:2]
    reason = " | ".join(
        f"{FEATURE_COLS[j]}({sv[i][j]:+.3f})" for j in top2
    )
    row = {
        "error_type"       : err_type,
        "true_label"       : int(y_test[idx]),
        "pred_label"       : int(y_pred_best[idx]),
        "score"            : round(float(y_prob_best[idx]), 4),
        "top_shap_reasons" : reason,
    }
    for c, col in enumerate(FEATURE_COLS):
        row[col] = X_test_raw[idx][c]
    rows.append(row)

error_df = pd.DataFrame(rows)
error_csv_path = REPORTS_DIR / "error_analysis.csv"
error_df.to_csv(error_csv_path, index=False)
print(f"  Saved error_analysis.csv ({len(rows)} rows)")

print("\n  Top reasons for False Positives:")
fp_df = error_df[error_df["error_type"] == "False Positive"]
if len(fp_df):
    for reason, cnt in fp_df["top_shap_reasons"].value_counts().head(3).items():
        print(f"    [{cnt}x] {reason}")

print("\n  Top reasons for False Negatives:")
fn_df = error_df[error_df["error_type"] == "False Negative"]
if len(fn_df):
    for reason, cnt in fn_df["top_shap_reasons"].value_counts().head(3).items():
        print(f"    [{cnt}x] {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Save evaluation_results.json
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Saving results JSON ──────────────────────────────────────────────────")
best_model_name = max(all_results, key=lambda k: all_results[k]["f1"])
best_metrics    = all_results[best_model_name]

results_json = {
    "best_model"            : best_model_name,
    "accuracy"              : best_metrics["accuracy"],
    "precision"             : best_metrics["precision"],
    "recall"                : best_metrics["recall"],
    "f1"                    : best_metrics["f1"],
    "auc"                   : best_metrics["auc"],
    "fpr"                   : best_metrics["fpr"],
    "threshold_dangerous"   : THRESHOLD_DANGEROUS,
    "threshold_suspicious"  : THRESHOLD_SUSPICIOUS,
    "optimal_threshold_cost": round(float(best_t_cost), 4),
    "fn_cost"               : FN_COST,
    "fp_cost"               : FP_COST,
    "test_set_size"         : int(len(y_test)),
    "all_models"            : all_results,
}

json_path = MODELS_DIR / "evaluation_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print("  Saved evaluation_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Human-readable report
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Writing evaluation_report.txt ────────────────────────────────────────")
lines = [
    "=" * 60,
    "PhishTrace — Model Evaluation Report",
    "=" * 60,
    "",
    f"Best model  : {best_model_name}",
    f"Test set    : {len(y_test):,} samples",
    "",
    "── Best Model Metrics ──────────────────────────────",
    f"  Accuracy   : {best_metrics['accuracy']:.4f}",
    f"  Precision  : {best_metrics['precision']:.4f}",
    f"  Recall     : {best_metrics['recall']:.4f}",
    f"  F1         : {best_metrics['f1']:.4f}",
    f"  AUC        : {best_metrics['auc']:.4f}",
    f"  FPR        : {best_metrics['fpr']:.4f}",
    "",
    "── Threshold Configuration ─────────────────────────",
    f"  Dangerous  : score >= {THRESHOLD_DANGEROUS}",
    f"  Suspicious : score >= {THRESHOLD_SUSPICIOUS}",
    f"  Cost-optimal threshold: {best_t_cost:.2f}",
    f"  (FN cost={FN_COST}, FP cost={FP_COST})",
    "",
    "── All Models Comparison ───────────────────────────",
]
for name, m in all_results.items():
    lines.append(f"  {name:<22}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  FPR={m['fpr']:.4f}")

lines += [
    "",
    "── Error Analysis Summary ──────────────────────────",
    f"  False Positives : {len(fp_idx)}",
    f"  False Negatives : {len(fn_idx)}",
    f"  Details         : reports/error_analysis.csv",
    "",
    "── Figures Generated ───────────────────────────────",
    "  confusion_matrix_<model>.png",
    "  roc_curves.png",
    "  threshold_analysis.png",
    "  threshold_cost.png",
    "  feature_importance_best.png",
    "",
    "=" * 60,
]

report_path = REPORTS_DIR / "evaluation_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print("\n✅ Evaluation complete!")