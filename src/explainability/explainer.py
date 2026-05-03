import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

# ── Paths ───────────────────────────────────────────────
FEATURES_CSV = "data/processed/phishtrace_features.csv"
MODELS_DIR   = "models"
FIGURES_DIR  = "reports/figures"
REPORT_PATH  = "reports/evaluation_report.txt"
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
    "num_at","num_question","num_equals","num_percent",
    "num_digits_in_domain","num_digits_in_path","last_path_segment_is_integer",
    "has_ip","has_https","num_subdomains",
    "hostname_length","path_length","double_slash",
    "num_suspicious_words"
]

# ── Load ────────────────────────────────────────────────
print("Loading data and model...")
df     = pd.read_csv(FEATURES_CSV)
X      = df[FEATURE_COLS]
y      = df["label"].values

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
model  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

# Use a sample for SHAP (full dataset is slow)
sample_idx = np.random.RandomState(42).choice(len(X), size=500, replace=False)
X_sample   = pd.DataFrame(scaler.transform(X.iloc[sample_idx]),
                           columns=FEATURE_COLS)

# ── SHAP ────────────────────────────────────────────────
print("Computing SHAP values (this may take a minute)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Handle both list and 3D array output from RandomForest
if isinstance(shap_values, list):
    sv = shap_values[1]          # list of 2 → take class 1
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]    # shape (500, 19, 2) → take class 1
else:
    sv = shap_values             # already (500, 19)

# ── Plot 1: Summary bar ─────────────────────────────────
print("Saving shap_bar.png...")
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_sample, plot_type="bar",
                  feature_names=FEATURE_COLS, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_bar.png"), dpi=150)
plt.close()

# ── Plot 2: Summary beeswarm ────────────────────────────
print("Saving shap_summary.png...")
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_sample, feature_names=FEATURE_COLS, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=150)
plt.close()

# ── Plot 3: Waterfall for one sample ────────────────────
print("Saving shap_waterfall_sample.png...")

base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(np.array(base_val).flat[1])
else:
    base_val = float(base_val)

# sv is now guaranteed 2D (500, 19) — take first row
sv0 = sv[0]
if hasattr(sv0, 'ndim') and sv0.ndim == 2:
    sv0 = sv0[:, 1]   # extra safety

explanation = shap.Explanation(
    values=sv0,
    base_values=base_val,
    data=X_sample.iloc[0].values,
    feature_names=FEATURE_COLS
)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_waterfall_sample.png"), dpi=150)
plt.close()

# ── Text report ─────────────────────────────────────────
mean_shap = np.abs(sv).mean(axis=0)
top_features = sorted(zip(FEATURE_COLS, mean_shap),
                      key=lambda x: x[1], reverse=True)

report = ["=" * 50,
          "PhishTrace — SHAP Feature Importance Report",
          "=" * 50, ""]
for rank, (feat, val) in enumerate(top_features, 1):
    report.append(f"{rank:2}. {feat:<25} mean|SHAP| = {val:.4f}")

report += ["",
           "Figures saved:",
           f"  - {FIGURES_DIR}/shap_bar.png",
           f"  - {FIGURES_DIR}/shap_summary.png",
           f"  - {FIGURES_DIR}/shap_waterfall_sample.png"]

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n".join(report))
print(f"\nReport saved to {REPORT_PATH}")