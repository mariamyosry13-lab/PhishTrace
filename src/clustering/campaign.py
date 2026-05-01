"""
PhishTrace — Campaign Clustering  (upgraded)
=============================================
Changes vs original:
  - Saves campaign_model.pkl + campaign_scaler.pkl → used real-time by app.py
  - assign_campaign(features_dict) → str  — assign any new URL to a campaign
  - Persists campaign summary to SQLite DB via database.db.save_campaigns()
  - Adds a brief DBSCAN comparison (printed, not replacing KMeans) to justify choice
  - KMeans N_CLUSTERS auto-estimated via silhouette score on a sample

Run
---
  cd <project_root>
  python src/clustering/campaign.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
FEATURES_CSV  = ROOT / "data" / "processed" / "phishtrace_features.csv"
CAMPAIGNS_CSV = ROOT / "data" / "processed" / "phishtrace_campaigns.csv"
MODELS_DIR    = ROOT / "models"
FIGURES_DIR   = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Add src to path for DB import ─────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "src"))
try:
    from database.db import init_db, save_campaigns as db_save_campaigns
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    print("[Campaign] database.db not found — DB persistence skipped")

FEATURE_COLS = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
    "num_at","num_question","num_equals","num_percent","num_digits",
    "has_ip","has_https","has_suspicious_word","num_subdomains",
    "hostname_length","path_length","double_slash","has_at_in_url",
    "num_suspicious_words",
]

BATCH_SIZE = 5000


# ═══════════════════════════════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading features...")
df          = pd.read_csv(FEATURES_CSV)
phishing_df = df[df["label"] == 1].copy().reset_index(drop=True)
print(f"Total phishing URLs: {len(phishing_df):,}")

X = phishing_df[FEATURE_COLS].values


# ═══════════════════════════════════════════════════════════════════════════════
#  Scale  (MiniBatch — memory safe for large datasets)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nScaling features...")
campaign_scaler = StandardScaler()
campaign_scaler.fit(X[:BATCH_SIZE])          # fit on first batch

X_scaled = np.zeros_like(X, dtype=np.float32)
for i in range(0, len(X), BATCH_SIZE):
    X_scaled[i:i+BATCH_SIZE] = campaign_scaler.transform(X[i:i+BATCH_SIZE])

joblib.dump(campaign_scaler, MODELS_DIR / "campaign_scaler.pkl")
print("  campaign_scaler.pkl saved")


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto-select N_CLUSTERS via silhouette on a sample
# ═══════════════════════════════════════════════════════════════════════════════
print("\nAuto-selecting number of clusters (silhouette on 3000-sample)...")
sample_idx  = np.random.RandomState(42).choice(len(X_scaled), size=3000, replace=False)
X_sample    = X_scaled[sample_idx]

candidates  = range(10, 31, 5)     # test 10, 15, 20, 25, 30
sil_scores  = {}

for k in candidates:
    km  = MiniBatchKMeans(n_clusters=k, batch_size=BATCH_SIZE,
                          random_state=42, n_init=3, verbose=0)
    lbl = km.fit_predict(X_sample)
    s   = silhouette_score(X_sample, lbl, sample_size=1000, random_state=42)
    sil_scores[k] = round(s, 4)
    print(f"  k={k:2d}  silhouette={s:.4f}")

N_CLUSTERS = max(sil_scores, key=sil_scores.get)
print(f"\n  Best k = {N_CLUSTERS}  (silhouette = {sil_scores[N_CLUSTERS]})")


# ═══════════════════════════════════════════════════════════════════════════════
#  MiniBatchKMeans — full dataset
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning MiniBatchKMeans (k={N_CLUSTERS})...")
kmeans = MiniBatchKMeans(
    n_clusters  = N_CLUSTERS,
    batch_size  = BATCH_SIZE,
    random_state= 42,
    n_init      = 3,
    verbose     = 0,
)
for i in range(0, len(X_scaled), BATCH_SIZE):
    kmeans.partial_fit(X_scaled[i:i+BATCH_SIZE])

labels = np.zeros(len(X_scaled), dtype=int)
for i in range(0, len(X_scaled), BATCH_SIZE):
    labels[i:i+BATCH_SIZE] = kmeans.predict(X_scaled[i:i+BATCH_SIZE])

phishing_df["cluster"]  = labels
phishing_df["campaign"] = phishing_df["cluster"].apply(
    lambda c: f"campaign_{c:03d}"
)

# Save model
joblib.dump(kmeans, MODELS_DIR / "campaign_model.pkl")
print("  campaign_model.pkl saved")


# ═══════════════════════════════════════════════════════════════════════════════
#  DBSCAN comparison (on sample only — for academic justification)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── DBSCAN vs KMeans comparison (on 3000-sample) ──────────────────────────")
dbscan = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
db_lbl = dbscan.fit_predict(X_sample)
n_clusters_db = len(set(db_lbl)) - (1 if -1 in db_lbl else 0)
noise_ratio   = (db_lbl == -1).sum() / len(db_lbl)

print(f"  DBSCAN  → {n_clusters_db} clusters found  |  {noise_ratio:.1%} noise points")
print(f"  KMeans  → {N_CLUSTERS} clusters (pre-set)  |  0% noise (all assigned)")
print("""
  Justification for choosing MiniBatchKMeans over DBSCAN:
  ┌─────────────────────────────┬────────────────┬────────────────┐
  │ Property                    │ KMeans (chosen)│ DBSCAN         │
  ├─────────────────────────────┼────────────────┼────────────────┤
  │ Complexity                  │ O(n·k·iter)    │ O(n²) worst    │
  │ Large datasets (100k+ URLs) │ Fast ✓         │ Slow ✗         │
  │ Real-time assignment        │ Yes ✓          │ No ✗           │
  │ Incremental (new URLs)      │ Yes ✓          │ No ✗           │
  │ Needs k upfront             │ Yes (auto sel) │ No ✓           │
  │ Detects outliers            │ No ✗           │ Yes ✓          │
  └─────────────────────────────┴────────────────┴────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  Campaign summary
# ═══════════════════════════════════════════════════════════════════════════════
summary = (phishing_df.groupby("campaign")
           .agg(size=("url", "count"))
           .sort_values("size", ascending=False)
           .reset_index())

print(f"Clusters: {N_CLUSTERS}  |  Total phishing URLs: {len(phishing_df):,}")
print("\nTop 10 campaigns:")
print(summary.head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
#  Persist to CSV + DB
# ═══════════════════════════════════════════════════════════════════════════════
phishing_df.to_csv(CAMPAIGNS_CSV, index=False)
print(f"\nSaved phishtrace_campaigns.csv")

if _DB_AVAILABLE:
    init_db()
    campaign_list = [
        {
            "campaign_name": row["campaign"],
            "size"         : int(row["size"]),
            "centroid"     : kmeans.cluster_centers_[
                                int(row["campaign"].split("_")[1])
                             ].tolist(),
        }
        for _, row in summary.iterrows()
    ]
    db_save_campaigns(campaign_list)
    print(f"Saved {len(campaign_list)} campaigns to SQLite DB")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Campaign sizes bar chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(summary["campaign"], summary["size"], color="#185FA5", alpha=0.8)
ax.set_xticks(range(len(summary)))
ax.set_xticklabels(summary["campaign"], rotation=45, ha="right", fontsize=8)
ax.set_title("Phishing Campaigns by Size", fontsize=12)
ax.set_ylabel("Number of URLs", fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "campaign_sizes.png", dpi=120)
plt.close(fig)

# Plot 2: PCA scatter (sample)
pca       = PCA(n_components=2, random_state=42)
X_pca     = pca.fit_transform(X_scaled[sample_idx])
lbl_samp  = labels[sample_idx]

fig, ax = plt.subplots(figsize=(10, 7))
cmap    = plt.cm.tab20(np.linspace(0, 1, N_CLUSTERS))
for i in range(N_CLUSTERS):
    mask = lbl_samp == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[cmap[i]], s=5, alpha=0.6, label=f"C{i}")
ax.set_title(f"Campaign Clusters — PCA (sample 3000, k={N_CLUSTERS})", fontsize=11)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(markerscale=3, fontsize=7, ncol=3, loc="best")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "campaign_clusters.png", dpi=120)
plt.close(fig)

# Plot 3: Silhouette score vs k
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(sil_scores.keys()), list(sil_scores.values()),
        marker="o", color="#185FA5", lw=2)
ax.axvline(N_CLUSTERS, color="#A32D2D", lw=1.5, linestyle="--",
           label=f"Selected k = {N_CLUSTERS}")
ax.set_xlabel("Number of clusters (k)", fontsize=11)
ax.set_ylabel("Silhouette score",        fontsize=11)
ax.set_title("Silhouette Score vs k",    fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "silhouette_vs_k.png", dpi=120)
plt.close(fig)

print("\nFigures saved: campaign_sizes.png | campaign_clusters.png | silhouette_vs_k.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API — used by app.py
# ═══════════════════════════════════════════════════════════════════════════════
def assign_campaign(features_dict: dict) -> str | None:
    """
    Assign a new URL's features to the nearest campaign cluster.

    Parameters
    ----------
    features_dict : dict  — output of extract_features(url)

    Returns
    -------
    str  — e.g. "campaign_007"
    None — if campaign model is not available
    """
    model_path  = MODELS_DIR / "campaign_model.pkl"
    scaler_path = MODELS_DIR / "campaign_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None
    km  = joblib.load(model_path)
    sc  = joblib.load(scaler_path)
    X   = np.array([[features_dict.get(c, 0) for c in FEATURE_COLS]])
    Xs  = sc.transform(X)
    cid = int(km.predict(Xs)[0])
    return f"campaign_{cid:03d}"


print("\n✅ Campaign clustering complete!")
print(f"   campaign_model.pkl   → {MODELS_DIR / 'campaign_model.pkl'}")
print(f"   campaign_scaler.pkl  → {MODELS_DIR / 'campaign_scaler.pkl'}")