import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "phishtrace_features.csv"
CAMPAIGNS_CSV = ROOT / "data" / "processed" / "phishtrace_campaigns.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path for local imports (guard duplicate insert)
_src_path = str(ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from config import FEATURE_COLS  # noqa: E402

try:
    from database.db import init_db, save_campaigns as db_save_campaigns  # noqa: E402

    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    print("[Campaign] database.db not found - DB persistence skipped")

BATCH_SIZE = 5000
RANDOM_STATE = 42

# Cached inference artifacts for assign_campaign()
_campaign_model = None
_campaign_scaler = None
_campaign_loaded = False


def _load_campaign_artifacts() -> tuple[object | None, object | None]:
    """Load campaign model/scaler once and cache for low-latency inference."""
    global _campaign_model, _campaign_scaler, _campaign_loaded

    if _campaign_loaded:
        return _campaign_model, _campaign_scaler

    model_path = MODELS_DIR / "campaign_model.pkl"
    scaler_path = MODELS_DIR / "campaign_scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        _campaign_model = joblib.load(model_path)
        _campaign_scaler = joblib.load(scaler_path)

    _campaign_loaded = True
    return _campaign_model, _campaign_scaler


def assign_campaign(features_dict: dict) -> str | None:
    """
    Assign a new URL's features to the nearest campaign cluster.

    Parameters
    ----------
    features_dict : dict -- output of extract_features(url)

    Returns
    -------
    str  -- e.g. "campaign_007"
    None -- if campaign model/scaler is unavailable
    """
    km, sc = _load_campaign_artifacts()
    if km is None or sc is None:
        return None

    X = np.array([[features_dict.get(c, 0) for c in FEATURE_COLS]], dtype=float)
    Xs = sc.transform(X)
    cid = int(km.predict(Xs)[0])
    return f"campaign_{cid:03d}"


def _choose_cluster_count(X_sample: np.ndarray) -> tuple[int, dict[int, float]]:
    """Pick k using silhouette when valid; fallback safely for tiny datasets."""
    n_sample = len(X_sample)
    max_k = min(30, n_sample - 1)
    min_k = 2

    if max_k < min_k:
        # n_sample is 0/1, caller should already guard this.
        return 1, {}

    candidates = [k for k in range(10, 31, 5) if min_k <= k <= max_k]
    if not candidates:
        candidates = [min_k]

    sil_scores: dict[int, float] = {}
    for k in candidates:
        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=BATCH_SIZE,
            random_state=RANDOM_STATE,
            n_init=3,
            verbose=0,
        )
        lbl = km.fit_predict(X_sample)

        # Silhouette requires at least 2 labels and fewer labels than samples.
        unique_lbl = np.unique(lbl)
        if len(unique_lbl) < 2 or len(unique_lbl) >= len(X_sample):
            continue

        s = silhouette_score(
            X_sample,
            lbl,
            sample_size=min(1000, len(X_sample)),
            random_state=RANDOM_STATE,
        )
        sil_scores[k] = round(float(s), 4)
        print(f"  k={k:2d}  silhouette={s:.4f}")

    if sil_scores:
        best_k = max(sil_scores, key=sil_scores.get)
    else:
        # Fallback for tiny or degenerate samples.
        best_k = min(max_k, 2)
        print("  silhouette unavailable on sample; falling back to k=2")

    return best_k, sil_scores


def main() -> None:
    print("Loading features...")
    df = pd.read_csv(FEATURES_CSV)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing FEATURE_COLS in features CSV: {missing}")

    phishing_df = df[df["label"] == 1].copy().reset_index(drop=True)
    if phishing_df.empty:
        raise ValueError("No phishing rows found (label=1). Cannot run campaign clustering.")

    X = phishing_df[FEATURE_COLS].values
    if len(X) < 2:
        raise ValueError("Need at least 2 phishing samples to build campaign clusters.")

    print(f"Total phishing URLs: {len(phishing_df):,}")

    print("\nScaling features...")
    # Fit on full X to avoid first-batch bias.
    campaign_scaler = StandardScaler()
    campaign_scaler.fit(X)

    X_scaled = np.zeros_like(X, dtype=np.float32)
    for i in range(0, len(X), BATCH_SIZE):
        X_scaled[i : i + BATCH_SIZE] = campaign_scaler.transform(X[i : i + BATCH_SIZE])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(campaign_scaler, MODELS_DIR / "campaign_scaler.pkl")
    print("  campaign_scaler.pkl saved")

    sample_size = min(3000, len(X_scaled))
    print(f"\nAuto-selecting number of clusters (silhouette on {sample_size}-sample)...")
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_scaled), size=sample_size, replace=False
    )
    X_sample = X_scaled[sample_idx]

    n_clusters, sil_scores = _choose_cluster_count(X_sample)
    if sil_scores:
        print(f"\n  Best k = {n_clusters}  (silhouette = {sil_scores[n_clusters]:.4f})")
    else:
        print(f"\n  Selected k = {n_clusters} (fallback)")

    print(f"\nRunning MiniBatchKMeans (k={n_clusters})...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        n_init=3,
        verbose=0,
    )
    for i in range(0, len(X_scaled), BATCH_SIZE):
        kmeans.partial_fit(X_scaled[i : i + BATCH_SIZE])

    labels = np.zeros(len(X_scaled), dtype=int)
    for i in range(0, len(X_scaled), BATCH_SIZE):
        labels[i : i + BATCH_SIZE] = kmeans.predict(X_scaled[i : i + BATCH_SIZE])

    phishing_df["cluster"] = labels
    phishing_df["campaign"] = phishing_df["cluster"].apply(lambda c: f"campaign_{c:03d}")

    joblib.dump(kmeans, MODELS_DIR / "campaign_model.pkl")
    print("  campaign_model.pkl saved")

    print(f"\nDBSCAN vs KMeans comparison (on {sample_size}-sample)")
    dbscan = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
    db_lbl = dbscan.fit_predict(X_sample)
    n_clusters_db = len(set(db_lbl)) - (1 if -1 in db_lbl else 0)
    noise_ratio = (db_lbl == -1).sum() / len(db_lbl)

    print(f"  DBSCAN -> {n_clusters_db} clusters found | {noise_ratio:.1%} noise points")
    print(f"  KMeans -> {n_clusters} clusters (pre-set) | 0% noise (all assigned)")
    print("  Justification for choosing MiniBatchKMeans over DBSCAN:")
    print("    - Complexity: KMeans O(n*k*iter) vs DBSCAN O(n^2) worst-case")
    print("    - Large datasets (100k+ URLs): KMeans faster")
    print("    - Real-time assignment: KMeans supports predict() for new URLs")
    print("    - Incremental updates: KMeans supports batch-style fitting")

    summary = (
        phishing_df.groupby("campaign")
        .agg(size=("url", "count"))
        .sort_values("size", ascending=False)
        .reset_index()
    )

    print(f"Clusters: {n_clusters}  |  Total phishing URLs: {len(phishing_df):,}")
    print("\nTop 10 campaigns:")
    print(summary.head(10).to_string(index=False))

    CAMPAIGNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    phishing_df.to_csv(CAMPAIGNS_CSV, index=False)
    print("\nSaved phishtrace_campaigns.csv")

    if _DB_AVAILABLE:
        init_db()
        campaign_list = [
            {
                "campaign_name": row["campaign"],
                "size": int(row["size"]),
                "centroid": kmeans.cluster_centers_[
                    int(row["campaign"].split("_")[1])
                ].tolist(),
            }
            for _, row in summary.iterrows()
        ]
        db_save_campaigns(campaign_list)
        print(f"Saved {len(campaign_list)} campaigns to SQLite DB")

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
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled[sample_idx])
    lbl_samp = labels[sample_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = lbl_samp == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[cmap[i]], s=5, alpha=0.6, label=f"C{i}")
    ax.set_title(f"Campaign Clusters - PCA (sample {sample_size}, k={n_clusters})", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3, fontsize=7, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "campaign_clusters.png", dpi=120)
    plt.close(fig)

    # Plot 3: Silhouette score vs k (only when we have score data)
    if sil_scores:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o", color="#185FA5", lw=2)
        ax.axvline(n_clusters, color="#A32D2D", lw=1.5, linestyle="--", label=f"Selected k = {n_clusters}")
        ax.set_xlabel("Number of clusters (k)", fontsize=11)
        ax.set_ylabel("Silhouette score", fontsize=11)
        ax.set_title("Silhouette Score vs k", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "silhouette_vs_k.png", dpi=120)
        plt.close(fig)
    else:
        print("Skipped silhouette_vs_k.png (silhouette scores unavailable for tiny sample)")

    print("\nFigures saved: campaign_sizes.png | campaign_clusters.png | silhouette_vs_k.png")
    print("\nCampaign clustering complete.")
    print(f"  campaign_model.pkl  -> {MODELS_DIR / 'campaign_model.pkl'}")
    print(f"  campaign_scaler.pkl -> {MODELS_DIR / 'campaign_scaler.pkl'}")


if __name__ == "__main__":
    main()
