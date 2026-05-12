# PhishTrace

PhishTrace is a phishing URL detection system that classifies URLs as **Safe**, **Suspicious**, or **Dangerous** using a combination of machine learning and rule-based scoring.

## How it works

Each URL goes through three stages:

1. **Feature extraction** — 19 structural features are extracted: URL length, subdomain count, HTTPS, suspicious TLD, typosquatting distance from known brands, phishing keywords in path, digits in domain, and more.
2. **ML scoring** — a trained Random Forest scores the URL (0–1). If BERT is installed, scores are blended 60% RF / 40% BERT.
3. **Rule adjustment** — hard-evidence signals (raw IP, brand in subdomain, @ redirect, suspicious TLD + typosquat) can push the score to Dangerous. Structural-only signals without hard evidence are capped in the Suspicious band. Clean URLs with no red flags get a 60% confidence reduction.

Thresholds: Safe < 0.45, Suspicious 0.45–0.75, Dangerous ≥ 0.75.

Each scan is saved to SQLite. Similar attack patterns are grouped into campaigns via KMeans clustering. SHAP values explain which features drove each prediction.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

pip install -r requirements.txt
python src/api/app.py
```

Open http://localhost:5000.

For BERT support (optional, requires ~400 MB download):

```bash
pip install "transformers>=4.40.0" "torch>=2.6.0"
```

## Retrain from scratch

```bash
python src/data/collect.py          # merge raw datasets
python src/features/extract.py      # build feature matrix
python src/models/train.py          # train RF, XGBoost, LR — picks best
python src/models/evaluate.py       # generate reports and figures
```

## Project layout

```
src/
  api/app.py            Flask API  (/analyze, /api/history, /api/campaigns)
  features/             URL feature extraction
  models/               training, evaluation, BERT wrapper
  clustering/           campaign detection
  data/                 data collection pipeline
frontend/               HTML / CSS / JS interface
models/                 saved model artifacts (.pkl, .json)
data/raw/               raw datasets (PhishTank, OpenPhish, Tranco)
reports/figures/        evaluation figures
  url_length_dist.png           URL length distribution (notebook 01)
  feature_correlation.png       feature correlation heatmap (notebook 02)
  roc_curves.png                ROC curves for all trained models
  threshold_analysis.png        precision / recall vs. threshold
  threshold_cost.png            cost-weighted threshold selection
  feature_importance_best.png   feature importances for the best model
  shap_bar.png                  global SHAP mean |value| bar chart
  shap_summary.png              SHAP beeswarm summary plot
  shap_waterfall_sample.png     SHAP waterfall for a single sample
  campaign_clusters.png         2-D cluster projection of campaigns
  campaign_sizes.png            bar chart of campaign sizes
  silhouette_vs_k.png           silhouette score vs. number of clusters K
```
