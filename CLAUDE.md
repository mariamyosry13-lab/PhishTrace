# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhishTrace is an AI-powered phishing URL detection system. It combines a Random Forest classifier (94% accuracy) with rule-based heuristics and SHAP explainability. URLs are classified as **safe**, **suspicious**, or **dangerous**, assigned to phishing campaigns via clustering, and stored in SQLite.

## Running the Application

```bash
# Set PYTHONPATH before running any src/ module
set PYTHONPATH=src   # Windows PowerShell: $env:PYTHONPATH = "src"

# Run the Flask API (serves frontend at http://localhost:5000)
python src/api/app.py
```

## Data & Model Pipeline (run in order)

```bash
python src/data/collect.py                # Download PhishTank + Tranco feeds → data/phishtrace_dataset.csv
python src/data/collect.py --refresh      # Force re-download

python src/features/extract.py            # Feature extraction (not typically run standalone)

python src/models/train.py                # Train RF/LR/XGB, saves best_model.pkl + scaler.pkl to models/

python src/clustering/campaign.py         # Fit MiniBatchKMeans, saves campaign_model.pkl + campaign_scaler.pkl

python src/explainability/explainer.py    # Generate SHAP plots → reports/
```

## Testing

```bash
pytest tests/
pytest tests/test_features.py            # Run a single test file
```

Note: current test files are mostly stubs.

## Architecture

### Request Flow

```
POST /analyze (url)
  → src/features/extract.py         # 19 ML features + 4 display features
  → models/best_model.pkl            # Random Forest probability score
  → Rule-based boosting              # +20% typosquatting, +15% IP/brand, etc. (max +30%)
  → models/campaign_model.pkl        # Assign to phishing campaign cluster
  → src/database/db.py               # Persist scan to SQLite
  → SHAP top-5 reasons               # From pre-loaded TreeExplainer
  → JSON response
```

### Key Modules

| Module | Role |
|--------|------|
| `src/api/app.py` | Flask routes, verdict logic, rule-based score boosting |
| `src/features/extract.py` | Extracts 19 ML features from a URL string |
| `src/features/unified_extractor.py` | Wraps extractor with 4 extra display features |
| `src/models/train.py` | Trains and selects best model; saves pkl artifacts |
| `src/clustering/campaign.py` | MiniBatchKMeans (k=25) for campaign grouping |
| `src/explainability/explainer.py` | SHAP TreeExplainer, outputs report PNGs |
| `src/database/db.py` | SQLite CRUD — `scans` and `campaigns` tables |
| `src/config.py` | Shared constants: thresholds, paths, split ratios |

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/analyze` | Analyze a URL, returns verdict + score + SHAP reasons |
| GET | `/api/dashboard` | Aggregate stats for the dashboard UI |
| GET | `/api/campaigns` | List detected phishing campaigns |
| GET | `/api/history` | Last N scanned URLs |
| GET | `/health` | Health check |

### Verdict Thresholds (`src/config.py`)

- `score >= 0.75` → **dangerous**
- `score >= 0.45` → **suspicious**
- `score < 0.45` → **safe**

### Feature Sets

**19 ML features** (fed to the model): `url_length`, `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_at`, `num_question`, `num_equals`, `num_percent`, `num_digits`, `has_ip`, `has_https`, `num_subdomains`, `hostname_length`, `path_length`, `double_slash`, `has_at_in_url`, `num_suspicious_words`, and one more — see `extract.py`.

**4 display-only features** (rule boosting + UI): `url_entropy`, `tld_suspicious`, `brand_impersonation`, `path_depth`.

### Frontend

Single-page app at `frontend/templates/index.html` with JavaScript routing between four views: Scanner, Dashboard, Campaigns, History. No build step — static files served by Flask.

### Persistence

- **SQLite** at `data/phishtrace.db` (auto-created on first run)
- **Trained models** at `models/best_model.pkl` (631 MB), `models/scaler.pkl`, `models/campaign_model.pkl`
- No migrations — schema is created fresh by `db.py` if the DB file is absent

## Tech Stack

- **Backend:** Python 3.11, Flask 3.1, Flask-CORS
- **ML:** scikit-learn 1.8, XGBoost 3.2, imbalanced-learn
- **Explainability:** SHAP 0.51
- **Data:** pandas 3.0, numpy 2.4, tldextract
- **Frontend:** Tailwind CSS, vanilla JS, FontAwesome (no build toolchain)
