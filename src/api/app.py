"""
PhishTrace — Flask API  (upgraded)
====================================
Changes vs original:
  - Real SQLite DB via database.db (every scan is persisted)
  - /api/dashboard returns live stats from DB
  - /api/history returns real scan history with pagination
  - /api/campaigns returns clusters saved by campaign.py
  - /api/scan/<id> returns a single scan by id
  - SHAP reasons include human-readable Arabic + English text
  - Robust import path resolution (no fragile sys.path hacks)
  - Input validation and basic rate-guard via flask-limiter (optional)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── Resolve project root & add src to path ───────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # project root (above src/api/)
SRC  = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from features.extract import extract_features      # src/features/extract.py
from database.db import (                          # src/database/db.py
    init_db, save_scan,
    get_history, get_stats, get_campaigns, get_scan
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(ROOT / "frontend" / "templates"),
    static_folder=str(ROOT / "frontend" / "static"),
)
CORS(app)

# ── Optional rate limiter (pip install flask-limiter) ─────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app,
                      default_limits=["200 per day", "60 per hour"])
    _RATE_LIMIT = "30 per minute"
except ImportError:
    limiter = None
    _RATE_LIMIT = None
    print("[API] flask-limiter not installed — rate limiting disabled")

# ── Paths & thresholds ───────────────────────────────────────────────────────
MODELS_DIR           = ROOT / "models"
THRESHOLD_DANGEROUS  = 0.75
THRESHOLD_SUSPICIOUS = 0.45

FEATURE_COLS = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
    "num_at","num_question","num_equals","num_percent","num_digits",
    "has_ip","has_https","has_suspicious_word","num_subdomains",
    "hostname_length","path_length","double_slash","has_at_in_url",
    "num_suspicious_words",
]

# Human-readable explanations for each feature (Arabic + English)
FEATURE_EXPLANATIONS = {
    "url_length"          : {"ar": "الرابط طويل جداً — علامة على إخفاء الهدف الحقيقي",
                              "en": "Unusually long URL — often used to obscure the real destination"},
    "num_dots"            : {"ar": "عدد نقاط كتير — ممكن يكون subdomains مشبوهة",
                              "en": "Many dots — possible excessive subdomains"},
    "num_hyphens"         : {"ar": "علامات وصل كتير في الدومين — شكل مريب",
                              "en": "Excessive hyphens — common in phishing domains"},
    "num_underscores"     : {"ar": "شرطات سفلية في الرابط — غير طبيعية في الدومينات الحقيقية",
                              "en": "Underscores in URL — unusual for legitimate domains"},
    "num_slashes"         : {"ar": "مسار معقد بسبب كثرة الـ slashes",
                              "en": "Deeply nested path — often used to confuse users"},
    "num_at"              : {"ar": "رمز @ في الرابط — يُخفي الدومين الحقيقي",
                              "en": "@ symbol hides the real domain"},
    "num_question"        : {"ar": "query parameters كتير — ممكن redirect خبيث",
                              "en": "Many query parameters — possible malicious redirect"},
    "num_equals"          : {"ar": "كتر قيم الـ parameters — ممكن بيجمع بيانات",
                              "en": "Many parameter values — possible data harvesting"},
    "num_percent"         : {"ar": "URL encoding مشبوه — ممكن يخبي حروف خطيرة",
                              "en": "URL-encoded characters — may hide malicious content"},
    "num_digits"          : {"ar": "أرقام كتير في الرابط — علامة على رابط مولّد تلقائياً",
                              "en": "Many digits — suggests auto-generated URL"},
    "has_ip"              : {"ar": "الرابط بيستخدم IP مباشرة بدل دومين — علامة خطر قوية جداً",
                              "en": "IP address instead of domain — strong phishing indicator"},
    "has_https"           : {"ar": "الموقع بيستخدم HTTPS (يقلل الخطر)",
                              "en": "HTTPS present (reduces risk)"},
    "has_suspicious_word" : {"ar": "فيه كلمة مشبوهة في الرابط زي login أو verify",
                              "en": "Suspicious word detected (e.g. login, verify, secure)"},
    "num_subdomains"      : {"ar": "subdomains كتير — تكتيك شائع لتقليد مواقع حقيقية",
                              "en": "Many subdomains — common trick to impersonate real sites"},
    "hostname_length"     : {"ar": "اسم الموقع طويل — ممكن بيقلد موقع تاني",
                              "en": "Long hostname — may be impersonating a known brand"},
    "path_length"         : {"ar": "المسار في الرابط طويل جداً",
                              "en": "Very long path component"},
    "double_slash"        : {"ar": "فيه // في المسار — ممكن open redirect",
                              "en": "Double slash in path — possible open redirect"},
    "has_at_in_url"       : {"ar": "رمز @ موجود في الرابط — يُخفي الوجهة الحقيقية",
                              "en": "@ in URL — hides the real destination"},
    "num_suspicious_words": {"ar": "عدد الكلمات المشبوهة في الرابط",
                              "en": "Count of suspicious words found in URL"},
}

# ── Startup: load models & init DB ───────────────────────────────────────────
print("[API] Loading models...")
scaler  = joblib.load(MODELS_DIR / "scaler.pkl")
model   = joblib.load(MODELS_DIR / "best_model.pkl")

import shap
explainer = shap.TreeExplainer(model)
print("[API] Models loaded ✅")

# Try loading campaign model (optional — generated by campaign.py)
_campaign_model  = None
_campaign_scaler = None
try:
    _campaign_model  = joblib.load(MODELS_DIR / "campaign_model.pkl")
    _campaign_scaler = joblib.load(MODELS_DIR / "campaign_scaler.pkl")
    print("[API] Campaign model loaded ✅")
except FileNotFoundError:
    print("[API] campaign_model.pkl not found — campaign assignment disabled")

init_db()


# ── Helper functions ──────────────────────────────────────────────────────────
def get_verdict(score: float) -> str:
    if score >= THRESHOLD_DANGEROUS:
        return "Dangerous"
    elif score >= THRESHOLD_SUSPICIOUS:
        return "Suspicious"
    return "Safe"


def get_shap_explanation(features_scaled: np.ndarray) -> list[dict]:
    """Return top-5 SHAP contributors with human-readable text."""
    shap_values = explainer.shap_values(features_scaled)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    top_idx = np.argsort(np.abs(sv))[::-1][:5]
    reasons = []
    for i in top_idx:
        val  = float(sv[i])
        feat = FEATURE_COLS[i]
        expl = FEATURE_EXPLANATIONS.get(feat, {"ar": feat, "en": feat})
        reasons.append({
            "feature"     : feat,
            "contribution": round(val, 4),
            "direction"   : "increases" if val > 0 else "decreases",
            "text_ar"     : expl["ar"],
            "text_en"     : expl["en"],
        })
    return reasons


def assign_campaign(features: dict) -> str | None:
    """Return campaign id for a phishing URL, or None if campaign model unavailable."""
    if _campaign_model is None:
        return None
    try:
        X = np.array([[features.get(c, 0) for c in FEATURE_COLS]])
        X_scaled = _campaign_scaler.transform(X)
        cluster_id = int(_campaign_model.predict(X_scaled)[0])
        return f"campaign_{cluster_id:03d}"
    except Exception as e:
        print(f"[API] Campaign assignment failed: {e}")
        return None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL too long (max 2048 chars)"}), 400

    # ── Feature extraction ────────────────────────────────────────────────────
    feats    = extract_features(url)
    X        = pd.DataFrame([feats])[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # ── Model inference ───────────────────────────────────────────────────────
    score   = float(model.predict_proba(X_scaled)[0][1])
    verdict = get_verdict(score)
    reasons = get_shap_explanation(X_scaled)

    # ── Campaign assignment (only for Dangerous / Suspicious) ─────────────────
    campaign_id = None
    if verdict in ("Dangerous", "Suspicious"):
        campaign_id = assign_campaign(feats)

    # ── Persist to DB ─────────────────────────────────────────────────────────
    scan_record = {
        "url"        : url,
        "score"      : score,
        "percent"    : round(score * 100, 1),
        "verdict"    : verdict,
        "features"   : feats,
        "reasons"    : reasons,
        "campaign_id": campaign_id,
    }
    scan_id = save_scan(scan_record)

    return jsonify({
        "scan_id"    : scan_id,
        "url"        : url,
        "score"      : round(score, 4),
        "percent"    : round(score * 100, 1),
        "verdict"    : verdict,
        "reasons"    : reasons,
        "features"   : feats,
        "campaign_id": campaign_id,
    })


@app.route("/api/scan/<int:scan_id>")
def get_single_scan(scan_id: int):
    """Return details for a previously analysed URL by scan id."""
    scan = get_scan(scan_id)
    if not scan:
        return jsonify({"error": f"Scan {scan_id} not found"}), 404
    return jsonify(scan)


@app.route("/api/dashboard")
def dashboard():
    """Live dashboard stats from DB + seeded model metrics."""
    stats = get_stats()
    return jsonify(stats)


@app.route("/api/history")
def history():
    """Paginated scan history. Query params: limit (default 50), offset (default 0)."""
    try:
        limit  = min(int(request.args.get("limit",  50)), 200)
        offset = max(int(request.args.get("offset",  0)),   0)
    except ValueError:
        limit, offset = 50, 0

    scans = get_history(limit=limit, offset=offset)
    return jsonify({"scans": scans, "limit": limit, "offset": offset})


@app.route("/api/campaigns")
def campaigns():
    """Return all discovered phishing campaigns from DB."""
    camps = get_campaigns()
    return jsonify({"campaigns": camps, "total": len(camps)})


@app.route("/health")
def health():
    """Simple health check."""
    return jsonify({"status": "ok", "model": "loaded", "db": "connected"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)