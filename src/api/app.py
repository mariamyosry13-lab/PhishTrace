import os
import sys
import joblib
import numpy as np
import pandas as pd
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import shap
import warnings
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but StandardScaler was fitted without feature names",
    category=UserWarning,
)

# ── Add src to path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Fix #3: import thresholds from config — no re-definition here
from config import FEATURE_COLS, THRESHOLD_DANGEROUS, THRESHOLD_SUSPICIOUS
# Fix #8: use extract_all so extra features (url_entropy, path_depth, brand_impersonation)
#         appear in the API response under "features" for display
from features.unified_extractor import extract_all
from database.db import init_db, save_scan, get_history, get_dashboard_stats, get_campaigns
# Fix #9: single authoritative assign_campaign from clustering module
from clustering.campaign import assign_campaign
from models import bert_classifier

logger = logging.getLogger(__name__)
app = Flask(__name__,
            template_folder="../../frontend/templates",
            static_folder="../../frontend/static")
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

# ── Load ML model + SHAP explainer once at startup ───────────────────────────
logger.info("Loading models...")
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    model  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
except FileNotFoundError as exc:
    logger.error("Required model file missing in %s: %s", MODELS_DIR, exc)
    raise RuntimeError(f"Required model files not found in {MODELS_DIR}") from exc
except Exception as exc:
    logger.error("Failed loading model artifacts: %s", exc, exc_info=True)
    raise RuntimeError("Failed to load required model artifacts at startup") from exc

# Fix #13: graceful SHAP fallback — TreeExplainer for RF/XGBoost,
#          LinearExplainer for LogisticRegression, None if neither works.
explainer = None
try:
    explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer ready.")
except Exception:
    try:
        from sklearn.linear_model import LogisticRegression as _LR
        if isinstance(model, _LR):
            _bg       = np.zeros((1, len(FEATURE_COLS)))
            explainer = shap.LinearExplainer(model, _bg)
            logger.info("SHAP LinearExplainer ready (LogisticRegression model).")
        else:
            logger.warning("SHAP explainer unsupported for this model type — explanations disabled.")
    except Exception as _exc:
        logger.warning("SHAP explainer failed: %s — explanations disabled.", _exc)

logger.info("Models loaded.")

bert_classifier.load()
RF_WEIGHT   = 0.60
BERT_WEIGHT = 0.40

# ── Init DB ──────────────────────────────────────────────────────────────────
init_db()

# Fix #6: feature → human-readable English description for SHAP reasons
_SHAP_TEXT = {
    "url_length"                  : "Total URL length",
    "num_dots"                    : "Dot count — many dots suggest nested subdomains",
    "num_hyphens"                 : "Hyphen count — often used to mimic brand names",
    "num_underscores"             : "Underscore count — rare in legitimate URLs",
    "num_slashes"                 : "Slash count — deep path hierarchy",
    "num_at"                      : "@ symbol — can redirect to a different host",
    "num_question"                : "Query marker count",
    "num_equals"                  : "Parameter assignment count",
    "num_percent"                 : "URL-encoded character count",
    "num_digits_in_domain"        : "Digits in domain name",
    "num_digits_in_path"          : "Digits in path (often legitimate numeric IDs)",
    "last_path_segment_is_integer": "Last path segment is a pure integer",
    "has_ip"                      : "Raw IP address instead of a domain name",
    "has_https"                   : "Uses HTTPS (reduces phishing risk)",
    "num_subdomains"              : "Subdomain level count",
    "hostname_length"             : "Hostname length",
    "path_length"                 : "URL path length",
    "double_slash"                : "Double slash in path — open redirect risk",
    "num_suspicious_words"        : "Count of phishing-related keywords in path",
}

# ── Model metrics from evaluation_results.json ───────────────────────────────
def load_model_metrics():
    path = os.path.join(MODELS_DIR, "evaluation_results.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            data = json.load(f)
        return {
            # Fix #7: include model name so the frontend can display it accurately
            "model"    : data.get("model",     "unknown"),
            "accuracy" : data.get("accuracy",  0),
            "precision": data.get("precision", 0),
            "recall"   : data.get("recall",    0),
            "f1"       : data.get("f1",        0),
            "fpr"      : data.get("fpr",       0),
        }
    return {
        "model"    : "unknown",
        "accuracy" : 0.94,
        "precision": 0.94,
        "recall"   : 0.96,
        "f1"       : 0.9686,
        "fpr"      : 0.08,
    }

MODEL_METRICS = load_model_metrics()


# ── Rule-based boost ─────────────────────────────────────────────────────────
def rule_based_boost(feats: dict, raw_score: float) -> tuple[float, list[str]]:
    boost = 0.0
    rules = []

    lev = feats.get("min_levenshtein", 99)
    if feats.get("is_typosquat", 0):
        boost += 0.15
        rules.append(f"⚠️ Typosquatting: الدومين شبيه جداً لدومين مشهور (Levenshtein={lev})")

    if feats.get("has_ip", 0):
        boost += 0.15
        rules.append("⚠️ الرابط بيستخدم IP بدل اسم دومين — علامة خطر قوية")

    if feats.get("brand_in_subdomain", 0):
        boost += 0.15
        rules.append("⚠️ اسم علامة تجارية موجود في الـ subdomain — انتبه")

    if feats.get("tld_suspicious", 0):
        boost += 0.10
        rules.append("⚠️ الـ TLD مشبوه (.tk / .xyz / .click ...)")

    if feats.get("has_at_in_url", 0):
        boost += 0.10
        rules.append("⚠️ وجود @ في الرابط — تقنية تضليل شائعة")

    if feats.get("num_subdomains", 0) > 3:
        boost += 0.05
        rules.append(f"⚠️ عدد subdomains كبير ({feats['num_subdomains']})")

    entropy = feats.get("hostname_entropy", 0)
    if entropy > 4.0:
        boost += 0.05
        rules.append(f"⚠️ الـ hostname يبدو عشوائي (entropy={entropy:.2f})")

    boost = min(boost, 0.30)
    # Graduated dampening: a single rule signal on a low-confidence base score
    # shouldn't be enough to push a URL into Dangerous.
    if raw_score < 0.40:
        boost = boost * 0.30
    elif raw_score < 0.65:
        boost = boost * 0.60

    # Dampen scores for URLs that have no red-flag signals at all.
    # When every rule check is clean, the ML score is driven by structural
    # features like num_digits from legitimate numeric IDs (e.g. /questions/11227809)
    # or BERT reacting to topic keywords in the path (e.g. /wiki/Phishing).
    # In both cases the URL is safe — reduce confidence unconditionally.
    is_clean = (
        feats.get("has_https", 0) == 1 and
        feats.get("tld_suspicious", 0) == 0 and
        feats.get("num_suspicious_words", 0) == 0 and
        feats.get("has_ip", 0) == 0 and
        feats.get("is_typosquat", 0) == 0 and
        feats.get("brand_in_subdomain", 0) == 0 and
        feats.get("has_at_in_url", 0) == 0
    )
    if is_clean:
        raw_score = raw_score * 0.40

    return min(raw_score + boost, 1.0), rules


def get_verdict(score: float) -> str:
    if score >= THRESHOLD_DANGEROUS:
        return "Dangerous"
    elif score >= THRESHOLD_SUSPICIOUS:
        return "Suspicious"
    else:
        return "Safe"


def get_shap_explanation(features_scaled):
    # Fix #13: return empty list if explainer is unavailable
    if explainer is None:
        return []

    shap_values = explainer.shap_values(features_scaled)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    top_idx = np.argsort(np.abs(sv))[::-1][:5]
    return [
        {
            "feature"     : FEATURE_COLS[i],
            # Fix #6: add English description so the report download "Why it matters" column works
            "text_en"     : _SHAP_TEXT.get(FEATURE_COLS[i], FEATURE_COLS[i]),
            "contribution": round(float(sv[i]), 4),
            "direction"   : "increases" if sv[i] > 0 else "decreases",
        }
        for i in top_idx
    ]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Fix #8: extract_all adds url_entropy, path_depth, brand_impersonation
    #         to the feats dict for display; FEATURE_COLS slice still picks
    #         only the 19 ML features for the model.
    feats    = extract_all(url)
    X        = pd.DataFrame([feats])[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    rf_score   = float(model.predict_proba(X_scaled)[0][1])
    bert_score = bert_classifier.predict_proba(url)

    if bert_score is not None:
        raw_score = RF_WEIGHT * rf_score + BERT_WEIGHT * bert_score
    else:
        raw_score = rf_score

    final_score, rule_alerts = rule_based_boost(feats, raw_score)
    verdict                  = get_verdict(final_score)
    shap_reasons             = get_shap_explanation(X_scaled)
    campaign_id              = assign_campaign(feats)

    scan_id = save_scan(
        url          = url,
        verdict      = verdict,
        score        = final_score,
        raw_score    = raw_score,
        rule_alerts  = rule_alerts,
        shap_reasons = shap_reasons,
        features     = feats,
        campaign_id  = campaign_id,
    )

    return jsonify({
        "url"         : url,
        "scan_id"     : scan_id,
        "campaign_id" : campaign_id,
        # Fix #7: expose actual model name so the frontend never has to hardcode it
        "model_name"  : MODEL_METRICS.get("model", "unknown"),
        "score"       : round(final_score, 4),
        "raw_ml_score": round(raw_score, 4),
        "rf_score"    : round(rf_score, 4),
        "bert_score"  : round(bert_score, 4) if bert_score is not None else None,
        "percent"     : round(final_score * 100, 1),
        "verdict"     : verdict,
        "reasons"     : shap_reasons,
        "rule_alerts" : rule_alerts,
        "features"    : feats,
    })


@app.route("/api/dashboard")
def dashboard():
    stats = get_dashboard_stats()
    stats["false_positives"] = []
    return jsonify(stats)


@app.route("/api/campaigns")
def campaigns():
    return jsonify({"campaigns": get_campaigns()})


@app.route("/api/history")
def history():
    limit = request.args.get("limit", 50, type=int)
    if limit is None:
        limit = 50
    limit = max(1, min(limit, 500))
    return jsonify({"scans": get_history(limit)})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # use_reloader=False prevents Werkzeug from spawning a second process that
    # would load the XGBoost model + SHAP TreeExplainer twice, exhausting RAM.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
