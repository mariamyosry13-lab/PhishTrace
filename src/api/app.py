import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Add src to path ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ✅ FIX: استخدم extract_features مباشرة — بترجع الـ extra features كمان
# (min_levenshtein, is_typosquat, hostname_entropy, brand_in_subdomain, ...)
from features.extract import extract_features

app = Flask(__name__,
            template_folder="../../frontend/templates",
            static_folder="../../frontend/static")
CORS(app)

# ── Paths ───────────────────────────────────────────────
MODELS_DIR = "models"

# ── Thresholds ──────────────────────────────────────────
# ✅ FIX: وحّدنا الـ thresholds مع test_model.py و evaluation_report
# كانوا 0.65 / 0.35 — وده كان بيخلي Safe URLs تتحسب Suspicious
THRESHOLD_DANGEROUS  = 0.75   # ✅ متطابق مع الـ tests و evaluation_report
THRESHOLD_SUSPICIOUS = 0.45   # ✅ متطابق مع الـ tests و evaluation_report

# ── Load models once at startup ─────────────────────────
print("Loading models...")
scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
model     = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
explainer = shap.TreeExplainer(model)
print("Models loaded ✅")

# ── Feature cols الأصلية اللي الموديل اتعلم عليها ─────
# ✅ NOTE: الموديل اتعلم على الـ 19 دول بس
# الـ extra features (min_levenshtein, is_typosquat, ...) بنستخدمها
# في rule_based_boost فقط — مش بندخّلها للموديل
FEATURE_COLS = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
    "num_at","num_question","num_equals","num_percent","num_digits",
    "has_ip","has_https","has_suspicious_word","num_subdomains",
    "hostname_length","path_length","double_slash","has_at_in_url",
    "num_suspicious_words"
]

# ── Rule-based score booster ─────────────────────────────
def rule_based_boost(feats: dict, raw_score: float) -> tuple[float, list[str]]:
    """
    بنضيف signals مبنية على rules فوق الـ ML score.
    بترجع (boosted_score, list of rule reasons).

    ✅ FIX: الـ feats دلوقتي بتيجي من extract_features الكاملة
    فكل الـ extra features موجودة (is_typosquat, hostname_entropy, ...)
    """
    boost = 0.0
    rules = []

    # Typosquatting: قريب جداً من دومين مشهور بس مش هو
    lev = feats.get("min_levenshtein", 99)
    if feats.get("is_typosquat", 0):
        boost += 0.25
        rules.append(f"⚠️ Typosquatting: الدومين شبيه جداً بدومين مشهور (Levenshtein={lev})")

    # IP بدل domain
    if feats.get("has_ip", 0):
        boost += 0.20
        rules.append("⚠️ الرابط بيستخدم IP بدل اسم دومين — علامة خطر قوية")

    # Brand في subdomain (مش في الـ main domain)
    # ✅ FIX: كان بيقرأ "brand_in_subdomain" بس ده اسمه في extract.py
    if feats.get("brand_in_subdomain", 0):
        boost += 0.20
        rules.append("⚠️ اسم علامة تجارية مشهورة موجود في الـ subdomain — انتبه")

    # TLD مشبوه
    if feats.get("tld_suspicious", 0):
        boost += 0.15
        rules.append("⚠️ الـ TLD (امتداد الدومين) مشبوه (.tk / .xyz / .click ...)")

    # @ في الرابط
    if feats.get("has_at_in_url", 0):
        boost += 0.15
        rules.append("⚠️ وجود @ في الرابط — تقنية تضليل شائعة")

    # كتير subdomains (أكتر من 3)
    if feats.get("num_subdomains", 0) > 3:
        boost += 0.10
        rules.append(f"⚠️ عدد subdomains كبير ({feats['num_subdomains']}) — مشبوه")

    # Entropy عالية في الـ hostname
    entropy = feats.get("hostname_entropy", 0)
    if entropy > 4.0:
        boost += 0.10
        rules.append(f"⚠️ الـ hostname يبدو عشوائي (entropy={entropy:.2f}) — ممكن domain generated")

    boosted = min(raw_score + boost, 1.0)
    return boosted, rules


def get_verdict(score: float) -> str:
    if score >= THRESHOLD_DANGEROUS:
        return "Dangerous"
    elif score >= THRESHOLD_SUSPICIOUS:
        return "Suspicious"
    else:
        return "Safe"


def get_shap_explanation(features_scaled):
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
        val = sv[i]
        reasons.append({
            "feature"     : FEATURE_COLS[i],
            "contribution": round(float(val), 4),
            "direction"   : "increases" if val > 0 else "decreases"
        })
    return reasons


# ── Routes ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json()
    url  = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # ✅ FIX: extract_features بترجع كل الـ features الكاملة
    # (الـ 19 الأصلية + is_typosquat + min_levenshtein + hostname_entropy + ...)
    feats = extract_features(url)

    # ✅ نبعت للموديل الـ 19 feature الأصلية بس
    X        = pd.DataFrame([feats])[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # ML score
    raw_score = float(model.predict_proba(X_scaled)[0][1])

    # ✅ FIX: بنبعت feats الكاملة لـ rule_based_boost
    # فالـ is_typosquat, hostname_entropy, brand_in_subdomain كلها متاحة
    final_score, rule_reasons = rule_based_boost(feats, raw_score)

    verdict      = get_verdict(final_score)
    shap_reasons = get_shap_explanation(X_scaled)

    return jsonify({
        "url"          : url,
        "score"        : round(final_score, 4),
        "raw_ml_score" : round(raw_score, 4),
        "percent"      : round(final_score * 100, 1),
        "verdict"      : verdict,
        "reasons"      : shap_reasons,
        "rule_alerts"  : rule_reasons,
        "features"     : feats
    })


@app.route("/api/dashboard")
def dashboard():
    return jsonify({
        "total_scans": 0, "dangerous": 0, "suspicious": 0, "safe": 0,
        "model_metrics": {"accuracy": 0.94, "precision": 0.94,
                          "recall": 0.96, "f1": 0.9686, "fpr": 0.08},
        "timeline": [], "false_positives": []
    })


@app.route("/api/campaigns")
def campaigns():
    return jsonify({"campaigns": []})


@app.route("/api/history")
def history():
    return jsonify({"scans": []})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)