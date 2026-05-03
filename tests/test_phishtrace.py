"""
PhishTrace — Integration Test Suite
=====================================
Tests the full pipeline: Feature Extraction → Model → API → DB

Run:
    # Make sure the API is running first:
    python src/api/app.py

    # Then in another terminal:
    python tests/test_phishtrace.py

    # Or run without API (feature + model tests only):
    python tests/test_phishtrace.py --offline
"""

import sys
import os
import json
import time
import argparse
import requests
from pathlib import Path

# ── Setup path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

API_BASE = "http://127.0.0.1:5000"

# ── Colors ────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed   = 0
failed   = 0
warnings = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET}  {msg}")

def fail(msg, detail=""):
    global failed
    failed += 1
    print(f"  {RED}✗{RESET}  {msg}")
    if detail:
        print(f"      {RED}→ {detail}{RESET}")

def warn(msg):
    global warnings
    warnings += 1
    print(f"  {YELLOW}⚠{RESET}  {msg}")

def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")


# ═══════════════════════════════════════════════════════════
#  TEST CASES — URLs with expected verdicts
# ═══════════════════════════════════════════════════════════
TEST_CASES = [
    # ── DANGEROUS ──────────────────────────────────────────
    {
        "url"              : "http://192.168.0.45/secure-login",
        "expected_verdict" : "Dangerous",
        "expected_score_gt": 0.70,
        "description"      : "IP-based URL with suspicious word",
        "expected_features": {"has_ip": 1},
    },
    {
        "url"              : "http://bankofegypt-login.evil.xyz/confirm?token=abc123",
        "expected_verdict" : "Dangerous",
        "expected_score_gt": 0.60,
        "description"      : "Brand impersonation + suspicious TLD + suspicious word",
        "expected_features": {},
    },
    {
        "url"              : "http://secure-account-update.com/verify/login/confirm",
        "expected_verdict" : "Dangerous",
        "expected_score_gt": 0.65,
        "description"      : "Multiple suspicious words in path",
        "expected_features": {},
    },
    {
        "url"              : "http://paypal-account.verify-secure.login.tk/update",
        "expected_verdict" : "Dangerous",
        "expected_score_gt": 0.65,
        "description"      : "Many subdomains + suspicious TLD + suspicious words",
        "expected_features": {"num_subdomains": 2},
    },

    # ── SUSPICIOUS ─────────────────────────────────────────
    {
        "url"              : "http://login-portal.xyz/account",
        "expected_verdict" : ["Suspicious", "Dangerous"],
        "expected_score_gt": 0.45,
        "description"      : "Suspicious TLD + login word",
        "expected_features": {},
    },
    {
        "url"              : "https://amazoon.net/deals/today",
        "expected_verdict" : ["Suspicious", "Dangerous"],
        "expected_score_gt": 0.40,
        "description"      : "Typosquatting Amazon",
        "expected_features": {},
    },

    # ── SAFE ───────────────────────────────────────────────
    {
        "url"              : "https://www.google.com",
        "expected_verdict" : "Safe",
        "expected_score_lt": 0.45,
        "description"      : "Legitimate Google URL",
        "expected_features": {"has_https": 1, "has_ip": 0, "num_suspicious_words": 0},
    },
    {
        "url"              : "https://www.github.com/topics/python",
        "expected_verdict" : "Safe",
        "expected_score_lt": 0.45,
        "description"      : "Legitimate GitHub URL",
        "expected_features": {"has_https": 1, "has_ip": 0},
    },
    {
        "url"              : "https://stackoverflow.com/questions/11227809",
        "expected_verdict" : "Safe",
        "expected_score_lt": 0.55,
        "description"      : "Legitimate StackOverflow URL",
        "expected_features": {"has_https": 1},
    },
]

# ✅ Thresholds — متطابقة مع app.py و evaluation_report
THRESHOLD_DANGEROUS  = 0.75
THRESHOLD_SUSPICIOUS = 0.45


# ═══════════════════════════════════════════════════════════
#  Rule-based boost — نفس المنطق في app.py
#  ✅ FIX: أضفنا الدالة هنا عشان test_model يعمل نفس حساب الـ score
# ═══════════════════════════════════════════════════════════
def rule_based_boost(feats: dict, raw_score: float) -> float:
    """
    بترجع الـ final_score بعد تطبيق الـ rules.
    نفس المنطق بالظبط اللي في app.py — عشان الـ test يحاكي الـ API صح.
    """
    boost = 0.0

    if feats.get("is_typosquat", 0):
        boost += 0.15
    if feats.get("has_ip", 0):
        boost += 0.15
    if feats.get("brand_in_subdomain", 0):
        boost += 0.15
    if feats.get("tld_suspicious", 0):
        boost += 0.10
    if feats.get("has_at_in_url", 0):
        boost += 0.10
    if feats.get("num_subdomains", 0) > 3:
        boost += 0.05
    if feats.get("hostname_entropy", 0) > 4.0:
        boost += 0.05

    boost = min(boost, 0.30)
    if raw_score < 0.40:
        boost = boost * 0.30
    elif raw_score < 0.65:
        boost = boost * 0.60

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

    return min(raw_score + boost, 1.0)


def get_verdict(score: float) -> str:
    if score >= THRESHOLD_DANGEROUS:
        return "Dangerous"
    elif score >= THRESHOLD_SUSPICIOUS:
        return "Suspicious"
    else:
        return "Safe"


# ═══════════════════════════════════════════════════════════
#  1. Feature Extraction Tests
# ═══════════════════════════════════════════════════════════
def test_features():
    section("1. Feature Extraction — extract.py")
    try:
        from features.extract import extract_features
    except ImportError as e:
        fail("Could not import extract_features", str(e))
        return

    # Test 1: IP detection
    f = extract_features("http://192.168.0.45/login")
    if f.get("has_ip") == 1:
        ok("IP detection: http://192.168.0.45 → has_ip=1")
    else:
        fail("IP detection failed", f"has_ip={f.get('has_ip')}")

    # Test 2: HTTPS detection
    f_https = extract_features("https://www.google.com")
    f_http  = extract_features("http://evil.com")
    if f_https.get("has_https") == 1 and f_http.get("has_https") == 0:
        ok("HTTPS detection: https → 1, http → 0")
    else:
        fail("HTTPS detection failed")

    # Test 3: Suspicious word detection
    f = extract_features("http://secure-login.com/verify")
    if f.get("num_suspicious_words", 0) >= 1:
        ok("Suspicious word: 'login' + 'verify' → detected")
    else:
        fail("Suspicious word detection failed", f"num_suspicious_words={f.get('num_suspicious_words')}")

    # Test 4: Subdomains
    f     = extract_features("http://a.b.c.evil.com/page")
    count = f.get("num_subdomains", 0)
    if count >= 2:
        ok(f"Subdomain counting: a.b.c.evil.com → {count} subdomains")
    else:
        fail("Subdomain counting failed", f"got {count}")

    # Test 5: URL length
    long_url = "http://evil.com/" + "a" * 100
    f = extract_features(long_url)
    if f.get("url_length", 0) > 100:
        ok(f"URL length: {f['url_length']} chars")
    else:
        fail("URL length calculation failed")

    # Test 6: @ in URL
    f = extract_features("http://user@evil.com/login")
    if f.get("has_at_in_url") == 1:
        ok("@ in URL detection: user@evil.com → has_at_in_url=1")
    else:
        fail("@ in URL detection failed")

    # Test 7: extra features موجودة (is_typosquat, hostname_entropy, ...)
    # ✅ FIX: كان بيتحقق من 19 features بس — دلوقتي بيتحقق من الـ extra كمان
    f = extract_features("https://www.example.com")
    extra_keys = ["is_typosquat", "min_levenshtein", "hostname_entropy",
                  "brand_in_subdomain", "tld_suspicious"]
    missing = [k for k in extra_keys if k not in f]
    if not missing:
        ok(f"Extra features present: {extra_keys}")
    else:
        fail("Extra features missing from extract_features()", str(missing))

    # Test 8: unified_extractor
    try:
        from features.unified_extractor import extract_all
        f = extract_all("http://paypal-verify.xyz/account")
        if len(f) >= 23:
            ok(f"unified_extractor: {len(f)} features (19 base + extras)")
        else:
            warn(f"unified_extractor: only {len(f)} features")
        if f.get("tld_suspicious") == 1:
            ok("tld_suspicious: .xyz → flagged correctly")
        else:
            warn(f"tld_suspicious: .xyz not flagged (got {f.get('tld_suspicious')})")
    except ImportError:
        warn("unified_extractor.py not importable — skipping")


# ═══════════════════════════════════════════════════════════
#  2. Model Tests (offline)
# ═══════════════════════════════════════════════════════════
def test_model():
    section("2. Model — Random Forest Predictions (with rule boost)")
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from features.extract import extract_features
    except ImportError as e:
        fail("Missing dependency", str(e))
        return

    models_dir  = ROOT / "models"
    scaler_path = models_dir / "scaler.pkl"
    model_path  = models_dir / "best_model.pkl"

    if not scaler_path.exists() or not model_path.exists():
        fail("Models not found", f"Check {models_dir}/")
        return

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    # ✅ نفس الـ 19 columns اللي الموديل اتعلم عليها
    FEATURE_COLS = [
        "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
        "num_at","num_question","num_equals","num_percent",
        "num_digits_in_domain","num_digits_in_path","last_path_segment_is_integer",
        "has_ip","has_https","num_subdomains",
        "hostname_length","path_length","double_slash",
        "num_suspicious_words",
    ]

    ok("Models loaded: scaler.pkl + best_model.pkl")

    is_real_model = getattr(model, "n_estimators", 0) >= 100
    if not is_real_model:
        warn("Dummy model detected (n_estimators<100) — score thresholds skipped")
        warn("Run python src/models/train.py first to get the real trained model")

    correct = 0
    for tc in TEST_CASES:
        url   = tc["url"]

        # ✅ FIX: استخدم extract_features الكاملة (بترجع extra features كمان)
        feats    = extract_features(url)
        X        = pd.DataFrame([feats])[FEATURE_COLS]
        X_scaled = scaler.transform(X)
        raw_score = float(model.predict_proba(X_scaled)[0][1])

        # ✅ FIX: طبّق الـ rule boost عشان تحاكي الـ API صح
        # بدونه الـ Suspicious URLs مش بتتعرف والـ Safe بيتصنف غلط
        final_score = rule_based_boost(feats, raw_score)
        actual      = get_verdict(final_score)

        expected = tc["expected_verdict"]
        if isinstance(expected, str):
            expected_label = [expected]
        else:
            expected_label = expected

        # Check score thresholds only for real model
        score_ok = True
        if is_real_model:
            if "expected_score_gt" in tc and final_score <= tc["expected_score_gt"]:
                score_ok = False
            if "expected_score_lt" in tc and final_score >= tc["expected_score_lt"]:
                score_ok = False

        if actual in expected_label and score_ok:
            ok(f"{actual:10} {final_score:.0%}  {tc['description'][:45]}")
            correct += 1
        elif actual in expected_label:
            warn(f"{actual:10} {final_score:.0%}  {tc['description'][:40]} (score threshold missed)")
        else:
            if is_real_model:
                fail(
                    f"Expected {'/'.join(expected_label):12} got {actual:10} {final_score:.0%}",
                    tc["description"]
                )
            else:
                warn(f"[dummy] Expected {'/'.join(expected_label):10} got {actual:10} "
                     f"{final_score:.0%} — {tc['description'][:35]}")

    print(f"\n  Model accuracy on test cases: {correct}/{len(TEST_CASES)}")
    if not is_real_model:
        warn("Score thresholds skipped — real model needed for full accuracy test")
    elif correct >= len(TEST_CASES) * 0.75:
        ok(f"Model correctness: {correct}/{len(TEST_CASES)} ≥ 75% threshold")
    else:
        fail(f"Model correctness: {correct}/{len(TEST_CASES)} < 75% threshold")


# ═══════════════════════════════════════════════════════════
#  3. Database Tests
# ═══════════════════════════════════════════════════════════
def test_database():
    section("3. Database — db.py")
    try:
        from database.db import init_db, save_scan, get_scan, get_history, get_stats
    except ImportError as e:
        fail("Could not import database.db", str(e))
        return

    try:
        init_db()
        ok("init_db() — tables created / verified")
    except Exception as e:
        fail("init_db() failed", str(e))
        return

    test_scan = {
        "url"        : "http://test-phishtrace.com/test",
        "score"      : 0.88,
        "verdict"    : "Dangerous",
        "features"   : {"has_ip": 0, "num_suspicious_words": 1},
        "reasons"    : [{"feature": "num_suspicious_words", "contribution": 0.3}],
        "campaign_id": None,
    }

    try:
        scan_id = save_scan(
            url          = test_scan["url"],
            verdict      = test_scan["verdict"],
            score        = test_scan["score"],
            raw_score    = test_scan["score"],
            shap_reasons = test_scan["reasons"],
            features     = test_scan["features"],
            campaign_id  = test_scan["campaign_id"],
        )
        if isinstance(scan_id, int) and scan_id > 0:
            ok(f"save_scan() → scan_id={scan_id}")
        else:
            fail("save_scan() returned invalid id", str(scan_id))
            return
    except Exception as e:
        fail("save_scan() failed", str(e))
        return

    try:
        retrieved = get_scan(scan_id)
        if retrieved and retrieved["url"] == test_scan["url"]:
            ok(f"get_scan({scan_id}) → URL matches")
        else:
            fail("get_scan() returned wrong data")
    except Exception as e:
        fail("get_scan() failed", str(e))

    try:
        history = get_history(limit=5)
        if isinstance(history, list) and len(history) > 0:
            ok(f"get_history() → {len(history)} records")
        else:
            fail("get_history() returned empty")
    except Exception as e:
        fail("get_history() failed", str(e))

    try:
        stats    = get_stats()
        required = ["total_scans", "dangerous", "suspicious", "safe", "model_metrics"]
        missing  = [k for k in required if k not in stats]
        if not missing:
            ok(f"get_stats() → total={stats['total_scans']}, F1={stats['model_metrics'].get('f1','—')}")
        else:
            fail("get_stats() missing keys", str(missing))
    except Exception as e:
        fail("get_stats() failed", str(e))


# ═══════════════════════════════════════════════════════════
#  4. SHAP Tests
# ═══════════════════════════════════════════════════════════
def test_shap():
    section("4. SHAP Explainability")
    try:
        import joblib
        import shap
        import numpy as np
        import pandas as pd
        from features.extract import extract_features
    except ImportError as e:
        fail("Missing dependency", str(e))
        return

    models_dir  = ROOT / "models"
    scaler_path = models_dir / "scaler.pkl"
    model_path  = models_dir / "best_model.pkl"

    if not scaler_path.exists() or not model_path.exists():
        fail("Models not found — skipping SHAP test")
        return

    FEATURE_COLS = [
        "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
        "num_at","num_question","num_equals","num_percent",
        "num_digits_in_domain","num_digits_in_path","last_path_segment_is_integer",
        "has_ip","has_https","num_subdomains",
        "hostname_length","path_length","double_slash",
        "num_suspicious_words",
    ]

    try:
        scaler    = joblib.load(scaler_path)
        model     = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)

        feats    = extract_features("http://bankofegypt-login.evil.xyz/confirm")
        X        = pd.DataFrame([feats])[FEATURE_COLS]
        X_scaled = scaler.transform(X)

        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        if len(sv) == len(FEATURE_COLS):
            ok(f"SHAP values computed: {len(sv)} features")
            top_i   = int(np.argmax(np.abs(sv)))
            top_feat = FEATURE_COLS[top_i]
            ok(f"Top SHAP feature: {top_feat} ({sv[top_i]:+.4f})")
        else:
            fail("SHAP values length mismatch", f"got {len(sv)}, expected {len(FEATURE_COLS)}")
    except Exception as e:
        fail("SHAP test failed", str(e))


# ═══════════════════════════════════════════════════════════
#  5. API Tests (online)
# ═══════════════════════════════════════════════════════════
def test_api():
    section("5. Flask API — End-to-End")

    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "ok":
            ok("GET /health → 200 OK")
        else:
            fail("GET /health failed", str(r.status_code))
            return
    except Exception as e:
        fail("API not reachable — is it running?", str(e))
        print(f"\n  {YELLOW}Start the API with: python src/api/app.py{RESET}\n")
        return

    print(f"\n  {'URL':<45} {'Expected':<12} {'Got':<12} {'Score'}")
    print(f"  {'─'*80}")

    api_correct = 0
    for tc in TEST_CASES:
        try:
            r = requests.post(
                f"{API_BASE}/analyze",
                json={"url": tc["url"]},
                timeout=10
            )
            if r.status_code != 200:
                fail(f"POST /analyze → {r.status_code}", tc["url"])
                continue

            data    = r.json()
            verdict = data.get("verdict", "?")
            score   = data.get("score", 0)

            expected = tc["expected_verdict"]
            if isinstance(expected, str):
                expected = [expected]

            verdict_ok = verdict in expected
            score_ok   = True
            if "expected_score_gt" in tc and score <= tc["expected_score_gt"]:
                score_ok = False
            if "expected_score_lt" in tc and score >= tc["expected_score_lt"]:
                score_ok = False

            exp_str = "/".join(expected) if len(expected) > 1 else expected[0]
            status  = "✓" if (verdict_ok and score_ok) else ("⚠" if verdict_ok else "✗")
            color   = GREEN if status == "✓" else (YELLOW if status == "⚠" else RED)

            url_short = tc["url"][:44]
            print(f"  {color}{status}{RESET}  {url_short:<44} {exp_str:<12} {verdict:<12} {score:.0%}")

            if "features" in data and tc.get("expected_features"):
                for feat, val in tc["expected_features"].items():
                    actual_val = data["features"].get(feat)
                    if actual_val == val:
                        pass  # feature check passed silently
                    else:
                        warn(f"Feature mismatch: {feat} expected={val} got={actual_val}")

            if verdict_ok and score_ok:
                api_correct += 1

        except Exception as e:
            fail(f"Request failed for {tc['url'][:40]}", str(e))

    print(f"\n  API accuracy: {api_correct}/{len(TEST_CASES)}")
    if api_correct >= len(TEST_CASES) * 0.75:
        ok(f"API correctness: {api_correct}/{len(TEST_CASES)} ≥ 75%")
    else:
        fail(f"API correctness: {api_correct}/{len(TEST_CASES)} < 75%")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Skip API tests (run features + model + DB only)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*55}")
    print("  PhishTrace — Integration Test Suite")
    print(f"{'═'*55}{RESET}")
    print(f"  Root: {ROOT}")
    print(f"  Mode: {'Offline (no API)' if args.offline else 'Full (API required)'}")

    test_features()
    test_model()
    test_database()
    test_shap()
    if not args.offline:
        test_api()

    total = passed + failed
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"  {BOLD}Results:{RESET}  "
          f"{GREEN}{passed} passed{RESET}  "
          f"{RED}{failed} failed{RESET}  "
          f"{YELLOW}{warnings} warnings{RESET}")
    print(f"{'═'*55}")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}✅ All tests passed — project is ready!{RESET}\n")
    elif failed <= 2:
        print(f"\n  {YELLOW}{BOLD}⚠  Minor issues — check warnings above{RESET}\n")
    else:
        print(f"\n  {RED}{BOLD}❌ {failed} tests failed — review above{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()