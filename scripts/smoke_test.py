"""
PhishTrace manual smoke test tool.

Moved from tests/test_phishtrace.py to keep pytest suite deterministic.
Run manually when you want an end-to-end diagnostic check:

    python scripts/smoke_test.py
    python scripts/smoke_test.py --offline
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def run_features_smoke() -> bool:
    from features.extract import extract_features

    urls = [
        "http://192.168.0.45/login",
        "https://www.google.com",
        "http://secure-login.com/verify",
    ]
    for url in urls:
        feats = extract_features(url)
        if not isinstance(feats, dict) or not feats:
            print(f"[FAIL] extract_features returned invalid payload for: {url}")
            return False
    print("[PASS] Feature extraction smoke")
    return True


def run_model_smoke() -> bool:
    import joblib
    import pandas as pd
    from features.extract import extract_features

    model_path = ROOT / "models" / "best_model.pkl"
    scaler_path = ROOT / "models" / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        print("[FAIL] Missing model/scaler artifacts")
        return False

    feature_cols = [
        "url_length", "num_dots", "num_hyphens", "num_underscores", "num_slashes",
        "num_at", "num_question", "num_equals", "num_percent",
        "num_digits_in_domain", "num_digits_in_path", "last_path_segment_is_integer",
        "has_ip", "has_https", "num_subdomains",
        "hostname_length", "path_length", "double_slash",
        "num_suspicious_words",
    ]
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    feats = extract_features("http://192.168.0.1/login/verify")
    X = pd.DataFrame([feats])[feature_cols]
    score = float(model.predict_proba(scaler.transform(X))[0][1])
    if not (0.0 <= score <= 1.0):
        print(f"[FAIL] Invalid score: {score}")
        return False
    print(f"[PASS] Model smoke (score={score:.4f})")
    return True


def run_api_smoke() -> bool:
    import requests

    base = "http://127.0.0.1:5000"
    try:
        health = requests.get(f"{base}/health", timeout=5)
        if health.status_code != 200:
            print(f"[FAIL] /health status={health.status_code}")
            return False
        resp = requests.post(
            f"{base}/analyze",
            json={"url": "http://192.168.0.1/secure-login"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"[FAIL] /analyze status={resp.status_code}")
            return False
        payload = resp.json()
        required = ("verdict", "score", "reasons", "rule_alerts", "scan_id")
        missing = [k for k in required if k not in payload]
        if missing:
            print(f"[FAIL] /analyze missing fields: {missing}")
            return False
    except Exception as exc:
        print(f"[FAIL] API smoke failed: {exc}")
        print("       Start API first with: python src/api/app.py")
        return False
    print("[PASS] API smoke")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Skip live API checks")
    args = parser.parse_args()

    ok = True
    ok &= run_features_smoke()
    ok &= run_model_smoke()
    if not args.offline:
        ok &= run_api_smoke()

    print("\nDone.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
