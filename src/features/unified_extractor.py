"""
PhishTrace — Unified Feature Extractor
========================================
Wraps extract.py and adds 4 extra features that improve model performance:

  url_entropy       — Shannon entropy of URL characters
                      (phishing URLs tend to have higher entropy due to
                       random-looking strings and hex encoding)

  tld_suspicious    — 1 if TLD is associated with cheap/abused registrations
                      (.xyz, .top, .click, .tk, .ml, .ga, .cf, .gq, .pw, ...)

  brand_impersonation — 1 if a known brand name appears in a subdomain
                        rather than the root domain
                        (e.g. paypal.evil.com → PayPal in subdomain = suspicious)

  path_depth        — number of '/' segments in the URL path
                      (phishing pages often bury the form deep in the path)

Usage
-----
from features.unified_extractor import extract_all, ALL_FEATURE_COLS

feats = extract_all("http://bankofegypt-login.example.com/confirm/form")
# → dict with 23 features (19 base + 4 extra)

Notes
-----
- The 4 extra features are NOT used by the trained model (scaler + best_model.pkl)
  which expects the original 19-column FEATURE_COLS.
- To use them you must retrain after adding them to the feature matrix.
- They ARE included in the /analyze API response under "features" for display only.
- unified_extractor.extract_all() is the single entry-point for the API.
"""

import re
import math
from urllib.parse import urlparse
from pathlib import Path
import sys

# ── Import base extractor ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features.extract import extract_features, SUSPICIOUS_WORDS   # noqa: E402

# ── Extra feature config ──────────────────────────────────────────────────────
SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "tk", "ml", "ga", "cf", "gq",
    "pw", "cc", "ws", "biz", "info", "su", "icu", "rest",
    "online", "site", "store", "fun", "live", "space",
}

KNOWN_BRANDS = {
    "google", "facebook", "paypal", "amazon", "apple", "microsoft",
    "bankofegypt", "cib", "nbe", "hsbc", "vodafone", "etisalat",
    "instagram", "twitter", "whatsapp", "netflix", "ebay", "dhl",
    "fedex", "ups", "usps", "linkedin", "dropbox", "icloud",
}

# 19 original + 4 extra
BASE_FEATURE_COLS = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
    "num_at","num_question","num_equals","num_percent","num_digits",
    "has_ip","has_https","has_suspicious_word","num_subdomains",
    "hostname_length","path_length","double_slash","has_at_in_url",
    "num_suspicious_words",
]

EXTRA_FEATURE_COLS = [
    "url_entropy",
    "tld_suspicious",
    "brand_impersonation",
    "path_depth",
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + EXTRA_FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
#  Extra feature functions
# ═══════════════════════════════════════════════════════════════════════════════
def calc_entropy(url: str) -> float:
    """
    Shannon entropy of characters in the URL.
    Higher entropy → more random-looking → more likely phishing.

    Max theoretical entropy for printable ASCII ≈ 6.57 bits.
    Typical legitimate URL: 3–4 bits.
    Typical phishing URL:   4–5.5 bits.
    """
    if not url:
        return 0.0
    freq = {}
    for c in url:
        freq[c] = freq.get(c, 0) + 1
    n = len(url)
    return round(-sum((cnt/n) * math.log2(cnt/n) for cnt in freq.values()), 4)


def check_tld_suspicious(url: str) -> int:
    """Return 1 if the TLD is in the suspicious list."""
    try:
        host  = urlparse(url).hostname or ""
        parts = host.rstrip(".").split(".")
        tld   = parts[-1].lower() if parts else ""
        return int(tld in SUSPICIOUS_TLDS)
    except Exception:
        return 0


def check_brand_impersonation(url: str) -> int:
    """
    Return 1 if a known brand appears in a subdomain but NOT in the
    registered domain (second-level domain).

    Example:
      paypal.evil-login.com  → brand 'paypal' in subdomain → 1
      paypal.com             → brand 'paypal' IS the domain → 0
      evil-paypal-login.net  → brand in full URL but as part of hostname → 1
    """
    try:
        host   = urlparse(url).hostname or ""
        parts  = host.rstrip(".").split(".")
        if len(parts) < 2:
            return 0
        # Registered domain = last two parts (e.g. evil.com)
        reg_domain = ".".join(parts[-2:]).lower()
        full_lower = host.lower()

        for brand in KNOWN_BRANDS:
            if brand in full_lower:
                # Brand in URL — check if it's actually the legitimate domain
                if full_lower == f"{brand}.com" or full_lower.endswith(f".{brand}.com"):
                    return 0   # legitimate domain
                return 1       # brand in subdomain/hostname = impersonation
        return 0
    except Exception:
        return 0


def calc_path_depth(url: str) -> int:
    """
    Number of non-empty path segments.
    /confirm/form/login → depth 3.
    Phishing pages often bury the form deep (depth > 4).
    """
    try:
        path   = urlparse(url).path or ""
        segs   = [s for s in path.split("/") if s]
        return len(segs)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Main public function
# ═══════════════════════════════════════════════════════════════════════════════
def extract_all(url: str) -> dict:
    """
    Extract all 23 features for a given URL.

    The 19 base features are used by the trained model.
    The 4 extra features are displayed in the API response for user insight.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict with keys from ALL_FEATURE_COLS (23 total)
    """
    base   = extract_features(url)
    extras = {
        "url_entropy"        : calc_entropy(url),
        "tld_suspicious"     : check_tld_suspicious(url),
        "brand_impersonation": check_brand_impersonation(url),
        "path_depth"         : calc_path_depth(url),
    }
    return {**base, **extras}


def get_feature_report(url: str) -> list[dict]:
    """
    Return a structured report of all features with values and human context.
    Useful for display in the frontend or for debugging.
    """
    feats = extract_all(url)

    explanations = {
        "url_length"          : "Length of the full URL",
        "num_dots"            : "Number of dots (subdomains / path separators)",
        "num_hyphens"         : "Number of hyphens in the URL",
        "num_underscores"     : "Number of underscores",
        "num_slashes"         : "Number of forward slashes",
        "num_at"              : "Presence of @ symbol",
        "num_question"        : "Number of query string markers (?)",
        "num_equals"          : "Number of parameter assignments (=)",
        "num_percent"         : "URL-encoded characters (%xx)",
        "num_digits"          : "Number of digits in the URL",
        "has_ip"              : "URL uses raw IP instead of domain name",
        "has_https"           : "URL uses HTTPS",
        "has_suspicious_word" : f"Contains suspicious word (e.g. {', '.join(SUSPICIOUS_WORDS[:3])}...)",
        "num_subdomains"      : "Number of subdomain levels",
        "hostname_length"     : "Length of the hostname",
        "path_length"         : "Length of the URL path",
        "double_slash"        : "Double slash in path (// — possible open redirect)",
        "has_at_in_url"       : "@ symbol present in URL",
        "num_suspicious_words": "Count of suspicious words in URL",
        "url_entropy"         : "Shannon entropy (randomness) of URL characters",
        "tld_suspicious"      : "Top-level domain is in the high-risk list",
        "brand_impersonation" : "Known brand name appears in subdomain (not root domain)",
        "path_depth"          : "Depth of the URL path (number of segments)",
    }

    report = []
    for feat in ALL_FEATURE_COLS:
        val = feats.get(feat, 0)
        report.append({
            "feature"    : feat,
            "value"      : val,
            "explanation": explanations.get(feat, ""),
            "is_extra"   : feat in EXTRA_FEATURE_COLS,
        })
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI demo
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    test_urls = [
        "https://bankofegypt-login.example.com/confirm/form",
        "http://192.168.0.45/secure-login",
        "https://www.google.com",
        "http://paypal-verify.xyz/account/update?id=12345",
    ]

    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        report = get_feature_report(url)
        for item in report:
            flag = " [EXTRA]" if item["is_extra"] else ""
            val  = item["value"]
            mark = " ⚠" if (isinstance(val, int) and val == 1
                             and item["feature"] not in ("has_https",)) else ""
            print(f"  {item['feature']:<25} = {val}{mark}{flag}")