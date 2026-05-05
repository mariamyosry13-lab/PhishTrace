"""
PhishTrace — Unified Feature Extractor
========================================
Wraps extract.py and adds 3 extra features not present in the base extractor:

  url_entropy       — Shannon entropy of URL characters
                      (phishing URLs tend to have higher entropy due to
                       random-looking strings and hex encoding)

  brand_impersonation — 1 if a known brand name appears in a subdomain
                        rather than the root domain
                        (e.g. paypal.evil.com → PayPal in subdomain = suspicious)

  path_depth        — number of '/' segments in the URL path
                      (phishing pages often bury the form deep in the path)

Note: tld_suspicious is computed by extract.py and inherited here unchanged.

Usage
-----
from features.unified_extractor import extract_all, ALL_FEATURE_COLS

feats = extract_all("http://bankofegypt-login.example.com/confirm/form")
# → dict with all keys from extract_features plus 4 unified extras (see ALL_FEATURE_COLS)

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
import tldextract
from urllib.parse import urlparse
from pathlib import Path
import sys

# ── Import base extractor ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features.extract import extract_features, SUSPICIOUS_WORDS, BRAND_NAMES  # noqa: E402

# Backward-compatible name; same object as features.extract.KNOWN_BRANDS / BRAND_NAMES.
KNOWN_BRANDS = BRAND_NAMES

# ── Extra feature config ──────────────────────────────────────────────────────

# 19 ML features (must match FEATURE_COLS in app.py / train.py)
BASE_FEATURE_COLS = [
    "url_length","num_dots","num_hyphens","num_underscores","num_slashes",
    "num_at","num_question","num_equals","num_percent",
    "num_digits_in_domain","num_digits_in_path","last_path_segment_is_integer",
    "has_ip","has_https","num_subdomains",
    "hostname_length","path_length","double_slash",
    "num_suspicious_words",
]

# Rule-boost / display fields from extract_features() (used by app.rule_based_boost)
RULE_BOOST_FEATURE_COLS = [
    "has_at_in_url",
    "min_levenshtein",
    "is_typosquat",
    "hostname_entropy",
    "digit_ratio_hostname",
    "query_length",
    "num_params",
    "has_port",
    "tld_suspicious",
    "brand_in_subdomain",
]

EXTRA_FEATURE_COLS = [
    "url_entropy",
    "brand_impersonation",
    "path_depth",
]

ALL_FEATURE_COLS = (
    BASE_FEATURE_COLS + RULE_BOOST_FEATURE_COLS + EXTRA_FEATURE_COLS
)


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


def check_brand_impersonation(url: str) -> int:
    """
    Return 1 if a known brand appears in the hostname but NOT as the
    registrable domain label (eTLD+1 domain part).

    Uses the public suffix list (tldextract) so legitimate ccTLD and
    multi-part suffix registrations are not flagged:
      paypal.com, google.de, amazon.co.jp, paypal.co.uk, amazon.com.au → 0
      paypal.evil.com, secure-paypal.net → 1
    """
    try:
        host = urlparse(url).hostname or ""
        if not host:
            return 0
        if re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host):
            return 0
        ext = tldextract.extract(host.rstrip("."))
        if not ext.suffix:
            return 0
        registered = ext.domain.lower()
        full_lower = host.lower()

        for brand in BRAND_NAMES:
            if brand not in full_lower:
                continue
            if registered == brand:
                return 0
            return 1
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
    Extract all features for a given URL.

    The 19 base features are used by the trained model.
    Rule-boost fields and unified extras are included for API display and reports.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict with keys from ALL_FEATURE_COLS plus any other extract_features keys
    """
    base   = extract_features(url)
    extras = {
        "url_entropy"        : calc_entropy(url),
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
        "url_length"                 : "Length of the full URL",
        "num_dots"                   : "Number of dots (subdomains / path separators)",
        "num_hyphens"                : "Number of hyphens in the URL",
        "num_underscores"            : "Number of underscores",
        "num_slashes"                : "Number of forward slashes",
        "num_at"                     : "Presence of @ symbol",
        "num_question"               : "Number of query string markers (?)",
        "num_equals"                 : "Number of parameter assignments (=)",
        "num_percent"                : "URL-encoded characters (%xx)",
        "num_digits_in_domain"       : "Number of digits in the hostname (phishing signal)",
        "num_digits_in_path"         : "Number of digits in the path/query (often legitimate IDs)",
        "last_path_segment_is_integer": "Last path segment is a pure integer (e.g. /questions/11227809)",
        "has_ip"                     : "URL uses raw IP instead of domain name",
        "has_https"                  : "URL uses HTTPS",
        "num_subdomains"             : "Number of subdomain levels",
        "hostname_length"            : "Length of the hostname",
        "path_length"                : "Length of the URL path",
        "double_slash"               : "Double slash in path (// — possible open redirect)",
        "has_at_in_url"              : "@ symbol present in URL",
        "num_suspicious_words"       : f"Count of suspicious words (e.g. {', '.join(SUSPICIOUS_WORDS[:3])}...)",
        "min_levenshtein"            : "Minimum Levenshtein distance from hostname to a known brand (typosquat signal)",
        "is_typosquat"               : "Hostname closely resembles a known brand name",
        "hostname_entropy"           : "Shannon entropy of the hostname (randomness)",
        "digit_ratio_hostname"       : "Fraction of hostname characters that are digits",
        "query_length"               : "Length of the query string",
        "num_params"                 : "Number of query parameters",
        "has_port"                   : "URL specifies a non-default port",
        "brand_in_subdomain"         : "Known brand string appears in subdomain labels (extract.py heuristic)",
        "url_entropy"                : "Shannon entropy (randomness) of URL characters",
        "tld_suspicious"             : "Top-level domain is in the high-risk list",
        "brand_impersonation"        : "Known brand name appears in subdomain (not root domain)",
        "path_depth"                 : "Depth of the URL path (number of segments)",
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