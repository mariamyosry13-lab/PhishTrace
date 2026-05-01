import re
import math
import pandas as pd
from urllib.parse import urlparse

# ── Suspicious words ────────────────────────────────────
SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "account", "update",
    "banking", "confirm", "password", "signin", "webscr",
    "paypal", "ebay", "amazon", "apple", "microsoft",
    "google", "facebook", "netflix", "support", "alert",
    "suspended", "limited", "unusual", "validate", "billing"
]

# ── Popular legit domains for typosquatting detection ──
POPULAR_DOMAINS = [
    "google", "facebook", "amazon", "paypal", "apple",
    "microsoft", "netflix", "instagram", "twitter", "linkedin",
    "yahoo", "gmail", "outlook", "bankofamerica", "chase",
    "wellsfargo", "ebay", "dropbox", "spotify", "reddit"
]

# ── Levenshtein distance ────────────────────────────────
def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1,
                            curr[j] + 1,
                            prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

def min_levenshtein_to_popular(hostname: str) -> int:
    """أقل مسافة تعديل بين الـ hostname وأي دومين مشهور"""
    # نشيل الـ TLD (.com / .net / إلخ)
    base = hostname.split(".")[0] if "." in hostname else hostname
    if not base:
        return 99
    return min(levenshtein(base, pop) for pop in POPULAR_DOMAINS)

# ── Shannon entropy ──────────────────────────────────────
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())

# ── Digit ratio ──────────────────────────────────────────
def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)

# ── Main extractor ───────────────────────────────────────
def extract_features(url: str) -> dict:
    try:
        parsed   = urlparse(url if url.startswith("http") else "http://" + url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        full     = url.lower()
        query    = parsed.query or ""
    except Exception:
        return {f: 0 for f in feature_names()}

    min_lev = min_levenshtein_to_popular(hostname)

    return {
        # ── الميزات الأصلية (نفس الأسماء) ──────────────
        "url_length"           : len(url),
        "num_dots"             : url.count("."),
        "num_hyphens"          : url.count("-"),
        "num_underscores"      : url.count("_"),
        "num_slashes"          : url.count("/"),
        "num_at"               : url.count("@"),
        "num_question"         : url.count("?"),
        "num_equals"           : url.count("="),
        "num_percent"          : url.count("%"),
        "num_digits"           : sum(c.isdigit() for c in url),
        "has_ip"               : int(bool(re.match(
                                     r"http[s]?://\d+\.\d+\.\d+\.\d+", url))),
        "has_https"            : int(parsed.scheme == "https"),
        "has_suspicious_word"  : int(any(w in full for w in SUSPICIOUS_WORDS)),
        "num_subdomains"       : max(len(hostname.split(".")) - 2, 0),
        "hostname_length"      : len(hostname),
        "path_length"          : len(path),
        "double_slash"         : int("//" in path),
        "has_at_in_url"        : int("@" in url),
        "num_suspicious_words" : sum(w in full for w in SUSPICIOUS_WORDS),

        # ── ميزات جديدة ─────────────────────────────────
        "min_levenshtein"      : min_lev,           # typosquatting
        "is_typosquat"         : int(1 <= min_lev <= 2),  # قريب جداً من دومين مشهور
        "hostname_entropy"     : round(shannon_entropy(hostname), 4),
        "digit_ratio_hostname" : round(digit_ratio(hostname), 4),
        "query_length"         : len(query),
        "num_params"           : len(query.split("&")) if query else 0,
        "has_port"             : int(bool(parsed.port)),
        "tld_suspicious"       : int(any(
                                     hostname.endswith(t)
                                     for t in [".tk", ".ml", ".ga", ".cf", ".gq",
                                               ".xyz", ".top", ".click", ".link"]
                                 )),
        "brand_in_subdomain"   : int(any(
                                     pop in hostname.split(".")[0]
                                     for pop in POPULAR_DOMAINS
                                     if len(hostname.split(".")) > 2
                                 )),
    }

def feature_names():
    return list(extract_features("http://example.com").keys())

def build_feature_matrix(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    print(f"Processing {len(df)} URLs...")
    features = df["url"].apply(extract_features).apply(pd.Series)
    features["label"] = df["label"].values
    features["url"]   = df["url"].values
    features.to_csv(out_path, index=False)
    print(f"Saved to {out_path} — shape: {features.shape}")
    return features

if __name__ == "__main__":
    build_feature_matrix(
        "data/processed/phishtrace_dataset.csv",
        "data/processed/phishtrace_features.csv"
    )