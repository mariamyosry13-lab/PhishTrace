import re
import math
import pandas as pd
from urllib.parse import urlparse

# ── Suspicious words (phishing behavior only — NO brand names here) ─────────
# ✅ FIX: شيلنا brand names (google, amazon, paypal...) من الـ list دي
# كانوا بيخلوا روابط زي mail.google.com تتحسب suspicious
SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "account", "update",
    "banking", "confirm", "password", "signin", "webscr",
    "support", "alert", "suspended", "limited", "unusual",
    "validate", "billing", "authenticate", "credential",
    "urgent", "expire", "click-here", "free-prize"
]

# ── Brand names — for typosquatting detection ONLY (not used in ML features) ─
# ✅ FIX: Brand names منفصلة — بنستخدمهم بس في is_typosquat و brand_in_subdomain
BRAND_NAMES = [
    "google", "facebook", "amazon", "paypal", "apple",
    "microsoft", "netflix", "instagram", "twitter", "linkedin",
    "yahoo", "gmail", "outlook", "bankofamerica", "chase",
    "wellsfargo", "ebay", "dropbox", "spotify", "reddit"
]

# للـ Levenshtein نستخدم BRAND_NAMES بدل POPULAR_DOMAINS
POPULAR_DOMAINS = BRAND_NAMES  # alias للـ backward compatibility

# ── Levenshtein distance ─────────────────────────────────────────────────────
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
    """أقل مسافة تعديل بين الـ hostname وأي brand مشهور"""
    base = hostname.split(".")[0] if "." in hostname else hostname
    if not base:
        return 99
    return min(levenshtein(base, brand) for brand in BRAND_NAMES)

# ── Shannon entropy ───────────────────────────────────────────────────────────
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())

# ── Digit ratio ───────────────────────────────────────────────────────────────
def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)

# ── Main extractor ────────────────────────────────────────────────────────────
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

    # ✅ FIX: نحسب brand_in_subdomain بشكل صح
    # بنشوف لو الـ brand موجود في subdomain بس (مش الـ main domain)
    parts = hostname.split(".")
    subdomain_str = ".".join(parts[:-2]) if len(parts) > 2 else ""
    brand_in_sub = int(any(brand in subdomain_str for brand in BRAND_NAMES))

    return {
        # ── الميزات الأصلية ──────────────────────────────────────────────────
        "url_length"           : len(url),
        "num_dots"             : url.count("."),
        "num_hyphens"          : url.count("-"),
        "num_underscores"      : url.count("_"),
        "num_slashes"          : url.count("/"),
        "num_at"               : url.count("@"),
        "num_question"         : url.count("?"),
        "num_equals"           : url.count("="),
        "num_percent"          : url.count("%"),
        # digits split by location: domain digits = phishing signal,
        # path digits = often legitimate numeric IDs
        "num_digits_in_domain" : sum(c.isdigit() for c in hostname),
        "num_digits_in_path"   : sum(c.isdigit() for c in (path + query)),
        # 1 when the last path segment is a pure integer e.g. /questions/11227809
        "last_path_segment_is_integer": int(
            bool([s for s in path.split("/") if s]) and
            [s for s in path.split("/") if s][-1].isdigit()
        ),
        "has_ip"               : int(bool(re.match(
                                     r"http[s]?://\d+\.\d+\.\d+\.\d+", url))),
        "has_https"            : int(parsed.scheme == "https"),

        # ✅ FIX: بس num_suspicious_words بيبقى في الـ ML features
        # شيلنا has_suspicious_word عشان كان redundant مع num_suspicious_words
        "num_subdomains"       : max(len(hostname.split(".")) - 2, 0),
        "hostname_length"      : len(hostname),
        "path_length"          : len(path),
        "double_slash"         : int("//" in path),
        "has_at_in_url"        : int("@" in url),

        # ✅ FIX: SUSPICIOUS_WORDS دلوقتي بدون brand names
        # فـ mail.google.com مش هتحسب suspicious من الكلمات دي
        "num_suspicious_words" : sum(w in full for w in SUSPICIOUS_WORDS),

        # ── ميزات extra (للـ rule_based_boost بس — مش للـ ML) ───────────────
        "min_levenshtein"      : min_lev,
        "is_typosquat"         : int(1 <= min_lev <= 2),
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
        "brand_in_subdomain"   : brand_in_sub,  # ✅ fixed logic above
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