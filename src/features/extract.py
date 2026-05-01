import re
import pandas as pd
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "account", "update",
    "banking", "confirm", "password", "signin", "webscr"
]

def extract_features(url: str) -> dict:
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        full     = url.lower()
    except Exception:
        return {f: 0 for f in feature_names()}

    return {
        "url_length"          : len(url),
        "num_dots"            : url.count("."),
        "num_hyphens"         : url.count("-"),
        "num_underscores"     : url.count("_"),
        "num_slashes"         : url.count("/"),
        "num_at"              : url.count("@"),
        "num_question"        : url.count("?"),
        "num_equals"          : url.count("="),
        "num_percent"         : url.count("%"),
        "num_digits"          : sum(c.isdigit() for c in url),
        "has_ip"              : int(bool(re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url))),
        "has_https"           : int(parsed.scheme == "https"),
        "has_suspicious_word" : int(any(w in full for w in SUSPICIOUS_WORDS)),
        "num_subdomains"      : max(len(hostname.split(".")) - 2, 0),
        "hostname_length"     : len(hostname),
        "path_length"         : len(path),
        "double_slash"        : int("//" in path),
        "has_at_in_url"       : int("@" in url),
        "num_suspicious_words": sum(w in full for w in SUSPICIOUS_WORDS),
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