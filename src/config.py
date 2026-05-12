from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "phishtrace.db"

THRESHOLD_DANGEROUS = 0.75
THRESHOLD_SUSPICIOUS = 0.45
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ML model inputs (19 columns) — must match phishtrace_features.csv and scaler.pkl
FEATURE_COLS = [
    "url_length", "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_at", "num_question", "num_equals", "num_percent",
    "num_digits_in_domain", "num_digits_in_path", "last_path_segment_is_integer",
    "has_ip", "has_https", "num_subdomains",
    "hostname_length", "path_length", "double_slash",
    "num_suspicious_words",
]