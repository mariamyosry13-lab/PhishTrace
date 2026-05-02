from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
DB_PATH = BASE_DIR / "data" / "phishtrace.db"

# Model settings
THRESHOLD_DANGEROUS = 0.75
THRESHOLD_SUSPICIOUS = 0.45
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5