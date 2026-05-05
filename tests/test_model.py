import sys
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import FEATURE_COLS

MODELS_DIR = ROOT / "models"


@unittest.skipUnless((MODELS_DIR / "best_model.pkl").exists(), "best_model.pkl not found")
class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import joblib
        import pandas as pd
        cls.joblib = joblib
        cls.pd = pd
        cls.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        cls.model = joblib.load(MODELS_DIR / "best_model.pkl")

    def _score(self, url):
        from features.extract import extract_features
        feats = extract_features(url)
        X = self.pd.DataFrame([feats])[FEATURE_COLS]
        return float(self.model.predict_proba(self.scaler.transform(X))[0][1])

    def test_model_loads(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.scaler)

    def test_model_has_expected_estimators(self):
        n = getattr(self.model, "n_estimators", None)
        if n is not None:
            self.assertGreaterEqual(n, 100, "Model appears to be a dummy/toy model")

    def test_phishing_url_scores_higher_than_safe(self):
        phish_score = self._score("http://192.168.0.1/login/verify?token=abc")
        safe_score = self._score("https://www.google.com")
        self.assertGreater(phish_score, safe_score)

    def test_predict_proba_returns_valid_probability(self):
        score = self._score("https://www.github.com")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
