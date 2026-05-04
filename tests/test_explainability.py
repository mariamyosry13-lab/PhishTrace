import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"
FEATURE_COLS = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at",
    "num_question",
    "num_equals",
    "num_percent",
    "num_digits_in_domain",
    "num_digits_in_path",
    "last_path_segment_is_integer",
    "has_ip",
    "has_https",
    "num_subdomains",
    "hostname_length",
    "path_length",
    "double_slash",
    "num_suspicious_words",
]


@unittest.skipUnless((MODELS_DIR / "best_model.pkl").exists(), "best_model.pkl not found")
@unittest.skipUnless((MODELS_DIR / "scaler.pkl").exists(), "scaler.pkl not found")
class TestExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import shap
            import joblib
            import pandas as pd
        except Exception as exc:
            raise unittest.SkipTest(f"Missing explainability deps: {exc}")

        from features.extract import extract_features

        cls.shap = shap
        cls.joblib = joblib
        cls.pd = pd
        cls._extract_features = staticmethod(extract_features)
        cls.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        cls.model = joblib.load(MODELS_DIR / "best_model.pkl")

    def test_shap_smoke(self):
        feats = self._extract_features("http://bankofegypt-login.evil.xyz/confirm")
        X = self.pd.DataFrame([feats])[FEATURE_COLS]
        X_scaled = self.scaler.transform(X)

        explainer = self.shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0])
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                class_idx = 1 if arr.shape[-1] >= 2 else 0
                sv = arr[0, :, class_idx]
            else:
                sv = arr[0]

        self.assertEqual(len(sv), len(FEATURE_COLS))
        top_idx = np.argsort(np.abs(sv))[::-1][:5]
        self.assertGreater(len(top_idx), 0)
        top_feats = [FEATURE_COLS[i] for i in top_idx]
        self.assertTrue(all(isinstance(name, str) and name for name in top_feats))


if __name__ == "__main__":
    unittest.main()
