import sys
import os
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODELS_DIR = ROOT / "models"


@unittest.skipUnless((MODELS_DIR / "best_model.pkl").exists(), "best_model.pkl not found")
class TestFlaskApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["PHISHTRACE_DB"] = cls._db_path
        from api.app import app
        app.config["TESTING"] = True
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls._db_path):
            os.unlink(cls._db_path)

    def test_health_endpoint(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_index_returns_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"html", r.data.lower())

    def test_analyze_safe_url(self):
        r = self.client.post("/analyze", json={"url": "https://www.google.com"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("verdict", data)
        self.assertIn("score", data)
        self.assertIn(data["verdict"], ("Safe", "Suspicious", "Dangerous"))

    def test_analyze_phishing_url(self):
        r = self.client.post("/analyze",
                             json={"url": "http://192.168.0.1/secure-login"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn(data["verdict"], ("Suspicious", "Dangerous"))

    def test_analyze_missing_url_returns_400(self):
        r = self.client.post("/analyze", json={})
        self.assertEqual(r.status_code, 400)

    def test_analyze_returns_reasons(self):
        r = self.client.post("/analyze", json={"url": "https://www.github.com"})
        data = r.get_json()
        self.assertIn("reasons", data)
        self.assertIsInstance(data["reasons"], list)

    def test_history_endpoint(self):
        self.client.post("/analyze", json={"url": "https://www.example.com"})
        r = self.client.get("/api/history")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("scans", data)

    def test_dashboard_endpoint(self):
        r = self.client.get("/api/dashboard")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        for key in ("total_scans", "dangerous", "suspicious", "safe"):
            self.assertIn(key, data)

    def test_campaigns_endpoint(self):
        r = self.client.get("/api/campaigns")
        self.assertEqual(r.status_code, 200)
        self.assertIn("campaigns", r.get_json())


if __name__ == "__main__":
    unittest.main()
