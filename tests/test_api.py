import sys
import os
import json
import tempfile
import unittest
import gc
import time
import sqlite3
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
        # Patch db module directly so isolation holds regardless of import order.
        # db.DB_PATH is a module-level constant read once at import; if database.db
        # was already imported by another test the env-var set above is ignored.
        import database.db as _db
        _db.DB_PATH = cls._db_path
        _db._model_metrics_cache = None  # reset cached metrics from any prior run
        _db.init_db()                    # ensure schema exists in this temp file
        app.config["TESTING"] = True
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        # Release references that may keep SQLite handles alive in tests.
        cls.client = None
        gc.collect()

        if not os.path.exists(cls._db_path):
            return

        # On Windows, unlink can fail briefly if SQLite handle teardown lags.
        # Retry after explicitly opening/closing a no-op connection to ensure
        # all connections are finalized before file deletion.
        for _ in range(10):
            try:
                os.unlink(cls._db_path)
                break
            except PermissionError:
                try:
                    conn = sqlite3.connect(cls._db_path)
                    conn.close()
                except sqlite3.Error:
                    pass
                gc.collect()
                time.sleep(0.05)

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
