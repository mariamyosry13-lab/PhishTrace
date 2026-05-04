import gc
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from database import db as dbmod


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["PHISHTRACE_DB"] = cls._db_path
        dbmod.DB_PATH = cls._db_path

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        if not os.path.exists(cls._db_path):
            return
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

    def test_init_db_creates_tables(self):
        dbmod.init_db()
        with dbmod.get_conn() as conn:
            scans = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='scans'"
            ).fetchone()
            campaigns = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='campaigns'"
            ).fetchone()
        self.assertIsNotNone(scans)
        self.assertIsNotNone(campaigns)

    def test_save_scan_and_get_scan(self):
        dbmod.init_db()
        scan_id = dbmod.save_scan(
            url="http://unit-test.example/login",
            verdict="Dangerous",
            score=0.9123,
            raw_score=0.8999,
            rule_alerts=["has_ip"],
            shap_reasons=[{"feature": "has_ip", "contribution": 0.3, "direction": "increases"}],
            features={"has_ip": 1, "num_suspicious_words": 2},
            campaign_id="campaign_007",
        )
        self.assertIsInstance(scan_id, int)
        self.assertGreater(scan_id, 0)

        row = dbmod.get_scan(scan_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["id"], scan_id)
        self.assertEqual(row["url"], "http://unit-test.example/login")
        self.assertEqual(row["verdict"], "Dangerous")
        self.assertEqual(row["campaign_id"], "campaign_007")
        self.assertIsInstance(row["rule_alerts"], list)
        self.assertIsInstance(row["shap_reasons"], list)

    def test_get_history_contains_saved_scan(self):
        dbmod.init_db()
        scan_id = dbmod.save_scan(
            url="https://history-test.example",
            verdict="Safe",
            score=0.1234,
            raw_score=0.1234,
            features={"has_ip": 0},
        )
        history = dbmod.get_history(limit=10)
        self.assertIsInstance(history, list)
        target = next((x for x in history if x.get("scan_id") == scan_id), None)
        self.assertIsNotNone(target)
        for key in (
            "scan_id",
            "url",
            "verdict",
            "score",
            "raw_score",
            "rule_alerts",
            "shap_reasons",
            "campaign_id",
            "scanned_at",
        ):
            self.assertIn(key, target)

    def test_get_stats_returns_expected_keys(self):
        dbmod.init_db()
        stats = dbmod.get_stats()
        for key in (
            "total_scans",
            "dangerous",
            "suspicious",
            "safe",
            "timeline",
            "model_metrics",
        ):
            self.assertIn(key, stats)
        self.assertIsInstance(stats["model_metrics"], dict)


if __name__ == "__main__":
    unittest.main()
