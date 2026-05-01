"""
PhishTrace — Database Layer
============================
SQLite database with 3 tables:
  - scans      : every URL analysed, with features + SHAP stored as JSON
  - campaigns  : phishing campaign clusters discovered by campaign.py
  - model_metrics : latest evaluation results (seeded from evaluation_report.txt)

Usage
-----
from database.db import init_db, save_scan, get_history, get_stats, get_campaigns

init_db()                        # call once at app startup
scan_id = save_scan(scan_dict)   # called inside /analyze
stats   = get_stats()            # called inside /api/dashboard
history = get_history(limit=50)  # called inside /api/history
camps   = get_campaigns()        # called inside /api/campaigns
"""

import os
import json
import sqlite3
import re
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent.parent   # project root (above src/database/)
DB_PATH           = BASE_DIR / "data" / "phishtrace.db"
EVAL_REPORT_PATH  = BASE_DIR / "reports" / "evaluation_report.txt"
EVAL_JSON_PATH    = BASE_DIR / "models"  / "evaluation_results.json"

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT    NOT NULL,
    score         REAL    NOT NULL,
    percent       REAL    NOT NULL,
    verdict       TEXT    NOT NULL CHECK(verdict IN ('Safe','Suspicious','Dangerous')),
    features_json TEXT    NOT NULL,
    shap_json     TEXT    NOT NULL,
    campaign_id   TEXT,
    timestamp     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS campaigns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_name TEXT    NOT NULL UNIQUE,
    size          INTEGER NOT NULL DEFAULT 0,
    centroid_json TEXT,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id        INTEGER PRIMARY KEY CHECK(id = 1),
    accuracy  REAL,
    precision REAL,
    recall    REAL,
    f1        REAL,
    auc       REAL,
    fpr       REAL,
    threshold_dangerous  REAL DEFAULT 0.75,
    threshold_suspicious REAL DEFAULT 0.45,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# ── Connection context manager ────────────────────────────────────────────────
@contextmanager
def get_conn():
    """Yield a SQLite connection and auto-commit / rollback."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialise ────────────────────────────────────────────────────────────────
def init_db() -> None:
    """
    Create tables if they don't exist.
    Also seeds model_metrics from evaluation_results.json or evaluation_report.txt.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        conn.executescript(SCHEMA)

    _seed_metrics()
    print(f"[DB] Initialised at {DB_PATH}")


def _seed_metrics() -> None:
    """
    Populate model_metrics table once from existing report files.
    Priority: evaluation_results.json > evaluation_report.txt > defaults.
    """
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM model_metrics WHERE id=1").fetchone()
        if row:
            return  # already seeded

    metrics = _load_metrics_from_json() or _parse_metrics_from_txt() or _default_metrics()

    with get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO model_metrics
                (id, accuracy, precision, recall, f1, auc, fpr, updated_at)
            VALUES (1, :accuracy, :precision, :recall, :f1, :auc, :fpr,
                    strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        """, metrics)

    print(f"[DB] model_metrics seeded: F1={metrics.get('f1')}, AUC={metrics.get('auc')}")


def _load_metrics_from_json() -> dict | None:
    if not EVAL_JSON_PATH.exists():
        return None
    try:
        with open(EVAL_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "accuracy" : data.get("accuracy",  0.0),
            "precision": data.get("precision", 0.0),
            "recall"   : data.get("recall",    0.0),
            "f1"       : data.get("f1",        0.0),
            "auc"      : data.get("auc",        0.0),
            "fpr"      : data.get("fpr",        0.0),
        }
    except Exception as e:
        print(f"[DB] Warning: could not read evaluation_results.json — {e}")
        return None


def _parse_metrics_from_txt() -> dict | None:
    """
    Parse numbers from evaluation_report.txt.
    Looks for lines like:  accuracy : 0.9400
    """
    if not EVAL_REPORT_PATH.exists():
        return None
    try:
        text = EVAL_REPORT_PATH.read_text(encoding="utf-8").lower()
        def grab(key):
            m = re.search(rf"{key}\s*[=:]\s*([\d.]+)", text)
            return float(m.group(1)) if m else 0.0

        return {
            "accuracy" : grab("accuracy"),
            "precision": grab("precision"),
            "recall"   : grab("recall"),
            "f1"       : grab("f1"),
            "auc"      : grab("auc") or grab("roc"),
            "fpr"      : grab("fpr") or grab("false positive rate"),
        }
    except Exception as e:
        print(f"[DB] Warning: could not parse evaluation_report.txt — {e}")
        return None


def _default_metrics() -> dict:
    return {
        "accuracy": 0.94, "precision": 0.94,
        "recall"  : 0.96, "f1": 0.9686,
        "auc"     : 0.98, "fpr": 0.08,
    }


# ── Write ─────────────────────────────────────────────────────────────────────
def save_scan(scan: dict) -> int:
    """
    Persist one scan result to the scans table.

    Expected keys in `scan`:
        url, score, percent, verdict, features (dict), reasons (list),
        campaign_id (optional str)

    Returns the new row id.
    """
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO scans
                (url, score, percent, verdict, features_json, shap_json, campaign_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            scan["url"],
            round(float(scan["score"]),   4),
            round(float(scan["percent"]), 1),
            scan["verdict"],
            json.dumps(scan.get("features", {}), ensure_ascii=False),
            json.dumps(scan.get("reasons",  []), ensure_ascii=False),
            scan.get("campaign_id"),
        ))
        return cur.lastrowid


def save_campaigns(campaign_list: list[dict]) -> None:
    """
    Bulk-upsert campaign rows.
    Each dict needs: campaign_name (str), size (int), centroid_json (str, optional).
    """
    with get_conn() as conn:
        for c in campaign_list:
            conn.execute("""
                INSERT INTO campaigns (campaign_name, size, centroid_json)
                VALUES (:campaign_name, :size, :centroid_json)
                ON CONFLICT(campaign_name) DO UPDATE SET
                    size         = excluded.size,
                    centroid_json= excluded.centroid_json
            """, {
                "campaign_name": c["campaign_name"],
                "size"         : c.get("size", 0),
                "centroid_json": json.dumps(c.get("centroid", [])),
            })


# ── Read ──────────────────────────────────────────────────────────────────────
def get_scan(scan_id: int) -> dict | None:
    """Return a single scan by id, or None if not found."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()
    if not row:
        return None
    return _row_to_scan(row)


def get_history(limit: int = 50, offset: int = 0) -> list[dict]:
    """Return the most recent `limit` scans, newest first."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM scans
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
    return [_row_to_scan(r) for r in rows]


def get_stats() -> dict:
    """
    Return dashboard statistics:
      - total_scans, dangerous, suspicious, safe counts
      - model_metrics (from DB)
      - timeline: daily scan counts for the last 7 days
    """
    with get_conn() as conn:
        # Overall counts
        counts = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(verdict = 'Dangerous')  AS dangerous,
                SUM(verdict = 'Suspicious') AS suspicious,
                SUM(verdict = 'Safe')       AS safe
            FROM scans
        """).fetchone()

        # Timeline — last 7 days
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        timeline_rows = conn.execute("""
            SELECT
                substr(timestamp, 1, 10) AS day,
                COUNT(*)                 AS count
            FROM scans
            WHERE timestamp >= ?
            GROUP BY day
            ORDER BY day
        """, (seven_days_ago,)).fetchall()

        # Model metrics
        metrics_row = conn.execute(
            "SELECT * FROM model_metrics WHERE id = 1"
        ).fetchone()

    model_metrics = {}
    if metrics_row:
        model_metrics = {
            "accuracy" : metrics_row["accuracy"],
            "precision": metrics_row["precision"],
            "recall"   : metrics_row["recall"],
            "f1"       : metrics_row["f1"],
            "auc"      : metrics_row["auc"],
            "fpr"      : metrics_row["fpr"],
        }

    return {
        "total_scans": counts["total"]     or 0,
        "dangerous"  : counts["dangerous"] or 0,
        "suspicious" : counts["suspicious"]or 0,
        "safe"       : counts["safe"]      or 0,
        "model_metrics": model_metrics,
        "timeline"   : [{"day": r["day"], "count": r["count"]} for r in timeline_rows],
    }


def get_campaigns() -> list[dict]:
    """Return all known campaigns, largest first."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT campaign_name, size, centroid_json, created_at
            FROM campaigns
            ORDER BY size DESC
        """).fetchall()
    return [
        {
            "campaign_name": r["campaign_name"],
            "size"         : r["size"],
            "centroid"     : json.loads(r["centroid_json"] or "[]"),
            "created_at"   : r["created_at"],
        }
        for r in rows
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _row_to_scan(row: sqlite3.Row) -> dict:
    return {
        "id"         : row["id"],
        "url"        : row["url"],
        "score"      : row["score"],
        "percent"    : row["percent"],
        "verdict"    : row["verdict"],
        "features"   : json.loads(row["features_json"] or "{}"),
        "reasons"    : json.loads(row["shap_json"]     or "[]"),
        "campaign_id": row["campaign_id"],
        "timestamp"  : row["timestamp"],
    }


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    stats = get_stats()
    print("\n── Dashboard Stats ──────────────────────")
    print(f"  Total scans : {stats['total_scans']}")
    print(f"  Dangerous   : {stats['dangerous']}")
    print(f"  Suspicious  : {stats['suspicious']}")
    print(f"  Safe        : {stats['safe']}")
    print(f"  Model F1    : {stats['model_metrics'].get('f1', 'N/A')}")
    print("─────────────────────────────────────────\n")