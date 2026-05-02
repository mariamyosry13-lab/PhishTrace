"""
PhishTrace — Database Layer (SQLite)
=====================================
بيحفظ كل scan في قاعدة بيانات SQLite محلية.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = os.environ.get("PHISHTRACE_DB", "phishtrace.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """بنعمل الـ tables لو مش موجودة."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT    NOT NULL,
                verdict     TEXT    NOT NULL,
                score       REAL    NOT NULL,
                raw_score   REAL    NOT NULL,
                rule_alerts TEXT    DEFAULT '[]',
                shap_reasons TEXT   DEFAULT '[]',
                features    TEXT    DEFAULT '{}',
                scanned_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS campaigns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                scan_ids    TEXT    DEFAULT '[]',
                created_at  TEXT    NOT NULL
            );
        """)
    print(f"DB initialised at: {DB_PATH}")


def save_scan(url, verdict, score, raw_score,
              rule_alerts=None, shap_reasons=None, features=None):
    """بنحفظ نتيجة الـ scan وبنرجع الـ id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO scans
               (url, verdict, score, raw_score, rule_alerts, shap_reasons, features, scanned_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                url,
                verdict,
                round(score, 4),
                round(raw_score, 4),
                json.dumps(rule_alerts  or []),
                json.dumps(shap_reasons or []),
                json.dumps(features     or {}),
                datetime.utcnow().isoformat(),
            )
        )
        return cur.lastrowid


def get_history(limit=50):
    """بنرجع آخر N scan."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        result.append({
            "id"          : r["id"],
            "url"         : r["url"],
            "verdict"     : r["verdict"],
            "score"       : r["score"],
            "raw_score"   : r["raw_score"],
            "rule_alerts" : json.loads(r["rule_alerts"]),
            "shap_reasons": json.loads(r["shap_reasons"]),
            "scanned_at"  : r["scanned_at"],
        })
    return result


def get_dashboard_stats():
    """بنرجع الإحصائيات للـ dashboard."""
    with get_conn() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        dangerous = conn.execute("SELECT COUNT(*) FROM scans WHERE verdict='Dangerous'").fetchone()[0]
        suspicious= conn.execute("SELECT COUNT(*) FROM scans WHERE verdict='Suspicious'").fetchone()[0]
        safe      = conn.execute("SELECT COUNT(*) FROM scans WHERE verdict='Safe'").fetchone()[0]

        # Timeline: آخر 10 scans مع التاريخ
        timeline_rows = conn.execute(
            "SELECT verdict, scanned_at FROM scans ORDER BY id DESC LIMIT 10"
        ).fetchall()

    timeline = [{"verdict": r["verdict"], "time": r["scanned_at"]} for r in timeline_rows]

    return {
        "total_scans": total,
        "dangerous"  : dangerous,
        "suspicious" : suspicious,
        "safe"       : safe,
        "timeline"   : timeline,
    }


def get_campaigns():
    """بنرجع الـ campaigns المحفوظة."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM campaigns ORDER BY id DESC"
        ).fetchall()
    return [
        {
            "id"        : r["id"],
            "name"      : r["name"],
            "scan_ids"  : json.loads(r["scan_ids"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]