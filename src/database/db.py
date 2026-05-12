import sqlite3
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DB = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "phishtrace.db")
)
DB_PATH = os.environ.get("PHISHTRACE_DB", _DEFAULT_DB)
logger = logging.getLogger(__name__)

_EVAL_PATH = Path(os.path.join(os.path.dirname(__file__), "..", "..", "models", "evaluation_results.json"))
_model_metrics_cache: dict | None = None


def _get_model_metrics() -> dict:
    global _model_metrics_cache
    if _model_metrics_cache is None:
        if _EVAL_PATH.exists():
            with open(_EVAL_PATH) as f:
                _model_metrics_cache = _safe_json_loads(f.read(), {})
        else:
            _model_metrics_cache = {}
    return _model_metrics_cache


def _safe_json_loads(value, default):
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def get_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
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
                campaign_id TEXT    DEFAULT NULL,
                scanned_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS campaigns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                scan_ids    TEXT    DEFAULT '[]',
                created_at  TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_scans_verdict
                ON scans (verdict);

            CREATE INDEX IF NOT EXISTS idx_scans_scanned_at
                ON scans (scanned_at);
        """)
    logger.info("DB initialised at: %s", DB_PATH)


def save_scan(url, verdict, score, raw_score,
              rule_alerts=None, shap_reasons=None, features=None, campaign_id=None):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO scans
               (url, verdict, score, raw_score, rule_alerts, shap_reasons, features, campaign_id, scanned_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                url,
                verdict,
                round(score, 4),
                round(raw_score, 4),
                json.dumps(rule_alerts  or []),
                json.dumps(shap_reasons or []),
                json.dumps(features     or {}),
                campaign_id,
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            )
        )
        return cur.lastrowid


def get_history(limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for r in rows:
        result.append({
            "scan_id"     : r["id"],
            "url"         : r["url"],
            "verdict"     : r["verdict"],
            "score"       : r["score"],
            "raw_score"   : r["raw_score"],
            "rule_alerts" : _safe_json_loads(r["rule_alerts"], []),
            "shap_reasons": _safe_json_loads(r["shap_reasons"], []),
            "campaign_id" : r["campaign_id"],
            "scanned_at"  : r["scanned_at"],
        })
    return result


def get_scan(scan_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        return None
    return {
        "id"          : row["id"],
        "url"         : row["url"],
        "verdict"     : row["verdict"],
        "score"       : row["score"],
        "raw_score"   : row["raw_score"],
        "rule_alerts" : _safe_json_loads(row["rule_alerts"], []),
        "shap_reasons": _safe_json_loads(row["shap_reasons"], []),
        "campaign_id" : row["campaign_id"],
        "scanned_at"  : row["scanned_at"],
    }


def get_stats():
    """Backwards-compatible alias: dashboard stats are the single source."""
    return get_dashboard_stats()


def get_dashboard_stats():
    with get_conn() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        dangerous = conn.execute("SELECT COUNT(*) FROM scans WHERE verdict='Dangerous'").fetchone()[0]
        suspicious= conn.execute("SELECT COUNT(*) FROM scans WHERE verdict='Suspicious'").fetchone()[0]
        safe      = conn.execute("SELECT COUNT(*) FROM scans WHERE verdict='Safe'").fetchone()[0]

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
        "model_metrics": _get_model_metrics(),
    }


def get_campaigns():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM campaigns ORDER BY id DESC"
        ).fetchall()
        result = []
        for r in rows:
            scan_id_rows = conn.execute(
                "SELECT id FROM scans WHERE campaign_id = ?", (r["name"],)
            ).fetchall()
            result.append({
                "id"        : r["id"],
                "name"      : r["name"],
                "scan_ids"  : [s["id"] for s in scan_id_rows],
                "created_at": r["created_at"],
            })
    return result


def save_campaigns(campaign_list):
    """Replace all campaigns — clustering always rebuilds from scratch."""
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    with get_conn() as conn:
        conn.execute("DELETE FROM campaigns")
        conn.executemany(
            "INSERT INTO campaigns (name, scan_ids, created_at) VALUES (?, ?, ?)",
            [
                (
                    c["campaign_name"],
                    json.dumps(c.get("scan_ids", []) if isinstance(c.get("scan_ids", []), list) else []),
                    now,
                )
                for c in campaign_list
            ]
        )
    logger.info("[DB] Saved %s campaigns.", len(campaign_list))


def get_phishing_scan_count() -> int:
    """Count of scans with a phishing verdict."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM scans WHERE verdict IN ('Dangerous', 'Suspicious')"
        ).fetchone()[0]


def get_phishing_scans_with_features() -> list:
    """Return [{id, features}] for all phishing scans that have stored features."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, features FROM scans WHERE verdict IN ('Dangerous', 'Suspicious')"
        ).fetchall()
    result = []
    for r in rows:
        feats = _safe_json_loads(r["features"], {})
        if feats:
            result.append({"id": r["id"], "features": feats})
    return result


def update_scan_campaigns(assignments: list) -> None:
    """Batch-update campaign_id for a list of (campaign_id, scan_id) pairs."""
    with get_conn() as conn:
        conn.executemany(
            "UPDATE scans SET campaign_id = ? WHERE id = ?",
            assignments,
        )
    logger.info("[DB] Updated campaign_id for %d scans.", len(assignments))