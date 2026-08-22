"""
LexGraph — Query Telemetry
Durable SQLite log of every query event (route, latency, success/failure),
so Session Analytics survives container/app restarts instead of living only
in st.session_state. Also backs the RAGAS drift-history table.
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("logs", "monitoring.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query_text TEXT NOT NULL,
    route TEXT NOT NULL,
    route_confidence REAL,
    route_latency_ms REAL,
    pipeline_latency_ms REAL,
    total_latency_ms REAL,
    success INTEGER NOT NULL,
    error_message TEXT,
    naive_confidence REAL,
    latency_breakdown_json TEXT
);

CREATE TABLE IF NOT EXISTS ragas_drift_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    metric TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    score REAL NOT NULL,
    baseline REAL,
    drift_detected INTEGER NOT NULL DEFAULT 0
);
"""


def init_db(path=DB_PATH):
    """Create the telemetry DB and tables if they don't exist yet. Idempotent."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


def log_query_event(
    query_text,
    route,
    route_confidence=None,
    route_latency_ms=None,
    pipeline_latency_ms=None,
    total_latency_ms=None,
    success=True,
    error_message=None,
    naive_confidence=None,
    latency_breakdown=None,
    path=DB_PATH,
):
    """Insert one durable record of a single pipeline execution for a query."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO query_events (
                timestamp, query_text, route, route_confidence, route_latency_ms,
                pipeline_latency_ms, total_latency_ms, success, error_message,
                naive_confidence, latency_breakdown_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                query_text,
                route,
                route_confidence,
                route_latency_ms,
                pipeline_latency_ms,
                total_latency_ms,
                1 if success else 0,
                error_message,
                naive_confidence,
                json.dumps(latency_breakdown) if latency_breakdown else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_events(limit=50, path=DB_PATH):
    """Return the most recent query events as a list of dicts, newest first."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM query_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_failure_rate(window=50, path=DB_PATH):
    """Fraction of the last `window` query events that failed (0.0-1.0)."""
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT success FROM query_events ORDER BY id DESC LIMIT ?", (window,)
        ).fetchall()
        if not rows:
            return 0.0
        failures = sum(1 for (success,) in rows if not success)
        return failures / len(rows)
    finally:
        conn.close()


def log_ragas_drift(metric, pipeline, score, baseline=None, drift_detected=False, path=DB_PATH):
    """Insert one tidy row (metric, pipeline, score) for the RAGAS drift job."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO ragas_drift_history (timestamp, metric, pipeline, score, baseline, drift_detected)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), metric, pipeline, score, baseline, 1 if drift_detected else 0),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_ragas_history(metric=None, pipeline=None, limit=200, path=DB_PATH):
    """Return RAGAS drift history rows, optionally filtered, oldest first."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM ragas_drift_history WHERE 1=1"
        params = []
        if metric:
            query += " AND metric = ?"
            params.append(metric)
        if pipeline:
            query += " AND pipeline = ?"
            params.append(pipeline)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
