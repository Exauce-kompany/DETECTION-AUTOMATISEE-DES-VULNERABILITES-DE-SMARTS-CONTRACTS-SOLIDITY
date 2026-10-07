"""SQLite history, with one closed connection and transaction per operation."""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# Keep existing installations' history in its original location.
DATABASE_PATH = Path(
    os.environ.get(
        "SMARTBUG_DATABASE_PATH", str(Path(__file__).resolve().parents[1] / "smartbug.db")
    )
)


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def transaction():
    connection = get_connection()
    try:
        # sqlite3's context manager commits/rolls back but does not close.
        with connection:
            yield connection
    finally:
        connection.close()


def init_database():
    with transaction() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_size_bytes INTEGER DEFAULT 0,
                predicted_label INTEGER NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                probability_vulnerable REAL NOT NULL,
                probability_non_vulnerable REAL NOT NULL,
                risk_score INTEGER DEFAULT 0,
                risk_level TEXT,
                tokens_detected INTEGER DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                unknown_tokens INTEGER DEFAULT 0,
                unknown_rate REAL DEFAULT 0,
                truncated INTEGER DEFAULT 0,
                code TEXT,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at)"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_analyses_verdict ON analyses(verdict)")


def save_analysis(
    filename,
    file_size_bytes,
    predicted_label,
    verdict,
    confidence,
    probability_vulnerable,
    probability_non_vulnerable,
    risk_score,
    risk_level,
    tokens_detected,
    tokens_used,
    unknown_tokens,
    unknown_rate,
    truncated,
    code,
):
    with transaction() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                filename, file_size_bytes, predicted_label, verdict, confidence,
                probability_vulnerable, probability_non_vulnerable, risk_score,
                risk_level, tokens_detected, tokens_used, unknown_tokens,
                unknown_rate, truncated, code, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                filename,
                file_size_bytes,
                predicted_label,
                verdict,
                confidence,
                probability_vulnerable,
                probability_non_vulnerable,
                risk_score,
                risk_level,
                tokens_detected,
                tokens_used,
                unknown_tokens,
                unknown_rate,
                int(truncated),
                code,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cursor.lastrowid


def get_all_analyses():
    with transaction() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM analyses ORDER BY id DESC")]


def get_analysis_by_id(analysis_id):
    with transaction() as connection:
        row = connection.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return dict(row) if row is not None else None


def get_analysis_statistics():
    with transaction() as connection:
        row = connection.execute("""
            SELECT COUNT(*) AS total_analyses,
                   COALESCE(SUM(predicted_label = 1), 0) AS vulnerable,
                   COALESCE(SUM(predicted_label = 0), 0) AS non_vulnerable,
                   COALESCE(AVG(confidence), 0) AS average_confidence
            FROM analyses
        """).fetchone()
        result = dict(row)
        result["average_confidence"] = round(result["average_confidence"], 4)
        return result


def delete_analysis(analysis_id):
    with transaction() as connection:
        return connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,)).rowcount > 0


def clear_history():
    with transaction() as connection:
        connection.execute("DELETE FROM analyses")
