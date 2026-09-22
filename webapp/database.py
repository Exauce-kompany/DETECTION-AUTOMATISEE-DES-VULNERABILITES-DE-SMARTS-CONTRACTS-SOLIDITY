from pathlib import Path
import os
import sqlite3
from datetime import datetime


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("SMARTBUG_DATABASE_PATH", str(BASE_DIR / "smartbug.db")))


# ============================================================
# CONNEXION
# ============================================================

def get_connection():
    """
    Ouvre une connexion vers la base SQLite SMART BUG.
    """

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALISATION
# ============================================================

def init_database():
    """
    Crée les tables nécessaires si elles n'existent pas encore.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
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
        """
    )


    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analyses_created_at
        ON analyses(created_at)
        """
    )


    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analyses_verdict
        ON analyses(verdict)
        """
    )


    connection.commit()
    connection.close()

    print(
        f"Base SMART BUG prête : {DATABASE_PATH}"
    )


# ============================================================
# AJOUT D'UNE ANALYSE
# ============================================================

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
    code
):
    """
    Enregistre une nouvelle analyse dans SQLite.
    """

    created_at = datetime.now().isoformat(
        timespec="seconds"
    )

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO analyses (
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
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            1 if truncated else 0,
            code,
            created_at
        )
    )


    analysis_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return analysis_id


# ============================================================
# HISTORIQUE COMPLET
# ============================================================

def get_all_analyses():
    """
    Retourne toutes les analyses,
    de la plus récente à la plus ancienne.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM analyses
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# UNE ANALYSE PRÉCISE
# ============================================================

def get_analysis_by_id(analysis_id):
    """
    Retourne une analyse précise à partir de son ID.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM analyses
        WHERE id = ?
        """,
        (
            analysis_id,
        )
    )

    row = cursor.fetchone()

    connection.close()


    if row is None:
        return None

    return dict(row)


# ============================================================
# STATISTIQUES
# ============================================================

def get_analysis_statistics():
    """
    Retourne les statistiques globales de l'historique.
    """

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM analyses
        """
    )

    total = cursor.fetchone()["total"]


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM analyses
        WHERE predicted_label = 1
        """
    )

    vulnerable = cursor.fetchone()["total"]


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM analyses
        WHERE predicted_label = 0
        """
    )

    non_vulnerable = cursor.fetchone()["total"]


    cursor.execute(
        """
        SELECT AVG(confidence) AS average_confidence
        FROM analyses
        """
    )

    result = cursor.fetchone()

    average_confidence = (
        result["average_confidence"]
        if result["average_confidence"] is not None
        else 0
    )


    connection.close()


    return {
        "total_analyses": total,
        "vulnerable": vulnerable,
        "non_vulnerable": non_vulnerable,
        "average_confidence": round(
            average_confidence,
            4
        )
    }


# ============================================================
# SUPPRESSION D'UNE ANALYSE
# ============================================================

def delete_analysis(analysis_id):
    """
    Supprime une analyse précise.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM analyses
        WHERE id = ?
        """,
        (
            analysis_id,
        )
    )

    deleted = cursor.rowcount > 0

    connection.commit()
    connection.close()

    return deleted


# ============================================================
# SUPPRESSION DE TOUT L'HISTORIQUE
# ============================================================

def clear_history():
    """
    Supprime toutes les analyses.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM analyses
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# TEST DIRECT
# ============================================================

if __name__ == "__main__":

    init_database()

    stats = get_analysis_statistics()

    print(
        "Statistiques :",
        stats
    )
