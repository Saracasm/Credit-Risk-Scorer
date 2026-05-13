"""
Database layer for credit risk scorer.

Uses SQLite for structured prediction history and applicant records.
Optionally uses ChromaDB for semantic similarity search across applicants.

SQLite tables:
  - predictions: all scored applicants with timestamps and results
  - conversations: agent chat history per session

ChromaDB collection (optional):
  - applicant_profiles: vectorized applicant descriptions for similarity search
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "credit_risk.db"


# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

def _get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a SQLite connection, creating the DB and tables if needed."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age REAL,
            monthly_income REAL,
            debt_ratio REAL,
            revolving_util REAL,
            n_30_59_late REAL,
            n_60_89_late REAL,
            n_90_late REAL,
            n_open_lines REAL,
            n_real_estate REAL,
            n_dependents REAL,
            probability REAL,
            predicted_class INTEGER,
            risk_level TEXT
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Prediction CRUD
# ---------------------------------------------------------------------------

def save_prediction(applicant: dict, probability: float, predicted_class: int) -> int:
    """Save a prediction to the database.

    Args:
        applicant: Dict with applicant parameters.
        probability: Default probability.
        predicted_class: 0 or 1.

    Returns:
        Row ID of the inserted prediction.
    """
    risk_level = "HIGH" if probability > 0.6 else "MEDIUM" if probability > 0.3 else "LOW"
    conn = _get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, age, monthly_income, debt_ratio, revolving_util,
                n_30_59_late, n_60_89_late, n_90_late, n_open_lines,
                n_real_estate, n_dependents, probability, predicted_class, risk_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                applicant.get("age"),
                applicant.get("monthly_income"),
                applicant.get("debt_ratio"),
                applicant.get("revolving_util"),
                applicant.get("n_30_59_late"),
                applicant.get("n_60_89_late"),
                applicant.get("n_90_late"),
                applicant.get("n_open_lines"),
                applicant.get("n_real_estate"),
                applicant.get("n_dependents"),
                probability,
                predicted_class,
                risk_level,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_all_predictions(limit: int = 500) -> pd.DataFrame:
    """Retrieve recent predictions as a DataFrame."""
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        return df
    finally:
        conn.close()


def get_prediction_stats() -> dict:
    """Get aggregate statistics from prediction history."""
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(probability) as avg_prob,
                MIN(probability) as min_prob,
                MAX(probability) as max_prob,
                SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) as low_count,
                SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) as med_count,
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_count
            FROM predictions
        """).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def save_message(session_id: str, role: str, content: str) -> None:
    """Save a chat message to the database."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO conversations (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now(timezone.utc).isoformat(), role, content),
        )
        conn.commit()
    finally:
        conn.close()


def get_conversation(session_id: str) -> list[dict]:
    """Retrieve all messages for a session."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversations WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Similar applicant search (ChromaDB — optional)
# ---------------------------------------------------------------------------

_chroma_collection = None


def _get_chroma_collection():
    """Lazily initialize a ChromaDB collection for applicant similarity search."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    try:
        import chromadb
    except ImportError:
        return None

    chroma_dir = PROJECT_ROOT / "data" / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    _chroma_collection = client.get_or_create_collection(
        name="applicant_profiles",
        metadata={"description": "Vectorized applicant profiles for similarity search"},
    )
    return _chroma_collection


def _applicant_to_text(applicant: dict, probability: float) -> str:
    """Convert applicant dict to a natural language description for embedding."""
    return (
        f"Loan applicant age {applicant.get('age', 'unknown')}, "
        f"monthly income ${applicant.get('monthly_income', 0):,.0f}, "
        f"debt-to-income ratio {applicant.get('debt_ratio', 0):.2f}, "
        f"credit utilization {applicant.get('revolving_util', 0):.0%}, "
        f"late payments: {applicant.get('n_30_59_late', 0):.0f} (30-59d), "
        f"{applicant.get('n_60_89_late', 0):.0f} (60-89d), "
        f"{applicant.get('n_90_late', 0):.0f} (90+d), "
        f"open credit lines {applicant.get('n_open_lines', 0):.0f}, "
        f"real estate loans {applicant.get('n_real_estate', 0):.0f}, "
        f"dependents {applicant.get('n_dependents', 0):.0f}. "
        f"Default probability: {probability:.1%}."
    )


def index_applicant(applicant: dict, probability: float, pred_id: int) -> bool:
    """Add an applicant profile to the vector store for similarity search.

    Args:
        applicant: Applicant parameters dict.
        probability: Predicted default probability.
        pred_id: Prediction row ID from SQLite.

    Returns:
        True if indexed successfully, False if ChromaDB not available.
    """
    collection = _get_chroma_collection()
    if collection is None:
        return False

    text = _applicant_to_text(applicant, probability)
    metadata = {
        "probability": probability,
        "risk_level": "HIGH" if probability > 0.6 else "MEDIUM" if probability > 0.3 else "LOW",
        "age": float(applicant.get("age", 0)),
        "monthly_income": float(applicant.get("monthly_income", 0)),
        "pred_id": pred_id,
    }
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[f"pred_{pred_id}"],
    )
    return True


def find_similar_applicants(applicant: dict, probability: float, n: int = 5) -> list[dict]:
    """Find similar past applicants using vector similarity.

    Args:
        applicant: Current applicant parameters.
        probability: Current prediction probability.
        n: Number of similar applicants to return.

    Returns:
        List of dicts with similar applicant info, or empty if ChromaDB unavailable.
    """
    collection = _get_chroma_collection()
    if collection is None:
        return []

    if collection.count() == 0:
        return []

    query_text = _applicant_to_text(applicant, probability)
    results = collection.query(query_texts=[query_text], n_results=min(n, collection.count()))

    similar = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        dist = results["distances"][0][i] if results["distances"] else None
        similar.append({
            "description": doc,
            "probability": meta.get("probability"),
            "risk_level": meta.get("risk_level"),
            "age": meta.get("age"),
            "monthly_income": meta.get("monthly_income"),
            "similarity_distance": dist,
        })
    return similar


def get_chroma_status() -> str:
    """Check if ChromaDB is available and return status string."""
    try:
        import chromadb
        coll = _get_chroma_collection()
        if coll is not None:
            return f"Active ({coll.count()} profiles indexed)"
        return "Unavailable"
    except ImportError:
        return "Not installed (pip install chromadb)"
