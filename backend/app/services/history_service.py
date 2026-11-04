"""
history_service.py: saves every prediction and reads them back for the History page.
Uses SQLite (built into Python): the whole database is one file, backend/data/app.db.
Each prediction remembers which user made it, so everyone sees only their own.
"""
import json
import sqlite3
from datetime import datetime

from ..config import REPO_ROOT

DB_PATH = REPO_ROOT / "backend" / "data" / "app.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)   # create backend/data/ if missing
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row                      # rows behave like dicts: row["prediction"]
    return conn


def init_db():
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prediction    REAL NOT NULL,
                features      TEXT NOT NULL,
                user_id       INTEGER
            )
        """)
        # Databases made before accounts existed have no user_id column:
        # add it. Those old predictions belong to nobody and stay hidden.
        columns = [r["name"] for r in conn.execute("PRAGMA table_info(predictions)")]
        if "user_id" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
        conn.commit()
    finally:
        conn.close()


def add_prediction(user_id: int, features: dict, prediction: float, model_version: str):
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO predictions (timestamp, model_version, prediction, features, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                model_version,
                prediction,
                json.dumps(features, ensure_ascii=False),
                user_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_recent(user_id: int, limit: int = 50) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT timestamp, model_version, prediction, features "
            "FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "timestamp": r["timestamp"],
            "model_version": r["model_version"],
            "prediction": r["prediction"],
            "features": json.loads(r["features"]),
        }
        for r in rows
    ]
