"""Small SQLite connection helper for future persistence features."""

from __future__ import annotations

import sqlite3
from pathlib import Path


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "career_navigator.sqlite3"


def get_connection(database_path: str | Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name enabled."""
    path = str(database_path)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection