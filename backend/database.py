"""
Canonical Database Configuration — Mortgage AI v3.1

Single source of truth for the SQLite database path.
All backend modules MUST import DATABASE_PATH from this module
instead of defining their own.

Usage:
    from database import DATABASE_PATH, get_connection
"""

import os
import sqlite3
from pathlib import Path

# Canonical database path: defaults to backend/mortgage.db
# Override via DATABASE_PATH environment variable for Docker/production.
_BACKEND_DIR = Path(__file__).resolve().parent
DATABASE_PATH = os.environ.get("DATABASE_PATH", str(_BACKEND_DIR / "mortgage.db"))


def get_connection() -> sqlite3.Connection:
    """Return a new sqlite3 connection to the canonical database."""
    return sqlite3.connect(DATABASE_PATH)
