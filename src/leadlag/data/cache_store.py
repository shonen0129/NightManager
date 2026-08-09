"""Transactional SQLite-backed cache store.

Provides an alternative to advisory file locks (``fcntl``/``msvcrt``) and
pickle files. SQLite serializes concurrent readers/writers through its own
transaction mechanism, removing the need for explicit file-level locking.

Public API::

    from leadlag.data.cache_store import SqliteCacheStore
    store = SqliteCacheStore("market_data/cache.sqlite")
    store.set("etf_data", data)
    data = store.get("etf_data")
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheStoreError(Exception):
    """Raised when a cache store operation fails."""


class SqliteCacheStore:
    """Key-value cache backed by a single SQLite file.

    Values are stored as BLOB pickles. The schema is a single table with a
    primary-key on ``key``. SQLite's WAL mode is enabled so that readers do
    not block writers and the database remains resilient to process crashes.
    """

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(
            str(self.path),
            timeout=self._timeout,
            isolation_level=None,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_store (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

    def set(self, key: str, value: Any) -> None:
        """Atomically store a value under ``key``."""
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache_store (key, value) VALUES (?, ?)",
                    (key, blob),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value stored under ``key``, or ``default``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM cache_store WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return pickle.loads(row[0])
        except Exception as exc:
            raise CacheStoreError(f"Failed to deserialize cache key {key!r}: {exc}") from exc

    def delete(self, key: str) -> bool:
        """Delete a key. Return True if it existed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM cache_store WHERE key = ?", (key,))
            return cur.rowcount > 0

    def keys(self) -> list[str]:
        """Return all cached keys."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key FROM cache_store").fetchall()
            return [r[0] for r in rows]

    def clear(self) -> None:
        """Remove all entries."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache_store")

    def meta(self, key: str) -> dict[str, Any] | None:
        """Return metadata (updated_at) for a key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT updated_at FROM cache_store WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return {"updated_at": row[0]}
