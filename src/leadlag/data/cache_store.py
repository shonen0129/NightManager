"""Transactional SQLite-backed cache store.

Provides an alternative to advisory file locks (``fcntl``/``msvcrt``) and
pickle files. SQLite serializes concurrent readers/writers through its own
transaction mechanism, removing the need for explicit file-level locking.

Public API::

    from leadlag.data.cache_store import SqliteCacheStore
    store = SqliteCacheStore("var/market_data/cache.sqlite")
    store.set("etf_data", data)
    data = store.get("etf_data")
"""

from __future__ import annotations

import io
import logging
import pickle
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class CacheStoreError(Exception):
    """Raised when a cache store operation fails."""


class SqliteCacheStore:
    """Key-value cache backed by a single SQLite file.

    Values are stored as BLOB pickles. The schema is a single table with a
    primary-key on ``key``. SQLite's WAL mode is enabled so that readers do
    not block writers and the database remains resilient to process crashes.

    DataFrames are serialized to a portable record (Parquet if pyarrow is
    installed, otherwise a pickle-bytes wrapper) before being pickled, so that
    the round-trip does not depend on DataFrame-specific pickle internals.
    """

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
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

    @staticmethod
    def _df_to_record(df: pd.DataFrame) -> dict[str, Any]:
        """Serialize a DataFrame to a portable record.

        Uses Parquet when pyarrow is available; otherwise falls back to a
        pickle-bytes wrapper. The record dict is itself pickled by ``set``.
        """
        buf = io.BytesIO()
        try:
            df.to_parquet(buf, engine="pyarrow", index=True)
            return {"__df_record__": True, "format": "parquet", "blob": buf.getvalue()}
        except Exception:
            buf = io.BytesIO()
            df.to_pickle(buf)
            return {"__df_record__": True, "format": "pickle", "blob": buf.getvalue()}

    @staticmethod
    def _record_to_df(record: dict[str, Any]) -> pd.DataFrame:
        """Deserialize a record produced by :meth:`_df_to_record`."""
        buf = io.BytesIO(record["blob"])
        fmt = record.get("format", "pickle")
        if fmt == "parquet":
            return pd.read_parquet(buf, engine="pyarrow")
        return pd.read_pickle(buf)

    def _prepare_for_storage(self, value: Any) -> Any:
        """Recursively replace DataFrames with portable records."""
        if isinstance(value, pd.DataFrame):
            return self._df_to_record(value)
        if isinstance(value, dict):
            return {k: self._prepare_for_storage(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._prepare_for_storage(v) for v in value]
        return value

    def _restore_from_storage(self, value: Any) -> Any:
        """Recursively restore DataFrames from records."""
        if isinstance(value, dict):
            if value.get("__df_record__"):
                return self._record_to_df(value)
            return {k: self._restore_from_storage(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._restore_from_storage(v) for v in value]
        return value

    def set(self, key: str, value: Any) -> None:
        """Atomically store a value under ``key``."""
        blob = pickle.dumps(
            self._prepare_for_storage(value), protocol=pickle.HIGHEST_PROTOCOL
        )
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
            value = pickle.loads(row[0])
        except Exception as exc:
            raise CacheStoreError(f"Failed to deserialize cache key {key!r}: {exc}") from exc
        return self._restore_from_storage(value)

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
