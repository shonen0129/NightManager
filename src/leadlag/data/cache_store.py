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

import base64
import io
import json
import logging
import pickle
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CacheStoreError(Exception):
    """Raised when a cache store operation fails."""


class SqliteCacheStore:
    """Key-value cache backed by a single SQLite file.

    Values are stored as JSON text. The schema is a single table with a
    primary-key on ``key``. SQLite's WAL mode is enabled so that readers do
    not block writers and the database remains resilient to process crashes.

    DataFrames are serialized to a portable record (Parquet if pyarrow is
    installed, otherwise a pickle-bytes wrapper) and then base64-encoded, so
    the round-trip does not depend on DataFrame-specific pickle internals and
    the top-level store is pure JSON.
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
        pickle-bytes wrapper. The binary blob is base64-encoded so the record
        can be embedded in JSON.
        """
        buf = io.BytesIO()
        try:
            df.to_parquet(buf, engine="pyarrow", index=True)
            blob = buf.getvalue()
            fmt = "parquet"
        except Exception:
            buf = io.BytesIO()
            df.to_pickle(buf)
            blob = buf.getvalue()
            fmt = "pickle"
        return {
            "__df_record__": True,
            "format": fmt,
            "blob": base64.b64encode(blob).decode("ascii"),
        }

    @staticmethod
    def _record_to_df(record: dict[str, Any]) -> pd.DataFrame:
        """Deserialize a record produced by :meth:`_df_to_record`."""
        blob = record["blob"]
        if isinstance(blob, str):
            blob = base64.b64decode(blob.encode("ascii"))
        buf = io.BytesIO(blob)
        fmt = record.get("format", "pickle")
        if fmt == "parquet":
            return pd.read_parquet(buf, engine="pyarrow")
        return pd.read_pickle(buf)

    @staticmethod
    def _index_to_record(index: pd.Index) -> dict[str, Any]:
        """Serialize a pandas Index (including DatetimeIndex and RangeIndex)."""
        if isinstance(index, pd.DatetimeIndex):
            return {
                "__index_record__": True,
                "index_type": "DatetimeIndex",
                "values": [dt.isoformat() for dt in index],
                "name": index.name,
                "freq": getattr(index, "freqstr", None),
            }
        if isinstance(index, pd.RangeIndex):
            return {
                "__index_record__": True,
                "index_type": "RangeIndex",
                "start": index.start,
                "stop": index.stop,
                "step": index.step,
                "name": index.name,
            }
        return {
            "__index_record__": True,
            "index_type": "Index",
            "values": SqliteCacheStore._prepare_for_storage_static(index.tolist()),
            "name": index.name,
        }

    @staticmethod
    def _prepare_for_storage_static(value: Any) -> Any:
        """Recursively replace DataFrames/Series/ndarrays/bytes with JSON-safe objects.

        Static variant used by :meth:`_index_to_record` to avoid needing a
        :class:`SqliteCacheStore` instance.
        """
        if isinstance(value, pd.DataFrame):
            return SqliteCacheStore._df_to_record(value)
        if isinstance(value, pd.Series):
            return {
                "__series_record__": True,
                "values": SqliteCacheStore._prepare_for_storage_static(value.tolist()),
                "index": SqliteCacheStore._index_to_record(value.index),
                "name": value.name,
            }
        if isinstance(value, np.ndarray):
            return {
                "__ndarray_record__": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "data": SqliteCacheStore._prepare_for_storage_static(value.tolist()),
            }
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return {
                "__bytes_record__": True,
                "blob": base64.b64encode(value).decode("ascii"),
            }
        if isinstance(value, dict):
            return {k: SqliteCacheStore._prepare_for_storage_static(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SqliteCacheStore._prepare_for_storage_static(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return float(value) if isinstance(value, np.floating) else int(value)
        return value

    def _prepare_for_storage(self, value: Any) -> Any:
        """Recursively replace DataFrames/Series/ndarrays/bytes with JSON-safe objects."""
        return self._prepare_for_storage_static(value)

    @staticmethod
    def _record_to_index(record: Any) -> pd.Index | None:
        """Deserialize an index record produced by :meth:`_index_to_record`."""
        if not isinstance(record, dict) or not record.get("__index_record__"):
            return None
        index_type = record.get("index_type", "Index")
        name = record.get("name")
        if index_type == "DatetimeIndex":
            values = record.get("values", [])
            freq = record.get("freq")
            try:
                return pd.DatetimeIndex(pd.to_datetime(values), name=name, freq=freq)
            except ValueError:
                return pd.DatetimeIndex(pd.to_datetime(values), name=name)
        if index_type == "RangeIndex":
            return pd.RangeIndex(
                start=record.get("start"),
                stop=record.get("stop"),
                step=record.get("step", 1),
                name=name,
            )
        values = SqliteCacheStore._restore_from_storage_static(record.get("values", []))
        return pd.Index(values, name=name)

    @staticmethod
    def _restore_from_storage_static(value: Any) -> Any:
        """Recursively restore DataFrames/Series/ndarrays/bytes from records.

        Static variant used by :meth:`_record_to_index` and
        :meth:`_restore_from_storage`.
        """
        if isinstance(value, dict):
            if value.get("__df_record__"):
                return SqliteCacheStore._record_to_df(value)
            if value.get("__series_record__"):
                index_record = value.get("index")
                index = SqliteCacheStore._record_to_index(index_record)
                values = SqliteCacheStore._restore_from_storage_static(value["values"])
                return pd.Series(values, index=index, name=value.get("name"))
            if value.get("__ndarray_record__"):
                arr = np.array(
                    SqliteCacheStore._restore_from_storage_static(value["data"]),
                    dtype=value.get("dtype"),
                )
                shape = value.get("shape")
                if shape is not None:
                    arr = arr.reshape(shape)
                return arr
            if value.get("__bytes_record__"):
                blob = value["blob"]
                if isinstance(blob, str):
                    blob = base64.b64decode(blob.encode("ascii"))
                return blob
            return {k: SqliteCacheStore._restore_from_storage_static(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SqliteCacheStore._restore_from_storage_static(v) for v in value]
        return value

    def _restore_from_storage(self, value: Any) -> Any:
        """Recursively restore DataFrames/Series/ndarrays/bytes from records."""
        return self._restore_from_storage_static(value)

    def _set_with_conn(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        """Serialize and insert *value* under *key* using an existing connection.

        The caller is responsible for ``BEGIN`` / ``COMMIT`` / ``ROLLBACK``
        management. This lets ``GapStore`` keep the gap index and the cache
        value in a single atomic transaction.
        """
        blob = json.dumps(
            self._prepare_for_storage(value),
            default=self._json_default,
            ensure_ascii=False,
        ).encode("utf-8")
        conn.execute(
            "INSERT OR REPLACE INTO cache_store (key, value) VALUES (?, ?)",
            (key, blob),
        )

    def set(self, key: str, value: Any) -> None:
        """Atomically store a value under ``key``."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                self._set_with_conn(conn, key, value)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Fallback JSON encoder for residual objects."""
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        if isinstance(value, (np.integer, np.floating)):
            return float(value) if isinstance(value, np.floating) else int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value stored under ``key``, or ``default``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM cache_store WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        blob = row[0]
        try:
            # New stores use JSON; old stores used pickle. Try JSON first, then
            # pickle for backward compatibility.
            value = json.loads(blob.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                value = pickle.loads(blob)
            except Exception as exc:
                raise CacheStoreError(
                    f"Failed to deserialize cache key {key!r}: {exc}"
                ) from exc
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
