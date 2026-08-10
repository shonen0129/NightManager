"""SQLite-backed gap matrix store.

Provides an alternative to per-date ``.npy`` files for ``mu_gap``,
``omega_gap``, and optional multi-horizon / rank-reversal matrices.
The store is backed by :class:`leadlag.data.cache_store.SqliteCacheStore`
so gap matrices are stored as versioned, picklable cache values while a
lightweight ``gap_matrices`` index table keeps track of dates, matrix types
and horizons for fast listing and look-ups.

Public API::

    from leadlag.data.gap_store import GapStore
    store = GapStore("var/market_data/gap_matrices.sqlite")
    store.save("2026-08-10", mu_gap, omega_gap, metadata={"sig_date": "2026-08-09"})
    mu, omega, meta = store.load("2026-08-10")

The legacy ``put`` / ``get`` / ``exists`` methods remain available for
per-matrix-type storage (e.g. multi-horizon ``mu_gap_h3`` / ``omega_gap_h3``
and ``rank_reversal`` signals).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)


class GapStoreError(Exception):
    """Raised when a gap store operation fails."""


class GapStore:
    """SQLite-backed store for gap-adjusted distribution matrices.

    Values are stored in a :class:`SqliteCacheStore`.  The local
    ``gap_matrices`` table acts as an index with one row per cached matrix
    so we can quickly list dates, check existence and resolve cache keys.

    Schema:
      - trade_date: YYYY-MM-DD
      - matrix_type: 'mu' | 'omega' | 'rank_reversal' | 'meta' | ...
      - horizon: int (``-1`` for the default non-horizon case)
      - cache_key: TEXT (foreign to ``cache_store.key``)
      - created_at: ISO timestamp
    """

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._cache = SqliteCacheStore(self.path, timeout=timeout)
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
                CREATE TABLE IF NOT EXISTS gap_matrices (
                    trade_date TEXT NOT NULL,
                    matrix_type TEXT NOT NULL,
                    horizon INTEGER NOT NULL DEFAULT -1,
                    cache_key TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (trade_date, matrix_type, horizon)
                )
            """)

    @staticmethod
    def _horizon_key(horizon: int | None) -> int:
        """SQLite does not treat NULLs as equal in UNIQUE/PK constraints, so
        store the sentinel ``-1`` for the default (non-horizon) case.
        """
        return -1 if horizon is None else int(horizon)

    @staticmethod
    def _cache_key(trade_date: str, matrix_type: str, horizon: int) -> str:
        """Return the canonical cache key for a matrix."""
        return f"gap:{trade_date}:{matrix_type}:{horizon}"

    def put(
        self,
        trade_date: str,
        matrix_type: str,
        data: np.ndarray | Any,
        horizon: int | None = None,
    ) -> None:
        """Store a matrix for a given trade date and type.

        The gap index and the cached matrix are written in a single SQLite
        transaction so a failure cannot leave an index row pointing to a
        missing cache value.
        """
        date_key = _normalise_date(trade_date)
        horizon_key = self._horizon_key(horizon)
        cache_key = self._cache_key(date_key, matrix_type, horizon_key)
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO gap_matrices
                    (trade_date, matrix_type, horizon, cache_key)
                    VALUES (?, ?, ?, ?)
                    """,
                    (date_key, matrix_type, horizon_key, cache_key),
                )
                self._cache._set_with_conn(conn, cache_key, data)
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                raise GapStoreError(
                    f"Failed to store {matrix_type} for {date_key}: {e}"
                ) from e

    def get(
        self,
        trade_date: str,
        matrix_type: str,
        horizon: int | None = None,
    ) -> np.ndarray | Any | None:
        """Return the stored matrix, or None if not found."""
        date_key = _normalise_date(trade_date)
        horizon_key = self._horizon_key(horizon)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cache_key FROM gap_matrices
                WHERE trade_date = ? AND matrix_type = ? AND horizon = ?
                """,
                (date_key, matrix_type, horizon_key),
            ).fetchone()
        if row is None:
            return None
        try:
            return self._cache.get(row[0])
        except Exception as e:
            raise GapStoreError(
                f"Failed to load {matrix_type} for {date_key}: {e}"
            ) from e

    def exists(
        self,
        trade_date: str,
        matrix_type: str,
        horizon: int | None = None,
    ) -> bool:
        date_key = _normalise_date(trade_date)
        horizon_key = self._horizon_key(horizon)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM gap_matrices
                WHERE trade_date = ? AND matrix_type = ? AND horizon = ?
                """,
                (date_key, matrix_type, horizon_key),
            ).fetchone()
        return row is not None

    def save(
        self,
        date: str,
        mu: np.ndarray,
        omega: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store the ``mu_gap`` / ``omega_gap`` pair for a trade date.

        The optional *metadata* dict (e.g. ``sig_date``) is stored together
        with the matrices and returned by :meth:`load`.
        """
        date_key = _normalise_date(date)
        meta = metadata if metadata is not None else {}
        self.put(date_key, "mu", mu)
        self.put(date_key, "omega", omega)
        self.put(date_key, "meta", meta)

    def load(self, date: str) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
        """Load the ``mu_gap`` / ``omega_gap`` pair and metadata for *date*.

        Returns ``(mu, omega, metadata)``.  Missing components are ``None``.
        """
        date_key = _normalise_date(date)
        mu = self.get(date_key, "mu")
        omega = self.get(date_key, "omega")
        meta = self.get(date_key, "meta")
        if meta is None:
            meta = {}
        if mu is None or omega is None:
            return None, None, None
        return cast(np.ndarray, mu), cast(np.ndarray, omega), cast(dict[str, Any], meta)

    def latest_date(self) -> str | None:
        """Return the most recent trade date with both ``mu`` and ``omega``."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT trade_date FROM gap_matrices
                WHERE matrix_type IN ('mu', 'omega') AND horizon = -1
                ORDER BY trade_date DESC
                LIMIT 1
                """
            ).fetchone()
        return row[0] if row else None

    def import_from_directory(
        self,
        gap_input_dir: str | Path,
        mu_pattern: str = "matrices/mu_gap_{date}.npy",
        omega_pattern: str = "matrices/omega_gap_{date}.npy",
        n_j: int = len(JP_TICKERS),
    ) -> dict[str, Any]:
        """Import all matching ``.npy`` files from a directory into the store.

        Returns a summary dict with ``imported`` and ``failed`` counts.
        """
        from leadlag.utils.gap_matrix_io import load_gap_matrices

        gap_input_dir = Path(gap_input_dir)
        if not gap_input_dir.exists():
            raise GapStoreError(f"Gap input directory not found: {gap_input_dir}")

        imported = 0
        failed = 0
        # Find all candidate date suffixes from mu files.
        # Filter with a regex so horizon files (e.g. mu_gap_h1_YYYYMMDD.npy)
        # are not mistaken for the default horizon=-1 matrices.
        import re as _re
        from pathlib import Path as _Path

        mu_basename = _Path(mu_pattern).name
        # re.escape does not escape braces, but to be robust we replace the
        # {date} placeholder with a safe token, escape the rest, then insert
        # the capture group.
        _date_placeholder = "__DATE_PLACEHOLDER__"
        escaped = _re.escape(mu_basename.replace("{date}", _date_placeholder))
        mu_re = _re.compile(
            r"^" + escaped.replace(_date_placeholder, r"(\d{8})") + r"$"
        )
        mu_files = sorted(
            f for f in gap_input_dir.glob(mu_pattern.replace("{date}", "*"))
            if mu_re.fullmatch(f.name)
        )
        for mu_file in mu_files:
            date_numeric = mu_file.stem.split("_")[-1]
            trade_date = _date_from_numeric(date_numeric)
            mu, omega, alerts = load_gap_matrices(
                gap_input_dir,
                trade_date,
                mu_pattern=mu_pattern,
                omega_pattern=omega_pattern,
                n_j=n_j,
            )
            if mu is None or omega is None:
                logger.warning("[%s] Skipped import: %s", trade_date, alerts)
                failed += 1
                continue
            self.save(trade_date, mu, omega)
            imported += 1

        return {"imported": imported, "failed": failed, "total_candidates": len(mu_files)}


def _normalise_date(date_str: str) -> str:
    """Return YYYY-MM-DD for any parseable date string."""
    return str(pd.to_datetime(date_str).date())


def _date_from_numeric(date_numeric: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    return f"{date_numeric[:4]}-{date_numeric[4:6]}-{date_numeric[6:8]}"


def is_gap_store_path(path: str | Path) -> bool:
    """Return True if *path* looks like a SQLite gap store."""
    return str(path).lower().endswith((".sqlite", ".sqlite3", ".db"))
