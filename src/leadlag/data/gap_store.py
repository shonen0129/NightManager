"""SQLite-backed gap matrix store.

Provides an alternative to per-date ``.npy`` files for ``mu_gap``,
``omega_gap``, and optional multi-horizon / rank-reversal matrices.
Matrices are stored as BLOB pickles in a single SQLite file with WAL mode.

The store is designed to be opt-in: callers can still use the file-based
``utils/gap_matrix_io`` helpers.  When ``gap_input_dir`` (or a new
``gap_store_path``) points at an ``.sqlite`` file, the V2 pipeline can read
from the store instead of the directory.
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from leadlag.data.tickers import JP_TICKERS

logger = logging.getLogger(__name__)


class GapStoreError(Exception):
    """Raised when a gap store operation fails."""


class GapStore:
    """SQLite-backed store for gap-adjusted distribution matrices.

    Schema:
      - trade_date: YYYY-MM-DD
      - matrix_type: 'mu' | 'omega' | 'rank_reversal' | ...
      - horizon: int or NULL (for h=1 default / non-horizon-aware matrices)
      - data: BLOB (pickled np.ndarray)
      - created_at: ISO timestamp
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
                CREATE TABLE IF NOT EXISTS gap_matrices (
                    trade_date TEXT NOT NULL,
                    matrix_type TEXT NOT NULL,
                    horizon INTEGER,
                    data BLOB NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (trade_date, matrix_type, horizon)
                )
            """)

    def put(
        self,
        trade_date: str,
        matrix_type: str,
        data: np.ndarray,
        horizon: int | None = None,
    ) -> None:
        """Store a matrix for a given trade date and type."""
        blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        date_key = _normalise_date(trade_date)
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO gap_matrices
                    (trade_date, matrix_type, horizon, data)
                    VALUES (?, ?, ?, ?)
                    """,
                    (date_key, matrix_type, horizon, blob),
                )
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                raise GapStoreError(f"Failed to store {matrix_type} for {date_key}: {e}") from e

    def get(
        self,
        trade_date: str,
        matrix_type: str,
        horizon: int | None = None,
    ) -> np.ndarray | None:
        """Return the stored matrix, or None if not found."""
        date_key = _normalise_date(trade_date)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT data FROM gap_matrices
                WHERE trade_date = ? AND matrix_type = ? AND horizon IS ?
                """,
                (date_key, matrix_type, horizon),
            ).fetchone()
        if row is None:
            return None
        try:
            return pickle.loads(row[0])
        except Exception as e:
            raise GapStoreError(f"Failed to load {matrix_type} for {date_key}: {e}") from e

    def exists(
        self,
        trade_date: str,
        matrix_type: str,
        horizon: int | None = None,
    ) -> bool:
        date_key = _normalise_date(trade_date)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM gap_matrices
                WHERE trade_date = ? AND matrix_type = ? AND horizon IS ?
                """,
                (date_key, matrix_type, horizon),
            ).fetchone()
        return row is not None

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
        mu_files = sorted(gap_input_dir.glob(mu_pattern.replace("{date}", "*.npy")))
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
            self.put(trade_date, "mu", mu)
            self.put(trade_date, "omega", omega)
            imported += 1

        return {"imported": imported, "failed": failed, "total_candidates": len(mu_files)}


def _normalise_date(date_str: str) -> str:
    """Return YYYY-MM-DD for any parseable date string."""
    import pandas as pd

    return str(pd.to_datetime(date_str).date())


def _date_from_numeric(date_numeric: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    return f"{date_numeric[:4]}-{date_numeric[4:6]}-{date_numeric[6:8]}"


def is_gap_store_path(path: str | Path) -> bool:
    """Return True if *path* looks like a SQLite gap store."""
    return str(path).lower().endswith((".sqlite", ".sqlite3", ".db"))
