"""SQLite-backed backtest result store.

Stores full backtest ``results`` dicts in a :class:`SqliteCacheStore` while
keeping queryable ``run_info`` / ``daily_pnl`` / ``daily_weights`` tables for
fast access to the most commonly inspected time series.

Public API::

    from leadlag.data.backtest_store import BacktestResultStore
    store = BacktestResultStore("var/results/backtest.sqlite")
    run_id = store.save_results(results)
    results = store.load_results(run_id)
    runs = store.list_runs()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd

from leadlag.data.cache_store import SqliteCacheStore

logger = logging.getLogger(__name__)


class BacktestStoreError(Exception):
    """Raised when a backtest store operation fails."""


class BacktestResultStore:
    """SQLite-backed store for full backtest output dicts.

    Tables:
      - ``run_info``: run_id, start_date, end_date, config_json, created_at.
      - ``daily_pnl``: per-run, per-date P&L record.
      - ``daily_weights``: per-run, per-date, per-ticker weight.

    Full ``results`` dicts are also cached under ``bt:{run_id}`` so they can be
    reloaded without reconstructing the individual tables.
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
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_info (
                    run_id INTEGER PRIMARY KEY,
                    start_date TEXT,
                    end_date TEXT,
                    config_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    run_id INTEGER NOT NULL,
                    trade_date TEXT NOT NULL,
                    daily_return REAL,
                    daily_return_gross REAL,
                    equity REAL,
                    drawdown REAL,
                    turnover REAL,
                    gross_exposure REAL,
                    fallback INTEGER,
                    slippage_cost REAL,
                    financing_cost REAL,
                    borrow_cost REAL,
                    reverse_cost REAL,
                    total_cost REAL,
                    PRIMARY KEY (run_id, trade_date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_weights (
                    run_id INTEGER NOT NULL,
                    trade_date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    weight REAL,
                    PRIMARY KEY (run_id, trade_date, ticker)
                )
            """)

    # -----------------------------------------------------------------------
    # New high-level API
    # -----------------------------------------------------------------------

    def save_results(self, results: dict, run_id: str | int | None = None) -> str:
        """Store a full ``results`` dict and return the ``run_id``.

        If *run_id* is omitted a new entry is created in ``run_info`` and the
        start/end dates are derived from ``results['daily_returns']``.
        """
        if run_id is None:
            run_id = self._create_run_info_from_results(results, config=None)
        run_id_str = str(run_id)
        try:
            self._cache.set(f"bt:{run_id_str}", results)
        except Exception as e:
            raise BacktestStoreError(
                f"Failed to cache results for run {run_id_str}: {e}"
            ) from e
        return run_id_str

    def load_results(self, run_id: str | int) -> dict[str, Any] | None:
        """Return the full ``results`` dict for *run_id*.

        Returns ``None`` when the run is not present in the cache.
        """
        run_id_str = str(run_id)
        try:
            return cast(dict[str, Any] | None, self._cache.get(f"bt:{run_id_str}"))
        except Exception as e:
            raise BacktestStoreError(
                f"Failed to load results for run {run_id_str}: {e}"
            ) from e

    def list_runs(self) -> list[str]:
        """Return a list of run ids stored in the database."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM run_info ORDER BY run_id"
            ).fetchall()
        return [str(r[0]) for r in rows]

    # -----------------------------------------------------------------------
    # Detailed-table API (legacy, used by tests and CSV-style consumers)
    # -----------------------------------------------------------------------

    def save_run(
        self,
        results: dict,
        config: Any | None = None,
    ) -> int | None:
        """Save a full backtest run and return the run_id.

        *results* must contain the keys produced by
        ``BacktestEngine.run_v2_backtest``:
          daily_returns, equity_curve, drawdown, daily_turnover,
          daily_gross_exps, daily_fallback, daily_slip_costs,
          daily_financing_costs, daily_borrow_costs, daily_reverse_costs,
          daily_costs, weights.
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                cur = conn.execute(
                    """
                    INSERT INTO run_info (start_date, end_date, config_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        _first_date(results.get("daily_returns")),
                        _last_date(results.get("daily_returns")),
                        json.dumps(_safe_config(config), default=str, ensure_ascii=False),
                    ),
                )
                run_id = cur.lastrowid

                index = results.get("daily_returns", pd.Series()).index
                equity = results.get("equity_curve", pd.Series(index=index, dtype=float))
                drawdown = results.get("drawdown", pd.Series(index=index, dtype=float))
                turnover = results.get("daily_turnover", pd.Series(index=index, dtype=float))
                gross = results.get("daily_gross_exps", pd.Series(index=index, dtype=float))
                fallback = results.get("daily_fallback", pd.Series(index=index, dtype=bool))
                slip = results.get("daily_slip_costs", pd.Series(index=index, dtype=float))
                financing = results.get("daily_financing_costs", pd.Series(index=index, dtype=float))
                borrow = results.get("daily_borrow_costs", pd.Series(index=index, dtype=float))
                reverse = results.get("daily_reverse_costs", pd.Series(index=index, dtype=float))
                costs = results.get("daily_costs", pd.Series(index=index, dtype=float))
                gross_returns = results.get("daily_returns_gross", pd.Series(index=index, dtype=float))

                rows = []
                for dt in index:
                    date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
                    rows.append((
                        run_id,
                        date_str,
                        float(results["daily_returns"].loc[dt]),
                        float(gross_returns.loc[dt]) if dt in gross_returns.index and not pd.isna(gross_returns.loc[dt]) else None,
                        float(equity.loc[dt]) if dt in equity.index and not pd.isna(equity.loc[dt]) else None,
                        float(drawdown.loc[dt]) if dt in drawdown.index and not pd.isna(drawdown.loc[dt]) else None,
                        float(turnover.loc[dt]) if dt in turnover.index and not pd.isna(turnover.loc[dt]) else None,
                        float(gross.loc[dt]) if dt in gross.index and not pd.isna(gross.loc[dt]) else None,
                        int(bool(fallback.loc[dt])) if dt in fallback.index and not pd.isna(fallback.loc[dt]) else 0,
                        float(slip.loc[dt]) if dt in slip.index and not pd.isna(slip.loc[dt]) else None,
                        float(financing.loc[dt]) if dt in financing.index and not pd.isna(financing.loc[dt]) else None,
                        float(borrow.loc[dt]) if dt in borrow.index and not pd.isna(borrow.loc[dt]) else None,
                        float(reverse.loc[dt]) if dt in reverse.index and not pd.isna(reverse.loc[dt]) else None,
                        float(costs.loc[dt]) if dt in costs.index and not pd.isna(costs.loc[dt]) else None,
                    ))

                conn.executemany(
                    """
                    INSERT OR REPLACE INTO daily_pnl
                    (run_id, trade_date, daily_return, daily_return_gross, equity, drawdown,
                     turnover, gross_exposure, fallback, slippage_cost, financing_cost,
                     borrow_cost, reverse_cost, total_cost)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

                weights_df = results.get("weights")
                if weights_df is not None and len(weights_df) > 0:
                    weight_rows = []
                    for dt, row in weights_df.iterrows():
                        date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
                        for ticker, w in row.items():
                            weight_rows.append((run_id, date_str, str(ticker), float(w)))
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO daily_weights
                        (run_id, trade_date, ticker, weight)
                        VALUES (?, ?, ?, ?)
                        """,
                        weight_rows,
                    )

                conn.execute("COMMIT")
                # Also cache the full results dict for the high-level API.
                self.save_results(results, run_id=run_id)
                return run_id
            except Exception as e:
                conn.execute("ROLLBACK")
                raise BacktestStoreError(f"Failed to save backtest run: {e}") from e

    def _create_run_info_from_results(
        self,
        results: dict,
        config: Any | None = None,
    ) -> int | None:
        """Insert a minimal run_info row for the high-level cache-only API."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO run_info (start_date, end_date, config_json)
                VALUES (?, ?, ?)
                """,
                (
                    _first_date(results.get("daily_returns")),
                    _last_date(results.get("daily_returns")),
                    json.dumps(_safe_config(config), default=str, ensure_ascii=False),
                ),
            )
            return cur.lastrowid

    def _resolve_run_id(self, conn: sqlite3.Connection, run_id: int | None) -> int | None:
        if run_id is not None:
            return run_id
        row = conn.execute("SELECT MAX(run_id) FROM run_info").fetchone()
        return row[0] if row and row[0] is not None else None

    def load_pnl(self, run_id: int | None = None) -> pd.DataFrame:
        """Return the daily P&L DataFrame for *run_id* (default: latest run)."""
        with self._connect() as conn:
            run_id = self._resolve_run_id(conn, run_id)
            if run_id is None:
                return pd.DataFrame()
            cur = conn.execute(
                """
                SELECT trade_date, daily_return, daily_return_gross, equity, drawdown,
                       turnover, gross_exposure, fallback, slippage_cost, financing_cost,
                       borrow_cost, reverse_cost, total_cost
                FROM daily_pnl
                WHERE run_id = ?
                ORDER BY trade_date
                """,
                (run_id,),
            )
            rows = cur.fetchall()
            columns = [c[0] for c in cur.description]
        df = pd.DataFrame(rows, columns=columns)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")
        return df

    def load_weights(self, run_id: int | None = None) -> pd.DataFrame:
        """Return daily weights as a wide DataFrame (dates x tickers) for *run_id*."""
        with self._connect() as conn:
            run_id = self._resolve_run_id(conn, run_id)
            if run_id is None:
                return pd.DataFrame()
            rows = conn.execute(
                """
                SELECT trade_date, ticker, weight
                FROM daily_weights
                WHERE run_id = ?
                ORDER BY trade_date, ticker
                """,
                (run_id,),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["trade_date", "ticker", "weight"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.pivot(index="trade_date", columns="ticker", values="weight")


def _first_date(series: pd.Series | None) -> str | None:
    if series is None or len(series) == 0:
        return None
    dt = series.index[0]
    return str(dt.date()) if hasattr(dt, "date") else str(dt)


def _last_date(series: pd.Series | None) -> str | None:
    if series is None or len(series) == 0:
        return None
    dt = series.index[-1]
    return str(dt.date()) if hasattr(dt, "date") else str(dt)


def _safe_config(config: Any) -> Any:
    if config is None:
        return None
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "__dict__"):
        return config.__dict__
    return config
