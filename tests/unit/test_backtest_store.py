"""Tests for ``leadlag.data.backtest_store``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data.backtest_store import BacktestResultStore


def _make_results():
    dates = pd.date_range("2020-01-06", periods=5, freq="B")
    return {
        "daily_returns": pd.Series([0.001, -0.002, 0.003, 0.0, -0.001], index=dates),
        "daily_returns_gross": pd.Series([0.002, -0.001, 0.004, 0.001, -0.001], index=dates),
        "equity_curve": pd.Series([1.001, 0.999, 1.002, 1.002, 1.001], index=dates),
        "drawdown": pd.Series([0.0, -0.002, 0.0, 0.0, -0.001], index=dates),
        "daily_turnover": pd.Series([0.5, 0.6, 0.5, 0.4, 0.5], index=dates),
        "daily_gross_exps": pd.Series([1.9, 2.0, 1.8, 1.9, 2.0], index=dates),
        "daily_fallback": pd.Series([False, False, False, True, False], index=dates),
        "daily_slip_costs": pd.Series([0.0001] * 5, index=dates),
        "daily_financing_costs": pd.Series([0.0002] * 5, index=dates),
        "daily_borrow_costs": pd.Series([0.0003] * 5, index=dates),
        "daily_reverse_costs": pd.Series([0.0004] * 5, index=dates),
        "daily_costs": pd.Series([0.001] * 5, index=dates),
        "weights": pd.DataFrame(
            np.random.RandomState(42).randn(5, 4) * 0.1,
            index=dates,
            columns=["A", "B", "C", "D"],
        ),
    }


def test_backtest_store_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bt.sqlite"
        store = BacktestResultStore(path)
        results = _make_results()

        run_id = store.save_run(results, config={"model": "v2"})
        assert run_id == 1

        pnl = store.load_pnl()
        assert len(pnl) == 5
        assert "daily_return" in pnl.columns
        assert "equity" in pnl.columns

        weights = store.load_weights()
        assert weights.shape == (5, 4)


def test_backtest_store_multi_run_audit_trail():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bt.sqlite"
        store = BacktestResultStore(path)
        results = _make_results()

        run_id_1 = store.save_run(results, config={"model": "v2", "run": 1})
        run_id_2 = store.save_run(results, config={"model": "v2", "run": 2})
        assert run_id_1 == 1
        assert run_id_2 == 2

        # Default load returns the latest run.
        pnl_latest = store.load_pnl()
        assert len(pnl_latest) == 5

        # Both runs remain queryable (audit trail).
        pnl_1 = store.load_pnl(run_id=1)
        pnl_2 = store.load_pnl(run_id=2)
        assert len(pnl_1) == 5
        assert len(pnl_2) == 5

        import sqlite3

        conn = sqlite3.connect(str(path))
        run_count = conn.execute("SELECT COUNT(*) FROM run_info").fetchone()[0]
        pnl_count = conn.execute("SELECT COUNT(*) FROM daily_pnl").fetchone()[0]
        conn.close()
        assert run_count == 2
        assert pnl_count == 10
