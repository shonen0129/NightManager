"""Tests for ``leadlag.data.backtest_store``."""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadlag.data.backtest_store import BacktestResultStore, BacktestStoreError


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
        "daily_overnight_returns": pd.Series([0.0005] * 5, index=dates),
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
        assert "overnight_return" in pnl.columns
        assert pnl["overnight_return"].tolist() == [0.0005] * 5

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


@pytest.mark.parametrize("operation", ["save_results", "load_results", "save_run", "invalid_config"])
def test_backtest_storage_failures_preserve_safe_exception_contract(tmp_path, monkeypatch, operation):
    secret = "SYNTHETIC_STORE_SECRET_34"
    store = BacktestResultStore(tmp_path / "backtest.sqlite")
    results = {"daily_returns": pd.Series([0.0], index=pd.DatetimeIndex(["2026-10-01"]))}

    def fail(*args, **kwargs):
        raise OSError(f"synthetic storage error: {secret}")

    monkeypatch.setattr(store._cache, "get" if operation == "load_results" else "set", fail)
    with pytest.raises(BacktestStoreError) as captured:
        if operation == "save_results":
            store.save_results(results, run_id=1)
        elif operation == "load_results":
            store.load_results(1)
        else:
            store.save_run(results, config={"risk": {"var_window": secret}} if operation == "invalid_config" else None)
    assert secret not in str(captured.value) + "".join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__
    expected_message = {
        "save_results": "Failed to cache backtest results.",
        "load_results": "Failed to load backtest results.",
        "save_run": "Failed to save backtest run.",
        "invalid_config": "Failed to save backtest run.",
    }[operation]
    assert str(captured.value) == expected_message
    if operation == "save_run":
        # The cache write happens after commit; its failure must not mask the
        # contract or undo the independently durable run audit trail.
        assert store.list_runs() == ["1"]
