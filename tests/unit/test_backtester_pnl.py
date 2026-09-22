"""Cost-model regressions for the V2 daily inventory ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine


def test_flat_transition_charges_liquidation_slippage() -> None:
    result = BacktestEngine._simulate_daily_pnl(
        weights=np.array([[1.0], [0.0]]),
        target_returns=np.zeros((2, 1)),
        gap_returns=np.zeros((2, 1)),
        sim_dates=pd.DatetimeIndex(["2026-01-05", "2026-01-06"]),
        slip=0.0005,
        financing_daily=0.0,
        borrow_daily=0.0,
        reverse_daily=0.0,
        alpha_long=0.75,
        alpha_short=0.5,
        side_leverage=1.5,
    )
    # Day 1: open 1.0 plus close 25%; day 2: liquidate the 75% carried
    # inventory.  The old sign-based formula charged zero on day 2.
    expected = [1.5 * 0.0005 * 1.25, 1.5 * 0.0005 * 0.75]
    np.testing.assert_allclose(result["slip_costs"], expected)


def test_backtest_target_uses_run_owned_open_910_returns() -> None:
    dates = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    frame = pd.DataFrame(index=dates)
    for ticker in JP_TICKERS:
        frame[f"jp_oc_{ticker}"] = 0.20
        frame[f"jp_gap_{ticker}"] = 0.0
        frame[f"jp_open_trade_{ticker}"] = 100.0

    open_910 = pd.DataFrame(0.10, index=dates, columns=JP_TICKERS)
    target, gap = BacktestEngine._compute_target_and_gap_returns(
        frame,
        dates,
        dates,
        open_910_returns=open_910,
    )

    np.testing.assert_allclose(target, (1.20 / 1.10) - 1.0)
    np.testing.assert_allclose(gap, 0.0)
