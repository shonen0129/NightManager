"""Unit regressions for Sprint 1 AUM 1M configuration and accounting contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.allocator import allocate_capital
from leadlag.core.pnl import simulate_daily_pnl
from research.diagnostics.sprint1_experiments import restore_dollar_neutrality


def test_config_loading():
    """Verify that configs/sprint1_aum1m_tachibana.yaml has the expected cost inputs."""
    config_path = ROOT / "configs" / "archive" / "sprint1_aum1m_tachibana.yaml"
    assert config_path.exists()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["aum_jpy"] == 1000000
    assert config["buy_interest_rate_annual"] == 0.025
    assert config["stock_borrow_fee_annual"] == 0.0115
    assert "broker_profile" in config
    assert config["broker_profile"]["margin_buy_interest_annual"] == 0.025


def test_rounding_to_lot_size_uses_canonical_allocator():
    """Exercise the maintained weight-to-quantity boundary, including the 1629.T lot."""
    result = allocate_capital(
        weights=np.array([0.10, -0.10, 0.05, -0.05]),
        tickers=["1617.T", "1629.T", "1630.T", "1631.T"],
        open_prices={
            "1617.T": 25000.0,
            "1629.T": 12000.0,
            "1630.T": 3000.0,
            "1631.T": 45000.0,
        },
        max_capital=1_000_000,
        side_leverage=1.5,
    )

    np.testing.assert_array_equal(result.quantities, [6, 10, 25, 1])
    np.testing.assert_allclose(result.allocated_amounts, [150000.0, 120000.0, 75000.0, 45000.0])


def test_credit_cost_calculation_uses_inventory_ledger():
    """Verify one calendar day of financing, borrow, and reverse costs on real inventory."""
    aum = 1_000_000.0
    buy_rate = 0.025
    borrow_rate = 0.0115
    reverse_fee_bps = 10.0

    result = simulate_daily_pnl(
        weights=np.zeros((1, 2)),
        target_returns=np.zeros((1, 2)),
        gap_returns=np.zeros((1, 2)),
        sim_dates=pd.DatetimeIndex(["2026-01-06"]),
        slip=0.0,
        financing_daily=buy_rate / 365.0,
        borrow_daily=borrow_rate / 365.0,
        reverse_daily=reverse_fee_bps / 10000.0,
        alpha_long=0.75,
        alpha_short=0.5,
        initial_holdings=np.array([100_000.0, -50_000.0]),
        initial_cash=950_000.0,
        initial_mark_date="2026-01-05",
    )

    np.testing.assert_allclose(result["financing_costs"][0] * aum, 100_000.0 * buy_rate / 365.0)
    np.testing.assert_allclose(result["borrow_costs"][0] * aum, 50_000.0 * borrow_rate / 365.0)
    np.testing.assert_allclose(
        result["reverse_costs"][0] * aum,
        50_000.0 * reverse_fee_bps / 10000.0,
    )
    np.testing.assert_allclose(
        result["gross_returns"],
        np.asarray(result["net_returns"]) + np.asarray(result["costs"]),
    )
    np.testing.assert_allclose(result["terminal_inventory"]["holdings"], [0.0, 0.0])


def test_short_unavailability_neutralization_uses_research_helper():
    """Exercise the same neutrality helper used by Sprint 1 constrained/stress paths."""
    weights = np.array([0.1, 0.2, 0.1, -0.1, -0.2, -0.1])
    unavailable = np.array([False, False, False, False, True, False])

    stressed = weights.copy()
    stressed[(stressed < 0.0) & unavailable] = 0.0
    result = restore_dollar_neutrality(stressed)

    np.testing.assert_allclose(result, [0.05, 0.10, 0.05, -0.10, 0.0, -0.10])
    assert np.isclose(np.sum(result), 0.0, atol=1e-12)
    assert np.sum(np.abs(result)) <= np.sum(np.abs(weights))
