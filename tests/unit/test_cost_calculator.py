"""Tests for the unified CostCalculator module."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from leadlag.cost.cost_calculator import (
    CostCalculator,
    CostConfig,
    CostBreakdown,
)
from leadlag.execution.slippage_model import CostSource


class TestCostConfig:
    def test_defaults(self):
        cfg = CostConfig()
        assert cfg.fallback_roundtrip_bps == 15.0
        assert cfg.buy_interest_rate_annual == 0.025
        assert cfg.stock_borrow_fee_annual == 0.0115
        assert cfg.holding_days == 1

    def test_from_config_empty(self):
        cfg = CostConfig.from_config({})
        assert cfg.fallback_roundtrip_bps == 15.0

    def test_from_config_with_values(self):
        cfg = CostConfig.from_config({
            "cost_aware_optimization": {
                "default_spread_fallback_roundtrip_bps": 20.0,
                "buy_interest_rate_annual": 0.03,
                "stock_borrow_fee_annual": 0.02,
                "holding_days": 3,
            }
        })
        assert cfg.fallback_roundtrip_bps == 20.0
        assert cfg.buy_interest_rate_annual == 0.03
        assert cfg.stock_borrow_fee_annual == 0.02
        assert cfg.holding_days == 3


class TestCostBreakdown:
    def test_total_roundtrip_bps(self):
        bd = CostBreakdown(
            entry_cost_bps=5.0,
            exit_cost_bps=7.5,
            financing_bps=0.68,
            borrow_bps=0.0,
            reverse_bps=2.0,
            cost_source=CostSource.FIXED_SPREAD_FALLBACK,
        )
        assert abs(bd.total_roundtrip_bps - 15.18) < 1e-10

    def test_total_roundtrip_decimal(self):
        bd = CostBreakdown(
            entry_cost_bps=100.0,
            exit_cost_bps=100.0,
            financing_bps=0.0,
            borrow_bps=0.0,
            reverse_bps=0.0,
            cost_source=CostSource.FIXED_SPREAD_FALLBACK,
        )
        assert abs(bd.total_roundtrip_decimal - 0.02) < 1e-10


class TestCostCalculator:
    def test_from_config(self):
        calc = CostCalculator.from_config({})
        assert calc.cfg.fallback_roundtrip_bps == 15.0

    def test_compute_cost_long_no_snapshot(self):
        calc = CostCalculator.from_config({})
        bd = calc.compute_cost(None, "BUY", 500_000)
        # Fallback: entry = 15/2 = 7.5, exit = 15/2 = 7.5
        assert abs(bd.entry_cost_bps - 7.5) < 1e-10
        assert abs(bd.exit_cost_bps - 7.5) < 1e-10
        # Long pays financing, no borrow
        assert bd.financing_bps > 0.0
        assert bd.borrow_bps == 0.0
        assert bd.reverse_bps == 0.0
        assert bd.cost_source == CostSource.FIXED_SPREAD_FALLBACK

    def test_compute_cost_short_no_snapshot(self):
        calc = CostCalculator.from_config({})
        bd = calc.compute_cost(None, "SELL", 500_000, reverse_fee_bps=3.0)
        assert abs(bd.entry_cost_bps - 7.5) < 1e-10
        assert abs(bd.exit_cost_bps - 7.5) < 1e-10
        # Short pays borrow + reverse, no financing
        assert bd.financing_bps == 0.0
        assert bd.borrow_bps > 0.0
        assert abs(bd.reverse_bps - 3.0) < 1e-10

    def test_compute_net_signal_long(self):
        calc = CostCalculator.from_config({})
        signal = 0.001  # 10 bps expected return
        net, bd = calc.compute_net_signal(signal, None, "BUY", 500_000)
        # net = signal - total_cost_decimal
        assert net < signal
        assert net == signal - bd.total_roundtrip_decimal

    def test_compute_net_signal_short(self):
        calc = CostCalculator.from_config({})
        signal = -0.001  # negative signal → short candidate
        net, bd = calc.compute_net_signal(signal, None, "SELL", 500_000, reverse_fee_bps=2.0)
        # net = -signal - total_cost_decimal = 0.001 - cost
        assert net < abs(signal)
        assert net == (-signal) - bd.total_roundtrip_decimal

    def test_financing_bps_daily_rate(self):
        calc = CostCalculator.from_config({})
        # 2.5% annual / 365 * 10000 = 0.6849... bps/day
        expected = (0.025 / 365.0) * 10000.0
        bd = calc.compute_cost(None, "BUY", 100_000)
        assert abs(bd.financing_bps - expected) < 1e-6

    def test_borrow_bps_daily_rate(self):
        calc = CostCalculator.from_config({})
        # 1.15% annual / 365 * 10000 = 0.315... bps/day
        expected = (0.0115 / 365.0) * 10000.0
        bd = calc.compute_cost(None, "SELL", 100_000)
        assert abs(bd.borrow_bps - expected) < 1e-6

    def test_holding_days_multiplier(self):
        cfg = CostConfig(holding_days=5)
        calc = CostCalculator(cfg)
        bd_long = calc.compute_cost(None, "BUY", 100_000)
        bd_short = calc.compute_cost(None, "SELL", 100_000)
        # 5 days of financing
        expected_fin = (0.025 / 365.0) * 10000.0 * 5
        expected_borrow = (0.0115 / 365.0) * 10000.0 * 5
        assert abs(bd_long.financing_bps - expected_fin) < 1e-6
        assert abs(bd_short.borrow_bps - expected_borrow) < 1e-6

    def test_with_lob_snapshot(self):
        """Test with a mock LOB snapshot to verify LOB path is used."""
        snap = MagicMock()
        snap.lob_available = True
        snap.ticker = "7203"
        snap.bid_price_1 = 2500.0
        snap.ask_price_1 = 2502.0
        snap.last_price = 2501.0
        snap.cost_source = "lob_snapshot"
        # Mock the LOB cost functions by setting up bid/ask sizes
        for i in range(1, 6):
            setattr(snap, f"bid_price_{i}", 2500.0 - (i - 1) * 2)
            setattr(snap, f"ask_price_{i}", 2502.0 + (i - 1) * 2)
            setattr(snap, f"bid_size_{i}", 1000)
            setattr(snap, f"ask_size_{i}", 1000)

        calc = CostCalculator.from_config({})
        bd = calc.compute_cost(snap, "BUY", 100_000)
        # LOB path should give different (likely lower) entry cost
        assert bd.cost_source == CostSource.LOB_SNAPSHOT
        assert bd.entry_cost_bps > 0.0
