"""tests/unit/test_runner_helpers.py

Unit tests for runner.helpers shared utilities.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from domain.models.types import RiskConfig
from runner.config import ProductionConfig
from runner.helpers import (
    auto_adjust_gross_exposure,
    build_risk_config,
    save_decision_output,
)


# ---------------------------------------------------------------------------
# build_risk_config
# ---------------------------------------------------------------------------


class TestBuildRiskConfig:
    def test_returns_risk_config_type(self):
        cfg = ProductionConfig()
        risk = build_risk_config(cfg)
        assert isinstance(risk, RiskConfig)

    def test_var_confidence_matches(self):
        cfg = ProductionConfig()
        risk = build_risk_config(cfg)
        assert risk.var_confidence == pytest.approx(cfg.var_confidence)

    def test_max_gross_exposure_matches(self):
        cfg = ProductionConfig()
        risk = build_risk_config(cfg)
        assert risk.max_gross_exposure == pytest.approx(cfg.max_gross_exposure)

    def test_max_net_exposure_matches(self):
        cfg = ProductionConfig()
        risk = build_risk_config(cfg)
        assert risk.max_net_exposure == pytest.approx(cfg.max_net_exposure)

    def test_all_thresholds_transferred(self):
        cfg = ProductionConfig()
        risk = build_risk_config(cfg)
        for attr in (
            "var_warning", "var_stop", "es_warning", "es_stop",
            "daily_loss_warning", "daily_loss_stop", "monthly_loss_stop",
        ):
            assert getattr(risk, attr) == pytest.approx(getattr(cfg, attr))


# ---------------------------------------------------------------------------
# auto_adjust_gross_exposure
# ---------------------------------------------------------------------------


class TestAutoAdjustGrossExposure:
    def _make_decision(self, weights: np.ndarray) -> dict:
        n = len(weights)
        from domain.portfolio.optimizer import classify_actions
        return {
            "tickers": [f"TK{i}" for i in range(n)],
            "weight": weights.copy(),
            "action": classify_actions(weights),
            "signal": np.zeros(n),
        }

    def test_not_adjusted_when_within_limit(self):
        w = np.zeros(28)
        w[:7] = 0.1
        w[7:14] = -0.1
        decision = self._make_decision(w)
        cfg = ProductionConfig()
        result = auto_adjust_gross_exposure(decision, cfg)
        assert result["gross_adjusted"] is False

    def test_adjusted_when_over_limit(self):
        w = np.zeros(28)
        w[:7] = 0.3
        w[7:14] = -0.3   # gross = 4.2 > 3.0
        decision = self._make_decision(w)
        cfg = ProductionConfig()
        result = auto_adjust_gross_exposure(decision, cfg)
        assert result["gross_adjusted"] is True
        assert result["gross_after"] <= 3.0 + 1e-6

    def test_adjusted_gross_is_at_limit(self):
        w = np.zeros(28)
        w[:7] = 0.3
        w[7:14] = -0.3
        decision = self._make_decision(w)
        cfg = ProductionConfig()
        result = auto_adjust_gross_exposure(decision, cfg)
        gross_after = result.get("gross_after", np.abs(result["weight"]).sum())
        assert gross_after == pytest.approx(cfg.max_gross_exposure, rel=1e-5)


# ---------------------------------------------------------------------------
# save_decision_output
# ---------------------------------------------------------------------------


class TestSaveDecisionOutput:
    def test_creates_csv_file(self):
        tickers = ["1617.T", "XLB"]
        df = pd.DataFrame(
            {
                "ticker": tickers,
                "action": ["BUY", "SELL"],
                "quantity": [100, 50],
                "open_price": [1500.0, 45.0],
            }
        )
        trade_date = pd.Timestamp("2024-01-15")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_decision_output(df, tmpdir, trade_date)
            assert os.path.exists(path)
            assert "20240115" in path
            assert path.endswith(".csv")

    def test_csv_content_matches_dataframe(self):
        tickers = ["1617.T", "XLB"]
        df = pd.DataFrame(
            {
                "ticker": tickers,
                "action": ["BUY", "SELL"],
                "quantity": [100, 50],
            }
        )
        trade_date = pd.Timestamp("2024-06-01")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_decision_output(df, tmpdir, trade_date)
            loaded = pd.read_csv(path)
            assert list(loaded["ticker"]) == tickers
            assert list(loaded["action"]) == ["BUY", "SELL"]
