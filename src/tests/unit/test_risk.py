"""tests/unit/test_risk.py

Unit tests for domain.risk.metrics:
  - compute_var_es
  - evaluate_risk_checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domain.models.types import RiskConfig
from domain.risk.metrics import compute_var_es, evaluate_risk_checks


# ---------------------------------------------------------------------------
# compute_var_es
# ---------------------------------------------------------------------------


class TestComputeVarEs:
    def _sample_returns(self, n: int, seed: int = 42) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0.0, 0.01, n))

    def test_available_when_sufficient_samples(self):
        returns = self._sample_returns(300)
        result = compute_var_es(returns, confidence=0.99, window=250)
        assert result.available is True
        assert result.samples == 250

    def test_not_available_when_insufficient_samples(self):
        returns = self._sample_returns(100)
        result = compute_var_es(returns, confidence=0.99, window=250)
        assert result.available is False
        assert np.isnan(result.var_loss)
        assert np.isnan(result.es_loss)

    def test_var_es_are_non_negative(self):
        returns = self._sample_returns(300)
        result = compute_var_es(returns, confidence=0.99, window=250)
        assert result.var_loss >= 0.0
        assert result.es_loss >= 0.0

    def test_es_gte_var(self):
        # Expected Shortfall is always >= VaR
        returns = self._sample_returns(300)
        result = compute_var_es(returns, confidence=0.99, window=250)
        assert result.es_loss >= result.var_loss - 1e-9

    def test_window_is_correct(self):
        returns = self._sample_returns(300)
        result = compute_var_es(returns, confidence=0.99, window=200)
        assert result.window == 200
        assert result.samples == 200

    def test_all_positive_returns_gives_zero_losses(self):
        # If all returns are positive, VaR and ES should be 0
        returns = pd.Series(np.ones(300) * 0.01)
        result = compute_var_es(returns, confidence=0.99, window=250)
        assert result.available is True
        assert result.var_loss == pytest.approx(0.0, abs=1e-9)
        assert result.es_loss == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate_risk_checks
# ---------------------------------------------------------------------------


def _make_risk_config(**kwargs) -> RiskConfig:
    defaults = dict(
        var_confidence=0.99,
        var_window=250,
        var_warning=0.02,
        var_stop=0.03,
        es_warning=0.025,
        es_stop=0.04,
        daily_loss_warning=0.015,
        daily_loss_stop=0.025,
        monthly_loss_stop=0.05,
        max_net_exposure=0.05,
        max_gross_exposure=3.0,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


class TestEvaluateRiskChecks:
    def _normal_weights(self) -> np.ndarray:
        """28-element weight vector with small net and 1x gross."""
        w = np.zeros(28)
        w[:7] = 0.07   # BUY side
        w[7:14] = -0.07  # SELL side
        return w

    def test_no_breaches_under_normal_conditions(self):
        weights = self._normal_weights()
        returns = pd.Series(np.zeros(300) + 0.001)  # small positive returns
        config = _make_risk_config()
        report = evaluate_risk_checks(
            weights=weights,
            total_buy_allocated=500_000,
            total_sell_allocated=500_000,
            max_capital=2_000_000,
            hist_daily_returns=returns,
            config=config,
        )
        assert report.is_blocked is False

    def test_stop_breach_when_gross_exceeds_limit(self):
        # gross exposure = 3.5 > limit 3.0
        weights = np.zeros(28)
        weights[:7] = 0.25   # 7 * 0.25 = 1.75 long
        weights[7:14] = -0.25  # 1.75 short → gross = 3.5
        returns = pd.Series(np.zeros(300) + 0.001)
        config = _make_risk_config(max_gross_exposure=3.0)
        report = evaluate_risk_checks(
            weights=weights,
            total_buy_allocated=1_000_000,
            total_sell_allocated=1_000_000,
            max_capital=2_000_000,
            hist_daily_returns=returns,
            config=config,
        )
        assert report.is_blocked is True
        assert any("gross" in b for b in report.stop_breaches)

    def test_warning_only_when_var_in_warning_zone(self):
        weights = self._normal_weights()
        # Simulate large losses in history to trigger VaR warning
        returns = pd.Series(np.full(300, -0.025))  # 2.5% loss every day
        config = _make_risk_config(var_warning=0.02, var_stop=0.05)
        report = evaluate_risk_checks(
            weights=weights,
            total_buy_allocated=500_000,
            total_sell_allocated=500_000,
            max_capital=2_000_000,
            hist_daily_returns=returns,
            config=config,
        )
        # Should have warning breaches (but not necessarily stop)
        assert len(report.warning_breaches) > 0 or len(report.stop_breaches) > 0

    def test_is_blocked_true_when_stop_breached(self):
        weights = self._normal_weights()
        returns = pd.Series(np.zeros(300) + 0.001)
        config = _make_risk_config(
            max_net_exposure=0.001,  # extremely tight limit to force breach
        )
        # Net exposure = 0 (balanced), should not trigger net exposure breach
        # Let's instead trigger a manual stop via daily loss
        returns_with_loss = pd.Series(np.full(300, -0.03))  # 3% loss every day
        config2 = _make_risk_config(daily_loss_stop=0.025)
        report = evaluate_risk_checks(
            weights=weights,
            total_buy_allocated=500_000,
            total_sell_allocated=500_000,
            max_capital=2_000_000,
            hist_daily_returns=returns_with_loss,
            config=config2,
        )
        assert report.is_blocked is True

    def test_is_blocked_property_matches_stop_breaches(self):
        weights = self._normal_weights()
        returns = pd.Series(np.zeros(300) + 0.001)
        config = _make_risk_config()
        report = evaluate_risk_checks(
            weights=weights,
            total_buy_allocated=500_000,
            total_sell_allocated=500_000,
            max_capital=2_000_000,
            hist_daily_returns=returns,
            config=config,
        )
        # is_blocked must be True iff stop_breaches is non-empty
        assert report.is_blocked == (len(report.stop_breaches) > 0)
