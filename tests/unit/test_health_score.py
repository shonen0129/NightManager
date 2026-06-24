"""Tests for Model Health Score module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.monitoring.health_score import (
    HealthScoreCalculator,
    HealthScore,
    HealthGrade,
    ComponentScore,
)


class TestHealthGrade:
    def test_grade_thresholds(self):
        calc = HealthScoreCalculator()
        assert calc._grade_from_score(90) == HealthGrade.EXCELLENT
        assert calc._grade_from_score(75) == HealthGrade.GOOD
        assert calc._grade_from_score(60) == HealthGrade.FAIR
        assert calc._grade_from_score(45) == HealthGrade.POOR
        assert calc._grade_from_score(30) == HealthGrade.CRITICAL


class TestHealthScoreCalculator:
    def _make_synthetic_data(self, n_days=120, n_assets=17, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
        signals = pd.DataFrame(
            rng.normal(0, 0.01, (n_days, n_assets)),
            index=dates,
            columns=[f"JP{i}" for i in range(n_assets)],
        )
        weights = np.zeros((n_days, n_assets))
        weights[:, :5] = 0.2
        weights[:, 12:] = -0.2
        weights_df = pd.DataFrame(weights, index=dates, columns=signals.columns)
        returns = rng.normal(0, 0.005, n_days)
        return signals, weights_df, returns

    def test_basic_compute(self):
        sig, w, ret = self._make_synthetic_data()
        calc = HealthScoreCalculator()
        score = calc.compute(sig, w, daily_returns=ret, target_gross=2.0)
        assert 0 <= score.score <= 100
        assert len(score.components) == 5
        assert score.grade in HealthGrade

    def test_healthy_data_scores_well(self):
        """Stable signals, low turnover, no fallback → high score."""
        sig, w, ret = self._make_synthetic_data(n_days=120)
        calc = HealthScoreCalculator()
        score = calc.compute(sig, w, daily_returns=ret, target_gross=2.0)
        # With constant weights, turnover should be 0 → high turnover score
        to_comp = [c for c in score.components if c.name == "turnover"][0]
        assert to_comp.score == 100.0
        # Gross is exactly 2.0 → high gross score
        gross_comp = [c for c in score.components if c.name == "gross_deviation"][0]
        assert gross_comp.score == 100.0
        # No fallback → 100
        fb_comp = [c for c in score.components if c.name == "fallback_rate"][0]
        assert fb_comp.score == 100.0

    def test_high_turnover_penalized(self):
        """High turnover should reduce turnover score (when turnover weight > 0)."""
        n_days, n_assets = 120, 17
        rng = np.random.default_rng(99)
        dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
        # Random weights → high turnover
        w = rng.normal(0, 0.1, (n_days, n_assets))
        # Make dollar-neutral
        w = w - w.mean(axis=1, keepdims=True)
        w_df = pd.DataFrame(w, index=dates, columns=[f"JP{i}" for i in range(n_assets)])
        sig = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=dates)
        # Explicitly enable turnover weight for this test
        calc = HealthScoreCalculator(
            ic_weight=0.30, turnover_weight=0.20,
            gross_dev_weight=0.15, fallback_weight=0.20, signal_drift_weight=0.15,
        )
        score = calc.compute(sig, w_df, daily_returns=rng.normal(0, 0.005, n_days))
        to_comp = [c for c in score.components if c.name == "turnover"][0]
        assert to_comp.score < 50.0

    def test_fallback_rate_penalized(self):
        """High fallback rate should reduce score."""
        sig, w, ret = self._make_synthetic_data()
        n = len(w)
        fallback = np.zeros(n, dtype=bool)
        fallback[n // 2:] = True  # 50% fallback
        calc = HealthScoreCalculator()
        score = calc.compute(sig, w, daily_returns=ret, fallback_flags=fallback)
        fb_comp = [c for c in score.components if c.name == "fallback_rate"][0]
        assert fb_comp.score == 0.0  # 50% fallback → 0

    def test_gross_deviation_penalized(self):
        """Gross far from target should reduce score."""
        sig, w, ret = self._make_synthetic_data()
        # Scale weights to gross=1.0 instead of 2.0
        w_scaled = w * 0.5
        calc = HealthScoreCalculator()
        score = calc.compute(sig, w_scaled, daily_returns=ret, target_gross=2.0)
        gross_comp = [c for c in score.components if c.name == "gross_deviation"][0]
        assert gross_comp.score < 50.0

    def test_signal_drift_detected(self):
        """Large signal distribution shift should reduce drift score."""
        n_days, n_assets = 120, 17
        dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
        sig = np.zeros((n_days, n_assets))
        sig[:100] = np.random.randn(100, n_assets) * 0.01
        sig[100:] = np.random.randn(20, n_assets) * 0.05 + 0.02  # shifted mean and std
        sig_df = pd.DataFrame(sig, index=dates)
        w = np.zeros((n_days, n_assets))
        w[:, :5] = 0.2
        w[:, 12:] = -0.2
        w_df = pd.DataFrame(w, index=dates)
        calc = HealthScoreCalculator()
        score = calc.compute(sig_df, w_df, daily_returns=np.random.randn(n_days))
        drift_comp = [c for c in score.components if c.name == "signal_drift"][0]
        assert drift_comp.score < 60.0

    def test_insufficient_data_handled(self):
        """Very short data should not crash."""
        sig = pd.DataFrame(np.zeros((5, 17)))
        w = pd.DataFrame(np.zeros((5, 17)))
        calc = HealthScoreCalculator()
        score = calc.compute(sig, w, daily_returns=np.zeros(5))
        assert 0 <= score.score <= 100

    def test_no_returns_handled(self):
        """Missing returns should not crash IC component."""
        sig, w, _ = self._make_synthetic_data()
        calc = HealthScoreCalculator()
        score = calc.compute(sig, w, daily_returns=None)
        ic_comp = [c for c in score.components if c.name == "ic_decay"][0]
        assert ic_comp.score == 50.0  # neutral when no data

    def test_weights_validation(self):
        """Weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HealthScoreCalculator(ic_weight=0.5, turnover_weight=0.5, gross_dev_weight=0.5)

    def test_turnover_excluded_by_default(self):
        """Default config should have turnover_weight=0 (day-trading model)."""
        calc = HealthScoreCalculator()
        assert calc.weights["turnover"] == 0.0
        # Active weights should sum to 1.0 (excluding turnover)
        active = {k: v for k, v in calc.weights.items() if v > 0}
        assert abs(sum(active.values()) - 1.0) < 1e-6

    def test_score_summary_string(self):
        sig, w, ret = self._make_synthetic_data()
        calc = HealthScoreCalculator()
        score = calc.compute(sig, w, daily_returns=ret)
        s = score.summary()
        assert "Health Score" in s
        assert "ic_decay" in s
        assert "turnover" in s

    def test_is_healthy_threshold(self):
        score = HealthScore(score=75.0, grade=HealthGrade.GOOD)
        assert score.is_healthy is True
        assert score.is_critical is False

    def test_is_critical_threshold(self):
        score = HealthScore(score=30.0, grade=HealthGrade.CRITICAL)
        assert score.is_healthy is False
        assert score.is_critical is True

    def test_component_weighted_score(self):
        comp = ComponentScore("test", 80.0, 0.3)
        assert abs(comp.weighted_score - 24.0) < 1e-10
