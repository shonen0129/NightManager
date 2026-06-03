
"""tests/unit/test_portfolio.py

Unit tests for domain.portfolio.optimizer:
  - adjust_gross_exposure
  - classify_actions
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.portfolio.optimizer import (
    adjust_gross_exposure,
    classify_actions,
)


# ---------------------------------------------------------------------------
# adjust_gross_exposure
# ---------------------------------------------------------------------------


class TestAdjustGrossExposure:
    def _weights(self, longs, shorts):
        """Helper: build weight array with given long/short values."""
        return np.array(longs + [-s for s in shorts], dtype=float)

    def test_no_adjustment_when_within_limit(self):
        # gross = 0.5 + 0.5 = 1.0, limit = 3.0 → no change
        w = self._weights([0.5], [0.5])
        result = adjust_gross_exposure(w, max_gross_exposure=3.0)
        assert not result.was_adjusted
        assert result.adjustment_factor == pytest.approx(1.0)
        assert result.gross_after == pytest.approx(result.gross_before)

    def test_adjustment_applied_when_over_limit(self):
        # gross = 2.0 + 2.0 = 4.0, limit = 3.0 → scale to 0.75
        w = self._weights([2.0], [2.0])
        result = adjust_gross_exposure(w, max_gross_exposure=3.0)
        assert result.was_adjusted
        assert result.gross_before == pytest.approx(4.0)
        assert result.gross_after == pytest.approx(3.0, rel=1e-6)
        assert result.adjustment_factor == pytest.approx(3.0 / 4.0)

    def test_zero_weights_no_adjustment(self):
        w = np.zeros(10)
        result = adjust_gross_exposure(w, max_gross_exposure=3.0)
        assert not result.was_adjusted
        assert result.gross_before == pytest.approx(0.0)

    def test_adjustment_factor_is_limit_over_gross(self):
        w = self._weights([1.5, 1.0], [1.5, 1.0])
        result = adjust_gross_exposure(w, max_gross_exposure=3.0)
        expected_gross = np.abs(w).sum()
        if expected_gross > 3.0:
            assert result.adjustment_factor == pytest.approx(3.0 / expected_gross)


# ---------------------------------------------------------------------------
# classify_actions
# ---------------------------------------------------------------------------


class TestClassifyActions:
    def test_positive_weights_are_buy(self):
        w = np.array([0.5, 0.3, 0.2])
        actions = classify_actions(w)
        assert all(a == "BUY" for a in actions)

    def test_negative_weights_are_sell(self):
        w = np.array([-0.5, -0.3])
        actions = classify_actions(w)
        assert all(a == "SELL" for a in actions)

    def test_zero_weights_are_hold(self):
        w = np.array([0.0, 0.0])
        actions = classify_actions(w)
        assert all(a == "HOLD" for a in actions)

    def test_mixed_weights(self):
        w = np.array([0.5, -0.3, 0.0, 0.2, -0.1])
        actions = classify_actions(w)
        assert actions[0] == "BUY"
        assert actions[1] == "SELL"
        assert actions[2] == "HOLD"
        assert actions[3] == "BUY"
        assert actions[4] == "SELL"

    def test_output_length_matches_input(self):
        w = np.random.randn(28)
        actions = classify_actions(w)
        assert len(actions) == 28
