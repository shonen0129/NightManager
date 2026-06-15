"""tests/unit/test_allocator.py

Unit tests for domain.portfolio.allocator.allocate_capital.
"""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.core.allocator import allocate_capital

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_opens(tickers: list[str], price: float = 1000.0) -> dict[str, float]:
    return {tk: price for tk in tickers}


def _make_tickers(n_jp: int = 17, n_us: int = 11) -> list[str]:
    jp = [f"{1617 + i}.T" for i in range(n_jp)]
    us_list = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "GLD"]
    return jp + us_list[:n_us]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllocateCapital:
    def test_quantities_are_integers(self):
        tickers = _make_tickers()
        w = np.zeros(28)
        w[:4] = 0.25
        w[4:8] = -0.25
        opens = _make_opens(tickers, 1000.0)
        result = allocate_capital(w, tickers, opens, max_capital=1_000_000)
        assert result.quantities.dtype == np.int64 or result.quantities.dtype == int
        assert all(isinstance(int(q), int) for q in result.quantities)

    def test_zero_capital_returns_all_zeros(self):
        tickers = _make_tickers()
        w = np.ones(28) * 0.1
        opens = _make_opens(tickers, 1000.0)
        result = allocate_capital(w, tickers, opens, max_capital=0)
        assert np.all(result.quantities == 0)
        assert result.buy_budget == pytest.approx(0.0)
        assert result.sell_budget == pytest.approx(0.0)

    def test_gross_budget_equals_buy_plus_sell(self):
        tickers = _make_tickers()
        w = np.zeros(28)
        w[:7] = 0.1
        w[7:14] = -0.1
        opens = _make_opens(tickers, 1000.0)
        result = allocate_capital(w, tickers, opens, max_capital=5_000_000)
        assert result.gross_budget == pytest.approx(
            result.buy_budget + result.sell_budget, rel=1e-9
        )

    def test_buy_budget_proportional_to_buy_weights(self):
        tickers = _make_tickers()
        # 3 long, 1 short → buy budget should be 3x sell budget
        w = np.zeros(28)
        w[:3] = 1.0   # 3 units long
        w[3] = -1.0   # 1 unit short
        opens = _make_opens(tickers, 1000.0)
        result = allocate_capital(w, tickers, opens, max_capital=1_000_000)
        assert result.buy_budget == pytest.approx(result.sell_budget * 3.0, rel=1e-6)

    def test_allocated_amounts_non_negative(self):
        tickers = _make_tickers()
        rng = np.random.default_rng(0)
        w = rng.normal(0, 0.1, 28)
        opens = _make_opens(tickers, 2000.0)
        result = allocate_capital(w, tickers, opens, max_capital=3_000_000)
        assert np.all(result.allocated_amounts >= 0)

    def test_no_allocation_for_hold_weights(self):
        tickers = _make_tickers()
        w = np.zeros(28)  # all HOLD
        opens = _make_opens(tickers, 1000.0)
        result = allocate_capital(w, tickers, opens, max_capital=1_000_000)
        assert np.all(result.quantities == 0)

    def test_1629_uses_10_lot_size(self):
        """1629.T has a minimum 10-share lot size."""
        tickers = ["1629.T", "1618.T"]
        w = np.array([1.0, 0.0])
        opens = {"1629.T": 300.0, "1618.T": 1000.0}
        result = allocate_capital(w, tickers, opens, max_capital=100_000)
        # Quantity must be a multiple of 10
        qty_1629 = int(result.quantities[0])
        assert qty_1629 >= 0
        if qty_1629 > 0:
            assert qty_1629 % 10 == 0

    def test_output_length_matches_tickers(self):
        tickers = _make_tickers()
        w = np.zeros(28)
        opens = _make_opens(tickers)
        result = allocate_capital(w, tickers, opens, max_capital=1_000_000)
        assert len(result.quantities) == len(tickers)
        assert len(result.allocated_amounts) == len(tickers)
