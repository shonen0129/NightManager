"""Unit tests for compute_jp_target_returns and build_5m_910_prices.

Tests focus on the h-day 9:10-to-close target definition and its fallbacks:
- h=1 and h>1 target formulas
- 5m 9:10 price vs open-to-close fallback
- Invalid / zero open guard
- No lookahead (start-day only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.data.preprocessor import (
    build_5m_910_prices,
    compute_jp_target_returns,
)
from leadlag.data.tickers import JP_TICKERS


@pytest.fixture
def simple_df_exec() -> pd.DataFrame:
    """Small deterministic df_exec for target tests."""
    n = 10
    dates = pd.bdate_range("2025-01-05", periods=n)
    data = {"sig_date": dates, "is_provisional": 0}
    df = pd.DataFrame(data, index=dates)

    for i, tk in enumerate(JP_TICKERS):
        # deterministic open and oc so close is known
        df[f"jp_open_trade_{tk}"] = 100.0 + i
        df[f"jp_oc_{tk}"] = np.linspace(0.01, 0.05, n)
        df[f"jp_gap_{tk}"] = 0.0
    return df


class TestComputeJpTargetReturns:
    """Unit tests for compute_jp_target_returns."""

    def test_h1_with_p_910_matches_close_over_p_910(self, simple_df_exec):
        """h=1 with p_910 should be close / p_910 - 1."""
        n = len(simple_df_exec)
        values = np.tile(
            np.array([101.0 + i for i in range(len(JP_TICKERS))], dtype=float),
            (n, 1),
        )
        p_910 = pd.DataFrame(
            values,
            index=simple_df_exec.index,
            columns=JP_TICKERS,
        )

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=1, p_910_df=p_910
        )

        for i, tk in enumerate(JP_TICKERS):
            open_ = 100.0 + i
            oc = simple_df_exec[f"jp_oc_{tk}"].values
            close = (1.0 + oc) * open_
            p = 101.0 + i
            expected = close / p - 1.0
            np.testing.assert_allclose(y[:, i], expected, rtol=1e-12)

    def test_h3_with_p_910_uses_start_day(self, simple_df_exec):
        """h=3 should use p_910 from row i-2 as the start price."""
        n = len(simple_df_exec)
        p_910 = pd.DataFrame(
            np.nan, index=simple_df_exec.index, columns=JP_TICKERS, dtype=float
        )
        for j, date in enumerate(simple_df_exec.index):
            p_910.loc[date] = [90.0 + j * 0.1] * len(JP_TICKERS)

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=3, p_910_df=p_910
        )

        # First 2 rows are NaN
        assert np.all(np.isnan(y[:2]))

        for i, tk in enumerate(JP_TICKERS):
            open_ = 100.0 + i
            oc = simple_df_exec[f"jp_oc_{tk}"].values
            close = (1.0 + oc) * open_
            for row in range(2, n):
                start_idx = row - 2
                p_start = 90.0 + start_idx * 0.1
                expected = close[row] / p_start - 1.0
                assert np.isclose(y[row, i], expected, rtol=1e-12)

    def test_h3_falls_back_to_open_when_p_910_missing(self, simple_df_exec):
        """h=3 should fall back to start-day open when p_910 is NaN."""
        n = len(simple_df_exec)
        p_910 = pd.DataFrame(
            np.nan, index=simple_df_exec.index, columns=JP_TICKERS, dtype=float
        )

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=3, p_910_df=p_910
        )

        assert np.all(np.isnan(y[:2]))

        for i, tk in enumerate(JP_TICKERS):
            open_ = 100.0 + i
            oc = simple_df_exec[f"jp_oc_{tk}"].values
            close = (1.0 + oc) * open_
            for row in range(2, n):
                open_start = 100.0 + i
                expected = close[row] / open_start - 1.0
                assert np.isclose(y[row, i], expected, rtol=1e-12)

    def test_invalid_open_returns_zero(self, simple_df_exec):
        """Zero or NaN start-day open should yield 0.0, not inf/NaN."""
        simple_df_exec.loc[simple_df_exec.index[0], "jp_open_trade_1617.T"] = 0.0
        simple_df_exec.loc[simple_df_exec.index[1], "jp_open_trade_1617.T"] = np.nan

        p_910 = pd.DataFrame(
            np.nan, index=simple_df_exec.index, columns=JP_TICKERS, dtype=float
        )

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=3, p_910_df=p_910
        )

        # row 2 depends on start day 0 (zero open) → fallback to 0 open gives 0.0
        assert y[2, JP_TICKERS.index("1617.T")] == 0.0
        # row 3 depends on start day 1 (NaN open) → 0.0
        assert y[3, JP_TICKERS.index("1617.T")] == 0.0

    def test_h1_no_p_910_df_matches_jp_oc(self, simple_df_exec):
        """h=1 without 5m data falls back to open-to-close = jp_oc."""
        # No 5m cache, so p_910 is unavailable for every date.
        y = compute_jp_target_returns(simple_df_exec, JP_TICKERS, horizon=1)
        for i, tk in enumerate(JP_TICKERS):
            np.testing.assert_allclose(
                y[:, i], simple_df_exec[f"jp_oc_{tk}"].values, rtol=1e-12
            )

    def test_lookahead_safety(self, simple_df_exec):
        """Target for row i must not use any data from rows > i."""
        n = len(simple_df_exec)
        p_910 = pd.DataFrame(
            np.nan, index=simple_df_exec.index, columns=JP_TICKERS, dtype=float
        )
        # Give a distinct p_910 only for future rows; historical rows are NaN.
        for j in range(n // 2, n):
            p_910.iloc[j] = 80.0

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=3, p_910_df=p_910
        )

        # For rows that depend only on NaN start p_910 (future p_910 unused),
        # the fallback to open-to-close should still hold.
        for i, tk in enumerate(JP_TICKERS):
            open_ = 100.0 + i
            oc = simple_df_exec[f"jp_oc_{tk}"].values
            close = (1.0 + oc) * open_
            for row in range(2, n // 2):
                expected = close[row] / open_ - 1.0
                assert np.isclose(y[row, i], expected, rtol=1e-12)

    def test_p_910_df_reindex_aligns_with_df_exec(self, simple_df_exec):
        """p_910_df with extra/missing index rows is reindexed to df_exec."""
        p_910 = pd.DataFrame(
            np.nan,
            index=simple_df_exec.index[2:],
            columns=JP_TICKERS,
            dtype=float,
        )
        for j, date in enumerate(p_910.index):
            p_910.loc[date] = 95.0

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=1, p_910_df=p_910
        )

        # First 2 rows of p_910 are NaN (reindexed) → fallback to open
        for i, tk in enumerate(JP_TICKERS):
            open_ = 100.0 + i
            oc = simple_df_exec[f"jp_oc_{tk}"].values
            close = (1.0 + oc) * open_
            expected_first2 = close[:2] / open_ - 1.0
            np.testing.assert_allclose(y[:2, i], expected_first2, rtol=1e-12)

            # Remaining rows use p_910=95.0
            expected_rest = close[2:] / 95.0 - 1.0
            np.testing.assert_allclose(y[2:, i], expected_rest, rtol=1e-12)


class TestBuild5m910Prices:
    """Smoke tests for build_5m_910_prices."""

    def test_empty_5m_cache_returns_all_nan(self, simple_df_exec, monkeypatch):
        """When 5m cache is empty, p_910_df is all NaN."""
        from leadlag.data import cache as _cache

        def _mock_load(_interval):
            return pd.DataFrame()

        monkeypatch.setattr(_cache, "load_intraday_cache", _mock_load)
        p_910 = build_5m_910_prices(simple_df_exec, JP_TICKERS)

        assert p_910.index.equals(simple_df_exec.index)
        assert list(p_910.columns) == list(JP_TICKERS)
        assert p_910.isna().all().all()

    def test_zero_or_infinite_open_gives_zero_target(self, simple_df_exec):
        """Zero or inf open prices (data quality outliers) must not yield inf targets."""
        simple_df_exec.loc[simple_df_exec.index[2], "jp_open_trade_1617.T"] = 0.0
        simple_df_exec.loc[simple_df_exec.index[3], "jp_open_trade_1617.T"] = np.inf

        p_910 = pd.DataFrame(
            np.nan, index=simple_df_exec.index, columns=JP_TICKERS, dtype=float
        )

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=3, p_910_df=p_910
        )

        assert np.isfinite(y[2:, JP_TICKERS.index("1617.T")]).all()
        assert not np.isinf(y).any()

    def test_zero_open_with_p910_gives_zero_target(self, simple_df_exec):
        """Zero open with a positive p_910 must not produce -1 target."""
        simple_df_exec.loc[simple_df_exec.index[2], "jp_open_trade_1617.T"] = 0.0

        p_910 = pd.DataFrame(
            100.0, index=simple_df_exec.index, columns=JP_TICKERS, dtype=float
        )

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=3, p_910_df=p_910
        )

        assert not np.isinf(y).any()
        assert y[2, JP_TICKERS.index("1617.T")] == 0.0

    def test_infinite_oc_gives_finite_target(self, simple_df_exec):
        """An infinite open-to-close return must be guarded by the valid mask."""
        simple_df_exec.loc[simple_df_exec.index[2], "jp_oc_1617.T"] = np.inf

        p_910 = pd.DataFrame(
            100.0, index=simple_df_exec.index, columns=JP_TICKERS, dtype=float
        )

        y = compute_jp_target_returns(
            simple_df_exec, JP_TICKERS, horizon=3, p_910_df=p_910
        )

        assert not np.isinf(y).any()
        assert not np.isnan(y[2:, :]).any()
