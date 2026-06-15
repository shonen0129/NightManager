"""tests/unit/test_ticker_registry.py

Unit tests for data.ticker_registry — the single source of truth for
US/JP ticker definitions, count constants, and conversion utilities.
"""

from __future__ import annotations

import pytest

from leadlag.data.tickers import (
    JP_TICKERS,
    JP_TICKERS_WITH_TOPIX,
    N_JP,
    N_JP_ASSETS,
    N_TOTAL,
    N_TOTAL_ASSETS,
    N_US,
    N_US_ASSETS,
    TOPIX_TICKER,
    US_TICKERS,
    is_jp_ticker,
    is_us_ticker,
    kabu_to_yf,
    lot_size_for,
    yf_to_kabu,
)

# ---------------------------------------------------------------------------
# Ticker list properties
# ---------------------------------------------------------------------------


class TestTickerLists:
    def test_us_tickers_count(self):
        assert len(US_TICKERS) == 15

    def test_jp_tickers_count(self):
        assert len(JP_TICKERS) == 17

    def test_topix_ticker_value(self):
        assert TOPIX_TICKER == "1306.T"

    def test_jp_tickers_with_topix_count(self):
        assert len(JP_TICKERS_WITH_TOPIX) == 18
        assert TOPIX_TICKER in JP_TICKERS_WITH_TOPIX

    def test_us_tickers_are_strings(self):
        assert all(isinstance(t, str) for t in US_TICKERS)

    def test_jp_tickers_end_with_dot_t(self):
        assert all(t.endswith(".T") for t in JP_TICKERS)

    def test_jp_tickers_no_duplicates(self):
        assert len(JP_TICKERS) == len(set(JP_TICKERS))

    def test_us_tickers_no_duplicates(self):
        assert len(US_TICKERS) == len(set(US_TICKERS))

    def test_topix_not_in_jp_tickers(self):
        # TOPIX is managed separately, not in the 17-ETF list
        assert TOPIX_TICKER not in JP_TICKERS


# ---------------------------------------------------------------------------
# Count constants
# ---------------------------------------------------------------------------


class TestCountConstants:
    def test_n_us(self):
        assert N_US == 15

    def test_n_jp(self):
        assert N_JP == 17

    def test_n_total(self):
        assert N_TOTAL == 32

    def test_n_us_alias(self):
        assert N_US_ASSETS == N_US

    def test_n_jp_alias(self):
        assert N_JP_ASSETS == N_JP

    def test_n_total_alias(self):
        assert N_TOTAL_ASSETS == N_TOTAL

    def test_n_total_is_sum(self):
        assert N_TOTAL == N_US + N_JP


# ---------------------------------------------------------------------------
# yf_to_kabu / kabu_to_yf
# ---------------------------------------------------------------------------


class TestTickerConversion:
    @pytest.mark.parametrize(
        "yf_code, kabu_code",
        [
            ("1617.T", "1617"),
            ("1629.T", "1629"),
            ("1306.T", "1306"),
        ],
    )
    def test_yf_to_kabu(self, yf_code, kabu_code):
        assert yf_to_kabu(yf_code) == kabu_code

    @pytest.mark.parametrize(
        "kabu_code, yf_code",
        [
            ("1617", "1617.T"),
            ("1629", "1629.T"),
            ("1306", "1306.T"),
        ],
    )
    def test_kabu_to_yf_jp(self, kabu_code, yf_code):
        assert kabu_to_yf(kabu_code) == yf_code

    def test_kabu_to_yf_us_unchanged(self):
        # US tickers have no .T suffix and pass through unchanged
        assert kabu_to_yf("XLB") == "XLB"
        assert kabu_to_yf("XLK") == "XLK"

    def test_roundtrip_jp(self):
        for tk in JP_TICKERS:
            assert kabu_to_yf(yf_to_kabu(tk)) == tk


# ---------------------------------------------------------------------------
# is_jp_ticker / is_us_ticker
# ---------------------------------------------------------------------------


class TestTickerClassification:
    def test_jp_tickers_are_jp(self):
        for tk in JP_TICKERS:
            assert is_jp_ticker(tk), f"{tk} should be JP"

    def test_topix_is_jp(self):
        assert is_jp_ticker(TOPIX_TICKER)

    def test_us_tickers_are_us(self):
        for tk in US_TICKERS:
            assert is_us_ticker(tk), f"{tk} should be US"

    def test_jp_tickers_are_not_us(self):
        for tk in JP_TICKERS:
            assert not is_us_ticker(tk), f"{tk} should NOT be US"

    def test_us_tickers_are_not_jp(self):
        for tk in US_TICKERS:
            assert not is_jp_ticker(tk), f"{tk} should NOT be JP"


# ---------------------------------------------------------------------------
# lot_size_for
# ---------------------------------------------------------------------------


class TestLotSize:
    def test_1629_lot_size_is_10(self):
        # 1629.T has special 10-share lot size
        assert lot_size_for("1629.T") == 10

    def test_standard_jp_lot_size_is_1(self):
        for tk in JP_TICKERS:
            if tk != "1629.T":
                assert lot_size_for(tk) == 1, f"{tk} should have lot size 1"

    def test_us_ticker_lot_size_is_1(self):
        for tk in US_TICKERS:
            assert lot_size_for(tk) == 1
