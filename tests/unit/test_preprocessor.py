"""Tests for the data preprocessor."""
from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS


def _make_raw(
    dates: pd.DatetimeIndex,
    us_values: float = 1.0,
    jp_open_values: float = 100.0,
    jp_close_values: float = 100.0,
) -> dict:
    us_close = pd.DataFrame(
        {tk: [us_values] * len(dates) for tk in US_TICKERS}, index=dates
    )
    jp_close = pd.DataFrame(
        {tk: [jp_close_values] * len(dates) for tk in JP_TICKERS},
        index=dates,
    )
    jp_close[TOPIX_TICKER] = jp_close_values
    jp_open = pd.DataFrame(
        {tk: [jp_open_values] * len(dates) for tk in JP_TICKERS},
        index=dates,
    )
    jp_open[TOPIX_TICKER] = jp_open_values
    return {"us_close": us_close, "jp_close": jp_close, "jp_open": jp_open}


def test_zero_open_does_not_create_inf_oc():
    """A zero JP open price must not create inf target returns."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    raw = _make_raw(dates, jp_open_values=100.0, jp_close_values=101.0)
    # Make one ticker open at 0.0 on the second day
    raw["jp_open"].loc[dates[1], "1619.T"] = 0.0

    df_exec = preprocess_data(raw)

    assert not np.isinf(df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].values).any()
    assert not np.isinf(df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values).any()
    # The broken ticker row should be skipped; if present it must be finite.
    if "jp_oc_1619.T" in df_exec.columns:
        assert np.isfinite(df_exec["jp_oc_1619.T"].values).all()


def test_zero_prev_close_does_not_create_inf_gap():
    """A zero previous close must not create inf gap returns."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    raw = _make_raw(dates, jp_open_values=100.0, jp_close_values=101.0)
    raw["jp_close"].loc[dates[0], "1619.T"] = 0.0

    df_exec = preprocess_data(raw)

    assert not np.isinf(df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values).any()


def test_zero_topix_open_does_not_create_inf_oc():
    """A zero TOPIX open must not create inf topix_oc_return."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    raw = _make_raw(dates, jp_open_values=100.0, jp_close_values=101.0)
    raw["jp_open"].loc[dates[1], TOPIX_TICKER] = 0.0

    df_exec = preprocess_data(raw)

    assert not np.isinf(df_exec["topix_oc_return"].values).any()
    assert not np.isinf(df_exec["topix_cc_trade"].values).any()
