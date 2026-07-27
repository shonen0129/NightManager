"""Unit tests for leadlag.core.pipeline build_common_inputs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from leadlag.core.pipeline import build_common_inputs
from leadlag.data.tickers import JP_TICKERS, US_TICKERS


def _make_df_exec(n_rows: int = 1500) -> pd.DataFrame:
    """Return a synthetic df_exec that covers the baseline period."""
    dates = pd.bdate_range("2009-06-01", periods=n_rows)
    df = pd.DataFrame(index=dates)
    rng = np.random.default_rng(42)

    for tk in US_TICKERS:
        df[f"us_cc_{tk}"] = rng.standard_normal(n_rows) * 0.01

    for tk in JP_TICKERS:
        df[f"jp_gap_{tk}"] = rng.standard_normal(n_rows) * 0.01
        df[f"jp_beta_{tk}"] = rng.standard_normal(n_rows) * 0.5
        df[f"jp_oc_{tk}"] = rng.standard_normal(n_rows) * 0.01

    df["topix_night_return"] = rng.standard_normal(n_rows) * 0.01
    df["topix_oc_return"] = rng.standard_normal(n_rows) * 0.01
    df["topix_cc_trade"] = (
        (1.0 + df["topix_night_return"]) * (1.0 + df["topix_oc_return"]) - 1.0
    )

    return df


def test_build_common_inputs_frac_diff_nan_warning(caplog: pytest.LogCaptureFixture) -> None:
    """NaNs in fractional-diff output must be filled with 0.0 and logged.

    This is a regression test for the 2026-07-27 production outage, where
    `build_common_inputs` called `logger.warning(...)` without `logger` being
    defined, raising `NameError` and aborting the gap-distribution step.
    """
    df_exec = _make_df_exec()
    n_u = len(US_TICKERS)
    n_j = len(JP_TICKERS)
    rng = np.random.default_rng(43)
    y_jp_target = rng.standard_normal((len(df_exec), n_j)) * 0.01

    # Inject NaNs into the tail of the US return series. These propagate
    # through the fractional-differencing filter and trigger the fill/warning
    # path that previously raised NameError when `logger` was undefined.
    n_nan = 10
    for tk in US_TICKERS:
        df_exec.loc[df_exec.index[-n_nan:], f"us_cc_{tk}"] = np.nan

    with caplog.at_level(logging.WARNING, logger="leadlag.core.pipeline"):
        inputs = build_common_inputs(
            df_exec=df_exec,
            y_jp_target=y_jp_target,
            n_u=n_u,
            n_j=n_j,
            ewma_half_life=30.0,
            beta_window=60,
            include_v4_prior=False,
            frac_diff_enabled=True,
            frac_diff_d=0.1,
            frac_diff_threshold=1e-5,
            frac_diff_window=100,
            frac_diff_normalize=None,
        )

    assert inputs.all_returns_raw.shape == (len(df_exec), n_u + n_j)
    assert not np.isnan(inputs.all_returns_raw[:, :n_u]).any()
    assert (
        "NaN values in fractional diff output will be filled with 0.0" in caplog.text
    )
