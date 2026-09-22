"""Shared multi-day return transformation for Step 2 and on-demand V2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, US_TICKERS


def compute_cumulative_returns(
    df_exec: pd.DataFrame,
    horizon: int,
    *,
    method: str = "cumprod",
) -> pd.DataFrame:
    """Return a copy with h-day cumulative return columns.

    The transformation is shared by the research gap generator and the
    production on-demand path so h=3/5 use identical historical inputs.
    """
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    if method not in {"cumprod", "sum"}:
        raise ValueError("method must be 'cumprod' or 'sum'")
    result = df_exec.copy(deep=True)
    if horizon == 1:
        return result
    columns = [
        *(f"us_cc_{ticker}" for ticker in US_TICKERS),
        *(f"jp_oc_{ticker}" for ticker in JP_TICKERS),
        *(f"jp_gap_{ticker}" for ticker in JP_TICKERS),
        "topix_night_return",
        "topix_oc_return",
        "topix_cc_trade",
    ]
    for column in columns:
        if column not in result.columns:
            continue
        window = result[column].rolling(horizon, min_periods=horizon)
        if method == "cumprod":
            result[column] = window.apply(lambda values: np.prod(1.0 + values) - 1.0, raw=True)
        else:
            result[column] = window.sum()
    return result


__all__ = ["compute_cumulative_returns"]
