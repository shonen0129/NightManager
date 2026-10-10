"""Pure JP target arithmetic. Intraday acquisition belongs to data.intraday_inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_one_day_target_returns(
    df_exec: pd.DataFrame,
    jp_tickers: list[str],
    open_910_returns: pd.DataFrame | dict[pd.Timestamp, dict[str, float]] | None = None,
) -> np.ndarray:
    """Compute one-day targets directly from open-to-09:10 returns.

    Missing 09:10 observations (NaN) fall back to the valid daily open, while
    explicitly invalid observations and invalid realized prices remain NaN.
    """
    jp_oc = df_exec[[f"jp_oc_{tk}" for tk in jp_tickers]].to_numpy(dtype=float)
    opens = df_exec[[f"jp_open_trade_{tk}" for tk in jp_tickers]].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        closes = opens * (1.0 + jp_oc)

    if open_910_returns is None:
        raise ValueError("h=1 target calculation requires explicit open_910_returns")

    if isinstance(open_910_returns, pd.DataFrame):
        returns_df = open_910_returns.reindex(index=df_exec.index, columns=jp_tickers)
    else:
        returns_df = pd.DataFrame(open_910_returns).T.reindex(
            index=df_exec.index, columns=jp_tickers
        )
    adjusted = returns_df.to_numpy(dtype=float)

    realized_valid = (
        np.isfinite(opens)
        & (opens > 0)
        & np.isfinite(jp_oc)
        & np.isfinite(closes)
        & (closes > 0)
    )
    quote_missing = np.isnan(adjusted)
    quote_valid = np.isfinite(adjusted) & (adjusted > -1.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        observed_target = (1.0 + jp_oc) / (1.0 + adjusted) - 1.0

    y_jp_target = np.full_like(jp_oc, np.nan, dtype=float)
    fallback_valid = realized_valid & quote_missing
    y_jp_target[fallback_valid] = jp_oc[fallback_valid]

    observed_valid = realized_valid & quote_valid & np.isfinite(observed_target)
    y_jp_target[observed_valid] = observed_target[observed_valid]
    return y_jp_target


def _compute_jp_target_returns_h(
    df_exec: pd.DataFrame,
    jp_tickers: list[str],
    horizon: int,
    p_910_df: pd.DataFrame | None,
    open_910_returns: pd.DataFrame | dict[pd.Timestamp, dict[str, float]] | None = None,
) -> np.ndarray:
    """Compute the h-day 9:10-to-close target return.

    For each row ``i`` (trade date) and ticker, the target is defined as:

        y_h[i, tk] = close_i / p_910_{i-h+1} - 1

    where ``close_i`` is derived from the h=1 open-to-close return and open price,
    and ``p_910_{i-h+1}`` is the 9:10 midpoint price on the starting day of the
    h-day window.  If ``p_910`` is unavailable for the start day, an explicit
    open-to-09:10 return is used to reconstruct it; when that is also unavailable,
    the open price on the start day is used.

    The first ``horizon - 1`` rows are NaN because the window is not yet complete.
    """
    n = len(df_exec)
    m = len(jp_tickers)
    open_cols = [f"jp_open_trade_{tk}" for tk in jp_tickers]
    oc_cols = [f"jp_oc_{tk}" for tk in jp_tickers]

    open_arr = df_exec[open_cols].values.astype(float)
    oc_arr = df_exec[oc_cols].values.astype(float)
    close_arr = (1.0 + oc_arr) * open_arr

    # Resolve the start-day 09:10 price while preserving the distinction between
    # a missing quote (NaN, eligible for daily-open fallback) and an explicitly
    # invalid quote (zero/negative/non-finite, which must remain invalid).
    p_910_arr = np.full((n, m), np.nan)
    fallback_to_open = np.ones((n, m), dtype=bool)
    quote_invalid = np.zeros((n, m), dtype=bool)

    if p_910_df is not None and not p_910_df.empty:
        aligned = p_910_df.reindex(index=df_exec.index, columns=jp_tickers)
        direct = aligned.to_numpy(dtype=float)
        direct_missing = np.isnan(direct)
        direct_valid = np.isfinite(direct) & (direct > 0)
        direct_invalid = ~direct_missing & ~direct_valid
        p_910_arr[direct_valid] = direct[direct_valid]
        quote_invalid |= direct_invalid
        fallback_to_open = direct_missing

    # The production adapter owns the 09:10 cache and exposes the explicit
    # open-to-09:10 return frame.  A directly supplied valid p_910_df takes
    # precedence.  Only a genuinely missing observation may fall back to open.
    if open_910_returns is not None:
        if isinstance(open_910_returns, pd.DataFrame):
            returns_df = open_910_returns.reindex(
                index=df_exec.index, columns=jp_tickers
            )
        else:
            returns_df = pd.DataFrame(open_910_returns).T.reindex(
                index=df_exec.index, columns=jp_tickers
            )
        open_to_910 = returns_df.to_numpy(dtype=float)
        observation_missing = np.isnan(open_to_910)
        with np.errstate(over="ignore", invalid="ignore"):
            derived_p_910 = open_arr * (1.0 + open_to_910)
        observation_valid = (
            np.isfinite(open_to_910)
            & (open_to_910 > -1.0)
            & np.isfinite(derived_p_910)
            & (derived_p_910 > 0)
            & np.isfinite(open_arr)
            & (open_arr > 0)
        )
        eligible = fallback_to_open & ~quote_invalid
        derived_use = eligible & observation_valid
        p_910_arr[derived_use] = derived_p_910[derived_use]
        quote_invalid |= eligible & ~observation_missing & ~observation_valid
        fallback_to_open = eligible & observation_missing

    # start-day arrays, shifted by (horizon - 1) rows
    p_start = np.full((n, m), np.nan)
    open_start = np.full((n, m), np.nan)
    start_fallback = np.zeros((n, m), dtype=bool)
    start_invalid = np.zeros((n, m), dtype=bool)
    if n >= horizon:
        source = slice(0, n - horizon + 1)
        target = slice(horizon - 1, None)
        p_start[target] = p_910_arr[source]
        open_start[target] = open_arr[source]
        start_fallback[target] = fallback_to_open[source]
        start_invalid[target] = quote_invalid[source]

    p_use = np.where(
        np.isfinite(p_start) & (p_start > 0),
        p_start,
        np.where(start_fallback & ~start_invalid, open_start, np.nan),
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = close_arr / p_use - 1.0

    realized_valid = (
        np.isfinite(open_arr)
        & (open_arr > 0)
        & np.isfinite(oc_arr)
        & np.isfinite(close_arr)
        & (close_arr > 0)
    )
    valid = (
        realized_valid
        & np.isfinite(p_use)
        & (p_use > 0)
        & ~start_invalid
        & np.isfinite(y)
    )
    y = np.where(valid, y, np.nan)

    # First (horizon - 1) rows have incomplete windows.
    if horizon > 1:
        y[: horizon - 1] = np.nan

    return y


def compute_jp_target_returns(
    df_exec: pd.DataFrame,
    jp_tickers: list[str],
    horizon: int = 1,
    p_910_df: pd.DataFrame | None = None,
    open_910_returns: pd.DataFrame | dict[pd.Timestamp, dict[str, float]] | None = None,
) -> np.ndarray:
    """Compute 9:10-to-close returns for JP assets, with Open-to-Close as fallback.

    Args:
        df_exec: Execution DataFrame with ``jp_oc_*`` and ``jp_open_trade_*``.
        jp_tickers: JP tickers to compute targets for.
        horizon: Number of trading days in the target window.  Defaults to 1,
            using explicit open-to-09:10 returns when prices are not supplied.
        p_910_df: Optional pre-built 9:10 midpoint prices (date × ticker).  When
            ``horizon > 1`` this is used to compute the start-day 9:10 price.  When
            both ``horizon == 1`` and ``p_910_df is None``, the return-based
            one-day calculation requires explicit ``open_910_returns``.
        open_910_returns: Explicit open-to-09:10 returns.  For ``horizon > 1``
            these reconstruct the start-day 9:10 price when ``p_910_df`` is
            missing or has a missing value.

    Returns:
        Array of target returns, shape (n_rows, n_tickers).  Leading ``horizon - 1``
        rows are NaN for ``horizon > 1``. Missing/invalid realized prices and
        explicitly invalid 09:10 observations remain NaN; a missing 09:10 quote
        may fall back to a finite positive daily open.
    """
    if horizon == 1 and p_910_df is None:
        return _compute_one_day_target_returns(
            df_exec, jp_tickers, open_910_returns=open_910_returns
        )
    return _compute_jp_target_returns_h(
        df_exec,
        jp_tickers,
        horizon,
        p_910_df,
        open_910_returns=open_910_returns,
    )
