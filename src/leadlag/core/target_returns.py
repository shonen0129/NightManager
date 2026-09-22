"""Pure JP target arithmetic. Intraday acquisition belongs to data.intraday_inputs."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def _compute_jp_target_returns_h1_legacy(
    df_exec: pd.DataFrame,
    jp_tickers: list[str],
    open_910_returns: pd.DataFrame | dict[pd.Timestamp, dict[str, float]] | None = None,
) -> np.ndarray:
    """Legacy h=1 9:10-to-close target computation preserved for exact backward compat.

    The arithmetic is kept separate from cache access so callers can provide the
    extracted values while retaining the exact historical definition.
    """
    jp_oc = df_exec[[f"jp_oc_{tk}" for tk in jp_tickers]].values
    y_jp_target = jp_oc.copy()

    if open_910_returns is None:
        raise ValueError("h=1 target calculation requires explicit open_910_returns")

    if isinstance(open_910_returns, pd.DataFrame):
        returns_df = open_910_returns.reindex(index=df_exec.index, columns=jp_tickers)
    else:
        returns_df = pd.DataFrame(open_910_returns).T.reindex(
            index=df_exec.index, columns=jp_tickers
        )
    adjusted = returns_df.to_numpy(dtype=float)
    valid = np.isfinite(adjusted)
    for t_idx in range(len(jp_tickers)):
        y_jp_target[:, t_idx] = np.where(
            valid[:, t_idx],
            (1.0 + jp_oc[:, t_idx]) / (1.0 + adjusted[:, t_idx]) - 1.0,
            y_jp_target[:, t_idx],
        )
    return cast(np.ndarray, y_jp_target)


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

    p_910_arr = np.full((n, m), np.nan)
    if p_910_df is not None and not p_910_df.empty:
        aligned = p_910_df.reindex(index=df_exec.index, columns=jp_tickers)
        p_910_arr = aligned.values.astype(float)

    # The production adapter owns the 09:10 cache and exposes the explicit
    # open-to-09:10 return frame.  For h>1, reconstruct the start-day 09:10
    # price from that return so on-demand fallback uses the same target
    # definition as the precomputed gap path.  A directly supplied p_910_df
    # takes precedence; invalid or missing values retain the open fallback.
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
        with np.errstate(over="ignore", invalid="ignore"):
            derived_p_910 = open_arr * (1.0 + open_to_910)
        derived_valid = (
            np.isfinite(derived_p_910)
            & (derived_p_910 > 0)
            & np.isfinite(open_arr)
            & (open_arr > 0)
        )
        direct_valid = np.isfinite(p_910_arr) & (p_910_arr > 0)
        p_910_arr = np.where(
            direct_valid,
            p_910_arr,
            np.where(derived_valid, derived_p_910, p_910_arr),
        )

    # start-day arrays, shifted by (horizon - 1) rows
    p_start = np.full((n, m), np.nan)
    open_start = np.full((n, m), np.nan)
    if n >= horizon:
        p_start[horizon - 1 :] = p_910_arr[: n - horizon + 1]
        open_start[horizon - 1 :] = open_arr[: n - horizon + 1]

    # Use p_910 when available and valid; otherwise fall back to open.
    p_use = np.where(
        np.isfinite(p_start) & (p_start > 0),
        p_start,
        open_start,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        y = close_arr / p_use - 1.0

    # Guard against invalid / zero denominators, NaN/Inf close values, and
    # non-positive open prices (which make both close and the target undefined).
    valid = (
        np.isfinite(p_use)
        & (p_use > 0)
        & np.isfinite(close_arr)
        & np.isfinite(open_arr)
        & (open_arr > 0)
        & np.isfinite(y)
    )
    y = np.where(valid, y, 0.0)

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
            which preserves the legacy h=1 definition for callers that do not
            pass ``p_910_df``.
        p_910_df: Optional pre-built 9:10 midpoint prices (date × ticker).  When
            ``horizon > 1`` this is used to compute the start-day 9:10 price.  When
            both ``horizon == 1`` and ``p_910_df is None``, the legacy h=1 path is
            used to guarantee backward compatibility.
        open_910_returns: Explicit open-to-09:10 returns.  For ``horizon > 1``
            these reconstruct the start-day 9:10 price when ``p_910_df`` is
            missing or has a missing value.

    Returns:
        Array of target returns, shape (n_rows, n_tickers).  Leading ``horizon - 1``
        rows are NaN for ``horizon > 1``.
    """
    if horizon == 1 and p_910_df is None:
        return _compute_jp_target_returns_h1_legacy(
            df_exec, jp_tickers, open_910_returns=open_910_returns
        )
    return _compute_jp_target_returns_h(
        df_exec,
        jp_tickers,
        horizon,
        p_910_df,
        open_910_returns=open_910_returns,
    )
