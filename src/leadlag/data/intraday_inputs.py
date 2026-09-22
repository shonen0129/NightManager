"""Intraday input adapters for 09:10 execution and target labels.

The adapter owns cache reads and the existing missing-bar fallbacks.  The
target arithmetic lives in :mod:`leadlag.core.target_returns` and accepts the
extracted open-to-09:10 returns as an explicit input.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.data.market_data_cache import load_intraday_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.utils.timestamps import (
    normalize_jst_date,
    normalize_jst_index,
    normalize_jst_timestamp,
)


def _normalize_bars_index(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Return 5-minute bars indexed by timezone-naive JST timestamps."""
    bars = df_5m.copy()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index)
    bars.index = pd.DatetimeIndex([normalize_jst_timestamp(value) for value in bars.index])
    return bars


def _execution_positions_by_jst_date(df_exec: pd.DataFrame) -> dict[pd.Timestamp, list[int]]:
    """Map each execution row to its JST calendar date without changing its index."""
    positions: dict[pd.Timestamp, list[int]] = {}
    for position, value in enumerate(df_exec.index):
        date = normalize_jst_date(value)
        positions.setdefault(date, []).append(position)
    return positions


def _day_data(df_5m: pd.DataFrame, dt: object) -> pd.DataFrame:
    """Return one calendar day's bars using the cache's existing date rule."""
    date = normalize_jst_date(dt)
    return df_5m[df_5m.index.normalize() == date]


def build_5m_910_prices(
    df_exec: pd.DataFrame,
    tickers: list[str] | None = None,
    df_5m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build trade-date × ticker 09:10 midpoint prices from 5m bars."""
    selected = list(JP_TICKERS if tickers is None else tickers)
    p_910 = pd.DataFrame(np.nan, index=df_exec.index, columns=selected, dtype=float)
    bars = load_intraday_cache("5m") if df_5m is None else df_5m
    if bars is None or bars.empty:
        return p_910
    bars = _normalize_bars_index(bars)
    execution_positions = _execution_positions_by_jst_date(df_exec)

    for dt_ts in pd.DatetimeIndex(bars.index).normalize().unique():
        positions = execution_positions.get(dt_ts)
        if not positions:
            continue
        day_data = _day_data(bars, dt_ts)
        idx_910 = dt_ts + pd.Timedelta(hours=9, minutes=10)
        if idx_910 not in day_data.index:
            continue
        row_910 = day_data.loc[idx_910]
        for column_index, ticker in enumerate(selected):
            high = row_910.get(("High", ticker))
            low = row_910.get(("Low", ticker))
            close = row_910.get(("Close", ticker))
            value = (high + low) / 2 if pd.notna(high) and pd.notna(low) else close
            if pd.notna(value) and np.isfinite(value):
                for position in positions:
                    p_910.iat[position, column_index] = float(value)
    return p_910


def build_open_910_returns(
    df_exec: pd.DataFrame,
    tickers: list[str] | None = None,
    df_5m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract open-to-09:10 returns using the execution-frame open.

    The denominator is ``jp_open_trade_*`` when it is available.  This keeps
    the explicit return compatible with the open used to reconstruct close
    prices in :mod:`leadlag.core.target_returns`.  A valid 5-minute open is
    retained as a compatibility fallback when the execution frame has no
    daily-open column.  Missing bars, prices, and non-positive prices remain
    NaN so the target arithmetic can apply its existing open-to-close fallback.
    Returning a date-indexed frame makes the input explicit and keeps the
    calculation layer free of cache access.
    """
    selected = list(JP_TICKERS if tickers is None else tickers)
    returns = pd.DataFrame(np.nan, index=df_exec.index, columns=selected, dtype=float)
    bars = load_intraday_cache("5m") if df_5m is None else df_5m
    if bars is None or bars.empty:
        return returns
    bars = _normalize_bars_index(bars)
    execution_positions = _execution_positions_by_jst_date(df_exec)

    for dt_ts in pd.DatetimeIndex(bars.index).normalize().unique():
        positions = execution_positions.get(dt_ts)
        if not positions:
            continue
        day_data = _day_data(bars, dt_ts)
        idx_910 = dt_ts + pd.Timedelta(hours=9, minutes=10)
        row_910 = day_data.loc[idx_910] if idx_910 in day_data.index else None
        for column_index, ticker in enumerate(selected):
            p_910 = np.nan
            if row_910 is not None:
                high = row_910.get(("High", ticker))
                low = row_910.get(("Low", ticker))
                close = row_910.get(("Close", ticker))
                if pd.notna(high) and pd.notna(low) and np.isfinite(high) and np.isfinite(low):
                    p_910 = (high + low) / 2
                elif pd.notna(close) and np.isfinite(close):
                    p_910 = close

            p_open_5m = np.nan
            for time_str in ("09:00:00", "09:05:00", "09:10:00"):
                idx_time = dt_ts + pd.Timedelta(time_str)
                if idx_time not in day_data.index:
                    continue
                row_time = day_data.loc[idx_time]
                op = row_time.get(("Open", ticker))
                cl = row_time.get(("Close", ticker))
                value = op if pd.notna(op) else cl
                if pd.notna(value) and np.isfinite(value):
                    p_open_5m = value
                    break

            # ``jp_oc_*`` and the target calculator use this execution-frame
            # open to reconstruct the close.  Use the same denominator for the
            # open-to-09:10 return; mixing it with an independently sourced
            # 5-minute open makes on-demand h=3/5 differ from direct p_910.
            p_exec_open = np.nan
            open_col = f"jp_open_trade_{ticker}"
            if open_col in df_exec.columns:
                for position in positions:
                    candidate = df_exec.iloc[position][open_col]
                    if np.isscalar(candidate) and pd.notna(candidate):
                        try:
                            candidate_float = float(cast(Any, candidate))
                        except (TypeError, ValueError):
                            candidate_float = np.nan
                        if np.isfinite(candidate_float) and candidate_float > 0:
                            p_exec_open = candidate_float
                            break

            denominator = p_exec_open if np.isfinite(p_exec_open) else p_open_5m
            if (
                pd.notna(p_910)
                and np.isfinite(p_910)
                and p_910 > 0
                and pd.notna(denominator)
                and np.isfinite(denominator)
                and denominator > 0
            ):
                for position in positions:
                    returns.iat[position, column_index] = float(p_910 / denominator - 1.0)
    return returns


def resolve_execution_prices(
    df_exec: pd.DataFrame,
    open_910_returns: pd.DataFrame,
    tickers: list[str] | None = None,
    required_index: pd.Index | list[object] | tuple[object, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve observed 09:10 prices or explicitly identified daily-open fallbacks.

    NaN represents an absent observation and may use a finite positive open.
    Infinite/impossible observations, absent dates/columns and absent opens
    are invalid inputs, not missing quotes to conceal. The raw return frame
    is never filled or changed, so cache fingerprints retain missingness.
    """
    selected = list(JP_TICKERS if tickers is None else tickers)
    if not isinstance(open_910_returns, pd.DataFrame) or open_910_returns.empty:
        raise ValueError("execution prices require explicit open_910_returns")
    frame = df_exec.copy(deep=False)
    observations = open_910_returns.copy(deep=False)
    frame.index = normalize_jst_index(frame.index)
    observations.index = normalize_jst_index(observations.index)
    if frame.index.has_duplicates or observations.index.has_duplicates:
        raise ValueError("execution price dates must be unique")
    dates = frame.index if required_index is None else pd.DatetimeIndex(
        [normalize_jst_date(value) for value in required_index]
    )
    if dates.empty or dates.hasnans or not dates.isin(frame.index).all() or not dates.isin(observations.index).all():
        raise ValueError("execution price snapshot must include requested dates")
    open_columns = [f"jp_open_trade_{ticker}" for ticker in selected]
    if not selected or not set(open_columns).issubset(frame.columns) or not set(selected).issubset(observations.columns):
        raise ValueError("execution price snapshot must include every requested ticker")
    opens = frame.loc[dates, open_columns].to_numpy(dtype=float)
    returns = observations.loc[dates, selected].to_numpy(dtype=float)
    if not (np.isfinite(opens) & (opens > 0)).all():
        raise ValueError("execution prices require finite positive daily opens")
    missing = np.isnan(returns)
    if not (missing | (np.isfinite(returns) & (returns > -1.0))).all():
        raise ValueError("invalid open-to-09:10 observation")
    with np.errstate(over="ignore", invalid="ignore"):
        prices = np.where(missing, opens, opens * (1.0 + returns))
    if not (np.isfinite(prices) & (prices > 0)).all():
        raise ValueError("resolved execution prices must be finite and positive")
    sources = np.where(missing, "daily_open_fallback", "observed_0910")
    return (
        pd.DataFrame(prices, index=dates, columns=selected),
        pd.DataFrame(sources, index=dates, columns=selected),
    )


def has_usable_execution_prices(
    open_910_returns: pd.DataFrame | None,
    df_exec: pd.DataFrame | None,
    tickers: list[str] | None = None,
    required_index: pd.Index | list[object] | tuple[object, ...] | None = None,
) -> bool:
    """Check explicit observed/fallback prices without any cache or provider I/O."""
    if df_exec is None or open_910_returns is None:
        return False
    if has_valid_open_910_returns(open_910_returns, df_exec, tickers, required_index):
        return True
    try:
        resolve_execution_prices(df_exec, open_910_returns, tickers, required_index)
    except (TypeError, ValueError, KeyError, OverflowError):
        return False
    return True


def has_valid_open_910_returns(
    open_910_returns: pd.DataFrame | None,
    df_exec: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    required_index: pd.Index | list[object] | tuple[object, ...] | None = None,
) -> bool:
    """Return whether an explicit 09:10 return frame is complete and finite.

    Strict typed runs must not treat an all-NaN (or partially missing) frame as
    proof that the 09:10 observation was supplied.  When ``required_index`` is
    supplied, only those decision dates are checked; historical rows may remain
    sparse because target arithmetic has an explicit per-cell fallback.  The
    full ``df_exec`` index remains the default for compatibility callers.
    """
    if not isinstance(open_910_returns, pd.DataFrame) or open_910_returns.empty:
        return False
    selected = list(JP_TICKERS if tickers is None else tickers)
    if not selected or any(ticker not in open_910_returns.columns for ticker in selected):
        return False
    aligned = open_910_returns.reindex(columns=selected)
    if required_index is not None:
        from leadlag.utils.timestamps import normalize_jst_date, normalize_jst_index

        try:
            required = [normalize_jst_date(value) for value in required_index]
            aligned.index = normalize_jst_index(aligned.index)
        except (TypeError, ValueError, OverflowError, OSError):
            return False
        aligned = aligned.reindex(required)
    elif df_exec is not None:
        if not isinstance(df_exec, pd.DataFrame) or df_exec.empty:
            return False
        aligned = aligned.reindex(index=df_exec.index)
    try:
        values = aligned.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(values.size and np.isfinite(values).all() and (values > -1.0).all())


def compute_jp_target_returns(
    df_exec: pd.DataFrame,
    jp_tickers: list[str],
    horizon: int = 1,
    p_910_df: pd.DataFrame | None = None,
    open_910_returns: pd.DataFrame | dict[pd.Timestamp, dict[str, float]] | None = None,
    allow_implicit_io: bool = True,
    required_index: pd.Index | list[object] | tuple[object, ...] | None = None,
) -> np.ndarray:
    """Delegate target arithmetic using explicit intraday inputs.

    ``open_910_returns`` is accepted from a run-owned snapshot.  When it is
    omitted, this compatibility adapter may load the local 5-minute cache only
    when ``allow_implicit_io`` is true.  Strict typed-input paths fail closed
    instead of reopening a cache owned by another run.
    """
    from leadlag.core.target_returns import compute_jp_target_returns as _compute
    compute_frame = df_exec
    compute_open_910 = open_910_returns
    compute_p_910 = p_910_df
    if required_index is not None:
        from leadlag.utils.timestamps import normalize_jst_index

        compute_frame = df_exec.copy()
        compute_frame.index = normalize_jst_index(compute_frame.index)
        if isinstance(open_910_returns, pd.DataFrame):
            compute_open_910 = open_910_returns.copy()
            compute_open_910.index = normalize_jst_index(compute_open_910.index)
        if isinstance(p_910_df, pd.DataFrame):
            compute_p_910 = p_910_df.copy()
            compute_p_910.index = normalize_jst_index(compute_p_910.index)

    open_910 = open_910_returns
    if p_910_df is None and open_910 is None:
        if not allow_implicit_io:
            raise ValueError(
                f"strict h={horizon} target calculation requires explicit open_910_returns"
            )
        open_910 = build_open_910_returns(df_exec, jp_tickers)
    elif (
        p_910_df is None
        and not allow_implicit_io
        and not has_usable_execution_prices(
            open_910, df_exec, jp_tickers, required_index=required_index
        )
    ):
        raise ValueError(
            f"strict h={horizon} target calculation requires complete finite "
            "open_910_returns or finite positive fallback opens"
        )
    if compute_open_910 is None:
        compute_open_910 = open_910
    return _compute(
        compute_frame,
        jp_tickers,
        horizon=horizon,
        p_910_df=compute_p_910,
        open_910_returns=compute_open_910,
    )


__all__ = [
    "build_5m_910_prices",
    "build_open_910_returns",
    "compute_jp_target_returns",
    "has_valid_open_910_returns",
    "has_usable_execution_prices",
    "resolve_execution_prices",
]
