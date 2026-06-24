"""Market data downloader — yfinance fetch + incremental update + NAV patch.

Responsible for:
- Downloading US and JP ETF OHLC data via yfinance
- Incremental (missing-only) updates to preserve manual patches
- Applying NAV patch for 1629.T (Yahoo Finance split anomaly workaround)
- Orchestrating cache reads/writes via leadlag.data.cache

Public API::

    from leadlag.data.fetcher import download_data
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from leadlag.data.cache import (
    etf_pkl_path,
    is_pkl_cache_valid,
    save_raw_cache,
)
from leadlag.data.tickers import (
    JP_TICKERS,
    JP_TICKERS_WITH_TOPIX,
    TOPIX_TICKER,
    US_TICKERS,
)

logger = logging.getLogger(__name__)

# Number of recent days to re-fetch during incremental update to absorb
# delayed / corrected data from Yahoo Finance
_CACHE_UPDATE_LOOKBACK_DAYS: int = 14


# ---------------------------------------------------------------------------
# 1629.T NAV patch (Yahoo Finance split anomaly)
# ---------------------------------------------------------------------------


def _load_1629_nav_per_share_from_csv(csv_path: str) -> pd.Series:
    """Load NAV (per share) series for 1629 from the vendor CSV.

    The CSV is CP932-encoded and contains a preamble before the header
    line that starts with 'Date,'.
    """
    with open(csv_path, encoding="cp932", errors="replace") as f:
        text = f.read()

    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines[:500]) if line.startswith("Date,")),
        None,
    )
    if header_idx is None:
        raise ValueError(
            f"Could not find 'Date' header in CSV file: {csv_path}. "
            f"Expected format: CSV with 'Date' column for NAV data."
        )

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    if "Date" not in df.columns or "Net Asset Value (per Share)" not in df.columns:
        raise ValueError(
            f"Unexpected CSV columns in {csv_path}. "
            f"Required columns: ['Date', 'Net Asset Value (per Share)']. "
            f"Found columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    nav = df.set_index("Date")["Net Asset Value (per Share)"].astype(float)
    nav.index = pd.to_datetime(nav.index).tz_localize(None).normalize()
    return nav


def _apply_1629_nav_patch(
    jp_close: pd.DataFrame,
    jp_open: pd.DataFrame,
    nav: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patch obviously-broken 1629.T prices using NAV (per share).

    Yahoo Finance occasionally emits split-related anomalies (e.g. Close ~0.55
    while surrounding days are ~280). We fix dates where Close is implausibly
    small by replacing Close with NAV and scaling Open by the same factor.
    """
    ticker = "1629.T"
    if ticker not in jp_close.columns or ticker not in jp_open.columns:
        return jp_close, jp_open

    jp_close = jp_close.copy()
    jp_open = jp_open.copy()

    jp_close.index = pd.to_datetime(jp_close.index).tz_localize(None).normalize()
    jp_open.index = pd.to_datetime(jp_open.index).tz_localize(None).normalize()

    close_series = pd.to_numeric(jp_close[ticker], errors="coerce")
    bad_dates = close_series[(close_series < 10) & np.isfinite(close_series)].index
    bad_dates = bad_dates.intersection(nav.index)

    if len(bad_dates) == 0:
        return jp_close, jp_open

    patched_dates: list[str] = []
    for dt in bad_dates:
        orig_close = float(close_series.loc[dt])
        target_close = float(nav.loc[dt])

        if not np.isfinite(orig_close) or orig_close == 0.0:
            continue

        factor = target_close / orig_close
        orig_open = jp_open.at[dt, ticker] if dt in jp_open.index else np.nan
        orig_open = float(orig_open) if np.isfinite(orig_open) else np.nan

        jp_close.at[dt, ticker] = target_close
        jp_open.at[dt, ticker] = orig_open * factor if np.isfinite(orig_open) else target_close
        patched_dates.append(dt.strftime("%Y-%m-%d"))

    if patched_dates:
        logger.warning(
            "Applied NAV patch for %s on %d dates: %s",
            ticker,
            len(patched_dates),
            patched_dates,
        )

    return jp_close, jp_open


def _try_apply_nav_patch(
    df_jp_close: pd.DataFrame,
    df_jp_open: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply 1629.T NAV patch if the vendor CSV exists (silently skips if absent)."""
    nav_csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "market_data", "ETF_1629.csv"
    )
    if not os.path.exists(nav_csv_path):
        return df_jp_close, df_jp_open
    try:
        nav_1629 = _load_1629_nav_per_share_from_csv(nav_csv_path)
        return _apply_1629_nav_patch(df_jp_close, df_jp_open, nav_1629)
    except Exception as e:
        logger.warning("Failed to apply 1629 NAV patch: %s", e)
        return df_jp_close, df_jp_open


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------


def _to_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _get_max_date(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return pd.to_datetime(df.index).tz_localize(None).normalize().max()


def _has_sufficient_topix_history(jp_close: pd.DataFrame, beta_window: int) -> bool:
    if not isinstance(jp_close, pd.DataFrame) or TOPIX_TICKER not in jp_close.columns:
        return False
    series = pd.to_numeric(jp_close[TOPIX_TICKER], errors="coerce")
    return int(series.notna().sum()) >= int(beta_window)


def _update_stale_cache_missing_only(
    stale_data: dict,
    start_date: str,
    end_date,
    beta_window: int = 60,
) -> dict:
    """Update only missing/new dates, preserving existing values.

    Downloads only from (last_cached_date - lookback) to end_date/now,
    then merges using combine_first (existing cells take priority).
    """
    if not isinstance(stale_data, dict):
        raise TypeError(f"Unexpected cache type: {type(stale_data)}")
    for k in ["us_close", "jp_close", "jp_open"]:
        if k not in stale_data:
            raise KeyError(f"Missing key {k} in stale cache")

    us_close_old = stale_data["us_close"]
    jp_close_old = stale_data["jp_close"]
    jp_open_old = stale_data["jp_open"]

    last_dates = [
        d
        for d in [
            _get_max_date(us_close_old),
            _get_max_date(jp_close_old),
            _get_max_date(jp_open_old),
        ]
        if d is not None
    ]
    if not last_dates:
        raise ValueError("Stale cache has no valid dates")

    last_cached_date = min(last_dates)

    start_bound = pd.to_datetime(start_date).tz_localize(None).normalize()
    update_start = last_cached_date - pd.Timedelta(days=_CACHE_UPDATE_LOOKBACK_DAYS)
    if update_start < start_bound:
        update_start = start_bound

    if end_date is not None:
        end_bound = pd.to_datetime(end_date).tz_localize(None).normalize()
        if end_bound <= last_cached_date:
            logger.info(
                "Cache expired but covers requested end_date=%s; using stale cache",
                end_date,
            )
            return stale_data

    logger.info(
        "Incremental update: downloading from %s (lookback=%dd)",
        update_start.date(),
        _CACHE_UPDATE_LOOKBACK_DAYS,
    )

    us_data = yf.download(
        US_TICKERS,
        start=update_start.strftime("%Y-%m-%d"),
        end=end_date,
        auto_adjust=False,
    )
    jp_data = yf.download(
        JP_TICKERS_WITH_TOPIX,
        start=update_start.strftime("%Y-%m-%d"),
        end=end_date,
        auto_adjust=False,
    )

    us_close_new = us_data["Close"]
    jp_close_new = jp_data["Close"]
    jp_open_new = jp_data["Open"]

    if us_close_new.empty or jp_close_new.empty or jp_open_new.empty:
        raise ValueError(
            "Incremental download returned empty market data (us_close/jp_close/jp_open)"
        )

    # Normalize indices and merge (old values take priority via combine_first)
    df_us_close = (
        _to_daily_index(us_close_old).combine_first(_to_daily_index(us_close_new)).sort_index()
    )
    df_jp_close = (
        _to_daily_index(jp_close_old).combine_first(_to_daily_index(jp_close_new)).sort_index()
    )
    df_jp_open = (
        _to_daily_index(jp_open_old).combine_first(_to_daily_index(jp_open_new)).sort_index()
    )

    df_jp_close, df_jp_open = _try_apply_nav_patch(df_jp_close, df_jp_open)

    return {
        "us_close": df_us_close,
        "jp_close": df_jp_close,
        "jp_open": df_jp_open,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_data(
    start_date: str = "2009-01-01",
    end_date=None,
    beta_window: int = 60,
) -> dict:
    """Download and cache ETF OHLC data.

    Strategy:
    1. If cache is valid (< CACHE_TTL_HOURS old) → return cached data
    2. If stale cache exists → try incremental update
    3. Otherwise → full re-download
    4. Falls back to stale cache if download fails

    Args:
        start_date: Download start date (used if no cache exists)
        end_date: Download end date (None = today)
        beta_window: Minimum TOPIX history required for beta computation

    Returns:
        Dict with keys "us_close", "jp_close", "jp_open" (all DataFrames)
    """
    pkl_path = etf_pkl_path()
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    if is_pkl_cache_valid(pkl_path):
        logger.info("Loading data from cache (valid)")
        return pd.read_pickle(pkl_path)

    has_stale_cache = os.path.exists(pkl_path)
    if has_stale_cache:
        cache_age = datetime.fromtimestamp(os.path.getmtime(pkl_path))
        logger.info("Cache expired (last modified: %s), re-downloading...", cache_age)

        try:
            stale_data = pd.read_pickle(pkl_path)
            updated = _update_stale_cache_missing_only(
                stale_data,
                start_date=start_date,
                end_date=end_date,
                beta_window=beta_window,
            )
            if not _has_sufficient_topix_history(updated.get("jp_close"), beta_window):
                raise ValueError("TOPIX proxy history is insufficient after incremental update")
            save_raw_cache(updated)
            logger.info("Cache updated (missing-only) and saved")
            return updated
        except Exception as e:
            logger.warning("Incremental cache update failed: %s", e)

    try:
        logger.info("Downloading US ETF data...")
        us_data = yf.download(
            US_TICKERS,
            start=start_date,
            end=end_date,
            auto_adjust=False,
        )
        logger.info("Downloading JP ETF data...")
        jp_data = yf.download(
            JP_TICKERS_WITH_TOPIX,
            start=start_date,
            end=end_date,
            auto_adjust=False,
        )

        us_close = us_data["Close"]
        jp_close = jp_data["Close"]
        jp_open = jp_data["Open"]

        if us_close.empty or jp_close.empty or jp_open.empty:
            raise ValueError("Downloaded market data is empty (us_close/jp_close/jp_open)")

        df_jp_close, df_jp_open = _try_apply_nav_patch(jp_close.copy(), jp_open.copy())

        data = {
            "us_close": us_close.copy(),
            "jp_close": df_jp_close,
            "jp_open": df_jp_open,
        }
        save_raw_cache(data)
        logger.info("Data downloaded and cached successfully")
        return data

    except Exception:
        if has_stale_cache and os.path.exists(pkl_path):
            logger.warning("Download failed. Falling back to stale cache: %s", pkl_path)
            return pd.read_pickle(pkl_path)
        raise


def update_intraday_cache(tickers: list[str] = JP_TICKERS) -> None:
    """Download and accumulate 1m and 5m intraday data for the given tickers.

    yfinance limits:
    - 1m data is only available for the last 7 days.
    - 5m data is only available for the last 60 days.
    This function downloads the maximum available period and merges it with existing cache.
    """
    from leadlag.data.cache import load_intraday_cache, save_intraday_cache

    for interval, period in [("1m", "7d"), ("5m", "60d")]:
        logger.info("Downloading %s data for period %s...", interval, period)
        try:
            new_data = yf.download(tickers, period=period, interval=interval, auto_adjust=False)
            if new_data.empty:
                logger.warning("No new data downloaded for interval %s.", interval)
                continue

            # Normalize index to timezone naive JST
            if new_data.index.tz is not None:
                new_data.index = new_data.index.tz_convert("Asia/Tokyo").tz_localize(None)

            # Load existing cache
            existing_data = load_intraday_cache(interval)

            if existing_data is not None and not existing_data.empty:
                # new_data takes precedence for overlapping indices, and appends new ones
                combined = new_data.combine_first(existing_data)
            else:
                combined = new_data

            combined = combined.sort_index()
            save_intraday_cache(combined, interval)

            # Also log date coverage
            start_dt = combined.index.min()
            end_dt = combined.index.max()
            logger.info(
                "Successfully updated %s cache. Total rows: %d (From %s to %s)",
                interval,
                len(combined),
                start_dt,
                end_dt,
            )
        except Exception as e:
            logger.error("Failed to update intraday cache for %s: %s", interval, e)
