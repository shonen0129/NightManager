"""SQLite-backed market data and df_exec cache.

Replaces ``.pkl`` / ``.npz`` caches for raw ETF OHLC, intraday data,
pre-processed ``df_exec``, and the VaR/ES daily-returns cache.

All stores are placed under ``var/market_data/`` by default.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from leadlag.config.paths import market_data
from leadlag.core.market_calendar import count_tse_bdays
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.schema import all_expected_columns

logger = logging.getLogger(__name__)

CACHE_TTL_HOURS = 12
_ETF_CACHE_FILENAME = "etf_prices.sqlite"
_DF_EXEC_CACHE_FILENAME = "df_exec.sqlite"

_ETF_RAW_KEY = "raw_ohlc"
_ETF_META_KEY = "raw_ohlc_meta"
_INTRADAY_KEY = "intraday_{interval}"
_DF_EXEC_KEY = "df_exec"
_DF_EXEC_META_KEY = "df_exec_meta"


def _etf_store_path() -> Path:
    return market_data(_ETF_CACHE_FILENAME)


def _df_exec_store_path() -> Path:
    return market_data(_DF_EXEC_CACHE_FILENAME)


def _resolve_store_path(cache_file: str | Path | None, default_filename: str) -> Path:
    if cache_file is None:
        return market_data(default_filename)
    p = Path(cache_file)
    if p.suffix.lower() in (".sqlite", ".sqlite3", ".db"):
        return p
    if p.is_dir():
        return p / default_filename
    raise ValueError(
        f"cache_file must be a directory or SQLite file path, got {p!r}"
    )


def _required_df_exec_columns() -> set[str]:
    return set(all_expected_columns())


def _check_df_exec_staleness(
    df_exec: pd.DataFrame,
    max_stale_bdays: int | None,
) -> None:
    """Raise RuntimeError if df_exec is older than max_stale_bdays TSE trading days."""
    if max_stale_bdays is None:
        return
    if df_exec is None or df_exec.empty:
        raise RuntimeError("df_exec cache is empty")
    last = df_exec.index.max()
    if pd.isna(last):
        raise RuntimeError("df_exec cache has no valid trade date")
    last = pd.to_datetime(last).normalize()
    today = pd.Timestamp.now().replace(tzinfo=None).normalize()
    if today < last:
        raise RuntimeError(f"df_exec cache last trade date {last.date()} is in the future")
    stale_bdays = count_tse_bdays(last, today)
    if stale_bdays > max_stale_bdays:
        raise RuntimeError(
            f"df_exec cache is stale: last={last.date()}, today={today.date()}, "
            f"{stale_bdays} TSE trading days old (max={max_stale_bdays})"
        )


def etf_pkl_path() -> str:
    """Return the canonical SQLite path for the raw ETF OHLC cache.

    The name is kept for backward compatibility; the file is a SQLite database.
    """
    return str(_etf_store_path())


def is_pkl_cache_valid(cache_file: str | Path | None = None) -> bool:
    """Return True if the raw ETF cache exists and is within TTL."""
    path = _resolve_store_path(cache_file, _ETF_CACHE_FILENAME)
    if not path.exists():
        return False
    store = SqliteCacheStore(path)

    if _ETF_RAW_KEY not in store.keys():
        return False

    meta = store.get(_ETF_META_KEY)
    updated_at = (meta or {}).get("updated_at")
    if updated_at is None:
        # Fallback to the store table metadata.
        meta = store.meta(_ETF_RAW_KEY)
        if meta is None:
            return False
        updated_at = meta.get("updated_at")
    if updated_at is None:
        return False

    age_hours = (
        datetime.now(UTC).replace(tzinfo=None)
        - datetime.fromisoformat(updated_at)
    ).total_seconds() / 3600
    return age_hours < CACHE_TTL_HOURS


def load_raw_cache() -> dict[str, Any]:
    """Load the raw OHLC dict from the ETF cache.

    Returns:
        Dict with keys "us_close", "jp_close", "jp_open" (all DataFrames).
    """
    store = SqliteCacheStore(_etf_store_path())
    data = store.get(_ETF_RAW_KEY)
    if data is None:
        raise FileNotFoundError(f"ETF data cache not found: {store.path}")
    return cast(dict[str, Any], data)


def save_raw_cache(data: dict) -> None:
    """Atomically write raw OHLC dict to the ETF cache."""
    store = SqliteCacheStore(_etf_store_path())
    store.set(_ETF_RAW_KEY, data)
    store.set(
        _ETF_META_KEY,
        {
            "keys": list(data.keys()),
            "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        },
    )
    logger.info("ETF cache saved: %s", store.path)


def load_jp_close_from_cache() -> pd.DataFrame:
    """Load only jp_close from the ETF cache (fast path for gap override)."""
    data = load_raw_cache()
    return data["jp_close"].copy()


def load_intraday_cache(interval: str) -> pd.DataFrame | None:
    """Load the intraday cache for a given interval ('1m' or '5m')."""
    store = SqliteCacheStore(_etf_store_path())
    key = _INTRADAY_KEY.format(interval=interval)
    return store.get(key)


def save_intraday_cache(data: pd.DataFrame, interval: str) -> None:
    """Atomically write intraday data to the ETF cache."""
    store = SqliteCacheStore(_etf_store_path())
    key = _INTRADAY_KEY.format(interval=interval)
    store.set(key, data)
    logger.info("Intraday cache (%s) saved: %s", interval, store.path)


def is_df_exec_cache_valid() -> bool:
    """Return True if the local df_exec cache exists and has required columns."""
    store = SqliteCacheStore(_df_exec_store_path())
    if _DF_EXEC_KEY not in store.keys():
        return False

    meta = store.get(_DF_EXEC_META_KEY)
    columns = set((meta or {}).get("columns", []))
    if not _required_df_exec_columns().issubset(columns):
        return False

    raw_path = _etf_store_path()
    if raw_path.exists() and store.path.exists():
        df_exec_meta = store.get(_DF_EXEC_META_KEY) or {}
        raw_store = SqliteCacheStore(_etf_store_path())
        raw_meta = raw_store.get(_ETF_META_KEY) or {}
        df_exec_updated_at = df_exec_meta.get("updated_at")
        raw_updated_at = raw_meta.get("updated_at")
        if df_exec_updated_at and raw_updated_at:
            if pd.to_datetime(df_exec_updated_at) < pd.to_datetime(raw_updated_at):
                logger.info(
                    "df_exec cache (updated_at=%s) is older than raw ETF cache (updated_at=%s); rebuild required",
                    df_exec_updated_at, raw_updated_at,
                )
                return False
        else:
            # Fallback to mtime if either cache lacks explicit updated_at.
            df_exec_mtime = os.stat(store.path).st_mtime_ns
            raw_mtime = os.stat(raw_path).st_mtime_ns
            if df_exec_mtime < raw_mtime:
                logger.info("df_exec cache is older than raw ETF cache; rebuild required")
                return False

    return True


def save_df_exec_to_local_cache(df_exec: pd.DataFrame) -> None:
    """Store a pre-processed df_exec under ``var/market_data/df_exec.sqlite``."""
    store = SqliteCacheStore(_df_exec_store_path())
    store.set(_DF_EXEC_KEY, df_exec)
    last_trade_date = df_exec.index.max()
    last_trade_date_str = (
        None if pd.isna(last_trade_date) else pd.to_datetime(last_trade_date).strftime("%Y-%m-%d")
    )
    store.set(
        _DF_EXEC_META_KEY,
        {
            "columns": list(df_exec.columns),
            "n_rows": len(df_exec),
            "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "last_trade_date": last_trade_date_str,
        },
    )
    # Keep the legacy decision cache in sync so callers using the old helpers
    # can still hit a pre-built df_exec.
    try:
        from leadlag.data import decision_cache

        decision_cache.save_decision_cache(df_exec)
    except Exception as exc:
        logger.warning("Failed to sync df_exec to decision cache: %s", exc)
    logger.info("df_exec cache saved: %s", store.path)


def load_df_exec_from_local_cache(max_stale_bdays: int | None = None) -> pd.DataFrame:
    """Load execution DataFrame from local data cache (no network).

    Args:
        max_stale_bdays: If set, raise RuntimeError when the cache's last trade
            date is more than this many TSE trading days in the past.  The
            default ``None`` keeps the old behaviour for research and tests.
    """

    def _validate(df_exec: pd.DataFrame | None, source: str) -> pd.DataFrame:
        if df_exec is None:
            raise RuntimeError(f"Loaded df_exec is None from {source}")
        _check_df_exec_staleness(df_exec, max_stale_bdays)
        return df_exec

    # 1. Prefer the dedicated df_exec SQLite cache.
    try:
        if is_df_exec_cache_valid():
            logger.info("[FAST MODE] Loading execution data from df_exec cache...")
            store = SqliteCacheStore(_df_exec_store_path())
            df_exec = store.get(_DF_EXEC_KEY)
            if df_exec is not None:
                return _validate(df_exec, "df_exec cache")
    except Exception as exc:
        logger.warning("[FAST MODE] df_exec cache not usable: %s", exc)

    # 2. Fall back to the legacy decision cache.
    try:
        from leadlag.data import decision_cache

        if decision_cache.is_decision_cache_valid():
            logger.info("[FAST MODE] Loading execution data from decision cache...")
            df_exec = decision_cache.load_decision_cache()
            save_df_exec_to_local_cache(df_exec)
            return _validate(df_exec, "decision cache")
    except Exception as exc:
        logger.warning("[FAST MODE] Decision cache not usable: %s", exc)

    # 3. Build from raw ETF cache.
    pkl_path = etf_pkl_path()
    if Path(pkl_path).exists():
        logger.info("[FAST MODE] Loading execution data from %s...", pkl_path)
        try:
            from leadlag.data.preprocessor import preprocess_data

            data = load_raw_cache()
            df_exec = preprocess_data(data, strict_validation=True)
            try:
                save_df_exec_to_local_cache(df_exec)
            except Exception as cache_err:
                logger.warning(
                    "[FAST MODE] Failed to refresh df_exec cache from raw ETF: %s",
                    cache_err,
                )
            return _validate(df_exec, "raw ETF cache")
        except Exception as e:
            logger.warning(
                "[FAST MODE] Failed to rebuild from raw ETF cache; "
                "trying stale df_exec cache fallback: %s",
                e,
            )

    # 4. Stale fallback: only allowed when max_stale_bdays is None.
    if max_stale_bdays is not None:
        raise RuntimeError(
            "[FAST MODE] Could not load a fresh df_exec cache and rebuilding from raw "
            "ETF cache also failed; stale cache fallback is disabled when max_stale_bdays "
            "is set. Prepare fresh caches via the non-fast path before running fast mode."
        )

    try:
        store = SqliteCacheStore(_df_exec_store_path())
        df_exec = store.get(_DF_EXEC_KEY)
        if df_exec is not None:
            logger.warning("[FAST MODE] Using existing df_exec cache as fallback; it may be stale.")
            return df_exec
    except Exception:
        pass

    try:
        from leadlag.data import decision_cache

        df_exec = decision_cache.load_decision_cache()
        logger.warning(
            "[FAST MODE] Using existing decision cache as fallback; it may be stale."
        )
        return df_exec
    except Exception as e:
        raise RuntimeError(
            "Local market-data cache not found/usable and no fallback is available. "
            "Prepare caches via the non-fast path before running fast mode."
        ) from e

