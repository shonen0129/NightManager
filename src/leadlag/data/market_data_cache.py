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
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.tickers import JP_TICKERS

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
    return p / default_filename


def _required_df_exec_columns() -> set[str]:
    return {"topix_night_return"} | {f"jp_beta_{tk}" for tk in JP_TICKERS}


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
        df_exec_mtime = os.path.getmtime(store.path)
        raw_mtime = os.path.getmtime(raw_path)
        if df_exec_mtime < raw_mtime:
            logger.info("df_exec cache is older than raw ETF cache; rebuild required")
            return False

    return True


def save_df_exec_to_local_cache(df_exec: pd.DataFrame) -> None:
    """Store a pre-processed df_exec under ``var/market_data/df_exec.sqlite``."""
    store = SqliteCacheStore(_df_exec_store_path())
    store.set(_DF_EXEC_KEY, df_exec)
    store.set(
        _DF_EXEC_META_KEY,
        {
            "columns": list(df_exec.columns),
            "n_rows": len(df_exec),
            "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
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


def load_df_exec_from_local_cache() -> pd.DataFrame:
    """Load execution DataFrame from local data cache (no network)."""
    # 1. Prefer the dedicated df_exec SQLite cache.
    try:
        if is_df_exec_cache_valid():
            logger.info("[FAST MODE] Loading execution data from df_exec cache...")
            store = SqliteCacheStore(_df_exec_store_path())
            return store.get(_DF_EXEC_KEY)
    except Exception as exc:
        logger.warning("[FAST MODE] df_exec cache not usable: %s", exc)

    # 2. Fall back to the legacy decision cache.
    try:
        from leadlag.data import decision_cache

        if decision_cache.is_decision_cache_valid():
            logger.info("[FAST MODE] Loading execution data from decision cache...")
            df_exec = decision_cache.load_decision_cache()
            save_df_exec_to_local_cache(df_exec)
            return df_exec
    except Exception as exc:
        logger.warning("[FAST MODE] Decision cache not usable: %s", exc)

    # 3. Build from raw ETF cache.
    pkl_path = etf_pkl_path()
    if Path(pkl_path).exists():
        logger.info("[FAST MODE] Loading execution data from %s...", pkl_path)
        try:
            from leadlag.data.preprocessor import preprocess_data

            data = load_raw_cache()
            df_exec = preprocess_data(data)
            try:
                save_df_exec_to_local_cache(df_exec)
            except Exception as cache_err:
                logger.warning(
                    "[FAST MODE] Failed to refresh df_exec cache from raw ETF: %s",
                    cache_err,
                )
            return df_exec
        except Exception as e:
            logger.warning(
                "[FAST MODE] Failed to rebuild from raw ETF cache; "
                "trying stale df_exec cache fallback: %s",
                e,
            )

    # 4. Stale fallback: return whatever df_exec cache exists, even if invalid.
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

