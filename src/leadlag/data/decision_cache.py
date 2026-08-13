"""SQLite-backed decision cache.

This module replaces the legacy ``.npz`` decision cache with a single
``SqliteCacheStore``. The cache holds either the full pre-processed
``df_exec`` (legacy fast-path) or per-date decision records keyed by date.
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
from leadlag.data.schema import all_expected_columns

logger = logging.getLogger(__name__)

_DECISION_CACHE_FILENAME = "decision_cache.sqlite"
_FULL_DF_EXEC_KEY = "decision_cache"
_FULL_META_KEY = "decision_cache_meta"
_DATE_KEY_PREFIX = "decision"
_META_PREFIX = "decision_meta"


def _format_date(date: Any) -> str:
    if isinstance(date, (pd.Timestamp, datetime)):
        return cast(str, date.strftime("%Y-%m-%d"))
    return str(date)


def _resolve_store_path(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is None:
        return market_data(_DECISION_CACHE_FILENAME)
    p = Path(cache_dir)
    # Treat explicit SQLite file extensions as a full store path.
    if p.suffix.lower() in (".sqlite", ".sqlite3", ".db"):
        return p
    if p.is_dir():
        return p / _DECISION_CACHE_FILENAME
    raise ValueError(
        f"cache_dir must be a directory or SQLite file path, got {p!r}"
    )


def _store(cache_dir: str | Path | None = None) -> SqliteCacheStore:
    return SqliteCacheStore(_resolve_store_path(cache_dir))


def _required_columns() -> set[str]:
    return set(all_expected_columns())


def _meta_key_for(key: str) -> str:
    if key == _FULL_DF_EXEC_KEY:
        return _FULL_META_KEY
    return f"{_META_PREFIX}_{key}"


def decision_cache_path() -> str:
    """Return the canonical SQLite path for the decision cache."""
    return str(_resolve_store_path(None))


def _raw_etf_cache_path() -> Path:
    return market_data("etf_prices.sqlite")


def save_decision_cache(
    df: pd.DataFrame, date: Any = None, cache_dir: str | Path | None = None
) -> str:
    """Store a DataFrame in the decision cache.

    When ``date`` is provided the record is stored under a per-date key,
    otherwise the full ``df_exec`` cache is updated.
    """
    store = _store(cache_dir)
    if date is None:
        key = _FULL_DF_EXEC_KEY
    else:
        key = f"{_DATE_KEY_PREFIX}_{_format_date(date)}"
    meta_key = _meta_key_for(key)

    last_trade_date = df.index.max()
    last_trade_date_str = (
        None if pd.isna(last_trade_date) else pd.to_datetime(last_trade_date).strftime("%Y-%m-%d")
    )
    store.set(key, df)
    store.set(
        meta_key,
        {
            "columns": list(df.columns),
            "n_rows": len(df),
            "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "date": _format_date(date) if date is not None else None,
            "last_trade_date": last_trade_date_str,
        },
    )
    logger.info("Decision cache saved: %s (key=%s, rows=%d)", store.path, key, len(df))
    return str(store.path)


def load_decision_cache(
    date: Any = None, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load a DataFrame from the decision cache.

    Returns the full ``df_exec`` when ``date`` is omitted, otherwise the
    per-date decision record for ``date``.
    """
    store = _store(cache_dir)
    if date is None:
        key = _FULL_DF_EXEC_KEY
    else:
        key = f"{_DATE_KEY_PREFIX}_{_format_date(date)}"

    value = store.get(key)
    if value is None:
        raise FileNotFoundError(f"Decision cache not found: {store.path} (key={key})")
    return value


def is_decision_cache_valid(
    date: Any = None, cache_dir: str | Path | None = None
) -> bool:
    """Return True if the decision cache exists and contains required columns."""
    store = _store(cache_dir)
    if date is None:
        key = _FULL_DF_EXEC_KEY
    else:
        key = f"{_DATE_KEY_PREFIX}_{_format_date(date)}"
    meta_key = _meta_key_for(key)

    if key not in store.keys():
        return False

    meta = store.get(meta_key)
    if not meta:
        return False

    # Column validation only makes sense for the full df_exec cache.
    # Per-date records have a different, narrower schema.
    if date is None:
        columns = set(meta.get("columns", []))
        if not _required_columns().issubset(columns):
            return False

    # For the full df_exec, also ensure it is not older than the raw ETF cache.
    if date is None:
        raw_path = _raw_etf_cache_path()
        if raw_path.exists() and store.path.exists():
            decision_mtime = os.stat(store.path).st_mtime_ns
            raw_mtime = os.stat(raw_path).st_mtime_ns
            if decision_mtime < raw_mtime:
                logger.info(
                    "Decision cache is older than raw ETF cache; rebuild required"
                )
                return False

    return True
