"""Data layer cache management compatibility shim.

All concrete cache I/O has moved to ``leadlag.data.decision_cache`` and
``leadlag.data.market_data_cache``. This module re-exports the public helpers
so existing callers continue to work without changing their imports.

The legacy advisory locking primitives, ``.npz`` writing, and ``.pkl`` I/O
have been removed. Concurrent access is now handled by SQLite transactions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from leadlag.data.decision_cache import (
    decision_cache_path as _decision_cache_path,
)
from leadlag.data.decision_cache import (
    is_decision_cache_valid as _is_decision_cache_valid,
)
from leadlag.data.decision_cache import (
    load_decision_cache as _load_decision_cache,
)
from leadlag.data.decision_cache import (
    save_decision_cache as _save_decision_cache,
)
from leadlag.data.market_data_cache import (
    etf_pkl_path,
    is_df_exec_cache_valid,
    is_pkl_cache_valid,
    load_df_exec_from_local_cache,
    load_intraday_cache,
    load_jp_close_from_cache,
    load_raw_cache,
    save_df_exec_to_local_cache,
    save_intraday_cache,
    save_raw_cache,
)

# ``decision_cache_path`` is exposed as a module-level function so tests and
# callers can monkeypatch it to redirect the decision cache store path.
decision_cache_path = _decision_cache_path


def save_decision_cache(
    df: pd.DataFrame,
    date: Any = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Store a DataFrame in the SQLite-backed decision cache."""
    if cache_dir is None:
        cache_dir = decision_cache_path()
    return _save_decision_cache(df, date=date, cache_dir=cache_dir)


def load_decision_cache(
    date: Any = None, cache_dir: str | Path | None = None
) -> pd.DataFrame | None:
    """Load a DataFrame from the SQLite-backed decision cache."""
    if cache_dir is None:
        cache_dir = decision_cache_path()
    return _load_decision_cache(date=date, cache_dir=cache_dir)


def is_decision_cache_valid(date: Any = None, cache_dir: str | Path | None = None) -> bool:
    """Return True if the decision cache exists and has required columns."""
    if cache_dir is None:
        cache_dir = decision_cache_path()
    return _is_decision_cache_valid(date=date, cache_dir=cache_dir)


__all__ = [
    "decision_cache_path",
    "etf_pkl_path",
    "is_decision_cache_valid",
    "is_df_exec_cache_valid",
    "is_pkl_cache_valid",
    "load_decision_cache",
    "load_df_exec_from_local_cache",
    "load_intraday_cache",
    "load_jp_close_from_cache",
    "load_raw_cache",
    "save_decision_cache",
    "save_df_exec_to_local_cache",
    "save_intraday_cache",
    "save_raw_cache",
]
