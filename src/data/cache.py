"""Data layer cache management.

Handles all pkl/npz cache I/O for market data (etf_data.pkl) and
the fast decision cache (decision_cache.npz).

The cache layout::

    data/
    ├── etf_data.pkl           — raw OHLC dict {"us_close", "jp_close", "jp_open"}
    └── decision_cache.npz     — preprocessed df_exec as numpy arrays for fast reload
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from data.ticker_registry import JP_TICKERS, TOPIX_TICKER

logger = logging.getLogger(__name__)

# Cache TTL (hours) before a re-download is triggered
CACHE_TTL_HOURS: int = 12

# Filename constants
_ETF_CACHE_FILENAME = "etf_data.pkl"
_DECISION_CACHE_FILENAME = "decision_cache.npz"
_ETF_1M_CACHE_FILENAME = "etf_1m_data.pkl"
_ETF_5M_CACHE_FILENAME = "etf_5m_data.pkl"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _data_dir() -> str:
    """Absolute path to the project ``data/`` directory."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "data")


def etf_pkl_path() -> str:
    return os.path.join(_data_dir(), _ETF_CACHE_FILENAME)


def etf_1m_pkl_path() -> str:
    return os.path.join(_data_dir(), _ETF_1M_CACHE_FILENAME)


def etf_5m_pkl_path() -> str:
    return os.path.join(_data_dir(), _ETF_5M_CACHE_FILENAME)


def decision_cache_path() -> str:
    return os.path.join(_data_dir(), _DECISION_CACHE_FILENAME)


# ---------------------------------------------------------------------------
# etf_data.pkl helpers
# ---------------------------------------------------------------------------

def is_pkl_cache_valid(cache_file: str | None = None) -> bool:
    """Return True if the pkl cache exists and is within TTL.

    Args:
        cache_file: Path to check. Defaults to ``etf_pkl_path()``.
    """
    path = cache_file or etf_pkl_path()
    if not os.path.exists(path):
        return False
    age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).total_seconds() / 3600
    return age_hours < CACHE_TTL_HOURS


def load_raw_cache() -> dict:
    """Load the raw OHLC dict from etf_data.pkl.

    Returns:
        Dict with keys "us_close", "jp_close", "jp_open".
    """
    path = etf_pkl_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"ETF data cache not found: {path}")
    return pd.read_pickle(path)


def save_raw_cache(data: dict) -> None:
    """Atomically write raw OHLC dict to etf_data.pkl."""
    path = etf_pkl_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        pd.to_pickle(data, tmp)
        os.replace(tmp, path)
        logger.info("ETF cache saved: %s", path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def load_jp_close_from_cache() -> pd.DataFrame:
    """Load only jp_close from etf_data.pkl (fast path for gap override).

    Note: Returns DataFrame with raw index — caller must normalize timezone/date.
    """
    data = load_raw_cache()
    return data["jp_close"].copy()


# ---------------------------------------------------------------------------
# Intraday data helpers
# ---------------------------------------------------------------------------

def load_intraday_cache(interval: str) -> pd.DataFrame | None:
    """Load the intraday cache for a given interval ('1m' or '5m').

    Returns:
        DataFrame if cache exists, otherwise None.
    """
    path = etf_1m_pkl_path() if interval == "1m" else etf_5m_pkl_path()
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def save_intraday_cache(data: pd.DataFrame, interval: str) -> None:
    """Atomically write intraday data to cache."""
    path = etf_1m_pkl_path() if interval == "1m" else etf_5m_pkl_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        pd.to_pickle(data, tmp)
        os.replace(tmp, path)
        logger.info("Intraday cache (%s) saved: %s", interval, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# decision_cache.npz helpers
# ---------------------------------------------------------------------------

def _decision_cache_has_required_columns(cache_path: str) -> bool:
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            numeric_cols = set(npz["numeric_cols"].tolist())
    except Exception:
        return False

    required = {"topix_night_return"}
    required.update({f"jp_beta_{tk}" for tk in JP_TICKERS})
    return required.issubset(numeric_cols)


def is_decision_cache_valid() -> bool:
    """Check if the decision cache exists and is in sync with etf_data.pkl.

    The decision cache is valid when:
    1. The .npz file exists
    2. The source etf_data.pkl exists
    3. The .npz is newer than etf_data.pkl (built from latest data)
    4. etf_data.pkl itself is within CACHE_TTL_HOURS
    5. Required columns (topix_night_return, jp_beta_*) are present
    """
    cache_path = decision_cache_path()
    pkl_path = etf_pkl_path()

    if not os.path.exists(cache_path) or not os.path.exists(pkl_path):
        return False

    if os.path.getmtime(cache_path) < os.path.getmtime(pkl_path):
        return False

    if not _decision_cache_has_required_columns(cache_path):
        return False

    return is_pkl_cache_valid(pkl_path)


def save_decision_cache(df_exec: pd.DataFrame) -> str:
    """Save df_exec as a fast-loading numpy cache (.npz).

    Stores numeric data, column names, date index, and sig_date separately
    so the full DataFrame can be reconstructed without pickle.

    Args:
        df_exec: Execution DataFrame from preprocessor.preprocess_data()

    Returns:
        Path to the saved cache file
    """
    cache_path = decision_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    numeric_cols = [c for c in df_exec.columns if c != "sig_date"]
    numeric_data = df_exec[numeric_cols].values.astype(np.float64)

    sig_dates = np.array(
        pd.to_datetime(df_exec["sig_date"]).dt.strftime("%Y-%m-%d").to_list(),
        dtype="U",
    )
    index_dates = np.array(df_exec.index.strftime("%Y-%m-%d").to_list(), dtype="U")

    np.savez_compressed(
        cache_path,
        numeric_data=numeric_data,
        numeric_cols=np.array(numeric_cols, dtype="U"),
        sig_dates=sig_dates,
        index_dates=index_dates,
    )

    logger.info("Decision cache saved: %s (%d rows)", cache_path, len(df_exec))
    return cache_path


def load_decision_cache() -> pd.DataFrame:
    """Load df_exec from the fast numpy cache.

    Returns:
        Reconstructed df_exec DataFrame identical to preprocess_data() output.
    """
    cache_path = decision_cache_path()
    npz = np.load(cache_path, allow_pickle=False)

    numeric_data = npz["numeric_data"]
    numeric_cols = list(npz["numeric_cols"])
    sig_dates = npz["sig_dates"]
    index_dates = npz["index_dates"]

    trade_index = pd.to_datetime(index_dates)
    trade_index.name = "trade_date"
    df_exec = pd.DataFrame(numeric_data, columns=numeric_cols, index=trade_index)
    df_exec.insert(0, "sig_date", pd.to_datetime(sig_dates))

    logger.info("Decision cache loaded: %d rows (fast path)", len(df_exec))
    return df_exec
