"""Data layer cache management and advisory locking services.

Handles all pkl/npz cache I/O for market data (etf_data.pkl),
the fast decision cache (decision_cache.npz), daily returns,
and file locks.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time as time_module
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS

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
    """Absolute path to the project ``market_data/`` directory."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "market_data")


def etf_pkl_path() -> str:
    return os.path.join(_data_dir(), _ETF_CACHE_FILENAME)


def etf_1m_pkl_path() -> str:
    return os.path.join(_data_dir(), _ETF_1M_CACHE_FILENAME)


def etf_5m_pkl_path() -> str:
    return os.path.join(_data_dir(), _ETF_5M_CACHE_FILENAME)


def decision_cache_path() -> str:
    return os.path.join(_data_dir(), _DECISION_CACHE_FILENAME)


# ---------------------------------------------------------------------------
# File locking primitives
# ---------------------------------------------------------------------------

if os.name == "nt":
    import msvcrt

    def _lock_file(file_obj, exclusive: bool, non_blocking: bool) -> None:
        if file_obj.writable():
            try:
                if os.fstat(file_obj.fileno()).st_size == 0:
                    file_obj.write(b"\0")
                    file_obj.flush()
            except OSError:
                pass
        try:
            file_obj.seek(0)
        except OSError:
            pass

        if exclusive:
            mode = msvcrt.LK_NBLCK if non_blocking else msvcrt.LK_LOCK
        else:
            mode = msvcrt.LK_NBRLCK if non_blocking else msvcrt.LK_RLCK
        try:
            msvcrt.locking(file_obj.fileno(), mode, 1)
        except OSError as exc:
            if non_blocking:
                raise BlockingIOError() from exc
            raise

    def _unlock_file(file_obj) -> None:
        try:
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_file(file_obj, exclusive: bool, non_blocking: bool) -> None:
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if non_blocking:
            lock_type |= fcntl.LOCK_NB
        fcntl.flock(file_obj, lock_type)

    def _unlock_file(file_obj) -> None:
        try:
            fcntl.flock(file_obj, fcntl.LOCK_UN)
        except OSError:
            pass


@contextlib.contextmanager
def file_lock(filepath: str, exclusive: bool = False):
    """Context manager for file-based locking to prevent concurrent cache corruption."""
    lock_desc = "exclusive" if exclusive else "shared"
    fd = open(filepath, "r+b" if exclusive else "rb")
    try:
        try:
            _lock_file(fd, exclusive=exclusive, non_blocking=True)
            logger.debug(f"Acquired {lock_desc} lock on {filepath}")
        except BlockingIOError:
            logger.warning(
                f"Could not acquire {lock_desc} lock on {filepath}; "
                "file may be in use by another process"
            )
            _lock_file(fd, exclusive=exclusive, non_blocking=False)
            logger.info(f"Acquired {lock_desc} lock on {filepath} after waiting")
        yield fd
    finally:
        _unlock_file(fd)
        fd.close()


@contextlib.contextmanager
def exclusive_lock(lock_path: str):
    """Acquire an exclusive advisory lock using a sidecar lock file."""
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    with open(lock_path, "a+b") as lock_file:
        _lock_file(lock_file, exclusive=True, non_blocking=False)
        try:
            yield
        finally:
            _unlock_file(lock_file)


@contextlib.contextmanager
def shared_lock(lock_path: str):
    """Acquire a shared advisory lock using a sidecar lock file."""
    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    with open(lock_path, "a+b") as lock_file:
        _lock_file(lock_file, exclusive=False, non_blocking=False)
        try:
            yield
        finally:
            _unlock_file(lock_file)


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
    age_hours = (
        datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    ).total_seconds() / 3600
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
# Daily returns cache (for VaR/ES)
# ---------------------------------------------------------------------------


def read_cache_with_lock(returns_cache: str, trade_date: pd.Timestamp) -> pd.Series | None:
    """Read cached daily returns with sidecar lock for consistency."""
    if not os.path.exists(returns_cache):
        return None

    cache_age = None
    try:
        cache_mtime = os.path.getmtime(returns_cache)
        cache_age = time_module.time() - cache_mtime
    except OSError:
        pass

    age_text = f"{cache_age:.0f}s" if cache_age is not None else "unknown"
    logger.info(f"Loading cached daily returns for VaR/ES (age: {age_text})...")

    lock_path = returns_cache + ".lock"
    with shared_lock(lock_path):
        if not os.path.exists(returns_cache):
            return None
        cached_df = pd.read_csv(returns_cache, index_col=0, parse_dates=True)

    hist_returns = cached_df["daily_return"]
    hist_returns = hist_returns[hist_returns.index < trade_date]
    logger.info(f"  Loaded {len(hist_returns)} days from cache")
    return hist_returns


def write_cache_with_lock(returns_cache: str, returns_data: pd.DataFrame) -> None:
    """Write daily returns cache with sidecar lock and atomic replace."""
    cache_dir = os.path.dirname(returns_cache)
    os.makedirs(cache_dir, exist_ok=True)

    lock_path = returns_cache + ".lock"
    tmp_path = f"{returns_cache}.tmp.{os.getpid()}"

    try:
        with exclusive_lock(lock_path):
            returns_data[["daily_return"]].to_csv(tmp_path)
            os.replace(tmp_path, returns_cache)
        logger.info(f"Daily returns cached to {returns_cache}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def get_hist_returns_for_risk(
    strategy: Any,
    config: Any,
    output_root: str,
    trade_date: pd.Timestamp,
) -> pd.Series:
    """Efficiently get historical daily returns for VaR/ES risk checks.

    Uses cache if available, otherwise runs full backtest and caches result.
    """
    cache_dir = os.path.join(output_root, ".cache")
    returns_cache = os.path.join(cache_dir, "daily_returns.csv")

    hist_returns = read_cache_with_lock(returns_cache, trade_date)
    if hist_returns is not None:
        return hist_returns

    logger.info("No return cache found; running full backtest for VaR/ES...")
    from leadlag.data.cache import load_df_exec_from_local_cache
    from leadlag.execution.backtester import BacktestEngine
    df_exec = load_df_exec_from_local_cache()
    out_res = BacktestEngine.run_backtest(strategy, df_exec, start_date=config.start_date)
    hist_results = pd.DataFrame(
        {"daily_return": out_res["daily_returns"]}, index=out_res["daily_returns"].index
    )

    write_cache_with_lock(returns_cache, hist_results)

    return pd.Series(
        hist_results.loc[
            hist_results.index < trade_date,
            "daily_return",
        ]
    )


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
    """Check if the decision cache exists and is in sync with etf_data.pkl."""
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
    """Save df_exec as a fast-loading numpy cache (.npz)."""
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
    """Load df_exec from the fast numpy cache."""
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


# ---------------------------------------------------------------------------
# Precomputed strategy cache validation
# ---------------------------------------------------------------------------


def is_strategy_cache_valid(cache_path: str, config=None) -> bool:
    """Check if precomputed strategy cache is in sync with market data and config."""
    if not os.path.exists(cache_path):
        return False

    pkl_path = etf_pkl_path()
    if not os.path.exists(pkl_path):
        return False

    try:
        cache_mtime = os.path.getmtime(cache_path)
        pkl_mtime = os.path.getmtime(pkl_path)
    except OSError:
        return False

    if cache_mtime < pkl_mtime:
        return False

    if config is None:
        return True

    def _get_val(key: str, default: Any = None) -> Any:
        if hasattr(config, key):
            return getattr(config, key)
        if isinstance(config, dict):
            if key in config:
                return config[key]
            for section in ["model", "ensemble", "portfolio", "costs", "residualization", "prior"]:
                if section in config and isinstance(config[section], dict) and key in config[section]:
                    return config[section][key]
            if key == "k" and "k" in config.get("model", {}):
                return config["model"]["k"]
            if key == "q" and "long_short_frac" in config.get("portfolio", {}):
                return config["portfolio"]["long_short_frac"]
            if key == "slippage_bps" and "slippage_bps_per_side" in config.get("costs", {}):
                return config["costs"]["slippage_bps_per_side"]
        return default

    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            required_keys = {
                "K": int(_get_val("k", 6)),
                "lambda_reg": float(_get_val("lambda_reg", 0.75)),
                "q": float(_get_val("q", 0.3)),
                "ewma_half_life": float(_get_val("ewma_half_life", 45)),
                "weight_mode": str(_get_val("weight_mode", "signal")),
                "dispersion_filter": bool(_get_val("dispersion_filter", False)),
                "dispersion_metric": str(_get_val("dispersion_metric", "long_short_mean_gap")),
                "v3_mode": str(_get_val("v3_mode", "static")),
                "include_v4_prior": bool(_get_val("include_v4_prior", True)),
                "signal_mode": str(_get_val("signal_mode", "gap_residual")),
                "gap_open_coef": float(_get_val("gap_open_coef", 0.70)),
                "topix_beta_coef": float(_get_val("topix_beta_coef", 0.6)),
                "beta_window": int(_get_val("beta_window", 60)),
                "corr_window": int(_get_val("corr_window", 60)),
                "lambda_lw": float(_get_val("lambda_lw", 0.5)),
                "lw_target": str(_get_val("lw_target", "equicorrelation")),
                "gamma": float(_get_val("gamma", 0.5)),
            }

            for key, expected in required_keys.items():
                if key not in npz.files:
                    logger.info("[FAST MODE] Cache missing key '%s'; rebuild required", key)
                    return False

                raw = np.asarray(npz[key])
                actual = raw.item() if raw.shape == () else raw

                if isinstance(expected, float):
                    try:
                        if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12):
                            logger.info(
                                "[FAST MODE] Cache config mismatch for %s: actual=%s expected=%s",
                                key,
                                actual,
                                expected,
                            )
                            return False
                    except (TypeError, ValueError):
                        return False
                elif isinstance(expected, bool):
                    if bool(actual) != expected:
                        logger.info(
                            "[FAST MODE] Cache config mismatch for %s: actual=%s expected=%s",
                            key,
                            actual,
                            expected,
                        )
                        return False
                elif isinstance(expected, int):
                    try:
                        if int(actual) != expected:
                            logger.info(
                                "[FAST MODE] Cache config mismatch for %s: actual=%s expected=%s",
                                key,
                                actual,
                                expected,
                            )
                            return False
                    except (TypeError, ValueError):
                        return False
                else:
                    if str(actual) != expected:
                        logger.info(
                            "[FAST MODE] Cache config mismatch for %s: actual=%s expected=%s",
                            key,
                            actual,
                            expected,
                        )
                        return False

            if str(config.signal_mode) == "gap_residual":
                for key in ("topix_night", "jp_beta"):
                    if key not in npz.files:
                        logger.info(
                            "[FAST MODE] Cache missing key '%s'; rebuild required",
                            key,
                        )
                        return False
    except Exception as e:
        logger.warning("[FAST MODE] Failed to validate cache config, rebuilding: %s", e)
        return False

    return True


def load_df_exec_from_local_cache() -> pd.DataFrame:
    """Load execution DataFrame from local data cache (no network)."""
    if is_decision_cache_valid():
        logger.info("[FAST MODE] Loading execution data from decision cache...")
        return load_decision_cache()

    pkl_path = etf_pkl_path()
    if os.path.exists(pkl_path):
        logger.info(f"[FAST MODE] Loading execution data from {pkl_path}...")
        try:
            from leadlag.data.preprocessor import preprocess_data

            data = pd.read_pickle(pkl_path)
            df_exec = preprocess_data(data)
            try:
                save_decision_cache(df_exec)
            except Exception as cache_err:
                logger.warning(
                    "[FAST MODE] Failed to refresh decision cache from local pkl: %s",
                    cache_err,
                )
            return df_exec
        except Exception as e:
            logger.warning(
                "[FAST MODE] Failed to rebuild from local etf_data.pkl; "
                "trying stale decision cache fallback: %s",
                e,
            )

    try:
        logger.warning("[FAST MODE] Using existing decision cache as fallback; it may be stale.")
        return load_decision_cache()
    except Exception as e:
        raise RuntimeError(
            "Local market-data cache not found/usable (market_data/etf_data.pkl) and "
            "decision cache fallback is unavailable. "
            "Prepare caches via non-fast path before running fast mode."
        ) from e
