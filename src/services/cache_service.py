"""Cache services: file locking, daily returns cache, precomputed strategy cache."""

from __future__ import annotations

import contextlib
import logging
import os
import time as time_module
from typing import Optional

import numpy as np
import pandas as pd

from data.cache import (
    etf_pkl_path as _get_etf_pkl_path,
    is_decision_cache_valid,
    load_decision_cache,
    save_decision_cache,
)
from data.preprocessor import preprocess_data

logger = logging.getLogger(__name__)


if os.name == "nt":
    import msvcrt

    def _lock_file(file_obj, exclusive: bool, non_blocking: bool) -> None:
        # Windows uses byte-range locks; lock the first byte.
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


# ---------------------------------------------------------------------------
# File locking primitives
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def file_lock(filepath: str, exclusive: bool = False):
    """Context manager for file-based locking to prevent concurrent cache corruption.

    Args:
        filepath: Path to the file to lock
        exclusive: If True, acquire exclusive lock (for writes).
                   If False, acquire shared lock (for reads).

    Yields:
        Open file handle
    """
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
# Daily returns cache (for VaR/ES)
# ---------------------------------------------------------------------------


def read_cache_with_lock(
    returns_cache: str, trade_date: pd.Timestamp
) -> Optional[pd.Series]:
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
    strategy,
    config,
    output_root: str,
    trade_date: pd.Timestamp,
) -> pd.Series:
    """Efficiently get historical daily returns for VaR/ES risk checks.

    Uses cache if available, otherwise runs full backtest and caches result.
    File locking is used to prevent corruption during concurrent access.

    Args:
        strategy: Initialized LeadLagStrategy instance
        config: ProductionConfig instance
        output_root: Root directory for cache files
        trade_date: Current trade date for filtering

    Returns:
        pd.Series of daily returns up to trade_date
    """
    cache_dir = os.path.join(output_root, ".cache")
    returns_cache = os.path.join(cache_dir, "daily_returns.csv")

    hist_returns = read_cache_with_lock(returns_cache, trade_date)
    if hist_returns is not None:
        return hist_returns

    # No cache: run full backtest and save cache
    logger.info("No return cache found; running full backtest for VaR/ES...")
    hist_results = strategy.run_backtest(start_date=config.start_date)

    # Save cache for future decision runs with exclusive lock
    write_cache_with_lock(returns_cache, hist_results)

    return pd.Series(
        hist_results.loc[
            hist_results.index < trade_date,
            "daily_return",
        ]
    )


# ---------------------------------------------------------------------------
# Precomputed strategy cache
# ---------------------------------------------------------------------------


def is_strategy_cache_valid(cache_path: str, config=None) -> bool:
    """Check if precomputed cache is in sync with market data and config.

    Args:
        cache_path: Path to strategy_cache.npz
        config: Optional ProductionConfig for configuration-based validation

    Returns:
        True if cache is valid and usable
    """
    if not os.path.exists(cache_path):
        return False

    pkl_path = _get_etf_pkl_path()
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

    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            required_keys = {
                "K": int(config.k),
                "lambda_reg": float(config.lambda_reg),
                "q": float(config.q),
                "ewma_half_life": (
                    float(config.ewma_half_life)
                    if config.ewma_half_life is not None
                    else -1.0
                ),
                "weight_mode": str(config.weight_mode),
                "dispersion_filter": bool(config.dispersion_filter),
                "dispersion_metric": str(config.dispersion_metric),
                "v3_mode": str(config.v3_mode),
                "include_v4_prior": bool(config.include_v4_prior),
                "signal_mode": str(config.signal_mode),
                "gap_open_coef": float(config.gap_open_coef),
                "topix_beta_coef": float(config.topix_beta_coef),
                "beta_window": int(config.beta_window),
                "corr_window": int(config.corr_window),
                "lambda_lw": float(config.lambda_lw),
                "lw_target": str(config.lw_target),
                "gamma": float(config.gamma),
            }

            for key, expected in required_keys.items():
                if key not in npz.files:
                    logger.info(
                        "[FAST MODE] Cache missing key '%s'; rebuild required", key
                    )
                    return False

                raw = np.asarray(npz[key])
                actual = raw.item() if raw.shape == () else raw

                if isinstance(expected, float):
                    try:
                        if not np.isclose(
                            float(actual), expected, rtol=0.0, atol=1e-12
                        ):
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
    """Load execution DataFrame from local data cache (no network).

    Tries decision cache first, falls back to etf_data.pkl.
    """
    if is_decision_cache_valid():
        logger.info("[FAST MODE] Loading execution data from decision cache...")
        return load_decision_cache()

    pkl_path = _get_etf_pkl_path()
    if os.path.exists(pkl_path):
        logger.info(f"[FAST MODE] Loading execution data from {pkl_path}...")
        try:
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
        logger.warning(
            "[FAST MODE] Using existing decision cache as fallback; " "it may be stale."
        )
        return load_decision_cache()
    except Exception as e:
        raise RuntimeError(
            "Local market-data cache not found/usable (data/etf_data.pkl) and "
            "decision cache fallback is unavailable. "
            "Prepare caches via non-fast path before running fast mode."
        ) from e
