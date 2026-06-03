"""Cache repository: file-based caching with locking."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

import pandas as pd

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


@contextmanager
def file_lock(filepath: str, exclusive: bool = False):
    """Context manager for file-based locking.

    Args:
        filepath: Path to the file to lock
        exclusive: If True, acquire exclusive lock (for writes).
                   If False, acquire shared lock (for reads).

    Yields:
        Open file handle
    """
    lock_desc = "exclusive" if exclusive else "shared"
    mode = "r+b" if exclusive else "rb"

    if exclusive and not os.path.exists(filepath):
        # Create file for exclusive lock
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pass  # Create empty file
        mode = "r+b"

    fd = open(filepath, mode)
    try:
        _lock_file(fd, exclusive=exclusive, non_blocking=True)
        logger.debug(f"Acquired {lock_desc} lock on {filepath}")
        yield fd
    except BlockingIOError:
        logger.warning(
            f"Could not acquire {lock_desc} lock on {filepath}; "
            "file may be in use by another process"
        )
        _lock_file(fd, exclusive=exclusive, non_blocking=False)
        try:
            logger.info(f"Acquired {lock_desc} lock on {filepath} after waiting")
            yield fd
        finally:
            _unlock_file(fd)
    finally:
        _unlock_file(fd)
        fd.close()


class CacheRepository:
    """File-based cache with locking for daily returns.

    This encapsulates file I/O and locking logic to prevent
    concurrent cache corruption.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.returns_cache = os.path.join(cache_dir, "daily_returns.csv")

    def ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def read_daily_returns(self, trade_date: pd.Timestamp) -> pd.Series | None:
        """Read cached daily returns.

        Args:
            trade_date: Current trade date for filtering

        Returns:
            pd.Series of daily returns up to trade_date, or None if cache doesn't exist
        """
        if not os.path.exists(self.returns_cache):
            return None

        cache_age = None
        try:
            cache_mtime = os.path.getmtime(self.returns_cache)
            cache_age = time.time() - cache_mtime
        except OSError:
            pass

        logger.info(
            f"Loading cached daily returns for VaR/ES (age: {cache_age:.0f}s if available)..."
        )

        with file_lock(self.returns_cache, exclusive=False) as f:
            cached_df = pd.read_csv(f, index_col=0, parse_dates=True)

        hist_returns = cached_df["daily_return"]
        hist_returns = hist_returns[hist_returns.index < trade_date]
        logger.info(f"  Loaded {len(hist_returns)} days from cache")
        return hist_returns

    def write_daily_returns(self, returns_data: pd.DataFrame) -> None:
        """Write daily returns cache.

        Args:
            returns_data: DataFrame with 'daily_return' column
        """
        self.ensure_cache_dir()

        # Write to temp file first, then atomic rename
        tmp_path = self.returns_cache + ".tmp"
        returns_data[["daily_return"]].to_csv(tmp_path)

        with open(tmp_path, "r+b") as f:
            try:
                _lock_file(f, exclusive=True, non_blocking=True)
                os.rename(tmp_path, self.returns_cache)
                logger.info(f"Daily returns cached to {self.returns_cache}")
            except BlockingIOError:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                logger.warning(
                    "Could not acquire exclusive lock for cache write; "
                    "skipping cache update"
                )

    @property
    def exists(self) -> bool:
        """Check if cache file exists."""
        return os.path.exists(self.returns_cache)