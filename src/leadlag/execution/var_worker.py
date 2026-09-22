"""Own temporary inputs and bounded cache-write processes for VaR replay."""
from __future__ import annotations

import multiprocessing
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from leadlag.data.cache_store import SqliteCacheStore
from leadlag.utils.threading import run_with_timeout


def run_snapshot_worker[T](function: Callable[[], T], owner: tempfile.TemporaryDirectory[str] | None, *, timeout: float) -> T:
    """The worker retains its inputs even after its caller's deadline expires."""
    def execute() -> T:
        try:
            return function()
        finally:
            if owner is not None:
                owner.cleanup()

    return run_with_timeout(execute, timeout=timeout, label="VaR/ES backtest")


def _cache_set_worker(
    path: str,
    key: str,
    value: Any,
    timeout: float,
) -> None:
    """Write one cache value in an isolated process that can be terminated."""
    SqliteCacheStore(path, timeout=max(0.01, timeout)).set(key, value)


def _set_cache_with_deadline(
    path: Path,
    key: str,
    value: Any,
    *,
    timeout: float,
) -> None:
    """Atomically write a cache entry without allowing late work to survive.

    A daemon thread cannot be cancelled while SQLite is blocked in an OS call.
    Cache writes are therefore isolated in a child process; a deadline expiry
    terminates the child before it can publish a late result to the shared DB.
    """
    # ``fork`` avoids importing the complete application during a short
    # deadline on Unix.  Windows has no fork context and uses spawn instead.
    context: Any
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_cache_set_worker,
        args=(str(path), key, value, timeout),
        daemon=True,
    )
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        raise TimeoutError(f"cache write exceeded {timeout}s deadline")
    if process.exitcode != 0:
        raise RuntimeError(
            f"cache write process failed with exit code {process.exitcode}"
        )
