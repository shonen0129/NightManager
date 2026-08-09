"""Centralized timeout defaults and helpers.

All network, lock, and long-running I/O calls should reference the constants
in this module instead of hard-coding their own values. This makes it easy to
audit and tune timeout behaviour from one place.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from leadlag.utils.threading import run_with_timeout

# Network and I/O timeouts (seconds)
YFINANCE_DOWNLOAD: float = 60.0
YFINANCE_TICKER_HISTORY: float = 30.0
MACRO_DOWNLOAD: float = 30.0
WEB_REQUEST: float = 10.0
KABU_REQUEST: float = 10.0
TACHIBANA_REQUEST: float = 30.0

# File and lock timeouts (seconds)
FILE_LOCK: float = 30.0
CACHE_OPERATION: float = 30.0

# Job-level guardrails
MAX_BACKTEST_STEP_TIMEOUT: float = 300.0

T = TypeVar("T")


def with_timeout(
    timeout: float,
    *,
    label: str = "operation",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that runs a function with a daemon-thread timeout.

    Note: the wrapped function should not rely on thread-local state or return
    non-picklable objects, because it is executed in a ``threading.Thread``.
    """

    def _decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> T:
            return run_with_timeout(
                lambda: fn(*args, **kwargs),
                timeout=timeout,
                label=label or fn.__name__,
            )

        return _wrapper

    return _decorator
