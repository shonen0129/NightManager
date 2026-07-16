"""Threading utilities — daemon-thread timeout wrapper for blocking calls."""

from __future__ import annotations

import logging
import threading
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_with_timeout(
    fn: Callable[[], T],
    timeout: float,
    *,
    label: str = "operation",
) -> T:
    """Run *fn* in a daemon thread, raise TimeoutError if it exceeds *timeout* seconds.

    Args:
        fn: Zero-argument callable to execute.
        timeout: Maximum seconds to wait.
        label: Human-readable label for timeout error messages.

    Returns:
        The return value of *fn*.

    Raises:
        TimeoutError: If *fn* does not complete within *timeout* seconds.
        Exception: Any exception raised by *fn* is re-raised.
    """
    result_box: dict = {}

    def _worker() -> None:
        try:
            result_box["value"] = fn()
        except Exception as e:
            result_box["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logger.error("%s timed out after %.1fs", label, timeout)
        raise TimeoutError(f"{label} exceeded {timeout}s timeout")
    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]
