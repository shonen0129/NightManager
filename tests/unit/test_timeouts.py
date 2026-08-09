"""Tests for the centralized timeout helpers."""

from __future__ import annotations

import time

import pytest

from leadlag.core.timeouts import with_timeout


def test_with_timeout_completes():
    @with_timeout(timeout=1.0, label="fast")
    def fast() -> int:
        return 42

    assert fast() == 42


def test_with_timeout_raises():
    @with_timeout(timeout=0.2, label="slow")
    def slow() -> int:
        time.sleep(1.0)
        return 42

    with pytest.raises(TimeoutError, match="slow"):
        slow()
