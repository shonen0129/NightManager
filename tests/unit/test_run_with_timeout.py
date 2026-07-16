"""Tests for run_with_timeout threading utility."""

from __future__ import annotations

import time

import pytest

from leadlag.utils.threading import run_with_timeout


class TestRunWithTimeout:
    def test_normal_completion(self):
        """Function returns its value when it completes within timeout."""
        result = run_with_timeout(lambda: 42, 5.0, label="test_ok")
        assert result == 42

    def test_timeout_raises_timeout_error(self):
        """TimeoutError is raised when function exceeds timeout."""
        def slow_fn():
            time.sleep(10)
            return "should never get here"

        with pytest.raises(TimeoutError, match="slow_op"):
            run_with_timeout(slow_fn, 0.5, label="slow_op")

    def test_exception_propagated(self):
        """Exceptions raised by fn are re-raised by run_with_timeout."""

        def failing_fn():
            raise ValueError("intentional failure")

        with pytest.raises(ValueError, match="intentional failure"):
            run_with_timeout(failing_fn, 5.0, label="failing_op")
