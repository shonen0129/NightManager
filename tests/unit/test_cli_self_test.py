"""Unit tests for the CLI self-test diagnostics."""

from __future__ import annotations

from leadlag.execution.self_test import run_self_tests


def test_run_self_tests_passes():
    """The bundled self-tests must all return 0."""
    assert run_self_tests() == 0
