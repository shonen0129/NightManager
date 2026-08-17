"""Tests for gap matrix I/O helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from leadlag.data.gap_store import GapStore
from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError
from leadlag.utils.gap_matrix_io import load_gap_matrices


def _build_gap_store(tmpdir: Path, trade_date: str = "2026-01-01") -> Path:
    """Create a SQLite GapStore with a default valid pair for the given date."""
    n_j = len(JP_TICKERS)
    store_path = tmpdir / "gap.sqlite"
    store = GapStore(store_path)
    mu = np.zeros(n_j)
    omega = np.eye(n_j)
    store.save(trade_date, mu, omega)
    return store_path


def test_load_gap_matrices_nonstrict_returns_warnings_with_arrays():
    """Non-strict mode must return the loaded arrays and the validation alerts."""
    n_j = len(JP_TICKERS)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        store_path = _build_gap_store(d)

        # Add an asymmetric Omega to the store to trigger a validation warning.
        store = GapStore(store_path)
        mu = np.zeros(n_j)
        omega = np.eye(n_j)
        omega[0, 1] = 1e-3
        store.save("2026-01-02", mu, omega)

        mu_out, omega_out, alerts = load_gap_matrices(
            store_path, "2026-01-02", strict=False
        )
        assert mu_out is not None
        assert omega_out is not None
        assert alerts
        assert any("symmetric" in a for a in alerts)


def test_load_gap_matrices_strict_raises_on_asymmetric_omega():
    n_j = len(JP_TICKERS)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        store_path = _build_gap_store(d)

        store = GapStore(store_path)
        mu = np.zeros(n_j)
        omega = np.eye(n_j)
        omega[0, 1] = 1e-3
        store.save("2026-01-02", mu, omega)

        with pytest.raises(DataValidationError):
            load_gap_matrices(store_path, "2026-01-02", strict=True)


def test_load_gap_matrices_nonstrict_returns_none_for_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        store_path = _build_gap_store(d)

        mu_out, omega_out, alerts = load_gap_matrices(
            store_path, "2026-01-03", strict=False
        )
        assert mu_out is None
        assert omega_out is None
        assert alerts
