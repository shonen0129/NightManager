"""Tests for gap matrix I/O helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError
from leadlag.utils.gap_matrix_io import load_gap_matrices


def test_load_gap_matrices_nonstrict_returns_warnings_with_arrays():
    """Non-strict mode must return the loaded arrays and the validation alerts."""
    n_j = len(JP_TICKERS)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "matrices").mkdir()
        mu = np.zeros(n_j)
        omega = np.eye(n_j)
        # Make Omega slightly non-symmetric (numerical noise level).
        omega[0, 1] = 1e-3
        np.save(d / "matrices" / "mu_gap_20260101.npy", mu)
        np.save(d / "matrices" / "omega_gap_20260101.npy", omega)

        mu_out, omega_out, alerts = load_gap_matrices(d, "2026-01-01", strict=False)
        assert mu_out is not None
        assert omega_out is not None
        assert alerts
        assert any("symmetric" in a for a in alerts)


def test_load_gap_matrices_strict_raises_on_asymmetric_omega():
    n_j = len(JP_TICKERS)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "matrices").mkdir()
        mu = np.zeros(n_j)
        omega = np.eye(n_j)
        omega[0, 1] = 1e-3
        np.save(d / "matrices" / "mu_gap_20260101.npy", mu)
        np.save(d / "matrices" / "omega_gap_20260101.npy", omega)

        with pytest.raises(DataValidationError):
            load_gap_matrices(d, "2026-01-01", strict=True)


def test_load_gap_matrices_nonstrict_returns_none_for_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        mu_out, omega_out, alerts = load_gap_matrices(d, "2026-01-01", strict=False)
        assert mu_out is None
        assert omega_out is None
        assert alerts
