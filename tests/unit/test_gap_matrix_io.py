"""Tests for gap matrix I/O helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from leadlag.data.gap_store import GapStore
from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError
from leadlag.utils.gap_matrix_io import load_gap_bundle, load_gap_matrices, save_gap_matrices


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
            store_path, "2026-01-02", strict=False, require_metadata=False
        )
        assert mu_out is not None
        assert omega_out is not None
        assert alerts
        assert any("symmetric" in a for a in alerts)


def test_load_gap_bundle_reads_mu_omega_metadata_from_one_store_snapshot():
    n_j = len(JP_TICKERS)
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "gap.sqlite"
        GapStore(store_path).save(
            "2026-01-02",
            np.zeros(n_j),
            np.eye(n_j),
            metadata={"sig_date": "2026-01-01"},
        )
        mu, omega, metadata, alerts = load_gap_bundle(
            store_path, "2026-01-02", strict=True
        )
        assert mu is not None and omega is not None
        assert metadata == {"sig_date": "2026-01-01"}
        assert alerts == []


def test_npy_bundle_reads_published_provenance_sidecar(tmp_path: Path):
    n_j = len(JP_TICKERS)
    assert save_gap_matrices(
        tmp_path,
        "2026-01-02",
        np.zeros(n_j),
        np.eye(n_j),
        metadata={"sig_date": "2026-01-01", "trade_date": "2026-01-02", "horizon": 1},
    )
    mu, omega, metadata, alerts = load_gap_bundle(tmp_path, "2026-01-02", strict=True)
    assert mu is not None and omega is not None
    assert metadata == {
        "sig_date": "2026-01-01",
        "trade_date": "2026-01-02",
        "horizon": 1,
    }
    assert alerts == []


def test_npy_bundle_rejects_partial_republication_with_old_manifest(tmp_path: Path):
    """A changed matrix cannot be paired with the previous provenance marker."""
    n_j = len(JP_TICKERS)
    assert save_gap_matrices(
        tmp_path,
        "2026-01-02",
        np.zeros(n_j),
        np.eye(n_j),
        metadata={"sig_date": "2026-01-01", "trade_date": "2026-01-02", "horizon": 1},
    )
    np.save(tmp_path / "matrices" / "mu_gap_20260102.npy", np.ones(n_j))

    mu, omega, metadata, alerts = load_gap_bundle(tmp_path, "2026-01-02", strict=False)
    assert mu is None and omega is None
    assert metadata is None
    assert any("consistency check failed" in alert for alert in alerts)


def test_npy_bundle_without_commit_manifest_is_unusable_in_compat_loader(tmp_path: Path):
    n_j = len(JP_TICKERS)
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    np.save(matrix_dir / "mu_gap_20260102.npy", np.zeros(n_j))
    np.save(matrix_dir / "omega_gap_20260102.npy", np.eye(n_j))

    mu, omega, alerts = load_gap_matrices(tmp_path, "2026-01-02", strict=False)
    assert mu is None and omega is None
    assert any(alert.startswith("[FATAL]") for alert in alerts)


def test_compat_loader_rejects_future_provenance_by_default(tmp_path: Path):
    n_j = len(JP_TICKERS)
    assert save_gap_matrices(
        tmp_path,
        "2026-01-02",
        np.zeros(n_j),
        np.eye(n_j),
        metadata={"sig_date": "2026-01-03", "trade_date": "2026-01-02"},
    )
    mu, omega, alerts = load_gap_matrices(tmp_path, "2026-01-02", strict=False)
    assert mu is None and omega is None
    assert any("not before" in alert for alert in alerts)


def test_npy_bundle_without_metadata_does_not_retain_previous_provenance(tmp_path: Path):
    n_j = len(JP_TICKERS)
    assert save_gap_matrices(
        tmp_path,
        "2026-01-02",
        np.zeros(n_j),
        np.eye(n_j),
        metadata={"sig_date": "2026-01-01"},
    )
    assert save_gap_matrices(tmp_path, "2026-01-02", np.ones(n_j), np.eye(n_j) * 2)

    mu, omega, metadata, alerts = load_gap_bundle(tmp_path, "2026-01-02", strict=True)
    assert mu is not None and omega is not None
    assert metadata is None
    assert alerts == []


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "sig_date": "2026-01-01",
            "signal_date": "2026-01-03",
            "trade_date": "2026-01-02",
        },
        {
            "sig_date": "2026-01-01",
            "trade_date": ["2026-01-02"],
        },
        {
            "sig_date": "2026-01-01",
            "trade_date": None,
        },
        {
            "sig_date": "2026-01-01",
            "horizon": float("inf"),
        },
        {
            "sig_date": "NaT",
        },
        {
            "sig_date": "",
        },
    ],
)
def test_compat_loader_rejects_conflicting_or_non_scalar_provenance(
    tmp_path: Path, metadata: dict
):
    n_j = len(JP_TICKERS)
    assert save_gap_matrices(
        tmp_path,
        "2026-01-02",
        np.zeros(n_j),
        np.eye(n_j),
        metadata=metadata,
    )

    mu, omega, alerts = load_gap_matrices(tmp_path, "2026-01-02", strict=False)

    assert mu is None and omega is None
    assert any(alert.startswith("[FATAL]") for alert in alerts)


def test_npy_bundle_normalizes_matching_signal_date_alias(tmp_path: Path):
    n_j = len(JP_TICKERS)
    assert save_gap_matrices(
        tmp_path,
        "2026-01-02",
        np.zeros(n_j),
        np.eye(n_j),
        metadata={
            "sig_date": "2026-01-01T14:00:00Z",
            "signal_date": "2026-01-01",
        },
    )

    mu, omega, alerts = load_gap_matrices(tmp_path, "2026-01-02", strict=False)
    assert mu is not None and omega is not None
    assert alerts == []


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
