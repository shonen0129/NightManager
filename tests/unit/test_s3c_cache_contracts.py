"""Regression tests for the S3c bundle and VaR cache contracts."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from leadlag.data.gap_store import GapStore
from leadlag.domain.gap_bundle import GapBundleRef
from leadlag.execution.var_cache import DeadlineBudget, VaRCacheIdentity, build_var_cache_key
from leadlag.utils.distribution_provenance import validate_distribution_provenance
from leadlag.utils.gap_matrix_io import (
    load_gap_bundle,
    load_gap_bundle_manifest,
    save_gap_matrices,
)
from leadlag.utils.gap_provenance import input_version, open_910_version


def test_gap_bundle_ref_round_trip_carries_versions() -> None:
    metadata = {
        "sig_date": "2026-09-15",
        "input_version": "df:abc",
        "model_version": "blpx:v2",
        "config_hash": "cfg:def",
        "ticker_order": ["A", "B"],
    }
    ref = GapBundleRef.from_payload(
        trade_date="2026-09-16",
        horizon=3,
        storage_format="npy",
        mu_bytes=b"mu",
        omega_bytes=b"omega",
        metadata_bytes=b"meta",
        metadata=metadata,
    )
    restored = GapBundleRef.from_dict(ref.to_dict())
    assert restored == ref
    assert restored.input_version == "df:abc"
    assert restored.model_version == "blpx:v2"
    assert restored.config_version == "cfg:def"
    assert restored.validate(
        trade_date="2026-09-16",
        horizon=3,
        storage_format="npy",
        mu_sha256=ref.mu_sha256,
        omega_sha256=ref.omega_sha256,
        metadata_sha256=ref.metadata_sha256,
    ) == []


def test_npy_bundle_manifest_exposes_horizon_and_versions(tmp_path) -> None:
    metadata = {
        "sig_date": "2026-09-15",
        "trade_date": "2026-09-16",
        "horizon": 3,
        "input_version": "df:abc",
        "model_version": "blpx:v2",
        "config_version": "cfg:def",
        "ticker_order": [f"T{i}" for i in range(17)],
    }
    assert save_gap_matrices(
        tmp_path,
        "2026-09-16",
        np.zeros(17),
        np.eye(17),
        mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        pattern_kwargs={"h": 3},
        metadata=metadata,
    )
    manifest_path = tmp_path / "matrices/.mu_gap_h3_20260916.bundle.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == "gap-bundle-v1"
    assert raw["horizon"] == 3
    assert raw["input_version"] == "df:abc"
    assert raw["ticker_order"] == metadata["ticker_order"]
    ref, errors = load_gap_bundle_manifest(
        tmp_path,
        "2026-09-16",
        mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        pattern_kwargs={"h": 3},
    )
    assert errors == []
    assert ref is not None and ref.horizon == 3
    mu, omega, loaded_metadata, alerts = load_gap_bundle(
        tmp_path,
        "2026-09-16",
        mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        pattern_kwargs={"h": 3},
        strict=True,
    )
    assert mu is not None and omega is not None
    assert loaded_metadata == metadata
    assert alerts == []


def test_bundle_identity_is_required_when_requested(tmp_path) -> None:
    assert save_gap_matrices(
        tmp_path,
        "2026-09-16",
        np.zeros(17),
        np.eye(17),
        metadata={
            "sig_date": "2026-09-15",
            "trade_date": "2026-09-16",
            "horizon": 1,
        },
    )
    mu, omega, metadata, alerts = load_gap_bundle(
        tmp_path,
        "2026-09-16",
        strict=False,
        require_metadata=True,
        require_identity=True,
    )
    assert mu is None and omega is None and metadata is None
    assert any("ticker_order" in alert for alert in alerts)


def test_sqlite_bundle_manifest_is_atomic_and_read_in_one_snapshot(tmp_path) -> None:
    store_path = tmp_path / "gap.sqlite"
    metadata = {"sig_date": "2026-09-15", "trade_date": "2026-09-16", "horizon": 1}
    assert save_gap_matrices(
        store_path,
        "2026-09-16",
        np.zeros(17),
        np.eye(17),
        metadata=metadata,
    )
    mu, omega, loaded_metadata, manifest = GapStore(store_path).load_horizon_bundle(
        "2026-09-16"
    )
    assert mu is not None and omega is not None
    assert loaded_metadata == metadata
    assert manifest is not None and manifest.storage_format == "sqlite"
    loaded_ref, errors = load_gap_bundle_manifest(store_path, "2026-09-16")
    assert loaded_ref == manifest
    assert errors == []


def test_sqlite_horizon_bundle_manifest_round_trip(tmp_path) -> None:
    store_path = tmp_path / "gap_horizon.sqlite"
    metadata = {
        "sig_date": "2026-09-15",
        "trade_date": "2026-09-16",
        "horizon": 3,
    }
    assert save_gap_matrices(
        store_path,
        "2026-09-16",
        np.zeros(17),
        np.eye(17),
        mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        pattern_kwargs={"h": 3},
        metadata=metadata,
    )
    ref, errors = load_gap_bundle_manifest(
        store_path,
        "2026-09-16",
        mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        pattern_kwargs={"h": 3},
    )
    assert errors == []
    assert ref is not None and ref.horizon == 3
    mu, omega, loaded_metadata, alerts = load_gap_bundle(
        store_path,
        "2026-09-16",
        mu_pattern="matrices/mu_gap_h{h}_{date}.npy",
        omega_pattern="matrices/omega_gap_h{h}_{date}.npy",
        pattern_kwargs={"h": 3},
        strict=True,
    )
    assert mu is not None and omega is not None
    assert loaded_metadata == metadata
    assert alerts == []


def test_legacy_sqlite_rewrite_invalidates_previous_manifest(tmp_path) -> None:
    store_path = tmp_path / "gap.sqlite"
    assert save_gap_matrices(
        store_path,
        "2026-09-16",
        np.zeros(17),
        np.eye(17),
        metadata={"sig_date": "2026-09-15", "trade_date": "2026-09-16"},
    )
    # A direct legacy writer has no manifest. It must not leave the previous
    # marker attached to the newly written matrices.
    GapStore(store_path).save("2026-09-16", np.ones(17), np.eye(17) * 2)
    ref, errors = load_gap_bundle_manifest(store_path, "2026-09-16")
    assert ref is None
    assert any("manifest missing" in error for error in errors)
    mu, omega, metadata, alerts = load_gap_bundle(
        store_path,
        "2026-09-16",
        strict=True,
        require_metadata=False,
    )
    assert mu is not None and omega is not None
    assert np.allclose(mu, np.ones(17))
    assert metadata == {}
    assert alerts == []


def test_var_cache_identity_includes_all_input_versions() -> None:
    identity = VaRCacheIdentity(
        effective_config={"x": 1},
        start_date="2015-01-05",
        slippage_bps=5.0,
        df_exec_hash="df",
        code_hash="code",
        overlay_identity="overlay",
        gap_input_hash="gap",
    )
    assert identity.cache_key == build_var_cache_key(
        effective_config={"x": 1},
        start_date="2015-01-05",
        slippage_bps=5.0,
        df_exec_hash="df",
        code_hash="code",
        overlay_identity="overlay",
        gap_input_hash="gap",
    )
    changed = identity.__class__(**{**identity.__dict__, "gap_input_hash": "gap-2"})
    assert changed.cache_key != identity.cache_key


def test_provenance_uses_jst_date_for_timezone_aware_trade_date() -> None:
    dates = pd.date_range("2025-06-01", periods=2, freq="D")
    frame = pd.DataFrame({"jp_oc_1617.T": [0.1, 0.2]}, index=dates)
    timezone_trade_date = "2025-06-01T15:00:00+00:00"  # 2025-06-02 00:00 JST

    assert input_version(frame, timezone_trade_date) == input_version(frame, "2025-06-02")
    assert open_910_version(frame, timezone_trade_date) == open_910_version(frame, "2025-06-02")


def test_distribution_provenance_converts_timezone_aware_signal_date_to_jst() -> None:
    # 15:00 UTC is midnight on the following JST trading date.  Removing the
    # offset without converting it first would incorrectly accept this as the
    # previous signal date.
    normalized, error = validate_distribution_provenance(
        {
            "sig_date": "2026-09-15T15:00:00+00:00",
            "trade_date": "2026-09-16",
            "horizon": 1,
        },
        "2026-09-16",
        1,
    )

    assert normalized is None
    assert error is not None and "signal date" in error


def test_deadline_budget_is_absolute() -> None:
    budget = DeadlineBudget.from_timeout(0.05)
    assert budget.remaining() > 0
    time.sleep(0.06)
    try:
        budget.remaining(label="test")
    except TimeoutError as exc:
        assert "test deadline exceeded" in str(exc)
    else:  # pragma: no cover - protects the timeout contract
        raise AssertionError("deadline did not expire")
