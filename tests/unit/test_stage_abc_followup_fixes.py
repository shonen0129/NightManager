"""Regression tests for the round-four Stage A-C fixes."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig
from leadlag.data.gap_store import GapStore
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution import var_history, var_inputs, var_worker
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.ml_order_overlay import MLOrderOverlayModel, save_overlay_model
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint


def _test_app_config(**kwargs):
    app = AppConfig(**kwargs)
    return app.model_copy(update={
        "v2": app.v2.model_copy(update={"ml_overlay_enabled": False}),
        "gap_distribution_dir": kwargs.get("gap_distribution_dir"),
    })


@pytest.fixture(autouse=True)
def isolated_var_config(monkeypatch):
    monkeypatch.setattr(var_history, "load_config_from_yaml", lambda *_args, **_kwargs: _test_app_config())


def _load_script(relative_path: str, name: str):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(name, root / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_overlay(tmp_path: Path) -> Path:
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="test"),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    save_overlay_model(
        model,
        tmp_path,
        training_metadata={
            "metadata_status": "verified",
            "train_start": "2015-01-05",
            "train_end": "2020-12-31",
            "data_hash": "data",
            "config_hash": "config",
        },
    )
    return tmp_path


def test_var_cache_key_uses_selected_overlay_object(tmp_path, monkeypatch):
    keys: list[str] = []
    dates = pd.DatetimeIndex(["2026-08-13"])
    frame = pd.DataFrame({"value": [1.0]}, index=dates)
    cached = pd.DataFrame({"daily_return": [0.01]}, index=dates)
    config = SimpleNamespace(start_date="2015-01-05", slippage_bps=None)
    model_a = SimpleNamespace(metadata={"artifact_version": "A", "model_sha256": "sha-a"})
    model_b = SimpleNamespace(metadata={"artifact_version": "B", "model_sha256": "sha-b"})

    monkeypatch.setattr(var_history, "load_df_exec_from_local_cache", lambda **_: frame)

    def cache_get(_store, key):
        keys.append(key)
        return cached

    monkeypatch.setattr(var_history.SqliteCacheStore, "get", cache_get)
    for model in (model_a, model_b):
        value = var_history.get_hist_returns_for_risk(
            strategy=None,
            config=config,
            output_root=str(tmp_path),
            trade_date=pd.Timestamp("2026-08-14"),
            overlay_model=model,
        )
        assert value.iloc[0] == 0.01
    assert len(keys) == 2
    assert keys[0] != keys[1]


def test_var_backtest_receives_same_selected_overlay_object(tmp_path, monkeypatch):
    frame = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2026-08-13"]))
    config = SimpleNamespace(start_date="2015-01-05", slippage_bps=None, var_history_timeout=5)
    model = SimpleNamespace(metadata={"artifact_version": "A", "model_sha256": "sha-a"})
    app_config = _test_app_config()
    captured: list[object] = []

    monkeypatch.setattr(var_history, "load_df_exec_from_local_cache", lambda **_: frame)
    monkeypatch.setattr(var_history, "load_config_from_yaml", lambda *_args, **_kwargs: app_config)
    monkeypatch.setattr(var_history.SqliteCacheStore, "get", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(var_history, "run_with_timeout", lambda function, **_: function())

    def fake_backtest(**kwargs):
        captured.append(kwargs["overlay_model"])
        return {"daily_returns": pd.Series([0.02], index=frame.index)}

    monkeypatch.setattr(BacktestEngine, "run_v2_backtest", fake_backtest)
    value = var_history.get_hist_returns_for_risk(
        strategy=None,
        config=config,
        output_root=str(tmp_path),
        trade_date=pd.Timestamp("2026-08-14"),
        overlay_model=model,
    )
    assert value.iloc[0] == 0.02
    assert captured == [model]


def test_var_cache_hit_respects_total_deadline(tmp_path, monkeypatch):
    """A slow cache read cannot return a normal series after the deadline."""
    frame = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2026-08-13"]))
    cached = pd.DataFrame(
        {"daily_return": [0.01]}, index=pd.DatetimeIndex(["2026-08-13"])
    )
    config = SimpleNamespace(
        start_date="2015-01-05", slippage_bps=None, var_history_timeout=0.05
    )

    monkeypatch.setattr(var_history, "load_df_exec_from_local_cache", lambda **_: frame)
    monkeypatch.setattr(var_inputs, "_file_manifest_fingerprint", lambda *_args, **_kwargs: "hash")

    def slow_get(*_args, **_kwargs):
        time.sleep(0.2)
        return cached

    monkeypatch.setattr(var_history.SqliteCacheStore, "get", slow_get)
    started = time.monotonic()
    result = var_history.get_hist_returns_for_risk(
        strategy=None,
        config=config,
        output_root=str(tmp_path),
        trade_date=pd.Timestamp("2026-08-14"),
    )

    assert result.empty
    assert time.monotonic() - started < 0.5


def test_var_input_fingerprint_respects_total_deadline(tmp_path, monkeypatch):
    """A slow run-input fingerprint cannot outlive the VaR preparation deadline."""
    frame = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2026-08-13"]))
    config = SimpleNamespace(
        start_date="2015-01-05", slippage_bps=None, var_history_timeout=0.05
    )
    app_config = _test_app_config()
    history = HistoricalInputs(frame, source="fingerprint-timeout-test")
    finished = threading.Event()
    original_fingerprint = type(history).fingerprint.fget

    def slow_fingerprint(self):
        try:
            time.sleep(0.4)
            return original_fingerprint(self)
        finally:
            finished.set()

    monkeypatch.setattr(var_history, "load_df_exec_from_local_cache", lambda **_: frame)
    monkeypatch.setattr(var_history, "load_config_from_yaml", lambda *_args, **_kwargs: app_config)
    monkeypatch.setattr(var_history.var_inputs, "_snapshot_gap_input", lambda *args, **kwargs: (None, "none", None))
    monkeypatch.setattr(var_history, "_build_var_historical_inputs", lambda *args, **kwargs: history)
    monkeypatch.setattr(type(history), "fingerprint", property(slow_fingerprint))

    started = time.monotonic()
    result = var_history.get_hist_returns_for_risk(
        strategy=None,
        config=config,
        output_root=str(tmp_path),
        trade_date=pd.Timestamp("2026-08-14"),
    )
    elapsed = time.monotonic() - started

    assert result.empty
    assert elapsed < 0.3
    assert not finished.is_set()
    assert finished.wait(timeout=1.0)


def test_var_late_cache_write_is_not_adopted(tmp_path, monkeypatch):
    """A cache write that reaches its deadline must return the blocking value."""
    frame = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2026-08-13"]))
    config = SimpleNamespace(
        start_date="2015-01-05", slippage_bps=None, var_history_timeout=1
    )
    app_config = _test_app_config()

    monkeypatch.setattr(var_history, "load_df_exec_from_local_cache", lambda **_: frame)
    monkeypatch.setattr(var_history, "load_config_from_yaml", lambda *_args, **_kwargs: app_config)
    monkeypatch.setattr(var_inputs, "_file_manifest_fingerprint", lambda *_args, **_kwargs: "hash")
    monkeypatch.setattr(var_history.SqliteCacheStore, "get", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(var_history, "run_with_timeout", lambda function, **_: function())
    monkeypatch.setattr(
        BacktestEngine,
        "run_v2_backtest",
        lambda **_: {"daily_returns": pd.Series([0.02], index=frame.index)},
    )

    def fail_cache_write(*_args, **_kwargs):
        raise TimeoutError("late write")

    monkeypatch.setattr(
        var_worker,
        "_set_cache_with_deadline",
        fail_cache_write,
    )

    result = var_history.get_hist_returns_for_risk(
        strategy=None,
        config=config,
        output_root=str(tmp_path),
        trade_date=pd.Timestamp("2026-08-14"),
    )

    assert result.empty


def test_gap_snapshot_keeps_cache_input_immutable_after_writer_update(tmp_path):
    source = tmp_path / "gap.sqlite"
    n_j = 17
    store = GapStore(source)
    store.save(
        "2026-08-13",
        np.ones(n_j) * 1.0,
        np.eye(n_j),
        metadata={"sig_date": "2026-08-12", "version": "A"},
    )

    snapshot_path, snapshot_fingerprint, temporary = var_inputs._snapshot_gap_input(source)
    assert snapshot_path is not None and temporary is not None
    try:
        store.save(
            "2026-08-13",
            np.ones(n_j) * 2.0,
            np.eye(n_j),
            metadata={"sig_date": "2026-08-12", "version": "B"},
        )
        loaded_mu, _loaded_omega, metadata = GapStore(snapshot_path).load("2026-08-13")
        assert loaded_mu is not None and loaded_mu[0] == 1.0
        assert metadata == {"sig_date": "2026-08-12", "version": "A"}
        assert snapshot_fingerprint == var_inputs._file_manifest_fingerprint(snapshot_path)
    finally:
        temporary.cleanup()


def test_gap_snapshot_respects_timeout_during_sqlite_lock(tmp_path):
    source = tmp_path / "locked.sqlite"
    GapStore(source).save(
        "2026-08-13",
        np.ones(17),
        np.eye(17),
        metadata={"sig_date": "2026-08-12"},
    )
    blocker = sqlite3.connect(source)
    blocker.execute("PRAGMA locking_mode=EXCLUSIVE")
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            var_inputs._snapshot_gap_input(source, timeout=0.25)
        assert time.monotonic() - started < 2.0
    finally:
        blocker.rollback()
        blocker.close()


def test_var_timeout_keeps_snapshot_until_worker_finishes(tmp_path, monkeypatch):
    source = tmp_path / "gap.sqlite"
    GapStore(source).save(
        "2026-08-13",
        np.ones(17),
        np.eye(17),
        metadata={"sig_date": "2026-08-12"},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps({
            "__base__": str(Path(__file__).resolve().parents[2] / "configs/production/production.yaml"),
            "ml_overlay_enabled": False,
            "gap_input_dir": str(source),
        })
    )
    frame = pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex(["2026-08-13"]))
    app_config = _test_app_config(gap_distribution_dir=str(source))
    config = SimpleNamespace(start_date="2015-01-05", slippage_bps=None, var_history_timeout=1)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    observed: dict[str, object] = {}

    monkeypatch.setattr(var_history, "load_df_exec_from_local_cache", lambda **_: frame)
    monkeypatch.setattr(var_history, "load_config_from_yaml", lambda *_args, **_kwargs: app_config)
    monkeypatch.setattr(var_history.SqliteCacheStore, "get", lambda *_args, **_kwargs: None)

    def fake_backtest(**kwargs):
        snapshot = Path(kwargs["gap_input_dir"])
        observed["path"] = snapshot
        observed["exists_at_start"] = snapshot.exists()
        started.set()
        release.wait(timeout=5)
        observed["exists_after_release"] = snapshot.exists()
        finished.set()
        return {"daily_returns": pd.Series([0.02], index=frame.index)}

    monkeypatch.setattr(BacktestEngine, "run_v2_backtest", fake_backtest)
    assert started.is_set() is False
    result = var_history.get_hist_returns_for_risk(
        strategy=None,
        config=config,
        output_root=str(tmp_path / "output"),
        trade_date=pd.Timestamp("2026-08-14"),
        config_path=config_path,
        gap_input_dir=source,
    )

    assert result.empty
    snapshot = observed["path"]
    assert isinstance(snapshot, Path)
    assert observed["exists_at_start"] is True
    assert snapshot.exists()
    release.set()
    assert finished.wait(timeout=3)
    assert observed["exists_after_release"] is True
    cleanup_deadline = time.monotonic() + 2.0
    while snapshot.exists() and time.monotonic() < cleanup_deadline:
        time.sleep(0.01)
    assert not snapshot.exists()


def test_dataframe_fingerprint_includes_column_identity():
    frame = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        columns=["us_cc_XLB", "us_cc_XLC"],
    )
    swapped = frame.copy()
    swapped.columns = ["us_cc_XLC", "us_cc_XLB"]
    assert dataframe_fingerprint(frame) != dataframe_fingerprint(swapped)


def test_verify_tool_fails_on_missing_active_model(tmp_path):
    module = _load_script(
        "src/research/scripts/experiments/verify_overlay_models.py", "verify_overlay_models_test"
    )
    artifact = _valid_overlay(tmp_path / "artifact")
    version = (artifact / "CURRENT").read_text(encoding="utf-8").strip()
    (artifact / "versions" / version / "model.pkl").unlink()
    assert module.verify_dir(artifact) is False


def test_migration_tool_does_not_scan_internal_versions(tmp_path):
    module = _load_script(
        "src/research/scripts/experiments/fix_overlay_metadata.py", "fix_overlay_metadata_test"
    )
    artifact = _valid_overlay(tmp_path / "artifact")
    assert list(module._iter_artifact_roots(tmp_path)) == [artifact]
