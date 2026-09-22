"""Round 4 offline probes: real entry points, temporary files, fake brokers."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Reuse historical probe definitions, without running their old top-level
# calls or overwriting historical evidence. These definitions already use
# temporary artifacts and fake brokers exclusively.
old_path = ROOT / "reports/20260913_stage_abc_round3_review/probe_review.py"
old = types.ModuleType("round3_probe_helpers")
old.__file__ = str(old_path)
sys.modules[old.__name__] = old
source = old_path.read_text().partition('\nprobe("missing_distribution_metadata"')[0]
exec(compile(source, str(old_path), "exec"), old.__dict__)

from leadlag.utils import gap_matrix_io as gap_io
from leadlag.execution import var_history
from leadlag.execution.backtester import BacktestEngine
from leadlag.runner.production import ProductionRunner

results = {}


def run(name, function):
    try:
        results[name] = function()
    except Exception as exc:
        results[name] = {"probe_error": repr(exc), "type": type(exc).__name__}


def outcome(function):
    try:
        value = function()
        return {"accepted": True, "returned_type": type(value).__name__}
    except Exception as exc:
        return {"accepted": False, "error_type": type(exc).__name__, "error": str(exc)}


def artifact_boundaries():
    cases = {}
    with tempfile.TemporaryDirectory(prefix="abc-r4-artifact-") as tmp:
        for name, update in {
            "null": {"train_end": None}, "nat_string": {"train_end": "NaT"},
            "nat_object": {"train_end": pd.NaT}, "empty": {"train_end": ""},
            "list": {"train_end": ["2020-12-31"]},
            "reversed": {"train_start": "2021-01-04"},
            "missing_hash": {"data_hash": " "},
            "future_label": {"label_asof_end": "2021-01-04"},
            "wrong_version": {"metadata_version": 3},
        }.items():
            metadata = {"metadata_version": 2, **old.train_meta("2020-12-31"), **update}
            model = old.dummy_overlay()
            object.__setattr__(model, "metadata", metadata)
            cases[name] = {
                "save": outcome(lambda: old.overlay.save_overlay_model(model, Path(tmp) / name, metadata)),
                "direct_apply": outcome(lambda: old.overlay.apply_overlay(
                    {"fallback": {}}, pd.DataFrame(), model, "2021-01-04")),
            }
        model_path = Path(tmp) / "valid"
        old.overlay.save_overlay_model(old.dummy_overlay(), model_path, old.train_meta("2020-12-31"))
        model = old.overlay.load_overlay_model(model_path)
        for name, date in (("train_end_day", "2020-12-31"), ("first_oos_day", "2021-01-04")):
            base = old.helpers._base_result(np.linspace(-1., 1., len(old.JP_TICKERS)))
            base["w_final"] = np.array([.2] * 5 + [-.2] * 5 + [0.] * 7)
            cases[name] = outcome(lambda: old.overlay.apply_overlay(
                base, old.helpers._make_df_exec(pd.Timestamp(date)), model, date))
    return cases


def legacy_rejection():
    return outcome(old.legacy_artifact_mix)


def npy_republication():
    cases = {}
    with tempfile.TemporaryDirectory(prefix="abc-r4-npy-") as tmp:
        root = Path(tmp)
        date = "2026-08-14"
        n = len(old.JP_TICKERS)
        mu_old, cov_old = np.linspace(-.01, .01, n), np.eye(n) * .001
        mu_new, cov_new = -mu_old, np.eye(n) * .002
        metadata_old = {"sig_date": "2026-08-13", "trade_date": date, "horizon": 1, "marker": "old"}
        metadata_new = {"sig_date": "2026-08-17", "trade_date": date, "horizon": 1, "marker": "new-future"}
        for mode in ("normal", "overwrite_without_metadata", "omega_write_failure", "sidecar_write_failure"):
            directory = root / mode
            assert gap_io.save_gap_matrices(directory, date, mu_old, cov_old, metadata=metadata_old)
            saved = True
            if mode == "overwrite_without_metadata":
                saved = gap_io.save_gap_matrices(directory, date, mu_new, cov_new)
            elif mode == "omega_write_failure":
                real_save = gap_io.np.save
                def failing_save(path, *args, **kwargs):
                    if Path(path).name.startswith("omega_gap"):
                        raise OSError("injected failure before omega replacement")
                    return real_save(path, *args, **kwargs)
                with patch.object(gap_io.np, "save", side_effect=failing_save):
                    saved = gap_io.save_gap_matrices(directory, date, mu_new, cov_new, metadata=metadata_new)
            elif mode == "sidecar_write_failure":
                with patch.object(Path, "write_text", side_effect=OSError("injected sidecar failure")):
                    saved = gap_io.save_gap_matrices(directory, date, mu_new, cov_new, metadata=metadata_new)
            mu, cov, metadata, alerts = gap_io.load_gap_bundle(directory, date)
            decision = old.generate_v2_production_portfolio(date, directory, old.config_for_probe())
            cases[mode] = {
                "save_returned": saved,
                "mu_version": "new" if np.array_equal(mu, mu_new) else "old",
                "omega_version": "new" if np.array_equal(cov, cov_new) else "old",
                "loaded_metadata": metadata, "load_alerts": alerts,
                "leakage": decision["leakage"], "fallback": decision["fallback"],
                "gross": float(np.abs(decision["w_final"]).sum()),
            }
    return cases


def fingerprint_scope():
    with tempfile.TemporaryDirectory(prefix="abc-r4-fingerprint-") as tmp:
        root = Path(tmp)
        old.overlay.save_overlay_model(old.dummy_overlay(), root, old.train_meta("2020-12-31"))
        first = var_history._active_overlay_artifact_fingerprint(root)
        staging = root / ".staging-interrupted"
        staging.mkdir()
        (staging / "model.pkl").write_bytes(b"inactive")
        second = var_history._active_overlay_artifact_fingerprint(root)
        old.overlay.save_overlay_model(old.dummy_overlay(), root, old.train_meta("2020-12-31"))
        third = var_history._active_overlay_artifact_fingerprint(root)
        return {"staging_changes_key": first != second, "current_switch_changes_key": second != third}


def var_version_race():
    # Stub only market-data loading and the expensive numerical BT loop.
    # The real VaR function, real artifact loader and real SQLite return cache
    # are used. The BT stub returns a marker for the version actually loaded.
    with tempfile.TemporaryDirectory(prefix="abc-r4-var-") as tmp:
        root = Path(tmp)
        artifact = root / "overlay"
        old.overlay.save_overlay_model(old.dummy_overlay(), artifact, old.train_meta("2020-12-31"))
        version_a = (artifact / "CURRENT").read_text().strip()
        old.overlay.save_overlay_model(old.dummy_overlay(), artifact, old.train_meta("2020-12-31"))
        version_b = (artifact / "CURRENT").read_text().strip()
        (artifact / "CURRENT").write_text(version_a + "\n")
        v2 = old.cfg.v2.model_copy(deep=True, update={
            "ml_overlay_model_dir": str(artifact), "ml_overlay_enabled": True,
        })
        app_cfg = old.cfg.model_copy(deep=True, update={"v2": v2})
        runner = ProductionRunner(app_cfg)
        dates = pd.bdate_range("2026-08-05", "2026-08-13")
        frame = pd.DataFrame({"input": np.arange(len(dates), dtype=float)}, index=dates)
        loaded_versions, keys = [], []
        original_get = var_history.SqliteCacheStore.get
        flip = True
        def cache_get(store, key, *args, **kwargs):
            nonlocal flip
            keys.append(key)
            if flip:
                (artifact / "CURRENT").write_text(version_b + "\n")
                flip = False
            return original_get(store, key, *args, **kwargs)
        def backtest_stub(**kwargs):
            model = kwargs.get("overlay_model")
            if model is None:
                model = old.overlay.load_overlay_model(Path(kwargs["cfg"].v2.ml_overlay_model_dir))
            version = model.metadata["artifact_version"]
            loaded_versions.append(version)
            value = .01 if version == version_a else .02
            return {"daily_returns": pd.Series(value, index=dates)}
        with patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame), \
             patch.object(var_history, "_load_yaml_with_base", return_value={"ml_overlay_enabled": True, "ml_overlay_model_dir": str(artifact)}), \
             patch.object(var_history, "load_config_from_yaml", return_value=app_cfg), \
             patch.object(var_history.SqliteCacheStore, "get", new=cache_get), \
             patch.object(BacktestEngine, "run_v2_backtest", side_effect=backtest_stub):
            args = (None, SimpleNamespace(start_date="2021-01-04"), str(root / "output"), pd.Timestamp("2026-08-14"))
            first = var_history.get_hist_returns_for_risk(*args)
            (artifact / "CURRENT").write_text(version_a + "\n")
            second = var_history.get_hist_returns_for_risk(*args)
        return {
            "runner_version": runner.model._overlay_model.metadata["artifact_version"],
            "version_a": version_a, "version_b": version_b,
            "backtest_loaded_versions": loaded_versions,
            "cache_keys": keys, "same_cache_key": keys[0] == keys[1],
            "first_value": float(first.iloc[-1]), "after_rollback_value": float(second.iloc[-1]),
            "expected_a_marker": .01, "backtest_count": len(loaded_versions),
        }


def migration_report():
    module_path = ROOT / "src/research/scripts/experiments/fix_overlay_metadata.py"
    spec = importlib.util.spec_from_file_location("round4_migration", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="abc-r4-migration-") as tmp:
        root = Path(tmp)
        artifact = root / "healthy"
        old.overlay.save_overlay_model(old.dummy_overlay(), artifact, old.train_meta("2020-12-31"))
        old.overlay.load_overlay_model(artifact)
        buf = io.StringIO()
        with patch.object(module, "BASE_DIR", root), contextlib.redirect_stdout(buf):
            code = module.main()
        return {"loader_accepts": True, "migration_exit": code, "stdout": buf.getvalue()}


def live_artifact_readiness():
    return outcome(lambda: ProductionRunner(old.cfg))


def verification_dangling_pointer():
    module_path = ROOT / "src/research/scripts/experiments/verify_overlay_models.py"
    spec = importlib.util.spec_from_file_location("round4_verification", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="abc-r4-verification-") as tmp:
        root = Path(tmp)
        artifact = root / "healthy"
        old.overlay.save_overlay_model(old.dummy_overlay(), artifact, old.train_meta("2020-12-31"))
        cases = {}
        for mode in ("healthy", "missing_active_model", "missing_active_version"):
            if mode == "missing_active_model":
                version = (artifact / "CURRENT").read_text().strip()
                (artifact / "versions" / version / "model.pkl").unlink()
            elif mode == "missing_active_version":
                (artifact / "CURRENT").write_text("nonexistent\n")
            buf = io.StringIO()
            with patch.object(module, "MODEL_DIRS", [artifact]), \
                 patch.object(module, "WF_BASE", root / "absent_wf"), contextlib.redirect_stdout(buf):
                code = module.main()
            cases[mode] = {
                "verification_exit": code, "stdout": buf.getvalue(),
                "loader": outcome(lambda: old.overlay.load_overlay_model(artifact)),
            }
        return cases


run("T01_artifact_boundaries", artifact_boundaries)
run("T02_legacy_rejection", legacy_rejection)
run("T03_missing_provenance", old.missing_distribution_provenance)
run("T04_fill_failure", old.fill_failure_status)
run("T05_initial_log_failure", old.failed_initial_log)
run("npy_republication", npy_republication)
run("active_fingerprint_scope", fingerprint_scope)
run("var_version_race", var_version_race)
run("migration_report", migration_report)
run("verification_dangling_pointer", verification_dangling_pointer)
run("live_artifact_readiness", live_artifact_readiness)
(OUT / "reproductions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
if any("probe_error" in value for value in results.values()):
    raise SystemExit(1)
