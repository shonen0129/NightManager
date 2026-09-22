"""Independent acceptance probes for the Round 7 fixes; temporary data only."""
from __future__ import annotations

from contextlib import ExitStack
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from leadlag.execution import var_history as vh
from leadlag.execution.backtester import BacktestEngine
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.validation import DataValidationError
from leadlag.utils.distribution_provenance import validate_distribution_provenance
from leadlag.utils.gap_matrix_io import load_gap_bundle, save_gap_matrices


def provenance_fields():
    results = []
    with tempfile.TemporaryDirectory(prefix="abc-r8-provenance-") as tmp:
        for backend in ("npy", "sqlite"):
            for horizon in (1, 3, 5):
                patterns = {} if horizon == 1 else {"mu_pattern": "matrices/mu_gap_h{h}_{date}.npy", "omega_pattern": "matrices/omega_gap_h{h}_{date}.npy", "pattern_kwargs": {"h": horizon}}
                for field in ("sig_date", "signal_date", "trade_date"):
                    for name, value in (("nat", "NaT"), ("empty", "")):
                        path = Path(tmp) / f"{backend}-{horizon}-{field}-{name}"
                        if backend == "sqlite":
                            path = path.with_suffix(".sqlite")
                        metadata = {"sig_date": "2026-08-13", "trade_date": "2026-08-14", "horizon": horizon, field: value}
                        assert save_gap_matrices(path, "2026-08-14", np.ones(17), np.eye(17), metadata=metadata, **patterns)
                        mu, omega, meta, alerts = load_gap_bundle(path, "2026-08-14", require_metadata=True, **patterns)
                        strict_rejected = False
                        try:
                            load_gap_bundle(path, "2026-08-14", require_metadata=True, strict=True, **patterns)
                        except DataValidationError:
                            strict_rejected = True
                        normalized, error = validate_distribution_provenance(metadata, "2026-08-14", horizon)
                        results.append({"backend": backend, "horizon": horizon, "field": field, "value": name, "passed": mu is None and omega is None and bool(alerts) and strict_rejected and normalized is None and bool(error)})
    direct = []
    for field in ("sig_date", "signal_date", "trade_date", "requested_trade_date"):
        for name, value in (("none", None), ("nat_object", pd.NaT), ("numpy_nat", np.datetime64("NaT")), ("nat_text", "NaT"), ("empty", "")):
            metadata = {"sig_date": "2026-08-13", "trade_date": "2026-08-14"}
            requested = "2026-08-14"
            if field == "requested_trade_date":
                requested = value
            else:
                metadata[field] = value
            normalized, error = validate_distribution_provenance(metadata, requested, 1)
            direct.append({"field": field, "value": name, "passed": normalized is None and bool(error)})
    return {"loader_cases": results, "direct_validator_cases": direct, "passed": all(x["passed"] for x in results + direct)}


def delayed_preparation(mode):
    """Hold a real operation boundary until after the caller has returned."""
    with tempfile.TemporaryDirectory(prefix="abc-r8-wait-") as tmp:
        root = Path(tmp)
        gap = root / "gap"
        assert save_gap_matrices(gap, "2026-08-13", np.ones(17), np.eye(17), metadata={"sig_date": "2026-08-12"})
        config_path = root / "config.yaml"
        config_path.write_text(json.dumps({"__base__": str(ROOT / "configs/production/production.yaml"), "ml_overlay_enabled": False, "gap_input_dir": str(gap)}))
        frame = pd.DataFrame({"value": [1.]}, index=pd.to_datetime(["2026-08-13"]))
        cfg = SimpleNamespace(start_date="2015-01-05", slippage_bps=None, var_history_timeout=.25)
        reached, release, finished = threading.Event(), threading.Event(), threading.Event()
        copied_paths, calls = [], []
        base_copy = vh._copy_file_with_deadline
        base_load = vh._load_yaml_with_base
        base_store_init = SqliteCacheStore._init_db

        def delayed(fn, *args, **kwargs):
            reached.set()
            assert release.wait(timeout=4)
            try:
                return fn(*args, **kwargs)
            finally:
                finished.set()

        def slow_copy(source, target, remaining):
            copied_paths.append(target.parents[1])
            return delayed(base_copy, source, target, remaining)

        def slow_load(*args, **kwargs):
            return delayed(base_load, *args, **kwargs)

        def slow_init(store):
            return delayed(base_store_init, store)

        def backtest(**kwargs):
            calls.append(True)
            return {"daily_returns": pd.Series([.01], index=frame.index)}

        with ExitStack() as stack:
            stack.enter_context(patch.object(vh, "load_df_exec_from_local_cache", return_value=frame))
            stack.enter_context(patch.object(BacktestEngine, "run_v2_backtest", side_effect=backtest))
            if mode == "directory_file_copy":
                stack.enter_context(patch.object(vh, "_copy_file_with_deadline", side_effect=slow_copy))
            elif mode == "config_load":
                stack.enter_context(patch.object(vh, "_load_yaml_with_base", side_effect=slow_load))
            elif mode == "cache_init":
                stack.enter_context(patch.object(SqliteCacheStore, "_init_db", new=slow_init))
            started = time.monotonic()
            value = vh.get_hist_returns_for_risk(None, cfg, str(root / "output"), pd.Timestamp("2026-08-14"), config_path=config_path, gap_input_dir=gap)
            elapsed = time.monotonic() - started
            result = {"fault_reached": reached.is_set(), "empty": value.empty, "configured_timeout": .25, "elapsed": elapsed, "worker_pending_after_return": not finished.is_set(), "backtest_calls": len(calls)}
            release.set()
            assert finished.wait(timeout=3)
            stop = time.monotonic() + 2
            while any(p.exists() for p in copied_paths) and time.monotonic() < stop:
                threading.Event().wait(.01)
            result["snapshot_removed_after_release"] = all(not p.exists() for p in copied_paths)
        store = SqliteCacheStore(root / "output/.cache/daily_returns.sqlite")
        result["cached_values_after_release"] = store.keys()
        result["passed"] = result["fault_reached"] and result["empty"] and elapsed < .65 and result["worker_pending_after_return"] and not calls and result["snapshot_removed_after_release"] and not store.keys()
        return result


def cache_write_process():
    """Real fork/spawn writer, warm parent reader, real SQLite lock and failure."""
    results = {}
    with tempfile.TemporaryDirectory(prefix="abc-r8-cache-") as tmp:
        path = Path(tmp) / "cache.sqlite"
        store = SqliteCacheStore(path)
        frame = pd.DataFrame({"daily_return": np.linspace(-.01, .01, 3000)}, index=pd.bdate_range("2015-01-05", periods=3000))
        children_before = {child.pid for child in multiprocessing.active_children()}
        runs = []
        for index in range(3):
            started = time.monotonic()
            vh._set_cache_with_deadline(path, f"normal-{index}", frame, timeout=3)
            loaded = store.get(f"normal-{index}")
            runs.append({"elapsed": time.monotonic()-started, "equal": frame.equals(loaded)})
        results["normal_after_warm_parent_read"] = runs
        conn = sqlite3.connect(path, isolation_level=None)
        conn.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        try:
            vh._set_cache_with_deadline(path, "locked", frame, timeout=.15)
            locked = {"error": None}
        except Exception as exc:
            locked = {"error": type(exc).__name__}
        locked["elapsed"] = time.monotonic()-started
        conn.execute("ROLLBACK")
        conn.close()
        threading.Event().wait(.3)
        locked["late_value_published"] = store.get("locked") is not None
        results["real_sqlite_write_lock"] = locked
        try:
            vh._set_cache_with_deadline(path, "normal-0", object(), timeout=3)
            results["serialization_error"] = {"error": None}
        except Exception as exc:
            results["serialization_error"] = {"error": type(exc).__name__, "message": str(exc)}
        results["serialization_error"]["prior_value_preserved"] = frame.equals(store.get("normal-0"))
        results["new_children_remaining"] = [p.pid for p in multiprocessing.active_children() if p.pid not in children_before]
        results["passed"] = all(x["equal"] for x in runs) and locked["error"] == "TimeoutError" and locked["elapsed"] < .6 and not locked["late_value_published"] and results["serialization_error"]["error"] == "RuntimeError" and results["serialization_error"]["prior_value_preserved"] and not results["new_children_remaining"]
    return results


if __name__ == "__main__":
    result = {"provenance_fields": provenance_fields(), "delayed_preparation": {mode: delayed_preparation(mode) for mode in ("directory_file_copy", "config_load", "cache_init")}, "cache_write_process": cache_write_process()}
    result["passed"] = result["provenance_fields"]["passed"] and all(x["passed"] for x in result["delayed_preparation"].values()) and result["cache_write_process"]["passed"]
    (OUT / "additional.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
