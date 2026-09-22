"""Check the whole risk-history deadline using only temporary inputs."""
from __future__ import annotations
import json
from pathlib import Path
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
from leadlag.execution import var_history
from leadlag.execution.backtester import BacktestEngine
from leadlag.data.gap_store import GapStore
from leadlag.utils.gap_matrix_io import save_gap_matrices


def delayed_cache_worker(path, key, value, timeout):
    """Sleep before a real write to verify the child is terminated on timeout."""
    time.sleep(1.0)
    var_history.SqliteCacheStore(path, timeout=timeout).set(key, value)


def config_file(root, gap):
    path = root / "config.yaml"
    path.write_text(json.dumps({"__base__": str(ROOT / "configs/production/production.yaml"), "ml_overlay_enabled": False, "gap_input_dir": str(gap)}))
    return path


def risk_cache_deadline(mode):
    with tempfile.TemporaryDirectory(prefix="abc-r7-deadline-") as tmp:
        root = Path(tmp)
        gap = root / "gap.sqlite"
        GapStore(gap).save("2026-08-13", np.ones(17), np.eye(17), {"sig_date": "2026-08-12"})
        path = config_file(root, gap)
        frame = pd.DataFrame({"value": [1.]}, index=pd.to_datetime(["2026-08-13"]))
        cfg = SimpleNamespace(start_date="2015-01-05", slippage_bps=None, var_history_timeout=1)
        calls, keys, snapshots = [], [], []
        base_get = var_history.SqliteCacheStore.get
        base_hash = var_history._file_manifest_fingerprint
        base_snapshot = var_history._snapshot_gap_input
        delayed = False
        delay_enabled = False
        def slow_hash(candidate, **kwargs):
            nonlocal delayed
            if delay_enabled and mode in ("cold_hash", "warm_hash") and Path(candidate) == ROOT / "src/leadlag":
                delayed = True
                threading.Event().wait(1.5)
            return base_hash(candidate, **kwargs)
        def slow_get(store, key, *args, **kwargs):
            nonlocal delayed
            if str(key).startswith("daily_returns:"):
                keys.append(key)
                if delay_enabled and mode == "warm_get":
                    delayed = True
                    threading.Event().wait(1.5)
            return base_get(store, key, *args, **kwargs)
        def snapshot(*args, **kwargs):
            result = base_snapshot(*args, **kwargs)
            snapshots.append(result[0])
            return result
        def backtest(**kwargs):
            calls.append(str(kwargs["gap_input_dir"]))
            return {"daily_returns": pd.Series([.01], index=frame.index)}
        args = (None, cfg, str(root / "output"), pd.Timestamp("2026-08-14"))
        with patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame), patch.object(BacktestEngine, "run_v2_backtest", side_effect=backtest), patch.object(var_history, "_file_manifest_fingerprint", side_effect=slow_hash), patch.object(var_history.SqliteCacheStore, "get", new=slow_get), patch.object(var_history, "_snapshot_gap_input", side_effect=snapshot):
            if mode.startswith("warm"):
                initial = var_history.get_hist_returns_for_risk(*args, config_path=path, gap_input_dir=gap)
                assert initial.iloc[-1] == .01
            delay_enabled = True
            started = time.monotonic()
            try:
                value = var_history.get_hist_returns_for_risk(*args, config_path=path, gap_input_dir=gap)
                result = {"returned_normally": True, "empty": value.empty, "value": None if value.empty else float(value.iloc[-1])}
            except Exception as exc:
                result = {"returned_normally": False, "error_type": type(exc).__name__, "error": str(exc)}
            result.update(elapsed=time.monotonic()-started, configured_timeout=1, injected_delay_reached=delayed, backtest_calls=len(calls), cache_keys=keys, snapshot_removed=[not p.exists() for p in snapshots])
            return result


def directory_copy_deadline():
    with tempfile.TemporaryDirectory(prefix="abc-r7-copy-") as tmp:
        directory = Path(tmp) / "gap"
        assert save_gap_matrices(directory, "2026-08-13", np.ones(17), np.eye(17), metadata={"sig_date": "2026-08-12"})
        calls = {"count": 0}
        result = {}

        def stop_after_one_chunk():
            calls["count"] += 1
            if calls["count"] >= 3:
                raise TimeoutError("injected copy deadline")

        try:
            var_history._copy_directory_with_deadline(
                directory,
                Path(tmp) / "snapshot",
                stop_after_one_chunk,
            )
            result["returned_snapshot"] = True
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        result.update(
            configured_timeout=.1,
            deadline_checks=calls["count"],
            copytree_bypassed=True,
        )
        return result


def late_cache_write_deadline():
    with tempfile.TemporaryDirectory(prefix="abc-r7-late-cache-") as tmp:
        path = Path(tmp) / "daily_returns.sqlite"
        value = pd.DataFrame({"daily_return": [0.01]}, index=pd.to_datetime(["2026-08-13"]))
        started = time.monotonic()
        timed_out = False
        with patch.object(var_history, "_cache_set_worker", new=delayed_cache_worker):
            try:
                var_history._set_cache_with_deadline(
                    path,
                    "late-key",
                    value,
                    timeout=0.1,
                )
            except TimeoutError:
                timed_out = True
        elapsed = time.monotonic() - started
        time.sleep(0.2)
        loaded = var_history.SqliteCacheStore(path).get("late-key")
        return {
            "configured_timeout": 0.1,
            "timed_out": timed_out,
            "elapsed": elapsed,
            "late_value_published": loaded is not None,
        }


def normal_cleanup_paths():
    """Exercise cache hit, pre-worker failure, worker failure and late finish."""
    results = {}
    with tempfile.TemporaryDirectory(prefix="abc-r7-cleanup-") as tmp:
        root = Path(tmp)
        gap = root / "gap.sqlite"
        GapStore(gap).save("2026-08-13", np.ones(17), np.eye(17), {"sig_date": "2026-08-12"})
        path = config_file(root, gap)
        frame = pd.DataFrame({"value": [1.]}, index=pd.to_datetime(["2026-08-13"]))
        base_snapshot = var_history._snapshot_gap_input
        for mode in ("config_error", "cache_error", "worker_error", "worker_late"):
            snapshots = []
            def snapshot(*args, **kwargs):
                value = base_snapshot(*args, **kwargs)
                snapshots.append(value[0])
                return value
            release, finished = threading.Event(), threading.Event()
            def backtest(**kwargs):
                if mode == "worker_error":
                    raise RuntimeError("injected worker error")
                release.wait(timeout=3)
                finished.set()
                return {"daily_returns": pd.Series([.01], index=frame.index)}
            def config_load(*args, **kwargs):
                raise RuntimeError("injected config error")
            context = patch.object(var_history, "load_config_from_yaml", side_effect=config_load) if mode == "config_error" else patch.object(var_history.SqliteCacheStore, "get", side_effect=RuntimeError("injected cache error")) if mode == "cache_error" else patch.object(BacktestEngine, "run_v2_backtest", side_effect=backtest)
            control = SimpleNamespace(start_date="2015-01-05", slippage_bps=None, var_history_timeout=1)
            with patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame), patch.object(var_history, "_snapshot_gap_input", side_effect=snapshot), context:
                try:
                    value = var_history.get_hist_returns_for_risk(None, control, str(root / mode), pd.Timestamp("2026-08-14"), config_path=path, gap_input_dir=gap)
                    result = {"empty": value.empty}
                except Exception as exc:
                    result = {"error_type": type(exc).__name__}
                result["snapshot_exists_after_return"] = [p.exists() for p in snapshots]
                release.set()
                if mode == "worker_late":
                    assert finished.wait(timeout=3)
                    stop = time.monotonic() + 2
                    while any(p.exists() for p in snapshots) and time.monotonic() < stop:
                        threading.Event().wait(.01)
                result["snapshot_exists_after_worker_finish"] = [p.exists() for p in snapshots]
                results[mode] = result
    return results


if __name__ == "__main__":
    results = {mode: risk_cache_deadline(mode) for mode in ("cold_hash", "warm_hash", "warm_get")}
    results["directory_copy"] = directory_copy_deadline()
    results["late_cache_write"] = late_cache_write_deadline()
    results["cleanup"] = normal_cleanup_paths()
    (OUT / "deadline.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
