"""Offline boundary checks for newly added bundle guards and gap snapshots."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import subprocess
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
from leadlag.utils import gap_matrix_io as gio
from leadlag.models.v2.distribution_source import _validate_distribution_metadata
from leadlag.models.v2 import gap_io
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.signal_enhancement import apply_multi_horizon_blend
from leadlag.execution.config import load_config_from_yaml


def config_file(root, gap):
    path = root / "config.yaml"
    path.write_text(json.dumps({"__base__": str(ROOT / "configs/production/production.yaml"), "ml_overlay_enabled": False, "gap_input_dir": str(gap)}))
    return path


def lock_child(root):
    frame = pd.DataFrame({"value": [1.0]}, index=pd.to_datetime(["2026-08-13"]))
    control = SimpleNamespace(start_date="2015-01-05", slippage_bps=None, var_history_timeout=1)
    calls = []
    def backtest(**kwargs):
        calls.append(str(kwargs["gap_input_dir"]))
        return {"daily_returns": pd.Series([.01], index=frame.index)}
    with patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame), patch.object(BacktestEngine, "run_v2_backtest", side_effect=backtest):
        print("READY", flush=True)
        start = time.monotonic()
        result = var_history.get_hist_returns_for_risk(None, control, str(root / "output"), pd.Timestamp("2026-08-14"), config_path=root / "config.yaml", gap_input_dir=root / "gap.sqlite")
        print(json.dumps({"elapsed": time.monotonic() - start, "configured_timeout": 1, "empty": result.empty, "backtest_calls": len(calls)}), flush=True)


def locked_snapshot():
    with tempfile.TemporaryDirectory(prefix="abc-r6-lock-") as tmp:
        root = Path(tmp)
        gap = root / "gap.sqlite"
        GapStore(gap).save("2026-08-13", np.ones(17), np.eye(17), {"sig_date": "2026-08-12"})
        config_file(root, gap)
        blocker = sqlite3.connect(gap)
        blocker.execute("PRAGMA locking_mode=EXCLUSIVE")
        blocker.execute("BEGIN EXCLUSIVE")
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--lock-child", str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            ready = child.stdout.readline().strip()
            assert ready == "READY", ready
            # Confirm that even sqlite connect(timeout=30) is not a total
            # backup deadline. All locks and files belong to this probe.
            try:
                child.wait(timeout=35)
                pending = False
            except subprocess.TimeoutExpired:
                pending = True
            blocker.rollback()
            blocker.close()
            output, error = child.communicate(timeout=15)
            child_result = json.loads(output.strip().splitlines()[-1])
            return {"still_pending_at_35_seconds": pending, "child_exit": child.returncode, "after_lock_release": child_result, "stderr": error}
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            try:
                blocker.close()
            except sqlite3.Error:
                pass


def metadata_boundaries():
    output = {}
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    cases = {
        "valid": {"sig_date": "2026-08-13", "trade_date": "2026-08-14"},
        "alias_future": {"sig_date": "2026-08-13", "signal_date": "2026-08-17", "trade_date": "2026-08-14"},
        "trade_date_null": {"sig_date": "2026-08-13", "trade_date": None},
        "nat_signal": {"sig_date": "NaT", "trade_date": "2026-08-14"},
        "empty_signal": {"sig_date": "", "trade_date": "2026-08-14"},
        "trade_date_list": {"sig_date": "2026-08-13", "trade_date": ["2026-08-14"]},
        "horizon_inf": {"sig_date": "2026-08-13", "horizon": float("inf")},
    }
    with tempfile.TemporaryDirectory(prefix="abc-r6-meta-") as tmp:
        for name, metadata in cases.items():
            for horizon in (1, 3, 5):
                directory = Path(tmp) / name / str(horizon)
                patterns = {} if horizon == 1 else {"mu_pattern": "matrices/mu_gap_h{h}_{date}.npy", "omega_pattern": "matrices/omega_gap_h{h}_{date}.npy", "pattern_kwargs": {"h": horizon}}
                meta = {"horizon": horizon, **metadata}
                mu = np.linspace(-.02, .02, 17)[::-1]
                assert gio.save_gap_matrices(directory, "2026-08-14", mu, np.eye(17) * .001, metadata=meta, **patterns)
                entry = {}
                try:
                    loaded_mu, omega, alerts = gio.load_gap_matrices(directory, "2026-08-14", **patterns)
                    entry["loader"] = {"available": loaded_mu is not None, "alerts": alerts}
                except Exception as exc:
                    entry["loader"] = {"error": type(exc).__name__, "message": str(exc)}
                try:
                    _, validation_alerts = _validate_distribution_metadata(meta, "2026-08-14", horizon)
                    entry["full_validator_alerts"] = validation_alerts
                except Exception as exc:
                    entry["full_validator_error"] = type(exc).__name__
                model = ProductionV2Model(cfg.v2, blpx_model=SimpleNamespace())
                model._current_gap_input_dir = directory
                with patch.object(gap_io, "_compute_ondemand", return_value=(np.zeros(17), np.eye(17))) as demand:
                    try:
                        actual, _ = model.compute_distribution("2026-08-14", pd.DataFrame(index=pd.to_datetime(["2026-08-14"])), {}, horizon=horizon)
                        entry["compute_distribution"] = {"accepted_cache": bool(np.array_equal(actual, mu))}
                    except Exception as exc:
                        entry["compute_distribution"] = {"error": type(exc).__name__, "message": str(exc)}
                    entry["compute_distribution"]["ondemand_calls"] = demand.call_count
                if horizon != 1:
                    try:
                        scores = np.linspace(-2., 2., 17)
                        actual, alerts = apply_multi_horizon_blend(scores, directory, "2026-08-14", (1, horizon), (.5, .5))
                        entry["blend"] = {"max_score_change": float(np.max(np.abs(actual - scores))), "alerts": alerts}
                    except Exception as exc:
                        entry["blend"] = {"error": type(exc).__name__, "message": str(exc)}
                output[f"{name}_h{horizon}"] = entry
    return output


def snapshot_timeout_lifetime():
    with tempfile.TemporaryDirectory(prefix="abc-r6-lifetime-") as tmp:
        root = Path(tmp)
        gap = root / "gap.sqlite"
        GapStore(gap).save("2026-08-13", np.ones(17), np.eye(17), {"sig_date": "2026-08-12"})
        config_path = config_file(root, gap)
        frame = pd.DataFrame({"value": [1.]}, index=pd.to_datetime(["2026-08-13"]))
        release, done = threading.Event(), threading.Event()
        observed = {}
        def backtest(**kwargs):
            path = kwargs["gap_input_dir"]
            observed["path"] = str(path)
            observed["exists_before_timeout"] = path.exists()
            release.wait(timeout=5)
            observed["exists_when_worker_resumes"] = path.exists()
            done.set()
            return {"daily_returns": pd.Series([.01], index=frame.index)}
        control = SimpleNamespace(start_date="2015-01-05", slippage_bps=None, var_history_timeout=1)
        with patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame), patch.object(BacktestEngine, "run_v2_backtest", side_effect=backtest):
            result = var_history.get_hist_returns_for_risk(None, control, str(root / "output"), pd.Timestamp("2026-08-14"), config_path=config_path, gap_input_dir=gap)
            observed["caller_returned_empty"] = result.empty
            observed["exists_after_caller_timeout"] = Path(observed["path"]).exists()
            release.set()
            assert done.wait(timeout=3)
        return observed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--lock-child":
        lock_child(Path(sys.argv[2]))
    else:
        result = {"metadata": metadata_boundaries(), "snapshot_timeout_lifetime": snapshot_timeout_lifetime(), "locked_snapshot": locked_snapshot()}
        (OUT / "boundaries.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
