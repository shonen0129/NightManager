"""Assess P2 severity and a memory-only candidate, without editing source."""
from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution import var_history
from leadlag.execution.config import load_config_from_yaml
from leadlag.utils.threading import run_with_timeout

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def measure(history: HistoricalInputs) -> dict:
    duration = []
    wrapped = []
    digests = set()
    for _ in range(5):
        start = time.monotonic()
        digests.add(history.fingerprint)
        duration.append(time.monotonic() - start)
        start = time.monotonic()
        digests.add(run_with_timeout(lambda: history.fingerprint, timeout=10, label="P2 benchmark"))
        wrapped.append(time.monotonic() - start)
    assert len(digests) == 1
    return {"direct_seconds": duration, "wrapped_seconds": wrapped,
            "direct_median_seconds": statistics.median(duration),
            "wrapped_median_seconds": statistics.median(wrapped),
            "same_digest": True}


def benchmark() -> dict:
    frame = pd.read_csv(ROOT / "tests/regression/baselines/df_exec_20260814.csv.gz",
                        index_col=0, parse_dates=True)
    baseline = measure(HistoricalInputs(frame, source="p2-fixed-frame-benchmark"))
    diag = pd.read_csv(ROOT / "tests/regression/baselines/full_history_diagnostics.csv")
    dates = pd.to_datetime(diag["trade_date"])
    ir_column = "pred_ir_gap_baseline_cost"
    ir = diag[ir_column].to_numpy(dtype=float)
    pit, pit_dates = {}, {}
    sim_dates = frame.index[frame.index >= pd.Timestamp("2015-01-05")]
    for dt in sim_dates:
        selected = dates < dt
        key = dt.strftime("%Y-%m-%d")
        pit[key] = ir[selected]
        pit_dates[key] = dates[selected].to_numpy()
    # Shape-only auxiliary observations: no trading or return calculation uses them.
    auxiliary = pd.DataFrame(np.nan, index=frame.index, columns=[f"x{i}" for i in range(17)])
    rich = HistoricalInputs(
        frame, source="p2-rich-shape-benchmark", pit_ir_history=pit,
        pit_history_trade_dates=pit_dates, open_910_returns=auxiliary,
        macro_prices=auxiliary, adr_features_frame=auxiliary,
        rank_reversal_signals=auxiliary,
        observed_at_by_date={d.strftime("%Y-%m-%d"): {"open_910_returns": f"{d.date()} 09:10"}
                             for d in sim_dates},
    )
    return {"frame_shape": list(frame.shape), "sim_dates": len(sim_dates),
            "diagnostic_rows": len(diag), "pit_elements": sum(len(v) for v in pit.values()),
            "fixed_frame": baseline, "rich_shape": measure(rich),
            "scope": "Actual fixed daily/diagnostic files; 4 synthetic 17-column auxiliary frames. Fingerprinting only, not live-run latency."}


def candidate_module():
    path = Path(var_history.__file__)
    original = path.read_text()
    before = "input_snapshot_hash = historical_input_snapshot.fingerprint"
    after = 'input_snapshot_hash = timed_call(lambda: historical_input_snapshot.fingerprint, "VaR/ES run-input fingerprint")'
    assert original.count(before) == 1
    module = types.ModuleType("leadlag.execution._p2_in_memory_candidate")
    module.__file__ = str(path)
    exec(compile(original.replace(before, after), str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(module, *, delay: float = 0.0, error: bool = False) -> dict:
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    app = app.model_copy(update={"v2": app.v2.model_copy(update={"ml_overlay_enabled": False})})
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2026-08-13", "2026-08-14"]))
    history = HistoricalInputs(frame)
    cached = pd.DataFrame({"daily_return": [0.01, 0.02]}, index=frame.index)
    actual_property = HistoricalInputs.fingerprint
    finished = threading.Event()
    cleanup_state = {}
    keys = []

    def fingerprint(self):
        try:
            if delay:
                time.sleep(delay)
            if error:
                raise ValueError("P2 injected fingerprint error")
            return actual_property.fget(self)
        finally:
            finished.set()

    def get_cache(_self, key):
        keys.append(key)
        return cached

    with tempfile.TemporaryDirectory(prefix="leadlag-p2-assessment-") as root:
        owner = tempfile.TemporaryDirectory(prefix="gap-", dir=root)
        snapshot_path = Path(owner.name)
        with (
            patch.object(module, "load_df_exec_from_local_cache", return_value=frame),
            patch.object(module, "load_config_from_yaml", return_value=app),
            patch.object(module.var_inputs, "_snapshot_gap_input", return_value=(snapshot_path, "gap", owner)),
            patch.object(module, "_build_var_historical_inputs", return_value=history),
            patch.object(HistoricalInputs, "fingerprint", property(fingerprint)),
            patch.object(module.var_inputs, "_file_manifest_fingerprint", return_value="code") as code_hash,
            patch.object(module.SqliteCacheStore, "get", get_cache),
            patch.object(module.var_worker, "run_snapshot_worker") as backtest,
            patch.object(module.var_worker, "_set_cache_with_deadline") as cache_write,
        ):
            start = time.monotonic()
            raised = None
            result = pd.Series(dtype=float)
            try:
                result = module.get_hist_returns_for_risk(
                    None, {"var_history_timeout": 0.1}, root, pd.Timestamp("2026-08-17")
                )
            except ValueError as exc:
                raised = str(exc)
            elapsed = time.monotonic() - start
            cleanup_state["snapshot_removed_at_return"] = not snapshot_path.exists()
            cleanup_state["hash_finished_at_return"] = finished.is_set()
            assert finished.wait(timeout=2)
            cleanup_state["hash_finished_after_wait"] = finished.is_set()
            result_record = {"elapsed_seconds": elapsed, "returned_values": result.tolist(),
                             "raised": raised, "cache_keys": keys,
                             "subsequent_code_hash_called": code_hash.called,
                             "backtest_started": backtest.called, "cache_written": cache_write.called,
                             **cleanup_state}
            assert not backtest.called and not cache_write.called
    return result_record


if __name__ == "__main__":
    candidate, source_hash = candidate_module()
    results = {"benchmark": benchmark(),
               "current_normal": run_case(var_history),
               "candidate_normal": run_case(candidate),
               "current_slow": run_case(var_history, delay=0.5),
               "candidate_slow": run_case(candidate, delay=0.5),
               "candidate_error": run_case(candidate, error=True)}
    assert results["current_normal"]["cache_keys"] == results["candidate_normal"]["cache_keys"]
    assert results["current_normal"]["returned_values"] == results["candidate_normal"]["returned_values"] == [0.01, 0.02]
    assert results["current_slow"]["elapsed_seconds"] >= 0.5
    assert results["candidate_slow"]["elapsed_seconds"] < 0.3
    assert not results["candidate_slow"]["hash_finished_at_return"]
    assert results["candidate_error"]["raised"] == "P2 injected fingerprint error"
    assert all(results[key]["snapshot_removed_at_return"] for key in results if key != "benchmark")
    results["source_sha256"] = source_hash
    assert hashlib.sha256(Path(var_history.__file__).read_bytes()).hexdigest() == source_hash
    results["source_unchanged"] = True
    (OUT / "p2_assessment_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
