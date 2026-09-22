"""Additional entry-point, cache and test-isolation checks for A-C review."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution import var_history
from leadlag.data.gap_store import GapStore
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.data.tickers import JP_TICKERS


def load_module(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


result = {}
cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
result["resolved_config"] = {
    "gap_input": cfg.gap_distribution_dir,
    "ml_enabled": cfg.v2.ml_overlay_enabled,
    "ml_model_dir": cfg.v2.ml_overlay_model_dir,
    "audit_failure_fallback": cfg.v2.fallback_on_audit_failure,
    "ondemand_fallback": cfg.v2.ondemand_fallback_enabled,
    "costs": cfg.v2.costs.model_dump(mode="json"),
    "blpx": {key: getattr(cfg.v2.blpx, key) for key in ("rho", "alpha_xx", "alpha_yx", "lambda_pca", "lambda_sector", "blp_window", "copula_enabled")},
}

# Capture the real Step 2 config resolution and baseline parameters, before
# any data retrieval/computation; all directories it creates are temporary.
step2 = load_module("tools/research/compute_gap_adjusted_distribution.py", "r5_step2")
class Captured(Exception):
    pass
with tempfile.TemporaryDirectory(prefix="abc-r5-step2-") as tmp:
    config_calls, baseline = [], []
    original_loader = step2.load_config_from_yaml
    def load_capture(path, **kwargs):
        value = original_loader(path, **kwargs)
        config_calls.append({"path": str(path), "strict": kwargs.get("strict"), "same_v2": value.v2 == cfg.v2})
        return value
    def info_capture(message, *args, **kwargs):
        if message.startswith("Baseline IR config:"):
            baseline.extend(args)
            raise Captured()
    with patch.object(sys, "argv", ["step2", "--config", "configs/production/production.yaml", "--output-dir", tmp, "--save-to-gap-store", "false"]), \
         patch.object(step2, "load_config_from_yaml", side_effect=load_capture), \
         patch.object(step2.logger, "info", side_effect=info_capture), \
         patch.object(step2, "download_data", side_effect=AssertionError("must stop before download")):
        try:
            step2.main()
        except Captured:
            pass
    result["F06_step2_config"] = {"config_calls": config_calls, "baseline_parameters": baseline}

with tempfile.TemporaryDirectory(prefix="abc-r5-wal-") as tmp:
    path = Path(tmp) / "gap.sqlite"
    store = GapStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        store.save("2026-08-14", np.ones(17), np.eye(17), {"version": "A"})
        old_hash = var_history._file_manifest_fingerprint(path)
        old_main = path.read_bytes()
        store.save("2026-08-14", np.ones(17) * 2, np.eye(17) * 2, {"version": "B"})
        new_hash = var_history._file_manifest_fingerprint(path)
        result["R05_WAL_fingerprint"] = {"main_bytes_unchanged": old_main == path.read_bytes(), "key_changed": old_hash != new_hash, "loaded_mu": float(store.load("2026-08-14")[0][0])}

# Exercise A7's actual default command guard. No legacy backtest is permitted.
a7 = load_module("src/research/scripts/experiments/experiment_a7_walkforward_dsr.py", "r5_a7")
with patch.object(sys, "argv", ["a7"]), patch.object(a7, "load_df_exec_from_local_cache", side_effect=AssertionError("guard must precede data loading")) as loader:
    try:
        a7.main()
        result["F18_A7_guard"] = {"blocked": False}
    except SystemExit as exc:
        result["F18_A7_guard"] = {"blocked": True, "reason": str(exc), "data_load_calls": loader.call_count}

# Run regression in this same process and inspect the patched function after
# pytest fixture teardown. This specifically checks the old R08 order leak.
import leadlag.models.production_v2 as production
original = production.download_macro_prices
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
    code = pytest.main([str(ROOT / "tests/regression/test_v2_baseline.py"), "-q", "--tb=short"])
(OUT / "regression_isolation.log").write_text(buffer.getvalue())
baseline = json.loads((ROOT / "tests/regression/baselines/v2_snapshot_v20260813.json").read_text())
weights = np.asarray(baseline["w_final"])
result["F22_R08_regression_isolation"] = {"pytest_exit": int(code), "macro_function_restored": original is production.download_macro_prices, "model_gross": float(np.abs(weights).sum()), "model_net": float(weights.sum()), "baseline_date": "2026-08-14"}
(OUT / "contracts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
