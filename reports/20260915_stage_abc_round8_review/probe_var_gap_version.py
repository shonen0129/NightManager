"""Check risk-cache input identity across an interleaved temporary gap update."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from leadlag.execution import var_history
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.backtester import BacktestEngine
from leadlag.data.gap_store import GapStore

with tempfile.TemporaryDirectory(prefix="abc-r5-var-gap-") as tmp:
    root = Path(tmp)
    gap = root / "gap.sqlite"
    writer = GapStore(gap)
    writer.save("2026-08-13", np.ones(17), np.eye(17), {"sig_date": "2026-08-12", "version": "A"})
    original_bytes = gap.read_bytes()
    config_path = root / "config.yaml"
    config_path.write_text(json.dumps({"__base__": str(ROOT / "configs/production/production.yaml"), "ml_overlay_enabled": False, "gap_input_dir": str(gap)}))
    cfg = load_config_from_yaml(config_path, strict=True)
    dates = pd.bdate_range("2026-08-05", "2026-08-13")
    frame = pd.DataFrame({"value": np.arange(len(dates), dtype=float)}, index=dates)
    keys, versions = [], []
    original_get = var_history.SqliteCacheStore.get
    first_lookup = True
    def get_and_commit(store, key, *args, **kwargs):
        global first_lookup
        if str(key).startswith("daily_returns:"):
            keys.append(key)
            if first_lookup:
                first_lookup = False
                writer.save("2026-08-13", np.ones(17) * 2, np.eye(17) * 2, {"sig_date": "2026-08-12", "version": "B"})
        return original_get(store, key, *args, **kwargs)
    def backtest(**kwargs):
        store = GapStore(kwargs["gap_input_dir"])
        mu, omega, meta = store.load("2026-08-13")
        versions.append(meta["version"])
        return {"daily_returns": pd.Series(float(mu[0]) / 100, index=dates)}
    with patch.object(var_history, "load_df_exec_from_local_cache", return_value=frame), \
         patch.object(var_history.SqliteCacheStore, "get", new=get_and_commit), \
         patch.object(BacktestEngine, "run_v2_backtest", side_effect=backtest):
        args = (None, cfg.strategy, str(root / "output"), pd.Timestamp("2026-08-14"))
        first = var_history.get_hist_returns_for_risk(*args, config_path=config_path, gap_input_dir=gap)
        # Restore the original SQLite snapshot only after all gap connections
        # are closed. This is an isolated temporary DB, never the live store.
        sidecars = [path.name for path in root.glob("gap.sqlite-*")]
        assert not sidecars, "cannot restore a DB while WAL/SHM state remains"
        gap.write_bytes(original_bytes)
        second = var_history.get_hist_returns_for_risk(*args, config_path=config_path, gap_input_dir=gap)
    result = {"cache_keys": keys, "same_key": keys[0] == keys[1], "backtest_versions": versions, "cache_key_input_version": "A", "first_marker": float(first.iloc[-1]), "after_snapshot_restore_marker": float(second.iloc[-1]), "expected_A_marker": .01, "note": "Actual VaR function, config loaders, SQLite cache and GapStore; numeric BT replaced with a version marker."}
(OUT / "var_gap_version.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
