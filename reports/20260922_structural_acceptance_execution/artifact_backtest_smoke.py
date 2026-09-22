"""Run one fixed, read-only V2 backtest day with the versioned artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml


def main() -> None:
    frame = pd.read_pickle(ROOT / "tests/regression/baselines/df_exec_20260814.pkl")
    if isinstance(frame, dict):
        frame = frame["df_exec"]
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    artifact = ROOT / "var/results/20260920_structural_completion/ml_overlay_retrained"
    gap_store = ROOT / "var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite"
    result = BacktestEngine.run_v2_backtest(
        cfg,
        gap_store,
        frame,
        start_date="2026-08-14",
        end_date="2026-08-14",
        n_jobs=1,
        overlay_model_dir=artifact,
    )
    weights = result["weights"]
    fallback = result.get("daily_fallback")
    turnover = result.get("turnover", result.get("turnovers"))
    if hasattr(turnover, "iloc"):
        turnover_value = float(turnover.iloc[0])
    elif isinstance(turnover, (list, tuple)) and turnover:
        turnover_value = float(turnover[0])
    elif turnover is None:
        turnover_value = None
    else:
        turnover_value = float(turnover)
    print(json.dumps({
        "status": "success",
        "date": "2026-08-14",
        "artifact_root": str(artifact),
        "gap_store": str(gap_store),
        "rows": int(len(weights)),
        "weight_columns": int(weights.shape[1]),
        "fallback": [bool(x) for x in fallback] if fallback is not None else None,
        "gross": float(weights.abs().sum(axis=1).iloc[0]),
        "net": float(weights.sum(axis=1).iloc[0]),
        "turnover": turnover_value,
        "result_keys": sorted(result),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
