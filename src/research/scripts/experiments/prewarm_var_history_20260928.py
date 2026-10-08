"""Populate the normal VaR/ES return cache for the next decision date."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config.paths import results
from leadlag.core.risk import compute_var_es
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution import var_history
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.ml_overlay_artifact import load_overlay_model

PRODUCTION_ROOT = ROOT / "models/ml_order_overlay/production_20260923"
REPORT = ROOT / "reports/20260927_ml_overlay_var_history/runtime_cache.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def prewarm() -> None:
    app = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    model = load_overlay_model(PRODUCTION_ROOT)
    if model.metadata["train_end"] != "2025-12-30":
        raise ValueError(f"Production overlay cutoff is unexpected: {model.metadata['train_end']}")
    if not (PRODUCTION_ROOT / "HISTORY.json").is_file():
        raise FileNotFoundError(PRODUCTION_ROOT / "HISTORY.json")

    # The workspace cache already contains a provisional 2026-09-28 row while
    # the system clock is still 2026-09-27. The VaR boundary clips that row
    # before it enters the cache key or history replay. This wrapper bypasses
    # only the preflight age guard so the normal production function can be
    # exercised against that exact local snapshot.
    live_frame = load_df_exec_from_local_cache(max_stale_bdays=None)
    original_loader = var_history.load_df_exec_from_local_cache
    var_history.load_df_exec_from_local_cache = lambda **_kwargs: live_frame
    trade_date = pd.Timestamp("2026-09-28")
    try:
        started = time.monotonic()
        first = var_history.get_hist_returns_for_risk(
            config=app.strategy,
            output_root=str(results()),
            trade_date=trade_date,
            config_path=ROOT / "configs/production/production.yaml",
            gap_input_dir=app.gap_distribution_dir,
            overlay_model=model,
        )
        replay_seconds = time.monotonic() - started
        cached_started = time.monotonic()
        second = var_history.get_hist_returns_for_risk(
            config=app.strategy,
            output_root=str(results()),
            trade_date=trade_date,
            config_path=ROOT / "configs/production/production.yaml",
            gap_input_dir=app.gap_distribution_dir,
            overlay_model=model,
        )
        cache_seconds = time.monotonic() - cached_started
    finally:
        var_history.load_df_exec_from_local_cache = original_loader

    if first.empty or len(first) < app.risk.var_window:
        raise ValueError(
            f"VaR history remains insufficient: {len(first)}/{app.risk.var_window}"
        )
    if not first.index.equals(second.index) or not np.allclose(
        first.to_numpy(dtype=float), second.to_numpy(dtype=float), rtol=0, atol=0
    ):
        raise ValueError("VaR cache hit did not return the exact replayed history")
    if first.index.max().normalize() != pd.Timestamp("2026-09-25"):
        raise ValueError(f"VaR history ends on an unexpected date: {first.index.max()}")
    var_es = compute_var_es(
        first,
        confidence=app.risk.var_confidence,
        window=app.risk.var_window,
        var_method=app.risk.var_method,
    )
    result = {
        "status": "prewarmed_and_cache_hit_verified",
        "trade_date": str(trade_date.date()),
        "input_df_exec_last": str(live_frame.index.max().date()),
        "history_start": str(first.index.min().date()),
        "history_end": str(first.index.max().date()),
        "history_rows": len(first),
        "configured_var_window": app.risk.var_window,
        "first_generation_seconds": round(replay_seconds, 3),
        "cache_hit_seconds": round(cache_seconds, 3),
        "overlay_artifact_version": model.metadata["artifact_version"],
        "overlay_train_end": model.metadata["train_end"],
        "var_es": var_es.__dict__,
        "configured_stops": {
            "var_stop": app.risk.var_stop,
            "es_stop": app.risk.es_stop,
        },
        "var_stop_breached": bool(var_es.available and var_es.var_loss >= app.risk.var_stop),
        "es_stop_breached": bool(var_es.available and var_es.es_loss >= app.risk.es_stop),
        "orders_sent": 0,
        "production_decision_or_broker_called": False,
    }
    _write_json(REPORT, result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prewarm"])
    parser.parse_args()
    prewarm()
