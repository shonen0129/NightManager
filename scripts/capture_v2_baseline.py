#!/usr/bin/env python
"""One-off script to capture the current V2 baseline for regression tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model


def _build_current_prices_from_df_exec(
    df_exec: pd.DataFrame,
    trade_date: str,
) -> dict[str, float] | None:
    """Build 09:10 current prices dict from ``jp_open_trade_*`` columns.

    Falls back to ``jp_close_{ticker} * (1 + jp_gap_{ticker})`` when the
    open column is missing.
    """
    if trade_date not in df_exec.index:
        return None
    row = df_exec.loc[trade_date]
    prices = {}
    for t in JP_TICKERS:
        open_col = f"jp_open_trade_{t}"
        gap_col = f"jp_gap_{t}"
        close_col = f"jp_close_{t}"
        if open_col in row.index and not pd.isna(row[open_col]):
            prices[t] = float(row[open_col])
        elif gap_col in row.index and not pd.isna(row[gap_col]) \
                and close_col in row.index and not pd.isna(row[close_col]):
            prices[t] = float(row[close_col]) * (1.0 + float(row[gap_col]))
    if len(prices) == len(JP_TICKERS):
        return prices
    return None


def main() -> int:
    app_config = load_config_from_yaml("configs/production/production.yaml")
    model = ProductionV2Model(app_config.v2)

    # Load a production-like df_exec fixture or the local cache.
    from leadlag.data.market_data_cache import load_df_exec_from_local_cache

    df_exec = load_df_exec_from_local_cache()
    if df_exec is None or df_exec.empty:
        raise RuntimeError("No df_exec available for baseline capture")

    trade_date = df_exec.index[-1].strftime("%Y-%m-%d")
    gap_input_dir = Path("var/live/pipeline_data/gap_adjusted_distribution/latest")

    current_prices = _build_current_prices_from_df_exec(df_exec, trade_date)
    result = model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        current_prices=current_prices,
    )

    baseline = {
        "w_final": result["w_final"].tolist(),
        "scores": result["scores"].tolist(),
        "pit_binning": result["pit_binning"],
        "summary": {k: v for k, v in result["summary"].items() if k not in (
            "trade_date", "version", "candidate"
        )},
    }

    out_dir = Path("tests/regression/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v2_snapshot_v20260813.json", "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    # Copy the gap matrices used for this trade_date so the regression test
    # can reproduce the exact same decision.
    import shutil
    out_matrices = out_dir / "matrices"
    out_matrices.mkdir(parents=True, exist_ok=True)
    src_matrices = gap_input_dir / "matrices"
    date_part = trade_date.replace("-", "")
    for src in src_matrices.glob(f"*_{date_part}.npy"):
        shutil.copy(src, out_matrices / src.name)
    # Also copy any h>1 multi-horizon and rank-reversal files.
    for src in src_matrices.glob(f"*_{date_part}_*.npy"):
        shutil.copy(src, out_matrices / src.name)

    # Copy diagnostics so regression tests can reproduce pit_binning exactly.
    # load_pit_ir_history prefers full_history_diagnostics.csv in gap_input_dir,
    # then in gap_input_dir.parent. Copy both locations.
    for diag_dir in [gap_input_dir, gap_input_dir.parent]:
        for diag_name in ["full_history_diagnostics.csv", "portfolio_gap_distribution_diagnostics.csv"]:
            diag_src = diag_dir / diag_name
            if diag_src.exists():
                shutil.copy(diag_src, out_dir / diag_name)

    print(f"Baseline captured for {trade_date}: {out_dir / 'v2_snapshot_v20260813.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
