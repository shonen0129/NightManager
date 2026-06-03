#!/usr/bin/env python3
"""Compare baseline vs custom w_6 backtests.

Usage: python tools/compare_w6.py [--start-date YYYY-MM-DD]

Creates results/compare_w6_<timestamp>/ with baseline/ and new_w6/ outputs.
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure src is importable like other modules
sys.path.insert(0, os.path.abspath("src"))

from backtest.runner import run_backtest_with_config
from data_loader import (
    load_decision_cache,
    is_decision_cache_valid,
    download_data,
    preprocess_data,
    save_decision_cache,
)
from domain.models.types import StrategyConfig
from config import STRATEGY_DEFAULTS, DEFAULT_START_DATE
from performance import calculate_metrics, generate_report


def build_strategy_config():
    return StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
        dispersion_metric=STRATEGY_DEFAULTS.get(
            "dispersion_metric", "long_short_mean_gap"
        ),
        v3_mode=STRATEGY_DEFAULTS["v3_mode"],
        ewma_half_life=STRATEGY_DEFAULTS["ewma_half_life"],
        lambda_lw=STRATEGY_DEFAULTS["lambda_lw"],
        lw_target=STRATEGY_DEFAULTS["lw_target"],
        corr_window=STRATEGY_DEFAULTS["corr_window"],
        include_v4_prior=STRATEGY_DEFAULTS["include_v4_prior"],
        signal_mode=STRATEGY_DEFAULTS["signal_mode"],
        gap_open_coef=STRATEGY_DEFAULTS["gap_open_coef"],
        gamma=STRATEGY_DEFAULTS.get("gamma", 0.5),
    )


def main(args):
    start_date = args.start_date or DEFAULT_START_DATE

    # Prepare output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("results", f"compare_w6_{ts}")
    os.makedirs(out_root, exist_ok=True)

    # Load or build df_exec
    if is_decision_cache_valid():
        print("Loading df_exec from decision cache...")
        df_exec = load_decision_cache()
    else:
        print(
            "Decision cache not found/invalid. Downloading/preprocessing data (may take time)..."
        )
        data = download_data()
        df_exec = preprocess_data(data)
        try:
            save_decision_cache(df_exec)
        except Exception:
            pass

    cfg = build_strategy_config()

    print("Running baseline backtest (current w_6)...")
    baseline_outdir = os.path.join(out_root, "baseline")
    os.makedirs(baseline_outdir, exist_ok=True)
    baseline_df = run_backtest_with_config(df_exec, cfg, start_date)
    baseline_df.to_csv(
        os.path.join(baseline_outdir, "daily_results.csv"), encoding="utf-8-sig"
    )
    baseline_metrics = calculate_metrics(baseline_df["daily_return"])
    generate_report(baseline_df, baseline_outdir)

    # New w_6 vector provided by user
    new_w6 = np.array(
        [
            +0.8,
            -0.3,
            +1.0,
            +0.3,
            +0.3,
            -0.5,
            -0.2,
            +0.4,
            -0.7,
            -0.2,
            -0.4,
            -0.4,
            +1.0,
            +0.3,
            +0.7,
            -0.2,
            -0.1,
            +0.6,
            +0.2,
            -0.3,
            -0.3,
            -0.8,
            -0.3,
            +0.8,
            -0.5,
            +0.2,
            +0.1,
            +0.3,
        ],
        dtype=float,
    )

    if new_w6.shape[0] != 28:
        raise ValueError("new_w6 must be length 28")

    print("Running backtest with new w_6 override...")
    new_outdir = os.path.join(out_root, "new_w6")
    os.makedirs(new_outdir, exist_ok=True)
    new_df = run_backtest_with_config(df_exec, cfg, start_date, w6_override=new_w6)
    new_df.to_csv(os.path.join(new_outdir, "daily_results.csv"), encoding="utf-8-sig")
    new_metrics = calculate_metrics(new_df["daily_return"])
    generate_report(new_df, new_outdir)

    # Summarize comparison
    keys = ["AR", "RISK", "R/R", "MDD", "Total Return", "Sharpe"]

    comp_rows = []
    print("\n=== Summary comparison (baseline vs new_w6) ===")
    print(f"Output folder: {out_root}")
    header = f"{'Metric':<20} {'Baseline':>16} {'New w_6':>16} {'Delta':>12}"
    print(header)
    print("-" * len(header))
    for k in keys:
        b = baseline_metrics.get(k, float("nan"))
        n = new_metrics.get(k, float("nan"))
        if k in ["MDD"]:
            b_disp = f"{b*100:>7.2f}%" if np.isfinite(b) else "   NaN"
            n_disp = f"{n*100:>7.2f}%" if np.isfinite(n) else "   NaN"
            delta = (n - b) * 100 if np.isfinite(b) and np.isfinite(n) else float("nan")
            delta_disp = f"{delta:>10.2f}%" if np.isfinite(delta) else "       NaN"
            print(f"{k:<20} {b_disp:>16} {n_disp:>16} {delta_disp:>12}")
        elif k in ["Sharpe"]:
            b_disp = f"{b:>8.3f}" if np.isfinite(b) else "   NaN"
            n_disp = f"{n:>8.3f}" if np.isfinite(n) else "   NaN"
            delta = n - b if np.isfinite(b) and np.isfinite(n) else float("nan")
            delta_disp = f"{delta:>10.3f}" if np.isfinite(delta) else "       NaN"
            print(f"{k:<20} {b_disp:>16} {n_disp:>16} {delta_disp:>12}")
        else:
            b_disp = f"{b*100:>7.2f}%" if np.isfinite(b) else "   NaN"
            n_disp = f"{n*100:>7.2f}%" if np.isfinite(n) else "   NaN"
            delta = (n - b) * 100 if np.isfinite(b) and np.isfinite(n) else float("nan")
            delta_disp = f"{delta:>10.2f}%" if np.isfinite(delta) else "       NaN"
            print(f"{k:<20} {b_disp:>16} {n_disp:>16} {delta_disp:>12}")

    # Save comparison CSV
    comp_df = pd.DataFrame({"Baseline": baseline_metrics, "New_w6": new_metrics}).T
    comp_df["Delta"] = comp_df.loc["New_w6"] - comp_df.loc["Baseline"]
    comp_df.to_csv(
        os.path.join(out_root, "metrics_comparison.csv"), encoding="utf-8-sig"
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    args = parser.parse_args()
    main(args)
