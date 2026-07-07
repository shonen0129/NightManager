#!/usr/bin/env python
"""Bayesian Online Update BLPX — Backtest Comparison.

Compares:
  1. Baseline: Residual-BLPX-RA v2 (current production)
  2. Bayesian: Same model + Bayesian online update on B_struct

Metrics: Sharpe, Rank IC, ICIR, MDD, Turnover, AR (annualized return).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.backtest_common import (
    compute_backtest_metrics,
    compute_rank_ic,
    load_cached_df_exec,
    load_config,
    run_backtest_with_costs,
)
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.models.bayesian_blpx import BayesianBLPXModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Bayesian BLPX Backtest Comparison")
    parser.add_argument("--config", default="configs/production/production_v2_primary_ruleD.yaml")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--output-dir", default="artifacts/bayesian_blpx_comparison")
    parser.add_argument("--modes", default="ic,cs_var,kalman",
                        help="Comma-separated Bayesian modes to test: ic,cs_var,kalman")
    # Bayesian params
    parser.add_argument("--eta-base", type=float, default=0.3)
    parser.add_argument("--ic-window", type=int, default=63)
    parser.add_argument("--ic-amplifier", type=float, default=5.0)
    parser.add_argument("--eta-min", type=float, default=0.05)
    parser.add_argument("--eta-max", type=float, default=0.80)
    # cs_var params
    parser.add_argument("--cs-var-window", type=int, default=63)
    parser.add_argument("--cs-var-scale", type=float, default=1.0)
    # kalman params
    parser.add_argument("--kalman-window", type=int, default=63)
    parser.add_argument("--kalman-q-scale", type=float, default=1.0)
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading df_exec from cache...")
    df_exec = load_cached_df_exec()
    logger.info("df_exec shape: %s", df_exec.shape)

    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    start_dt = pd.to_datetime(args.start_date)
    start_idx = max(df_exec.index.searchsorted(start_dt), 60)

    cfg = load_config(str(ROOT / args.config))

    # --- Baseline ---
    logger.info("=== Baseline: Residual-BLPX-RA v2 (no Bayesian update) ===")
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    t0 = time.perf_counter()
    results_base = run_backtest_with_costs(model_base, df_exec, start_date=args.start_date, slippage_bps=args.slippage_bps)
    logger.info("Baseline backtest: %.1fs", time.perf_counter() - t0)

    pred_base = model_base.predict_signals(df_exec)
    metrics_base = compute_backtest_metrics(
        results_base, signals_df=pred_base["signals"], y_target=y_target,
        sim_dates=sim_dates, start_idx=start_idx, include_rank_ic=True)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% AR=%.2f%%",
                metrics_base["Sharpe_net"], metrics_base["Mean_Rank_IC"],
                metrics_base["ICIR"], metrics_base["MDD"] * 100,
                metrics_base["AR_net"] * 100)

    # --- Run all requested Bayesian modes ---
    all_bayes_results = {}  # mode -> (metrics, pred, results)
    for mode in modes:
        logger.info("\n=== Bayesian mode=%s eta_base=%.2f ===", mode, args.eta_base)
        cfg_bayes = dict(cfg)
        cfg_bayes["bayesian_enabled"] = True
        cfg_bayes["bayesian_mode"] = mode
        cfg_bayes["bayesian_eta_base"] = args.eta_base
        cfg_bayes["bayesian_ic_window"] = args.ic_window
        cfg_bayes["bayesian_ic_amplifier"] = args.ic_amplifier
        cfg_bayes["bayesian_eta_min"] = args.eta_min
        cfg_bayes["bayesian_eta_max"] = args.eta_max
        cfg_bayes["bayesian_cs_var_window"] = args.cs_var_window
        cfg_bayes["bayesian_cs_var_scale"] = args.cs_var_scale
        cfg_bayes["bayesian_kalman_window"] = args.kalman_window
        cfg_bayes["bayesian_kalman_q_scale"] = args.kalman_q_scale

        model_bayes = BayesianBLPXModel(cfg_bayes)
        t0 = time.perf_counter()
        results_bayes = run_backtest_with_costs(model_bayes, df_exec, start_date=args.start_date, slippage_bps=args.slippage_bps)
        logger.info("Bayesian[%s] backtest: %.1fs", mode, time.perf_counter() - t0)

        pred_bayes = model_bayes.predict_signals(df_exec)
        metrics_bayes = compute_backtest_metrics(
            results_bayes, signals_df=pred_bayes["signals"], y_target=y_target,
            sim_dates=sim_dates, start_idx=start_idx, include_rank_ic=True)
        logger.info("Bayesian[%s]: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% AR=%.2f%%",
                    mode, metrics_bayes["Sharpe_net"], metrics_bayes["Mean_Rank_IC"],
                    metrics_bayes["ICIR"], metrics_bayes["MDD"] * 100,
                    metrics_bayes["AR_net"] * 100)
        all_bayes_results[mode] = (metrics_bayes, pred_bayes, results_bayes)

    # --- Comparison Table ---
    n_modes = len(modes)
    col_w = 12
    header_modes = "  ".join(f"{m:>12}" for m in modes)
    print("\n" + "=" * (40 + col_w * (n_modes + 1)))
    print("BAYESIAN MODES vs BASELINE COMPARISON")
    print("=" * (40 + col_w * (n_modes + 1)))
    print(f"{'Metric':<20} {'Baseline':>12}  {header_modes}")
    print("-" * (40 + col_w * (n_modes + 1)))
    for key in ["AR_net", "AR_gross", "Vol_net", "Sharpe_net", "Sharpe_monthly",
                "MDD", "Turnover", "GrossExp", "Mean_Rank_IC", "ICIR", "IC_positive_rate"]:
        bv = metrics_base.get(key, np.nan)
        vals = [all_bayes_results[m][0].get(key, np.nan) for m in modes]
        if key in ["MDD"]:
            row = f"{key:<20} {bv*100:>11.2f}%  " + "  ".join(f"{v*100:>11.2f}%" for v in vals)
        elif key in ["Mean_Rank_IC"]:
            row = f"{key:<20} {bv:>12.4f}  " + "  ".join(f"{v:>12.4f}" for v in vals)
        elif key in ["ICIR", "Sharpe_net", "Sharpe_monthly"]:
            row = f"{key:<20} {bv:>12.2f}  " + "  ".join(f"{v:>12.2f}" for v in vals)
        elif key in ["Turnover", "GrossExp"]:
            row = f"{key:<20} {bv:>12.4f}  " + "  ".join(f"{v:>12.4f}" for v in vals)
        else:
            row = f"{key:<20} {bv*100:>11.2f}%  " + "  ".join(f"{v*100:>11.2f}%" for v in vals)
        print(row)
    # Delta row for key metrics
    print("-" * (40 + col_w * (n_modes + 1)))
    for key in ["Sharpe_net", "Mean_Rank_IC", "ICIR", "AR_net"]:
        bv = metrics_base.get(key, np.nan)
        deltas = [all_bayes_results[m][0].get(key, np.nan) - bv for m in modes]
        if key in ["Mean_Rank_IC"]:
            row = f"{key + ' delta':<20} {'':>12}  " + "  ".join(f"{d:>+12.4f}" for d in deltas)
        elif key in ["ICIR", "Sharpe_net"]:
            row = f"{key + ' delta':<20} {'':>12}  " + "  ".join(f"{d:>+12.2f}" for d in deltas)
        else:
            row = f"{key + ' delta':<20} {'':>12}  " + "  ".join(f"{d*100:>+11.2f}%" for d in deltas)
        print(row)

    # --- Save results ---
    results_base["daily_returns"].to_csv(output_dir / "baseline_daily_returns.csv")
    results_base["equity_curve"].to_csv(output_dir / "baseline_equity_curve.csv")
    for mode in modes:
        _, pred_bayes, results_bayes = all_bayes_results[mode]
        results_bayes["daily_returns"].to_csv(output_dir / f"bayesian_{mode}_daily_returns.csv")
        results_bayes["equity_curve"].to_csv(output_dir / f"bayesian_{mode}_equity_curve.csv")
        diag = pred_bayes.get("bayesian_diagnostics")
        if diag is not None and len(diag) > 0:
            diag.to_csv(output_dir / f"bayesian_{mode}_diagnostics.csv")

    # Save metrics
    all_metrics = {"baseline": metrics_base}
    for mode in modes:
        all_metrics[mode] = all_bayes_results[mode][0]
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(output_dir / "metrics_comparison.csv")

    # Print diagnostics for each mode
    for mode in modes:
        diag = all_bayes_results[mode][1].get("bayesian_diagnostics")
        if diag is not None and len(diag) > 0:
            print(f"\n[{mode}] eta stats: mean={diag['eta'].mean():.4f} "
                  f"std={diag['eta'].std():.4f} "
                  f"min={diag['eta'].min():.4f} max={diag['eta'].max():.4f}")
            if "cs_var" in diag.columns:
                cs_col = diag["cs_var"].replace(0, np.nan).dropna()
                if len(cs_col) > 0:
                    print(f"[{mode}] cs_var stats: mean={cs_col.mean():.6f} "
                          f"std={cs_col.std():.6f}")
            if "rolling_ic" in diag.columns:
                ic_col = diag["rolling_ic"].dropna()
                if len(ic_col) > 0:
                    print(f"[{mode}] rolling IC: mean={ic_col.mean():.4f} "
                          f"std={ic_col.std():.4f}")

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
