#!/usr/bin/env python
"""Experiment script for Asymmetric Propagation and Meta-Learning Model Confidence.

Benchmarks different asymmetry parameters (scalar delta, covariance delta, post-gap delta,
asymmetric gap overrides) and meta-learning configurations.
Saves results under artifacts/asymmetric_propagation/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Add src/ to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine
from leadlag.reporting.metrics import calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PHASE1A_BEST = {
    "rho": 0.01,
    "alpha_xx": 0.20,
    "alpha_yx": 0.15,
    "alpha_yy": 0.50,
    "lambda_pca": 0.10,
    "lambda_sector": 0.60,
    "beta_conf": 0.25,
    "winsor_sigma": 3.0,
    "blp_window": 504,
    "ewma_halflife": 120,
    "sector_eta": 0.5,
    "sector_gamma": 4.0,
}

PHASE1B_WEIGHTS = {
    "raw_blpx": 0.8,
    "raw_pca": 0.2,
    "residual_pca": 0.0,
    "residual_blpx": 0.0,
}

def build_config(asymmetry_params: dict | None = None) -> dict:
    """Load canonical production.yaml and override with phase1 baseline & asymmetry parameters."""
    prod_path = ROOT / "configs" / "production" / "production.yaml"
    with open(prod_path) as f:
        cfg = yaml.safe_load(f)

    # Set up ensemble weight overrides to match PHASE1B_WEIGHTS
    cfg["signal_components"] = {
        "raw_blpx": {"enabled": True, "weight": PHASE1B_WEIGHTS["raw_blpx"]},
        "raw_pca": {"enabled": True, "weight": PHASE1B_WEIGHTS["raw_pca"]},
        "residual_pca": {"enabled": False, "weight": PHASE1B_WEIGHTS["residual_pca"]},
        "residual_blpx": {"enabled": False, "weight": PHASE1B_WEIGHTS["residual_blpx"]},
    }

    # Override blpx parameters with PHASE1A_BEST
    if "blpx" not in cfg:
        cfg["blpx"] = {}
    for k, v in PHASE1A_BEST.items():
        cfg["blpx"][k] = v

    # Override asymmetry and meta-learning parameters
    if asymmetry_params:
        for k, v in asymmetry_params.items():
            cfg["blpx"][k] = v

    return cfg

def run_variant(
    cfg: dict,
    df_exec: pd.DataFrame,
    start_date: str,
    end_date: str,
    slippage_bps: float = 5.0,
) -> dict[str, float]:
    """Run backtest for a single configuration variant and return standard metrics."""
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    model._start_date = start_date
    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=start_date, end_date=end_date, slippage_bps=slippage_bps
    )

    # Calculate standard returns metrics
    m = calculate_metrics(results["daily_returns"])

    # Calculate average daily turnover
    avg_turnover = float(results["daily_turnover"].mean())

    # Compute daily Spearman IC of combined signals against targets
    from leadlag.models.sre import compute_jp_target_returns
    y_jp_target_full = compute_jp_target_returns(df_exec, JP_TICKERS)
    y_jp_target_df = pd.DataFrame(y_jp_target_full, index=df_exec.index, columns=JP_TICKERS)
    y_jp_target = y_jp_target_df.loc[results["signals"].index].values
    sigs = results["signals"].values
    from scipy.stats import spearmanr

    daily_ics = []
    for t_idx in range(len(sigs)):
        sig_t = sigs[t_idx]
        y_t = y_jp_target[t_idx]
        valid = np.isfinite(sig_t) & np.isfinite(y_t)
        if np.sum(valid) >= 5 and np.std(sig_t[valid]) > 1e-8 and np.std(y_t[valid]) > 1e-8:
            daily_ics.append(spearmanr(sig_t[valid], y_t[valid])[0])
        else:
            daily_ics.append(0.0)

    mean_ic = float(np.mean(daily_ics))
    std_ic = float(np.std(daily_ics))
    icir = mean_ic / std_ic * np.sqrt(245) if std_ic > 0 else 0.0

    return {
        "AR": m.get("AR", 0.0),
        "Sharpe": m.get("Sharpe", 0.0),
        "MDD": m.get("MDD", 0.0),
        "Turnover": avg_turnover,
        "IC": mean_ic,
        "ICIR": icir,
    }

def print_results_table(results: list[dict], title: str):
    """Print results list as a clean text table."""
    print(f"\n=== {title} ===")
    headers = ["Variant", "Sharpe", "AR (%)", "MDD (%)", "Turnover (%)", "IC", "ICIR"]
    print(f"{headers[0]:<35} | {headers[1]:>8} | {headers[2]:>8} | {headers[3]:>8} | {headers[4]:>12} | {headers[5]:>8} | {headers[6]:>8}")
    print("-" * 105)
    for r in results:
        print(
            f"{r['Variant']:<35} | "
            f"{r['Sharpe']:>8.2f} | "
            f"{r['AR'] * 100:>8.2f} | "
            f"{r['MDD'] * 100:>8.2f} | "
            f"{r['Turnover'] * 100:>12.2f} | "
            f"{r['IC']:>8.4f} | "
            f"{r['ICIR']:>8.2f}"
        )

def main():
    parser = argparse.ArgumentParser(description="Asymmetric Propagation Experiment Runner")
    parser.add_argument("--start-date", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--end-date", default="2022-12-31", help="Backtest end date")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps per side")
    parser.add_argument("--wfo", action="store_true", help="Run 6-fold walk-forward validation (2020-2025)")
    args = parser.parse_args()

    # Create output directory
    out_dir = ROOT / "artifacts" / "asymmetric_propagation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and preprocess data
    logger.info("Loading market data...")
    raw_data = download_data(beta_window=60)
    df_exec = preprocess_data(raw_data, beta_window=60)

    # Compute TOPIX returns and align
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    all_results = []

    # --- Baseline ---
    logger.info("Running Baseline (No Asymmetry)...")
    baseline_cfg = build_config()
    baseline_metrics = run_variant(baseline_cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
    baseline_metrics["Variant"] = "Baseline (No Asymmetry)"
    all_results.append(baseline_metrics)

    # --- Stage 1: Scalar Delta ---
    logger.info("Running Stage 1: Scalar Delta...")
    scalar_deltas = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5]
    for d in scalar_deltas:
        logger.info(f"  asymmetry_delta = {d}")
        cfg = build_config({"asymmetry_mode": "scalar", "asymmetry_delta": d})
        m = run_variant(cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
        m["Variant"] = f"Scalar Delta (d={d})"
        all_results.append(m)

    # --- Stage 2: Covariance ---
    logger.info("Running Stage 2: Covariance...")
    cfg_cov = build_config({"asymmetry_mode": "covariance", "asymmetry_delta": 0.0})
    m_cov = run_variant(cfg_cov, df_exec, args.start_date, args.end_date, args.slippage_bps)
    m_cov["Variant"] = "Covariance Delta (d=0)"
    all_results.append(m_cov)

    # --- Stage 3: Covariance + Delta ---
    logger.info("Running Stage 3: Covariance + Delta...")
    cov_deltas = [-0.1, -0.05, 0.05, 0.1, 0.2]
    for d in cov_deltas:
        logger.info(f"  covariance asymmetry_delta = {d}")
        cfg = build_config({"asymmetry_mode": "covariance", "asymmetry_delta": d})
        m = run_variant(cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
        m["Variant"] = f"Covariance Delta (d={d})"
        all_results.append(m)

    # --- Stage 4: Post-gap Delta ---
    logger.info("Running Stage 4: Post-gap Delta...")
    post_gap_deltas = [0.05, 0.10, 0.20, 0.30]
    for pgd in post_gap_deltas:
        logger.info(f"  post-gap delta = {pgd}")
        cfg = build_config({
            "asymmetry_post_gap_mode": "signal_split",
            "asymmetry_post_gap_delta": pgd,
            "asymmetry_delta": 0.0
        })
        m = run_variant(cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
        m["Variant"] = f"Post-gap Delta (pg={pgd})"
        all_results.append(m)

    # --- Stage 5: Combined Propagation ---
    logger.info("Running Stage 5: Combined Propagation...")
    combined_post_gaps = [0.10, 0.20, 0.30]
    for pgd in combined_post_gaps:
        logger.info(f"  scalar delta=0.30 + post-gap delta = {pgd}")
        cfg = build_config({
            "asymmetry_mode": "scalar",
            "asymmetry_delta": 0.30,
            "asymmetry_post_gap_mode": "signal_split",
            "asymmetry_post_gap_delta": pgd
        })
        m = run_variant(cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
        m["Variant"] = f"Scalar(0.3) + Post-gap({pgd})"
        all_results.append(m)

    # --- Stage 6: Asymmetric Gap ---
    logger.info("Running Stage 6: Asymmetric Gap...")
    gap_negs = [0.55, 0.60, 0.65, 0.75, 0.80]
    for gn in gap_negs:
        logger.info(f"  gap_open_coef_neg = {gn}")
        cfg = build_config({
            "gap_open_coef_neg": gn,
            "topix_beta_coef_neg": 0.6,
            "asymmetry_delta": 0.0
        })
        m = run_variant(cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
        m["Variant"] = f"Asym Gap (gap_neg={gn})"
        all_results.append(m)

    # --- Stage 7: Combined Gap + Propagation ---
    logger.info("Running Stage 7: Combined Gap + Propagation...")
    combined_gaps = [0.60, 0.65, 0.75]
    for gn in combined_gaps:
        logger.info(f"  scalar delta=0.30 + gap_open_coef_neg = {gn}")
        cfg = build_config({
            "asymmetry_mode": "scalar",
            "asymmetry_delta": 0.30,
            "gap_open_coef_neg": gn,
            "topix_beta_coef_neg": 0.6
        })
        m = run_variant(cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
        m["Variant"] = f"Scalar(0.3) + Gap_neg({gn})"
        all_results.append(m)

    # --- Additional Stage: Meta-Learning Model ---
    logger.info("Running Additional Stage: Meta-Learning Dynamic Ensemble Weight...")
    meta_cfg = build_config({
        "meta_learning_enabled": True,
        "meta_learning_model_type": "logistic_regression",
        "meta_learning_train_window": 252,
        "meta_learning_smooth_factor": 0.2, # smoothed for overnight carryover compat
    })
    m_meta = run_variant(meta_cfg, df_exec, args.start_date, args.end_date, args.slippage_bps)
    m_meta["Variant"] = "Meta-Learning Ensemble (smooth=0.2)"
    all_results.append(m_meta)

    # Output and Save Results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "experiment_asymmetric_propagation_results.csv", index=False)
    print_results_table(all_results, "Asymmetric Propagation and Meta-Learning Backtest Benchmark")

    # Pick best performing variant (by highest Sharpe)
    best_variant = None
    best_sharpe = -999.0
    for r in all_results:
        if "Baseline" in r["Variant"]:
            continue
        if r["Sharpe"] > best_sharpe:
            best_sharpe = r["Sharpe"]
            best_variant = r

    if best_variant:
        logger.info(f"\nBest variant found: {best_variant['Variant']} with Sharpe: {best_variant['Sharpe']:.2f}")

    # --- Stage 8: Walk-Forward Verification (if requested) ---
    if args.wfo and best_variant:
        logger.info("\nRunning Walk-Forward Optimization (6 folds, 2020-2025)...")
        wfo_results = []
        
        best_name = best_variant["Variant"]
        wfo_params = {}
        if "Scalar Delta" in best_name:
            d_val = float(best_name.split("d=")[1].replace(")", ""))
            wfo_params = {"asymmetry_mode": "scalar", "asymmetry_delta": d_val}
        elif "Covariance Delta" in best_name:
            d_val = float(best_name.split("d=")[1].replace(")", ""))
            wfo_params = {"asymmetry_mode": "covariance", "asymmetry_delta": d_val}
        elif "Post-gap Delta" in best_name:
            pg_val = float(best_name.split("pg=")[1].replace(")", ""))
            wfo_params = {"asymmetry_post_gap_mode": "signal_split", "asymmetry_post_gap_delta": pg_val, "asymmetry_delta": 0.0}
        elif "Scalar(0.3) + Post-gap" in best_name:
            pg_val = float(best_name.split("Post-gap(")[1].replace(")", ""))
            wfo_params = {"asymmetry_mode": "scalar", "asymmetry_delta": 0.30, "asymmetry_post_gap_mode": "signal_split", "asymmetry_post_gap_delta": pg_val}
        elif "Asym Gap" in best_name:
            gn_val = float(best_name.split("gap_neg=")[1].replace(")", ""))
            wfo_params = {"gap_open_coef_neg": gn_val, "topix_beta_coef_neg": 0.6, "asymmetry_delta": 0.0}
        elif "Scalar(0.3) + Gap_neg" in best_name:
            gn_val = float(best_name.split("Gap_neg(")[1].replace(")", ""))
            wfo_params = {"asymmetry_mode": "scalar", "asymmetry_delta": 0.30, "gap_open_coef_neg": gn_val, "topix_beta_coef_neg": 0.6}
        elif "Meta-Learning" in best_name:
            wfo_params = {"meta_learning_enabled": True, "meta_learning_model_type": "logistic_regression", "meta_learning_train_window": 252, "meta_learning_smooth_factor": 0.2}

        wfo_cfg = build_config(wfo_params)

        for year in range(2020, 2026):
            wfo_start = f"{year}-01-01"
            wfo_end = f"{year}-12-31"
            logger.info(f"  Running Fold {year}: {wfo_start} to {wfo_end}...")
            
            # Baseline fold
            base_fold_cfg = build_config()
            m_base = run_variant(base_fold_cfg, df_exec, wfo_start, wfo_end, args.slippage_bps)
            m_base["Fold"] = str(year)
            m_base["Model"] = "Baseline"
            wfo_results.append(m_base)

            # Best variant fold
            m_best = run_variant(wfo_cfg, df_exec, wfo_start, wfo_end, args.slippage_bps)
            m_best["Fold"] = str(year)
            m_best["Model"] = best_name
            wfo_results.append(m_best)

        wfo_df = pd.DataFrame(wfo_results)
        wfo_df.to_csv(out_dir / "wfo_asymmetric_propagation_results.csv", index=False)

        # Print WFO results summary
        print("\n=== Walk-Forward Optimization (6 folds, 2020-2025) ===")
        print(f"{'Fold':<6} | {'Model':<35} | {'Sharpe':>8} | {'AR (%)':>8} | {'MDD (%)':>8} | {'IC':>8} | {'ICIR':>8}")
        print("-" * 90)
        for r in wfo_results:
            print(
                f"{r['Fold']:<6} | "
                f"{r['Model']:<35} | "
                f"{r['Sharpe']:>8.2f} | "
                f"{r['AR'] * 100:>8.2f} | "
                f"{r['MDD'] * 100:>8.2f} | "
                f"{r['IC']:>8.4f} | "
                f"{r['ICIR']:>8.2f}"
            )

if __name__ == "__main__":
    main()
