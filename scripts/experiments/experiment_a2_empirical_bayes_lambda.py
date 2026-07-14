#!/usr/bin/env python
"""A2: BLP Empirical Bayes λ推定実験スクリプント.

設計仕様（docs/design/A_theory_design_specs.md A2参照）:
  Tikhonov解のMAP解釈に基づき、λ_pca / λ_sector / ρ の最適値を探索する。
  各パラメータを現行値の {0.5x, 1x, 2x} で摂動（log-scale 3点）。

  Stage 1: 1パラメータずつ摂動（9 backtests）
  Stage 2: Stage 1の最良組合せで1 backtest

  過学習ガード: 3点のみ、±20%摂動での感度分析を併記。

Usage:
  python3 scripts/experiments/experiment_a2_empirical_bayes_lambda.py \
    --start-date 2015-01-05 --output-dir reports/sprint_a2_empirical_bayes
"""

from __future__ import annotations

import argparse
import copy
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245

# Current production values
BASE_LAMBDA_PCA = 0.10
BASE_LAMBDA_SECTOR = 0.60
BASE_RHO = 0.01

# Grid: 3 log-scale points per parameter
LAMBDA_PCA_GRID = [0.05, 0.10, 0.20]
LAMBDA_SECTOR_GRID = [0.30, 0.60, 1.20]
RHO_GRID = [0.005, 0.01, 0.02]


def compute_metrics(daily_returns: pd.Series, name: str | None = None) -> dict:
    dr = daily_returns.dropna()
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(dr.shape[0])
    m = {"Sharpe_net": sharpe, "AR_net": ar, "Vol_net": vol, "MDD": mdd, "n_days": len(dr)}
    if name:
        m["variant"] = name
    return m


def compute_rank_ic(signals_df: pd.DataFrame, y_target: np.ndarray, sim_dates: pd.DatetimeIndex, start_idx: int) -> tuple[float, float]:
    """Compute mean rank IC and ICIR."""
    ics = []
    T_sig = len(signals_df)
    T_target = len(y_target)
    T_min = min(T_sig, T_target, len(sim_dates))
    for t in range(start_idx, T_min):
        s = signals_df.iloc[t].values
        y = y_target[t]
        valid = np.isfinite(s) & np.isfinite(y)
        if valid.sum() >= 5 and np.std(s[valid]) > 1e-8 and np.std(y[valid]) > 1e-8:
            ics.append(float(spearmanr(s[valid], y[valid])[0]))
    if not ics:
        return 0.0, 0.0
    mean_ic = float(np.mean(ics))
    std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 1e-8
    icir = mean_ic / std_ic * np.sqrt(252) if std_ic > 1e-8 else 0.0
    return mean_ic, icir


def run_single_backtest(cfg: dict, df_exec: pd.DataFrame, start_date: str, slippage_bps: float = 5.0) -> dict:
    """Run a single backtest and return metrics + results."""
    model = SectorRelativeEnsembleBLPEnhancedModel(copy.deepcopy(cfg))
    model._start_date = start_date
    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=start_date, slippage_bps=slippage_bps,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    return results


def make_config_with_lambda(cfg_base: dict, lambda_pca: float, lambda_sector: float, rho: float) -> dict:
    """Create config with specified λ values."""
    cfg = copy.deepcopy(cfg_base)
    if "blpx" not in cfg:
        cfg["blpx"] = {}
    cfg["blpx"]["lambda_pca"] = lambda_pca
    cfg["blpx"]["lambda_sector"] = lambda_sector
    cfg["blpx"]["rho"] = rho
    return cfg


def main():
    parser = argparse.ArgumentParser(description="A2: Empirical Bayes Lambda Estimation")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--output-dir", default="reports/sprint_a2_empirical_bayes")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg_base = yaml.safe_load(f)

    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()
    T = len(df_exec)
    sim_dates = df_exec.index
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime(args.start_date)), 60)

    all_results = []

    # === Baseline ===
    logger.info("=== Baseline (production λ) ===")
    t0 = time.perf_counter()
    results = run_single_backtest(cfg_base, df_exec, args.start_date, args.slippage_bps)
    metrics = compute_metrics(results["daily_returns"], "baseline_production")
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
    metrics["Mean_Rank_IC"] = mean_ic
    metrics["ICIR"] = icir
    metrics["lambda_pca"] = BASE_LAMBDA_PCA
    metrics["lambda_sector"] = BASE_LAMBDA_SECTOR
    metrics["rho"] = BASE_RHO
    all_results.append(metrics)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f (%.1fs)",
                metrics["Sharpe_net"], mean_ic, icir, time.perf_counter() - t0)

    # === Stage 1: One-at-a-time sensitivity ===
    logger.info("\n=== Stage 1: One-at-a-time sensitivity ===")

    # Vary lambda_pca
    for val in LAMBDA_PCA_GRID:
        if val == BASE_LAMBDA_PCA:
            continue
        label = f"lambda_pca={val}"
        logger.info("Testing %s", label)
        cfg = make_config_with_lambda(cfg_base, val, BASE_LAMBDA_SECTOR, BASE_RHO)
        t0 = time.perf_counter()
        results = run_single_backtest(cfg, df_exec, args.start_date, args.slippage_bps)
        metrics = compute_metrics(results["daily_returns"], label)
        mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
        metrics["Mean_Rank_IC"] = mean_ic
        metrics["ICIR"] = icir
        metrics["lambda_pca"] = val
        metrics["lambda_sector"] = BASE_LAMBDA_SECTOR
        metrics["rho"] = BASE_RHO
        all_results.append(metrics)
        logger.info("%s: Sharpe=%.4f IC=%.4f (%.1fs)", label, metrics["Sharpe_net"], mean_ic, time.perf_counter() - t0)

    # Vary lambda_sector
    for val in LAMBDA_SECTOR_GRID:
        if val == BASE_LAMBDA_SECTOR:
            continue
        label = f"lambda_sector={val}"
        logger.info("Testing %s", label)
        cfg = make_config_with_lambda(cfg_base, BASE_LAMBDA_PCA, val, BASE_RHO)
        t0 = time.perf_counter()
        results = run_single_backtest(cfg, df_exec, args.start_date, args.slippage_bps)
        metrics = compute_metrics(results["daily_returns"], label)
        mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
        metrics["Mean_Rank_IC"] = mean_ic
        metrics["ICIR"] = icir
        metrics["lambda_pca"] = BASE_LAMBDA_PCA
        metrics["lambda_sector"] = val
        metrics["rho"] = BASE_RHO
        all_results.append(metrics)
        logger.info("%s: Sharpe=%.4f IC=%.4f (%.1fs)", label, metrics["Sharpe_net"], mean_ic, time.perf_counter() - t0)

    # Vary rho
    for val in RHO_GRID:
        if val == BASE_RHO:
            continue
        label = f"rho={val}"
        logger.info("Testing %s", label)
        cfg = make_config_with_lambda(cfg_base, BASE_LAMBDA_PCA, BASE_LAMBDA_SECTOR, val)
        t0 = time.perf_counter()
        results = run_single_backtest(cfg, df_exec, args.start_date, args.slippage_bps)
        metrics = compute_metrics(results["daily_returns"], label)
        mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
        metrics["Mean_Rank_IC"] = mean_ic
        metrics["ICIR"] = icir
        metrics["lambda_pca"] = BASE_LAMBDA_PCA
        metrics["lambda_sector"] = BASE_LAMBDA_SECTOR
        metrics["rho"] = val
        all_results.append(metrics)
        logger.info("%s: Sharpe=%.4f IC=%.4f (%.1fs)", label, metrics["Sharpe_net"], mean_ic, time.perf_counter() - t0)

    # === Stage 2: Best combination ===
    # Find best per-parameter from Stage 1
    results_df = pd.DataFrame(all_results)
    best_lp = results_df.loc[results_df["Sharpe_net"].idxmax(), "lambda_pca"]
    best_ls = results_df.loc[results_df["Sharpe_net"].idxmax(), "lambda_sector"]
    best_rho_val = results_df.loc[results_df["Sharpe_net"].idxmax(), "rho"]

    # Actually, find best per parameter independently
    lp_results = results_df[results_df["lambda_sector"] == BASE_LAMBDA_SECTOR]
    ls_results = results_df[results_df["lambda_pca"] == BASE_LAMBDA_PCA]
    rho_results = results_df[results_df["lambda_pca"] == BASE_LAMBDA_PCA]

    best_lp = float(lp_results.loc[lp_results["Sharpe_net"].idxmax(), "lambda_pca"])
    best_ls = float(ls_results.loc[ls_results["Sharpe_net"].idxmax(), "lambda_sector"])
    best_rho_val = float(rho_results.loc[rho_results["Sharpe_net"].idxmax(), "rho"])

    logger.info("\n=== Stage 2: Best combination λ_pca=%.3f λ_sector=%.3f ρ=%.4f ===",
                best_lp, best_ls, best_rho_val)
    cfg_best = make_config_with_lambda(cfg_base, best_lp, best_ls, best_rho_val)
    t0 = time.perf_counter()
    results = run_single_backtest(cfg_best, df_exec, args.start_date, args.slippage_bps)
    metrics = compute_metrics(results["daily_returns"], f"best_combo_lp{best_lp}_ls{best_ls}_rho{best_rho_val}")
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
    metrics["Mean_Rank_IC"] = mean_ic
    metrics["ICIR"] = icir
    metrics["lambda_pca"] = best_lp
    metrics["lambda_sector"] = best_ls
    metrics["rho"] = best_rho_val
    all_results.append(metrics)
    logger.info("Best combo: Sharpe=%.4f IC=%.4f (%.1fs)", metrics["Sharpe_net"], mean_ic, time.perf_counter() - t0)

    # === Save results ===
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    # === Sensitivity analysis: ±20% perturbation of best ===
    logger.info("\n=== Sensitivity: ±20%% perturbation of best combo ===")
    sensitivity_results = []
    for lp_mult, ls_mult, rho_mult in itertools.product([0.8, 1.0, 1.2], repeat=3):
        lp = best_lp * lp_mult
        ls = best_ls * ls_mult
        rv = best_rho_val * rho_mult
        label = f"sens_lp{lp:.3f}_ls{ls:.3f}_rho{rv:.4f}"
        cfg = make_config_with_lambda(cfg_base, lp, ls, rv)
        results = run_single_backtest(cfg, df_exec, args.start_date, args.slippage_bps)
        metrics = compute_metrics(results["daily_returns"], label)
        metrics["lambda_pca"] = lp
        metrics["lambda_sector"] = ls
        metrics["rho"] = rv
        sensitivity_results.append(metrics)
        logger.info("%s: Sharpe=%.4f", label, metrics["Sharpe_net"])

    sens_df = pd.DataFrame(sensitivity_results)
    sens_df.to_csv(out_dir / "sensitivity_analysis.csv", index=False)

    # === Report ===
    base_sharpe = all_results[0]["Sharpe_net"]
    best_combo_sharpe = all_results[-1]["Sharpe_net"]
    improvement = (best_combo_sharpe - base_sharpe) / base_sharpe * 100

    # Check stability
    sens_sharpes = [r["Sharpe_net"] for r in sensitivity_results]
    sens_range = (max(sens_sharpes) - min(sens_sharpes)) / max(sens_sharpes) * 100

    report_lines = [
        "# A2: Empirical Bayes Lambda Estimation Report\n",
        f"## Data: {T} rows, start={args.start_date}\n",
        f"\n## Stage 1: One-at-a-time sensitivity\n",
        results_df.to_string(index=False),
        f"\n\n## Stage 2: Best combination\n",
        f"λ_pca={best_lp}, λ_sector={best_ls}, ρ={best_rho_val}\n",
        f"Sharpe: {best_combo_sharpe:.4f} vs baseline {base_sharpe:.4f} ({improvement:+.1f}%)\n",
        f"\n## Sensitivity analysis (±20% perturbation)\n",
        sens_df.to_string(index=False),
        f"\nSensitivity range: {sens_range:.1f}% of max Sharpe\n",
        f"\n## Verdict\n",
    ]

    if improvement > 1.0 and sens_range < 15.0:
        report_lines.append(f"Best combo improves Sharpe by {improvement:.1f}% and is stable (range {sens_range:.1f}%).\n")
        report_lines.append("Recommend: adopt new λ values with walkforward validation.\n")
    elif improvement > 1.0 and sens_range >= 15.0:
        report_lines.append(f"Best combo improves Sharpe by {improvement:.1f}% but is FRAGILE (range {sens_range:.1f}%).\n")
        report_lines.append("Recommend: keep current λ. Improvement is not robust.\n")
    else:
        report_lines.append(f"No improvement from λ optimization ({improvement:+.1f}%).\n")
        report_lines.append("Current production λ values are near-optimal. No change needed.\n")

    report_text = "\n".join(report_lines)
    (out_dir / "a2_empirical_bayes_report.md").write_text(report_text)
    logger.info("Report saved to %s/a2_empirical_bayes_report.md", out_dir)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
