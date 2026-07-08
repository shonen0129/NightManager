#!/usr/bin/env python
"""Compare macro sensitivity matrices: original vs derived (w4/w5/w6-based).

Runs backtests with:
  1. Original MACRO_SENS_MATRIX (hand-tuned domain knowledge)
  2. MACRO_SENS_MATRIX_DERIVED (w4/w5/w6 JP abs + domain corrections)
  3. Macro confidence disabled (baseline)

Reports Sharpe, AR, MDD, turnover, and scale distribution statistics.

Usage:
  python3 scripts/experiments/compare_sensitivity_matrix.py
  python3 scripts/experiments/compare_sensitivity_matrix.py --start-date 2015-01-05
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import load_execution_data
from leadlag.core.macro import (
    MACRO_NAMES,
    MACRO_SECTOR_MAPPING,
    MACRO_SECTOR_MAPPING_DERIVED,
    MACRO_SENS_MATRIX,
    MACRO_SENS_MATRIX_DERIVED,
)
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.reporting.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _set_macro_params(cfg: dict, sens_matrix: str | None = None,
                      enabled: bool = True,
                      direction: bool = False,
                      sigma_yy_inflation: bool = False) -> dict:
    cfg = copy.deepcopy(cfg)
    blpx = cfg.setdefault("blpx", {})
    blpx["macro_confidence_enabled"] = enabled
    blpx["macro_direction_enabled"] = direction
    blpx["macro_sigma_yy_inflation_enabled"] = sigma_yy_inflation
    # minvar must be enabled for sigma_yy inflation to have effect
    if sigma_yy_inflation:
        blpx["minvar_enabled"] = True
        blpx["minvar_alpha"] = float(blpx.get("minvar_alpha", 0.5))
    if sens_matrix is not None:
        blpx["macro_sens_matrix"] = sens_matrix
    elif "macro_sens_matrix" in blpx:
        del blpx["macro_sens_matrix"]
    return cfg


def _run_backtest(cfg: dict, df_exec: pd.DataFrame, start_date: str) -> dict:
    costs = cfg.get("costs", {})
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=float(costs.get("slippage_bps_per_side", 5.0)),
        overnight_alpha_long=float(costs.get("overnight_alpha_long", 0.75)),
        overnight_alpha_short=float(costs.get("overnight_alpha_short", 0.5)),
        buy_interest_annual=float(costs.get("buy_interest_annual", 0.025)),
        borrow_fee_annual=float(costs.get("borrow_fee_annual", 0.0115)),
        reverse_fee_bps=float(costs.get("reverse_fee_bps", 2.0)),
    )
    metrics = calculate_metrics(results["daily_returns"])
    return {
        "metrics": metrics,
        "daily_returns": results["daily_returns"],
        "model": model,
    }


def _format_sens_matrix(matrix: np.ndarray, tickers: list[str]) -> str:
    lines = []
    header = f"{'Ticker':<10} {'USDJPY':>8} {'CLF':>8} {'TNX':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, tk in enumerate(tickers):
        lines.append(f"{tk:<10} {matrix[i, 0]:>8.2f} {matrix[i, 1]:>8.2f} {matrix[i, 2]:>8.2f}")
    return "\n".join(lines)


def _sens_diff_stats(matrix_a: np.ndarray, matrix_b: np.ndarray) -> dict:
    diff = matrix_a - matrix_b
    abs_diff = np.abs(diff)
    return {
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "n_changed": int(np.sum(abs_diff > 0.01)),
        "correlation": float(np.corrcoef(matrix_a.flatten(), matrix_b.flatten())[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Macro Sensitivity Matrices")
    parser.add_argument("--config", default="configs/production/production.yaml",
                        help="Path to config YAML")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--output-dir", default="reports/sprint_macro_kappa",
                        help="Output directory")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg = _load_config(config_path)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print sensitivity matrix comparison
    logger.info("=== Sensitivity Matrix Comparison ===")
    logger.info("\nOriginal (MACRO_SENS_MATRIX):\n%s", _format_sens_matrix(MACRO_SENS_MATRIX, JP_TICKERS))
    logger.info("\nDerived (MACRO_SENS_MATRIX_DERIVED):\n%s", _format_sens_matrix(MACRO_SENS_MATRIX_DERIVED, JP_TICKERS))

    diff_stats = _sens_diff_stats(MACRO_SENS_MATRIX, MACRO_SENS_MATRIX_DERIVED)
    logger.info("\nDiff stats: %s", diff_stats)

    # Load data
    logger.info("[1/4] Loading execution data...")
    beta_window = cfg.get("residualization", {}).get("beta_window", 60)
    beta_ewma_halflife = cfg.get("residualization", {}).get("beta_ewma_halflife")
    beta_shrinkage = cfg.get("residualization", {}).get("beta_shrinkage", 0.0)
    beta_winsor_sigma = cfg.get("residualization", {}).get("beta_winsor_sigma")
    df_exec = load_execution_data(
        beta_window=beta_window,
        beta_ewma_halflife=beta_ewma_halflife,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )

    # Run 6 configurations: baseline, magnitude-only, A (direction), E (sigma_yy), A+E, disabled
    configs = [
        ("baseline", _set_macro_params(cfg, sens_matrix=None, enabled=True)),
        ("direction_A", _set_macro_params(cfg, sens_matrix=None, enabled=True, direction=True)),
        ("sigma_yy_E", _set_macro_params(cfg, sens_matrix=None, enabled=True, sigma_yy_inflation=True)),
        ("dir_A+sigE", _set_macro_params(cfg, sens_matrix=None, enabled=True, direction=True, sigma_yy_inflation=True)),
        ("disabled", _set_macro_params(cfg, enabled=False)),
    ]

    results_summary = []
    all_daily_returns = {}

    n_configs = len(configs)
    for idx, (label, test_cfg) in enumerate(configs):
        logger.info("[2/%d] Running backtest: %s ...", n_configs + 1, label)
        res = _run_backtest(test_cfg, df_exec, args.start_date)
        m = res["metrics"]
        dr = res["daily_returns"]
        if isinstance(dr, pd.DataFrame):
            dr = dr.iloc[:, 0]
        all_daily_returns[label] = dr

        # Get scale stats if available
        scales = getattr(res["model"], "_macro_scales", None)
        if scales is not None:
            scale_flat = scales.flatten()
            scale_stats = {
                "scale_mean": float(np.mean(scale_flat)),
                "scale_std": float(np.std(scale_flat)),
                "scale_max": float(np.max(scale_flat)),
                "scale_p95": float(np.percentile(scale_flat, 95)),
            }
        else:
            scale_stats = {"scale_mean": 1.0, "scale_std": 0.0, "scale_max": 1.0, "scale_p95": 1.0}

        results_summary.append({
            "config": label,
            "sharpe": m.get("Sharpe", np.nan),
            "AR": m.get("AR", np.nan),
            "RISK": m.get("RISK", np.nan),
            "MDD": m.get("MDD", np.nan),
            "turnover": m.get("Turnover", np.nan),
            **scale_stats,
        })

    # Save results
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(out_dir / "sensitivity_matrix_comparison.csv", index=False)
    logger.info("[3/4] Results:\n%s", df_results.to_string(index=False))

    # Save daily returns for further analysis
    dr_df = pd.DataFrame(all_daily_returns)
    dr_df.to_csv(out_dir / "daily_returns_comparison.csv")

    # Compute pairwise return correlation
    corr_matrix = dr_df.corr()
    logger.info("[4/4] Daily return correlations:\n%s", corr_matrix.to_string())

    # Generate report
    report_path = out_dir / "sensitivity_matrix_report.md"
    _generate_report(df_results, diff_stats, corr_matrix, report_path, args)
    logger.info("Report saved to: %s", report_path)


def _generate_report(df_results: pd.DataFrame, diff_stats: dict,
                     corr_matrix: pd.DataFrame, report_path: Path, args) -> None:
    lines = [
        "# Macro Sensitivity Matrix Comparison Report",
        "",
        f"**Config**: `{args.config}`",
        f"**Start date**: {args.start_date}",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        "## 1. Matrix Difference Statistics",
        "",
        f"- Mean absolute difference: {diff_stats['mean_abs_diff']:.4f}",
        f"- Max absolute difference: {diff_stats['max_abs_diff']:.4f}",
        f"- Number of changed entries (>0.01): {diff_stats['n_changed']}/{MACRO_SENS_MATRIX.size}",
        f"- Correlation between matrices: {diff_stats['correlation']:.4f}",
        "",
        "## 2. Backtest Results",
        "",
        "| Config | Sharpe | AR | RISK | MDD | Turnover | Scale Mean | Scale Std | Scale Max | Scale P95 |",
        "|--------|--------|----|------|-----|----------|------------|-----------|-----------|----------|",
    ]
    for _, row in df_results.iterrows():
        lines.append(
            f"| {row['config']} | {row['sharpe']:.4f} | "
            f"{row['AR']*100:.2f}% | {row['RISK']*100:.2f}% | "
            f"{row['MDD']*100:.2f}% | {row['turnover']:.2f} | "
            f"{row['scale_mean']:.4f} | {row['scale_std']:.4f} | "
            f"{row['scale_max']:.4f} | {row['scale_p95']:.4f} |"
        )

    # Correlation table with dynamic columns
    corr_cols = list(corr_matrix.columns)
    header = "| | " + " | ".join(corr_cols) + " |"
    sep = "|---|" + "|".join(["--------"] * len(corr_cols)) + "|"
    lines.extend(["", "## 3. Daily Return Correlations", "", header, sep])
    for idx in corr_matrix.index:
        vals = " | ".join(f"{corr_matrix.loc[idx, c]:.4f}" for c in corr_cols)
        lines.append(f"| {idx} | {vals} |")

    # Summary
    config_names = list(df_results["config"])
    lines.extend(["", "## 4. Key Findings", ""])

    base_sharpe = df_results.loc[df_results["config"] == "baseline", "sharpe"].values
    disab_sharpe = df_results.loc[df_results["config"] == "disabled", "sharpe"].values

    for cn in config_names:
        sr = df_results.loc[df_results["config"] == cn, "sharpe"].values
        if len(sr) > 0:
            lines.append(f"- {cn} Sharpe: {sr[0]:.4f}")

    if len(base_sharpe) > 0 and len(disab_sharpe) > 0:
        lines.append(f"- Macro confidence effect (baseline - disabled): {base_sharpe[0] - disab_sharpe[0]:+.4f}")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
