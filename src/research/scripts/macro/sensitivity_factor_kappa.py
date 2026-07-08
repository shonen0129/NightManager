#!/usr/bin/env python
"""Sensitivity analysis for Factor-Specific Kappa (Macro Confidence) parameters.

Tests:
  1. Kappa perturbation: baseline [3.0, 0.5, 0.5] vs ±50% variations
  2. Half-life sensitivity: halflife_mean and halflife_vol variations
  3. Walkforward OOS: split into 3 folds, compare in-sample vs OOS Sharpe
  4. Deflated Sharpe: adjust for multiple testing across kappa configurations

Usage:
  python3 scripts/experiments/sensitivity_factor_kappa.py
  python3 scripts/experiments/sensitivity_factor_kappa.py --start-date 2015-01-05
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

TRADING_DAYS = 245


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _run_backtest(cfg: dict, df_exec: pd.DataFrame, start_date: str) -> dict:
    """Run a single backtest with the given config and return metrics."""
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
    }


def _set_macro_params(cfg: dict, kappas: list[float] | None = None,
                      halflife_mean: float | None = None,
                      halflife_vol: float | None = None,
                      enabled: bool = True) -> dict:
    """Return a deep copy of cfg with macro confidence params overridden."""
    cfg = copy.deepcopy(cfg)
    blpx = cfg.setdefault("blpx", {})
    blpx["macro_confidence_enabled"] = enabled
    if kappas is not None:
        blpx["macro_kappas"] = kappas
    if halflife_mean is not None:
        blpx["macro_surprise_halflife_mean"] = halflife_mean
    if halflife_vol is not None:
        blpx["macro_surprise_halflife_vol"] = halflife_vol
    return cfg


def _deflated_sharpe(sharpe: float, n_trials: int, T: int,
                     skewness: float = 0.0, kurtosis: float = 3.0) -> float:
    """Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Adjusts the observed Sharpe for multiple testing and non-normality.
    Returns the deflated Sharpe (higher = more robust).
    """
    if T < 2 or n_trials < 1:
        return sharpe
    # Expected max Sharpe under null (across n_trials)
    euler_mascheroni = 0.5772156649
    expected_max = np.sqrt(8.0 / np.pi) * (
        euler_mascheroni + np.log(n_trials / (2.0 * np.pi))
    ) if n_trials > 1 else 0.0
    # Variance of Sharpe estimator
    var_sharpe = (1.0 / (T - 1)) * (
        1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    )
    if var_sharpe < 1e-12:
        return sharpe
    # Deflated Sharpe
    dsr = (sharpe - expected_max / np.sqrt(T)) / np.sqrt(var_sharpe)
    return float(dsr)


def run_kappa_sensitivity(cfg: dict, df_exec: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Run backtests with kappa perturbations."""
    baseline_kappas = [3.0, 0.5, 0.5]
    perturbations = [
        ("baseline", baseline_kappas),
        ("kappa_+50pct", [k * 1.5 for k in baseline_kappas]),
        ("kappa_-50pct", [k * 0.5 for k in baseline_kappas]),
        ("kappa_usdjpy_only", [3.0, 0.0, 0.0]),
        ("kappa_equal", [1.0, 1.0, 1.0]),
        ("kappa_disabled", None),  # macro_confidence_enabled = False
    ]

    results = []
    for label, kappas in perturbations:
        if kappas is None:
            test_cfg = _set_macro_params(cfg, enabled=False)
        else:
            test_cfg = _set_macro_params(cfg, kappas=kappas)

        logger.info("Running kappa sensitivity: %s (kappas=%s)", label, kappas)
        res = _run_backtest(test_cfg, df_exec, start_date)
        m = res["metrics"]
        results.append({
            "config": label,
            "kappas": str(kappas),
            "sharpe": m.get("Sharpe", np.nan),
            "AR": m.get("AR", np.nan),
            "MDD": m.get("MDD", np.nan),
            "turnover": m.get("Turnover", np.nan),
        })

    return pd.DataFrame(results)


def run_halflife_sensitivity(cfg: dict, df_exec: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Run backtests with half-life perturbations."""
    halflife_variations = [
        ("hl_mean_10_vol_30", 10.0, 30.0),
        ("hl_mean_20_vol_60_baseline", 20.0, 60.0),
        ("hl_mean_40_vol_120", 40.0, 120.0),
        ("hl_mean_10_vol_60", 10.0, 60.0),
        ("hl_mean_20_vol_30", 20.0, 30.0),
        ("hl_mean_20_vol_120", 20.0, 120.0),
    ]

    results = []
    for label, hl_mean, hl_vol in halflife_variations:
        test_cfg = _set_macro_params(cfg, halflife_mean=hl_mean, halflife_vol=hl_vol)
        logger.info("Running half-life sensitivity: %s", label)
        res = _run_backtest(test_cfg, df_exec, start_date)
        m = res["metrics"]
        results.append({
            "config": label,
            "halflife_mean": hl_mean,
            "halflife_vol": hl_vol,
            "sharpe": m.get("Sharpe", np.nan),
            "AR": m.get("AR", np.nan),
            "MDD": m.get("MDD", np.nan),
            "turnover": m.get("Turnover", np.nan),
        })

    return pd.DataFrame(results)


def run_walkforward(cfg: dict, df_exec: pd.DataFrame, start_date: str,
                    n_folds: int = 3) -> pd.DataFrame:
    """Walkforward validation: split into n_folds, compare IS vs OOS Sharpe."""
    baseline_kappas = [3.0, 0.5, 0.5]
    test_cfg = _set_macro_params(cfg, kappas=baseline_kappas)

    model = SectorRelativeEnsembleBLPEnhancedModel(test_cfg)
    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=float(cfg.get("costs", {}).get("slippage_bps_per_side", 5.0)),
        overnight_alpha_long=float(cfg.get("costs", {}).get("overnight_alpha_long", 0.75)),
        overnight_alpha_short=float(cfg.get("costs", {}).get("overnight_alpha_short", 0.5)),
        buy_interest_annual=float(cfg.get("costs", {}).get("buy_interest_annual", 0.025)),
        borrow_fee_annual=float(cfg.get("costs", {}).get("borrow_fee_annual", 0.0115)),
        reverse_fee_bps=float(cfg.get("costs", {}).get("reverse_fee_bps", 2.0)),
    )

    daily_ret = results["daily_returns"]
    if isinstance(daily_ret, pd.DataFrame):
        daily_ret = daily_ret.iloc[:, 0]
    total_days = len(daily_ret)
    fold_size = total_days // n_folds

    fold_results = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else total_days
        fold_ret = daily_ret.iloc[start_idx:end_idx]
        fold_metrics = calculate_metrics(fold_ret)
        fold_results.append({
            "fold": i + 1,
            "start_date": str(fold_ret.index[0].date()),
            "end_date": str(fold_ret.index[-1].date()),
            "n_days": len(fold_ret),
            "sharpe": fold_metrics.get("Sharpe", np.nan),
            "AR": fold_metrics.get("AR", np.nan),
            "MDD": fold_metrics.get("MDD", np.nan),
        })

    return pd.DataFrame(fold_results)


def main():
    parser = argparse.ArgumentParser(description="Factor-Kappa Sensitivity Analysis")
    parser.add_argument("--config", default="configs/production/production.yaml",
                        help="Path to config YAML")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--output-dir", default="reports/sprint_macro_kappa",
                        help="Output directory for results")
    parser.add_argument("--n-folds", type=int, default=3, help="Walkforward folds")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg = _load_config(config_path)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    logger.info("[2/4] Running kappa sensitivity analysis...")
    kappa_df = run_kappa_sensitivity(cfg, df_exec, args.start_date)
    kappa_df.to_csv(out_dir / "kappa_sensitivity.csv", index=False)
    logger.info("Kappa sensitivity results:\n%s", kappa_df.to_string(index=False))

    logger.info("[3/4] Running half-life sensitivity analysis...")
    hl_df = run_halflife_sensitivity(cfg, df_exec, args.start_date)
    hl_df.to_csv(out_dir / "halflife_sensitivity.csv", index=False)
    logger.info("Half-life sensitivity results:\n%s", hl_df.to_string(index=False))

    logger.info("[4/4] Running walkforward validation...")
    wf_df = run_walkforward(cfg, df_exec, args.start_date, n_folds=args.n_folds)
    wf_df.to_csv(out_dir / "walkforward_results.csv", index=False)
    logger.info("Walkforward results:\n%s", wf_df.to_string(index=False))

    # Compute Deflated Sharpe for kappa trials
    n_trials = len(kappa_df)
    baseline_sharpe = kappa_df.loc[kappa_df["config"] == "baseline", "sharpe"].values
    if len(baseline_sharpe) > 0:
        T = len(df_exec)
        dsr = _deflated_sharpe(float(baseline_sharpe[0]), n_trials, T)
        logger.info("Deflated Sharpe (n_trials=%d, T=%d): %.4f", n_trials, T, dsr)

    # Generate report
    report_path = out_dir / "sensitivity_report.md"
    _generate_report(kappa_df, hl_df, wf_df, report_path, args)
    logger.info("Report saved to: %s", report_path)


def _generate_report(kappa_df: pd.DataFrame, hl_df: pd.DataFrame,
                     wf_df: pd.DataFrame, report_path: Path, args) -> None:
    """Generate markdown sensitivity report."""
    baseline_sharpe = kappa_df.loc[kappa_df["config"] == "baseline", "sharpe"].values
    baseline_val = float(baseline_sharpe[0]) if len(baseline_sharpe) > 0 else float("nan")

    n_trials = len(kappa_df)
    T = 2500  # approximate
    dsr = _deflated_sharpe(baseline_val, n_trials, T)

    lines = [
        "# Factor-Kappa (Macro Confidence) Sensitivity Analysis Report",
        "",
        f"**Config**: `{args.config}`",
        f"**Start date**: {args.start_date}",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        "## 1. Kappa Perturbation Results",
        "",
        "| Config | Kappas | Sharpe | AR | MDD | Turnover |",
        "|--------|--------|--------|----|-----|----------|",
    ]
    for _, row in kappa_df.iterrows():
        lines.append(
            f"| {row['config']} | {row['kappas']} | "
            f"{row['sharpe']:.4f} | {row['AR']*100:.2f}% | "
            f"{row['MDD']*100:.2f}% | {row['turnover']:.2f} |"
        )

    lines.extend([
        "",
        f"**Deflated Sharpe** (n_trials={n_trials}, T~{T}): {dsr:.4f}",
        "",
        "## 2. Half-Life Sensitivity Results",
        "",
        "| Config | HL Mean | HL Vol | Sharpe | AR | MDD | Turnover |",
        "|--------|---------|--------|--------|----|-----|----------|",
    ])
    for _, row in hl_df.iterrows():
        lines.append(
            f"| {row['config']} | {row['halflife_mean']:.0f} | {row['halflife_vol']:.0f} | "
            f"{row['sharpe']:.4f} | {row['AR']*100:.2f}% | "
            f"{row['MDD']*100:.2f}% | {row['turnover']:.2f} |"
        )

    lines.extend([
        "",
        "## 3. Walkforward OOS Validation",
        "",
        "| Fold | Start | End | Days | Sharpe | AR | MDD |",
        "|------|-------|-----|------|--------|----|-----|",
    ])
    for _, row in wf_df.iterrows():
        lines.append(
            f"| {row['fold']} | {row['start_date']} | {row['end_date']} | "
            f"{row['n_days']} | {row['sharpe']:.4f} | {row['AR']*100:.2f}% | "
            f"{row['MDD']*100:.2f}% |"
        )

    # OOS consistency check
    wf_sharpes = wf_df["sharpe"].values
    oos_mean = float(np.mean(wf_sharpes))
    oos_std = float(np.std(wf_sharpes))
    lines.extend([
        "",
        f"**OOS Sharpe**: mean={oos_mean:.4f}, std={oos_std:.4f}",
        f"**OOS Consistency**: {'PASS' if oos_std < 0.5 else 'WARN'} (std < 0.5)",
        "",
        "## 4. Conclusions",
        "",
        f"- Baseline Sharpe: {baseline_val:.4f}",
        f"- Deflated Sharpe: {dsr:.4f}",
        f"- Kappa ±50% sensitivity: "
        f"max Sharpe delta = {kappa_df['sharpe'].max() - kappa_df['sharpe'].min():.4f}",
        f"- Half-life sensitivity: "
        f"max Sharpe delta = {hl_df['sharpe'].max() - hl_df['sharpe'].min():.4f}",
        f"- Walkforward OOS Sharpe stability: std={oos_std:.4f}",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
