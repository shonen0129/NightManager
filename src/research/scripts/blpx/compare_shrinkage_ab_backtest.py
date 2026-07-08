"""Compare old vs new shrinkage parameters via backtest.

Old: lambda_lw=0.5, lambda_reg=0.75 (raw weight 12.5%)
New: lambda_lw=0.3, lambda_reg=0.30 (raw weight 49.0%)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    load_cached_df_exec,
    run_backtest_with_costs,
)
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.reporting.metrics import calculate_metrics

CONFIGS = {
    "old (lw=0.5, reg=0.75)": {"lambda_lw": 0.5, "lambda_reg": 0.75},
    "new (lw=0.3, reg=0.30)": {"lambda_lw": 0.3, "lambda_reg": 0.30},
}


def main():
    print("=" * 80)
    print("Shrinkage Parameter A/B Backtest Comparison")
    print("=" * 80)

    print("\n[Loading data...]")
    df_exec = load_cached_df_exec()
    print(f"  shape: {df_exec.shape}")

    results_all = {}

    for label, params in CONFIGS.items():
        print(f"\n[Backtest: {label}]")
        cfg = {
            "lambda_lw": params["lambda_lw"],
            "lambda_reg": params["lambda_reg"],
            "min_raw_weight": 0.0,  # disable guardrail to test pure parameter effect
        }
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        results = run_backtest_with_costs(model, df_exec, start_date="2015-01-05")
        daily_ret = results["daily_returns"]
        metrics = calculate_metrics(daily_ret)
        results_all[label] = {
            "metrics": metrics,
            "daily_ret": daily_ret,
            "equity": results["equity_curve"],
            "turnover": results["daily_turnover"].mean(),
            "gross_exp": results["daily_gross_exps"].mean(),
        }
        print(f"  AR:      {metrics['AR']:.4f}")
        print(f"  RISK:    {metrics['RISK']:.4f}")
        print(f"  Sharpe:  {metrics['Sharpe']:.4f}")
        print(f"  R/R:     {metrics['R/R']:.4f}")
        print(f"  MDD:     {metrics['MDD']:.4f}")
        print(f"  Total:   {metrics['Total Return']:.4f}")
        print(f"  Turnover:{results_all[label]['turnover']:.4f}")

    # Comparison table
    print("\n" + "=" * 80)
    print("[Comparison Table]")
    print(f"{'Metric':<15} {'Old (0.5/0.75)':>15} {'New (0.3/0.30)':>15} {'Delta':>15}")
    print("-" * 65)

    old_m = results_all["old (lw=0.5, reg=0.75)"]["metrics"]
    new_m = results_all["new (lw=0.3, reg=0.30)"]["metrics"]

    for key in ["AR", "RISK", "Sharpe", "R/R", "MDD", "Total Return"]:
        o = old_m[key]
        n = new_m[key]
        d = n - o
        print(f"{key:<15} {o:>15.4f} {n:>15.4f} {d:>+15.4f}")

    old_t = results_all["old (lw=0.5, reg=0.75)"]["turnover"]
    new_t = results_all["new (lw=0.3, reg=0.30)"]["turnover"]
    print(f"{'Turnover':<15} {old_t:>15.4f} {new_t:>15.4f} {new_t-old_t:>+15.4f}")

    # Also test with guardrail enabled on old params
    print(f"\n[Backtest: old params + guardrail (min_raw_weight=0.30)]")
    cfg_guard = {
        "lambda_lw": 0.5,
        "lambda_reg": 0.75,
        "min_raw_weight": 0.30,
    }
    model_g = SectorRelativeEnsembleBLPEnhancedModel(cfg_guard)
    results_g = run_backtest_with_costs(model_g, df_exec, start_date="2015-01-05")
    metrics_g = calculate_metrics(results_g["daily_returns"])
    print(f"  AR:      {metrics_g['AR']:.4f}")
    print(f"  RISK:    {metrics_g['RISK']:.4f}")
    print(f"  Sharpe:  {metrics_g['Sharpe']:.4f}")
    print(f"  R/R:     {metrics_g['R/R']:.4f}")
    print(f"  MDD:     {metrics_g['MDD']:.4f}")
    print(f"  Total:   {metrics_g['Total Return']:.4f}")

    print(f"\n{'Metric':<15} {'Old no guard':>15} {'Old+guard':>15} {'New params':>15}")
    print("-" * 65)
    for key in ["AR", "RISK", "Sharpe", "R/R", "MDD", "Total Return"]:
        o = old_m[key]
        g = metrics_g[key]
        n = new_m[key]
        print(f"{key:<15} {o:>15.4f} {g:>15.4f} {n:>15.4f}")

    print("\n" + "=" * 80)
    print("[SRE (Pure PCA) Model Comparison]")
    print("=" * 80)

    sre_results = {}
    for label, params in CONFIGS.items():
        print(f"\n[SRE Backtest: {label}]")
        cfg = {
            "lambda_lw": params["lambda_lw"],
            "lambda_reg": params["lambda_reg"],
            "min_raw_weight": 0.0,
        }
        model = SectorRelativeEnsembleModel(cfg)
        results = run_backtest_with_costs(model, df_exec, start_date="2015-01-05")
        daily_ret = results["daily_returns"]
        metrics = calculate_metrics(daily_ret)
        sre_results[label] = metrics
        print(f"  AR:      {metrics['AR']:.4f}")
        print(f"  RISK:    {metrics['RISK']:.4f}")
        print(f"  Sharpe:  {metrics['Sharpe']:.4f}")
        print(f"  R/R:     {metrics['R/R']:.4f}")
        print(f"  MDD:     {metrics['MDD']:.4f}")
        print(f"  Total:   {metrics['Total Return']:.4f}")

    # SRE with guardrail
    print(f"\n[SRE Backtest: old params + guardrail (min_raw_weight=0.30)]")
    cfg_guard_sre = {
        "lambda_lw": 0.5,
        "lambda_reg": 0.75,
        "min_raw_weight": 0.30,
    }
    model_g_sre = SectorRelativeEnsembleModel(cfg_guard_sre)
    results_g_sre = run_backtest_with_costs(model_g_sre, df_exec, start_date="2015-01-05")
    metrics_g_sre = calculate_metrics(results_g_sre["daily_returns"])
    print(f"  AR:      {metrics_g_sre['AR']:.4f}")
    print(f"  RISK:    {metrics_g_sre['RISK']:.4f}")
    print(f"  Sharpe:  {metrics_g_sre['Sharpe']:.4f}")
    print(f"  R/R:     {metrics_g_sre['R/R']:.4f}")
    print(f"  MDD:     {metrics_g_sre['MDD']:.4f}")
    print(f"  Total:   {metrics_g_sre['Total Return']:.4f}")

    print(f"\n{'Metric':<15} {'Old no guard':>15} {'Old+guard':>15} {'New params':>15}")
    print("-" * 65)
    old_sre = sre_results["old (lw=0.5, reg=0.75)"]
    new_sre = sre_results["new (lw=0.3, reg=0.30)"]
    for key in ["AR", "RISK", "Sharpe", "R/R", "MDD", "Total Return"]:
        o = old_sre[key]
        g = metrics_g_sre[key]
        n = new_sre[key]
        print(f"{key:<15} {o:>15.4f} {g:>15.4f} {n:>15.4f}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
