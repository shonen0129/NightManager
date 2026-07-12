"""Compare Frobenius norm prior scaling (spec) vs raw priors (current impl).

Spec (docs/モデル技術仕様書.md:247-253):
  B_pca_scaled = B_pca * ||B_blp||_F / ||B_pca||_F
  B_sector_scaled = M_sector * ||B_blp||_F / ||M_sector||_F

Current implementation uses raw B_pca and M_sector in the Tikhonov RHS.
This script runs both variants with the production config and compares.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    compute_backtest_metrics,
    load_execution_data,
    run_backtest_with_costs,
)
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

CONFIG_PATH = ROOT / "configs" / "production" / "production.yaml"
START_DATE = "2015-01-05"


def main():
    print("=" * 80)
    print("Frobenius Norm Prior Scaling A/B Comparison")
    print("=" * 80)

    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    costs = base_cfg.get("costs", {})
    cost_kwargs = dict(
        slippage_bps=float(costs.get("slippage_bps_per_side", 5.0)),
        overnight_alpha_long=float(costs.get("overnight_alpha_long", 0.75)),
        overnight_alpha_short=float(costs.get("overnight_alpha_short", 0.5)),
        buy_interest_annual=float(costs.get("buy_interest_annual", 0.025)),
        borrow_fee_annual=float(costs.get("borrow_fee_annual", 0.0115)),
        reverse_fee_bps=float(costs.get("reverse_fee_bps", 2.0)),
    )

    beta_cfg = base_cfg.get("residualization", {})
    print("\n[1/3] Loading execution data...")
    df_exec = load_execution_data(
        beta_window=int(beta_cfg.get("beta_window", 60)),
        beta_ewma_halflife=beta_cfg.get("beta_ewma_halflife"),
        beta_shrinkage=float(beta_cfg.get("beta_shrinkage", 0.0)),
        beta_winsor_sigma=beta_cfg.get("beta_winsor_sigma"),
    )
    print(f"  df_exec shape: {df_exec.shape}")

    variants = {
        "A: raw priors (current)": False,
        "B: frobenius scaled (spec)": True,
    }

    all_metrics = {}
    all_results = {}

    for label, use_scaling in variants.items():
        print(f"\n[2/3] Backtest: {label}")
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("blpx", {})["frobenius_scale_priors"] = use_scaling

        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        results = run_backtest_with_costs(model, df_exec, start_date=START_DATE, **cost_kwargs)
        metrics = compute_backtest_metrics(results, name=label)
        all_metrics[label] = metrics
        all_results[label] = results

        print(f"  Sharpe (net):  {metrics['Sharpe_net']:.4f}")
        print(f"  Sharpe (gross):{metrics['Sharpe_gross']:.4f}")
        print(f"  AR (net):      {metrics['AR_net']*100:.2f}%")
        print(f"  AR (gross):    {metrics['AR_gross']*100:.2f}%")
        print(f"  MDD:           {metrics['MDD']*100:.2f}%")
        print(f"  Turnover:      {metrics['Turnover']:.4f}")
        print(f"  GrossExp:      {metrics['GrossExp']:.4f}")

    print("\n" + "=" * 80)
    print("[3/3] Comparison Table")
    print(f"{'Metric':<20} {'A: Raw':>15} {'B: Frobenius':>15} {'Delta':>15}")
    print("-" * 70)

    a = all_metrics["A: raw priors (current)"]
    b = all_metrics["B: frobenius scaled (spec)"]

    for key in ["Sharpe_net", "Sharpe_gross", "AR_net", "AR_gross", "MDD", "Turnover", "GrossExp"]:
        av = a[key]
        bv = b[key]
        dv = bv - av
        if key in ["AR_net", "AR_gross", "MDD"]:
            print(f"{key:<20} {av*100:>14.2f}% {bv*100:>14.2f}% {dv*100:>+14.2f}%")
        else:
            print(f"{key:<20} {av:>15.4f} {bv:>15.4f} {dv:>+15.4f}")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
