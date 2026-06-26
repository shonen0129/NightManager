"""Experiment Round 2: Parameter sensitivity around sector prior.

Tests different lambda_sector, alpha_yx, and blp_window combinations
to find if a better configuration exists than the current production settings.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BlendSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Blend static M_sector with rolling cross-correlation."""

    def __init__(self, config, blend_alpha=0.5):
        super().__init__(config)
        self.blend_alpha = blend_alpha

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        C_YX = corr[self.n_u:, :self.n_u]
        if C_YX.shape != B_blp.shape:
            return np.zeros(B_blp.shape)
        static = self.M_sector
        if static.shape != C_YX.shape:
            return C_YX.copy()
        return (1.0 - self.blend_alpha) * static + self.blend_alpha * C_YX


def build_config(yaml_path: str, overrides: dict | None = None) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for key, val in overrides.items():
            if key in ("lambda_pca", "lambda_sector", "alpha_xx", "alpha_yx",
                        "alpha_yy", "rho", "blp_window", "blp_ewma_halflife",
                        "winsor_sigma", "beta_conf"):
                cfg.setdefault("blpx", {})[key] = val
            else:
                cfg[key] = val
    return cfg


def run_variant(name, model, df_exec, start_date="2015-01-01"):
    t0 = time.perf_counter()
    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=start_date,
        overnight_alpha_long=0.75,
        overnight_alpha_short=0.5,
        buy_interest_annual=0.025,
        borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    elapsed = time.perf_counter() - t0

    dr = results["daily_returns"]
    dr_gross = results["daily_returns_gross"]

    ar = float(dr.mean() * 245)
    ar_gross = float(dr_gross.mean() * 245)
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    vol_gross = float(dr_gross.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan
    sharpe_gross = ar_gross / vol_gross if vol_gross > 0 else np.nan

    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(results["daily_turnover"].mean())
    gross_exp = float(results["daily_gross_exps"].mean())

    monthly = (1.0 + dr).groupby(dr.index.year * 12 + dr.index.month).prod() - 1.0
    if len(monthly) > 1:
        monthly_sharpe = float(
            (monthly.mean() / monthly.std(ddof=1)) * np.sqrt(12.0)
        )
    else:
        monthly_sharpe = np.nan

    # Cost ratio
    total_cost = float(results["daily_costs"].mean() * 245)

    metrics = {
        "name": name,
        "AR_net": ar,
        "AR_gross": ar_gross,
        "Vol_net": vol,
        "Sharpe_net": sharpe,
        "Sharpe_gross": sharpe_gross,
        "Sharpe_monthly": monthly_sharpe,
        "MDD": mdd,
        "Turnover": turnover,
        "GrossExp": gross_exp,
        "Cost_annual": total_cost,
        "elapsed_s": elapsed,
    }
    return metrics


def main():
    yaml_path = str(ROOT / "configs" / "production.yaml")

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec shape: %s", df_exec.shape)

    all_metrics = []

    # --- Group A: Lambda_sector sensitivity ---
    logger.info("=== Group A: Lambda_sector sensitivity ===")
    for ls in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
        cfg = build_config(yaml_path, overrides={"lambda_sector": ls})
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        name = f"ls_{ls:.2f}"
        m = run_variant(name, model, df_exec)
        all_metrics.append(m)
        logger.info("%s: Sharpe=%.4f  AR=%.2f%%  MDD=%.2f%%", name, m["Sharpe_net"], m["AR_net"]*100, m["MDD"]*100)

    # --- Group B: Alpha_yx sensitivity (cross-correlation shrinkage) ---
    logger.info("=== Group B: Alpha_yx sensitivity ===")
    for ayx in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        cfg = build_config(yaml_path, overrides={"alpha_yx": ayx})
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        name = f"ayx_{ayx:.2f}"
        m = run_variant(name, model, df_exec)
        all_metrics.append(m)
        logger.info("%s: Sharpe=%.4f  AR=%.2f%%  MDD=%.2f%%", name, m["Sharpe_net"], m["AR_net"]*100, m["MDD"]*100)

    # --- Group C: BLP window sensitivity ---
    logger.info("=== Group C: BLP window sensitivity ===")
    for bw in [63, 126, 189, 252, 378, 504]:
        cfg = build_config(yaml_path, overrides={"blp_window": bw})
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        name = f"bw_{bw}"
        m = run_variant(name, model, df_exec)
        all_metrics.append(m)
        logger.info("%s: Sharpe=%.4f  AR=%.2f%%  MDD=%.2f%%", name, m["Sharpe_net"], m["AR_net"]*100, m["MDD"]*100)

    # --- Group D: Blend alpha sensitivity (static + rolling) ---
    logger.info("=== Group D: Blend alpha sensitivity ===")
    for ba in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
        cfg = build_config(yaml_path)
        model = BlendSectorModel(cfg, blend_alpha=ba)
        name = f"blend_{ba:.2f}"
        m = run_variant(name, model, df_exec)
        all_metrics.append(m)
        logger.info("%s: Sharpe=%.4f  AR=%.2f%%  MDD=%.2f%%", name, m["Sharpe_net"], m["AR_net"]*100, m["MDD"]*100)

    # --- Group E: Combinations ---
    logger.info("=== Group E: Best combinations ===")
    combos = [
        {"lambda_sector": 0.30, "alpha_yx": 0.10},
        {"lambda_sector": 0.30, "alpha_yx": 0.0},
        {"lambda_sector": 0.20, "alpha_yx": 0.10, "blp_window": 189},
        {"lambda_sector": 0.40, "alpha_yx": 0.0, "blp_window": 189},
        {"lambda_sector": 0.30, "alpha_yx": 0.0, "blp_window": 126},
        {"lambda_sector": 0.50, "alpha_yx": 0.0, "blp_window": 252},
        {"lambda_sector": 0.0, "alpha_yx": 0.0, "blp_window": 189},
        {"lambda_sector": 0.0, "alpha_yx": 0.0, "blp_window": 126},
    ]
    for i, combo in enumerate(combos):
        cfg = build_config(yaml_path, overrides=combo)
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        name = f"combo_{i}"
        m = run_variant(name, model, df_exec)
        m["params"] = str(combo)
        all_metrics.append(m)
        logger.info("%s %s: Sharpe=%.4f  AR=%.2f%%  MDD=%.2f%%", name, combo, m["Sharpe_net"], m["AR_net"]*100, m["MDD"]*100)

    # Summary
    print("\n" + "=" * 140)
    print("PARAMETER SENSITIVITY EXPERIMENT — RESULTS SUMMARY")
    print("=" * 140)
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.set_index("name")

    display = df_metrics.copy()
    display["AR_net"] = (display["AR_net"] * 100).round(2)
    display["AR_gross"] = (display["AR_gross"] * 100).round(2)
    display["Vol_net"] = (display["Vol_net"] * 100).round(2)
    display["Sharpe_net"] = display["Sharpe_net"].round(4)
    display["Sharpe_gross"] = display["Sharpe_gross"].round(4)
    display["Sharpe_monthly"] = display["Sharpe_monthly"].round(4)
    display["MDD"] = (display["MDD"] * 100).round(2)
    display["Turnover"] = display["Turnover"].round(4)
    display["GrossExp"] = display["GrossExp"].round(4)
    display["Cost_annual"] = (display["Cost_annual"] * 100).round(2)
    display["elapsed_s"] = display["elapsed_s"].round(1)

    print(display.to_string())
    print()

    # Save
    output_dir = ROOT / "artifacts" / "sector_prior_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_dir / "parameter_sensitivity.csv")
    print(f"Results saved to {output_dir}")

    # Best overall
    best = df_metrics["Sharpe_net"].idxmax()
    print(f"\nBest Sharpe (net): {best} ({df_metrics.loc[best, 'Sharpe_net']:.4f})")
    if "params" in df_metrics.columns:
        print(f"  Params: {df_metrics.loc[best, 'params']}")

    # Baseline reference
    baseline = df_metrics.loc["ls_0.40", "Sharpe_net"]
    print(f"Baseline (ls_0.40): {baseline:.4f}")
    print(f"Improvement: {df_metrics.loc[best, 'Sharpe_net'] - baseline:+.4f}")


if __name__ == "__main__":
    main()
