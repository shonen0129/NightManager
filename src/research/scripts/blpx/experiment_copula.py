#!/usr/bin/env python
"""Copula correlation blending A/B backtest.

Runs backtest with and without t-copula correlation blending using
the production config as baseline, then compares metrics.

Uses cached market data (etf_data.pkl) to avoid yfinance dependency.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    load_config,
    run_backtest_with_costs,
)
from leadlag.data.preprocessor import preprocess_data
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


def load_cached_data() -> dict:
    """Load market data from etf_data.pkl cache directly."""
    pkl_path = ROOT / "market_data" / "etf_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Market data cache not found: {pkl_path}\n"
            "Run download_data() first or install yfinance."
        )
    logger.info("Loading cached market data from %s", pkl_path)
    return pd.read_pickle(pkl_path)


def run_single_backtest(
    cfg: dict,
    df_exec: pd.DataFrame,
    start_date: str,
    costs: dict,
) -> dict:
    """Run a single backtest with given config and return results dict."""
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    return run_backtest_with_costs(
        model, df_exec, start_date=start_date,
        slippage_bps=float(costs.get("slippage_bps_per_side", 5.0)),
        overnight_alpha_long=float(costs.get("overnight_alpha_long", 0.75)),
        overnight_alpha_short=float(costs.get("overnight_alpha_short", 0.5)),
        buy_interest_annual=float(costs.get("buy_interest_annual", 0.025)),
        borrow_fee_annual=float(costs.get("borrow_fee_annual", 0.0115)),
        reverse_fee_bps=float(costs.get("reverse_fee_bps", 2.0)),
    )


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    for key, v in metrics.items():
        if key in ["AR", "RISK", "MDD", "Total Return"]:
            print(f"  {key}: {v*100:.2f}%")
        elif key == "Sharpe":
            print(f"  {key}: {v:.4f}")
        else:
            print(f"  {key}: {v:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Copula A/B Backtest")
    parser.add_argument(
        "--config", default="configs/production/production.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument(
        "--output-dir", default="results/copula_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--blend-weight", type=float, default=0.3,
        help="Copula blend weight (0=Pearson only, 1=Copula only)",
    )
    parser.add_argument(
        "--dynamic-blend", action="store_true", default=True,
        help="Enable dynamic stress-based blend weighting",
    )
    parser.add_argument(
        "--no-dynamic-blend", dest="dynamic_blend", action="store_false",
        help="Disable dynamic blend (use fixed weight)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run parameter sweep instead of single A/B",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    logger.info("Loading base config from %s", config_path)
    base_cfg = load_config(config_path)

    costs = base_cfg.get("costs", {})
    beta_window = base_cfg.get("residualization", {}).get("beta_window", 60)
    beta_ewma_halflife = base_cfg.get("residualization", {}).get("beta_ewma_halflife")
    beta_shrinkage = base_cfg.get("residualization", {}).get("beta_shrinkage", 0.0)
    beta_winsor_sigma = base_cfg.get("residualization", {}).get("beta_winsor_sigma")

    logger.info("[1/4] Loading cached market data...")
    raw_data = load_cached_data()

    logger.info("[2/4] Preprocessing aligned execution dataset...")
    df_exec = preprocess_data(
        raw_data,
        beta_window=beta_window,
        beta_ewma_halflife=beta_ewma_halflife,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )

    # Compute TOPIX returns for residualization
    from leadlag.data.tickers import TOPIX_TICKER
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (
        (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Baseline: copula disabled ---
    logger.info("[3/4] Running baseline backtest (Pearson only)...")
    cfg_baseline = dict(base_cfg)
    cfg_baseline["copula_enabled"] = False

    t0 = time.time()
    results_baseline = run_single_backtest(
        cfg_baseline, df_exec, args.start_date, costs
    )
    t_baseline = time.time() - t0
    metrics_baseline = calculate_metrics(results_baseline["daily_returns"])
    print_metrics(f"Baseline (Pearson) [{t_baseline:.1f}s]", metrics_baseline)

    # Save baseline artifacts
    results_baseline["daily_returns"].to_csv(
        out_dir / "baseline_daily_net_returns.csv", header=["net_return"]
    )
    results_baseline["equity_curve"].to_csv(
        out_dir / "baseline_equity_curve.csv", header=["equity"]
    )

    if args.sweep:
        # --- Parameter sweep: minvar × copula combinations ---
        sweep_configs = [
            {"label": "minvar_a0.3", "minvar": True, "minvar_alpha": 0.3, "copula": False},
            {"label": "minvar_a0.5", "minvar": True, "minvar_alpha": 0.5, "copula": False},
            {"label": "minvar_a0.8", "minvar": True, "minvar_alpha": 0.8, "copula": False},
            {"label": "minvar_a1.0", "minvar": True, "minvar_alpha": 1.0, "copula": False},
            {"label": "copula_dyn1.0", "minvar": False, "copula": True, "blend_weight": 1.0, "dynamic_blend": True},
            {"label": "copula+minvar_a0.5", "minvar": True, "minvar_alpha": 0.5, "copula": True, "blend_weight": 1.0, "dynamic_blend": True},
            {"label": "copula+minvar_a0.8", "minvar": True, "minvar_alpha": 0.8, "copula": True, "blend_weight": 1.0, "dynamic_blend": True},
        ]

        sweep_results = []
        for sc in sweep_configs:
            label = sc["label"]
            logger.info("Running copula sweep: %s ...", label)
            cfg_c = dict(base_cfg)
            cfg_c["minvar_enabled"] = sc.get("minvar", False)
            cfg_c["minvar_alpha"] = sc.get("minvar_alpha", 0.5)
            cfg_c["copula_enabled"] = sc.get("copula", False)
            if sc.get("copula", False):
                cfg_c["copula_blend_weight"] = sc.get("blend_weight", 1.0)
                cfg_c["copula_dynamic_blend"] = sc.get("dynamic_blend", True)
                cfg_c["copula_nu_init"] = 5.0
                cfg_c["copula_stress_threshold"] = 1.5

            t0 = time.time()
            try:
                results_c = run_single_backtest(cfg_c, df_exec, args.start_date, costs)
                t_c = time.time() - t0
                metrics_c = calculate_metrics(results_c["daily_returns"])
                print_metrics(f"Copula {label} [{t_c:.1f}s]", metrics_c)

                row = {"label": label, "time_sec": t_c}
                row.update(metrics_c)
                row["delta_sharpe"] = metrics_c.get("Sharpe", np.nan) - metrics_baseline.get("Sharpe", np.nan)
                row["delta_ar"] = metrics_c.get("AR", np.nan) - metrics_baseline.get("AR", np.nan)
                row["delta_mdd"] = metrics_c.get("MDD", np.nan) - metrics_baseline.get("MDD", np.nan)
                sweep_results.append(row)

                results_c["daily_returns"].to_csv(
                    out_dir / f"copula_{label}_daily_returns.csv", header=["net_return"]
                )
            except Exception as e:
                t_c = time.time() - t0
                logger.error("Sweep %s failed after %.1fs: %s", label, t_c, e)
                sweep_results.append({"label": label, "time_sec": t_c, "error": str(e)})

        # Save sweep table
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df.to_csv(out_dir / "sweep_results.csv", index=False)
        print("\n" + "=" * 80)
        print("=== Sweep Summary ===")
        print("=" * 80)
        print(f"{'Label':<15} {'Sharpe':>8} {'dSharpe':>8} {'AR%':>8} {'dAR%':>8} {'MDD%':>8} {'dMDD%':>8} {'Time':>7}")
        print("-" * 72)
        for row in sweep_results:
            if "error" in row:
                print(f"{row['label']:<15} {'ERROR':>8}")
                continue
            print(f"{row['label']:<15} {row.get('Sharpe', 0):>8.4f} {row.get('delta_sharpe', 0):>+8.4f} "
                  f"{row.get('AR', 0)*100:>7.2f}% {row.get('delta_ar', 0)*100:>+7.2f}% "
                  f"{row.get('MDD', 0)*100:>7.2f}% {row.get('delta_mdd', 0)*100:>+7.2f}% "
                  f"{row.get('time_sec', 0):>6.1f}s")

        logger.info("Sweep artifacts saved in: %s", out_dir)
        return

    # --- Single A/B comparison ---
    logger.info("[4/4] Running copula backtest (blend_weight=%.2f, dynamic=%s)...",
                args.blend_weight, args.dynamic_blend)
    cfg_copula = dict(base_cfg)
    cfg_copula["copula_enabled"] = True
    cfg_copula["copula_blend_weight"] = args.blend_weight
    cfg_copula["copula_dynamic_blend"] = args.dynamic_blend
    cfg_copula["copula_nu_init"] = 5.0
    cfg_copula["copula_stress_threshold"] = 1.5

    t0 = time.time()
    results_copula = run_single_backtest(
        cfg_copula, df_exec, args.start_date, costs
    )
    t_copula = time.time() - t0
    metrics_copula = calculate_metrics(results_copula["daily_returns"])
    print_metrics(f"Copula (w={args.blend_weight}, dynamic={args.dynamic_blend}) [{t_copula:.1f}s]",
                  metrics_copula)

    # Save copula artifacts
    results_copula["daily_returns"].to_csv(
        out_dir / "copula_daily_net_returns.csv", header=["net_return"]
    )
    results_copula["equity_curve"].to_csv(
        out_dir / "copula_equity_curve.csv", header=["equity"]
    )

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("=== Comparison: Copula vs Baseline ===")
    print("=" * 60)

    keys = ["AR", "RISK", "Sharpe", "MDD", "Total Return", "R/R"]
    print(f"{'Metric':<15} {'Baseline':>12} {'Copula':>12} {'Delta':>12}")
    print("-" * 51)
    for key in keys:
        b = metrics_baseline.get(key, np.nan)
        c = metrics_copula.get(key, np.nan)
        delta = c - b
        if key in ["AR", "RISK", "MDD", "Total Return"]:
            print(f"{key:<15} {b*100:>11.2f}% {c*100:>11.2f}% {delta*100:>+11.2f}%")
        elif key == "Sharpe":
            print(f"{key:<15} {b:>12.4f} {c:>12.4f} {delta:>+12.4f}")
        else:
            print(f"{key:<15} {b:>12.2f} {c:>12.2f} {delta:>+12.2f}")

    # Save comparison summary
    comparison = {
        "baseline": metrics_baseline,
        "copula": metrics_copula,
        "config": {
            "blend_weight": args.blend_weight,
            "dynamic_blend": args.dynamic_blend,
            "start_date": args.start_date,
        },
        "timing_sec": {
            "baseline": t_baseline,
            "copula": t_copula,
        },
    }
    with open(out_dir / "comparison_summary.yaml", "w") as f:
        yaml.dump(comparison, f, default_flow_style=False)

    # Save daily returns comparison
    comparison_df = pd.DataFrame({
        "baseline": results_baseline["daily_returns"],
        "copula": results_copula["daily_returns"],
    })
    comparison_df.to_csv(out_dir / "daily_returns_comparison.csv")

    logger.info("Artifacts saved in: %s", out_dir)


if __name__ == "__main__":
    main()
