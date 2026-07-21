#!/usr/bin/env python3
"""Fractional Differentiation Optimization Experiment.

Grid-searches over fractional differencing order d (0.1–1.0) and evaluates
each configuration via backtest using the production model.

For each d value, records:
  - ADF test statistic and p-value (stationarity)
  - Hurst exponent (long memory)
  - Net Sharpe ratio
  - Max drawdown
  - Turnover

Outputs:
  - CSV summary table
  - d vs Sharpe plot
  - d vs ADF statistic plot
  - Cumulative return comparison (baseline vs best vs d=0.5)
  - Drawdown comparison
  - Monthly return heatmap

Usage:
    python3 scripts/experiments/experiment_fractional_diff_optimization.py
    python3 scripts/experiments/experiment_fractional_diff_optimization.py --start-date 2015-01-05
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    CostParams,
    compute_backtest_metrics,
    load_cached_df_exec,
    load_execution_data,
    run_backtest_with_costs,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.features.fractional_diff import (
    adf_test,
    find_optimal_d,
    fractional_diff,
    fractional_diff_df,
    hurst_exponent,
)
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
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "fractional_diff"


def load_config(config_path: str | None = None) -> dict:
    """Load production config YAML."""
    if config_path is None:
        config_path = ROOT / "configs" / "production" / "production.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_backtest(
    cfg: dict,
    df_exec: pd.DataFrame,
    start_date: str,
    d: float,
    threshold: float = 1e-5,
    window: int = 100,
) -> dict:
    """Run a single backtest with fractional diff order d.

    Returns dict with metrics and results.
    """
    # Deep copy config to avoid mutation
    cfg_copy = copy.deepcopy(cfg)

    # Set fractional diff config
    if "features" not in cfg_copy:
        cfg_copy["features"] = {}
    cfg_copy["features"]["fractional_diff"] = {
        "enabled": d > 0.0,
        "d": d,
        "threshold": threshold,
        "window": window,
    }

    # Build model with modified config
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg_copy)

    # Extract cost params
    costs = cfg_copy.get("costs", {})
    slip_bps = float(costs.get("slippage_bps_per_side", 5.0))
    alpha_long = float(costs.get("overnight_alpha_long", 0.75))
    alpha_short = float(costs.get("overnight_alpha_short", 0.5))
    buy_interest = float(costs.get("buy_interest_annual", 0.025))
    borrow_fee = float(costs.get("borrow_fee_annual", 0.0115))
    reverse_fee = float(costs.get("reverse_fee_bps", 2.0))

    t0 = time.perf_counter()
    results = run_backtest_with_costs(
        model,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=slip_bps,
        overnight_alpha_long=alpha_long,
        overnight_alpha_short=alpha_short,
        buy_interest_annual=buy_interest,
        borrow_fee_annual=borrow_fee,
        reverse_fee_bps=reverse_fee,
    )
    elapsed = time.perf_counter() - t0

    metrics = compute_backtest_metrics(results, name=f"d={d:.1f}")
    metrics["elapsed_s"] = elapsed
    metrics["d"] = d

    return {"metrics": metrics, "results": results, "model": model}


def compute_stationarity_metrics(
    df_exec: pd.DataFrame,
    d: float,
    threshold: float = 1e-5,
    window: int = 100,
) -> dict:
    """Compute ADF and Hurst for each US ticker with fractional diff order d."""
    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    us_df = df_exec[us_cols]

    fd_df = fractional_diff_df(us_df, d=d, threshold=threshold, window=window).fillna(0.0)

    adf_stats = []
    adf_pvals = []
    hurst_vals = []

    for col in us_cols:
        series = fd_df[col]
        adf = adf_test(series)
        h = hurst_exponent(series)
        adf_stats.append(adf["statistic"])
        adf_pvals.append(adf["p_value"])
        hurst_vals.append(h)

    return {
        "adf_stat_mean": float(np.nanmean(adf_stats)),
        "adf_stat_median": float(np.nanmedian(adf_stats)),
        "adf_pval_mean": float(np.nanmean(adf_pvals)),
        "adf_pval_median": float(np.nanmedian(adf_pvals)),
        "hurst_mean": float(np.nanmean(hurst_vals)),
        "hurst_median": float(np.nanmedian(hurst_vals)),
        "n_stationary": int(sum(1 for p in adf_pvals if p < 0.05)),
        "n_total": len(adf_pvals),
    }


def plot_d_vs_sharpe(df_results: pd.DataFrame, out_path: Path):
    """Plot d vs net Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_results["d"], df_results["Sharpe_net"], "bo-", linewidth=2, markersize=8)
    ax.axhline(
        df_results[df_results["d"] == 1.0]["Sharpe_net"].values[0]
        if (df_results["d"] == 1.0).any()
        else 0,
        color="red", linestyle="--", alpha=0.5, label="Baseline (d=1.0)"
    )
    ax.set_xlabel("Fractional Differencing Order (d)", fontsize=12)
    ax.set_ylabel("Net Sharpe Ratio", fontsize=12)
    ax.set_title("Fractional Differentiation: d vs Net Sharpe", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_d_vs_adf(df_results: pd.DataFrame, out_path: Path):
    """Plot d vs ADF statistic."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_results["d"], df_results["adf_stat_mean"], "gs-", linewidth=2, markersize=8, label="ADF stat (mean)")
    ax1.axhline(-2.86, color="red", linestyle="--", alpha=0.5, label="5% critical value")
    ax1.set_xlabel("Fractional Differencing Order (d)", fontsize=12)
    ax1.set_ylabel("ADF Statistic (mean across tickers)", fontsize=12, color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df_results["d"], df_results["hurst_mean"], "m^--", linewidth=1.5, markersize=6, label="Hurst (mean)")
    ax2.axhline(0.5, color="blue", linestyle=":", alpha=0.3, label="H=0.5 (random walk)")
    ax2.set_ylabel("Hurst Exponent (mean)", fontsize=12, color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    fig.suptitle("Fractional Differentiation: Stationarity & Memory", fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cumulative_returns(
    results_dict: dict[str, dict],
    out_path: Path,
):
    """Plot cumulative returns for multiple d values."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, data in results_dict.items():
        dr = data["results"]["daily_returns"]
        wealth = (1.0 + dr).cumprod()
        ax.plot(wealth.index, wealth.values, linewidth=1.5, label=label)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Wealth (starting at 1.0)", fontsize=12)
    ax.set_title("Cumulative Returns: Fractional Diff Comparison", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_drawdowns(results_dict: dict[str, dict], out_path: Path):
    """Plot drawdown comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, data in results_dict.items():
        dd = data["results"]["drawdown"]
        ax.plot(dd.index, dd.values * 100, linewidth=1.0, label=label, alpha=0.8)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title("Drawdown Comparison: Fractional Diff", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_monthly_heatmap(daily_returns: pd.Series, out_path: Path, title: str = ""):
    """Plot monthly returns heatmap."""
    monthly = daily_returns.groupby(
        [daily_returns.index.year, daily_returns.index.month]
    ).apply(lambda x: (1 + x).prod() - 1)
    monthly = monthly.unstack(level=1)
    monthly.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in monthly.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        monthly.values * 100,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-5,
        vmax=5,
    )
    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels(monthly.columns)
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(f"Monthly Returns Heatmap: {title}", fontsize=14)
    plt.colorbar(im, ax=ax, label="Return (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_sensitivity_analysis(
    cfg: dict,
    df_exec: pd.DataFrame,
    start_date: str,
    best_d: float,
) -> pd.DataFrame:
    """Run sensitivity analysis on window, threshold, and sector groups."""
    logger.info("=== Sensitivity Analysis ===")
    rows = []

    # Window sensitivity
    for window in [60, 126, 252]:
        logger.info("  Window=%d...", window)
        res = run_single_backtest(cfg, df_exec, start_date, d=best_d, window=window)
        m = res["metrics"]
        rows.append({
            "parameter": "window",
            "value": window,
            "Sharpe_net": m["Sharpe_net"],
            "MDD": m["MDD"],
            "Turnover": m["Turnover"],
        })

    # Threshold sensitivity
    for thresh in [1e-3, 1e-5, 1e-7]:
        logger.info("  Threshold=%s...", thresh)
        res = run_single_backtest(cfg, df_exec, start_date, d=best_d, threshold=thresh)
        m = res["metrics"]
        rows.append({
            "parameter": "threshold",
            "value": thresh,
            "Sharpe_net": m["Sharpe_net"],
            "MDD": m["MDD"],
            "Turnover": m["Turnover"],
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fractional Differentiation Optimization Experiment"
    )
    parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--start-date",
        default="2015-01-05",
        help="Backtest start date",
    )
    parser.add_argument(
        "--d-step",
        type=float,
        default=0.1,
        help="Step size for d grid search",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached df_exec instead of downloading",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    logger.info("Loading config from %s", args.config)
    cfg = load_config(args.config)

    # Load data
    if args.use_cache:
        logger.info("Loading cached df_exec...")
        df_exec = load_cached_df_exec()
    else:
        logger.info("Downloading and preprocessing data...")
        beta_window = cfg.get("residualization", {}).get("beta_window", 60)
        df_exec = load_execution_data(beta_window=beta_window)

    logger.info("df_exec shape: %s, date range: %s to %s",
                df_exec.shape, df_exec.index[0], df_exec.index[-1])

    # --- Phase 1: Grid search over d ---
    d_values = np.arange(0.1, 1.01, args.d_step)
    logger.info("=== Phase 1: Grid Search (d=%.1f to %.1f, step=%.2f) ===",
                d_values[0], d_values[-1], args.d_step)

    all_results = []
    detailed_results = {}
    stationarity_rows = []

    for d_val in d_values:
        d_float = float(round(d_val, 2))
        logger.info("--- d=%.2f ---", d_float)

        # Stationarity metrics
        stat = compute_stationarity_metrics(df_exec, d=d_float)
        stationarity_rows.append({"d": d_float, **stat})

        # Backtest
        bt = run_single_backtest(cfg, df_exec, args.start_date, d=d_float)
        m = bt["metrics"]
        logger.info(
            "  Sharpe_net=%.4f, MDD=%.2f%%, Turnover=%.4f, AR=%.2f%%",
            m["Sharpe_net"], m["MDD"] * 100, m["Turnover"], m["AR_net"] * 100,
        )

        all_results.append({
            "d": d_float,
            "Sharpe_net": m["Sharpe_net"],
            "Sharpe_gross": m["Sharpe_gross"],
            "AR_net": m["AR_net"],
            "AR_gross": m["AR_gross"],
            "Vol_net": m["Vol_net"],
            "MDD": m["MDD"],
            "Turnover": m["Turnover"],
            "GrossExp": m["GrossExp"],
            "n_days": m["n_days"],
            "adf_stat_mean": stat["adf_stat_mean"],
            "adf_pval_mean": stat["adf_pval_mean"],
            "hurst_mean": stat["hurst_mean"],
            "n_stationary": stat["n_stationary"],
        })

        detailed_results[f"d={d_float:.2f}"] = bt

    df_grid = pd.DataFrame(all_results)
    df_stat = pd.DataFrame(stationarity_rows)

    # Save grid results
    df_grid.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)
    df_stat.to_csv(OUTPUT_DIR / "stationarity_metrics.csv", index=False)
    logger.info("Grid search results saved to %s", OUTPUT_DIR / "grid_search_results.csv")

    # --- Find best d ---
    best_row = df_grid.loc[df_grid["Sharpe_net"].idxmax()]
    best_d = float(best_row["d"])
    baseline_sharpe = float(df_grid[df_grid["d"] == 1.0]["Sharpe_net"].values[0])
    best_sharpe = float(best_row["Sharpe_net"])
    sharpe_improvement = best_sharpe - baseline_sharpe

    logger.info("=== Grid Search Results ===")
    logger.info("  Baseline (d=1.0): Sharpe=%.4f", baseline_sharpe)
    logger.info("  Best (d=%.2f): Sharpe=%.4f", best_d, best_sharpe)
    logger.info("  Improvement: +%.4f", sharpe_improvement)

    # --- Phase 2: Detailed comparison plots ---
    logger.info("=== Phase 2: Generating Plots ===")

    # d vs Sharpe
    plot_d_vs_sharpe(df_grid, OUTPUT_DIR / "d_vs_sharpe.png")
    # d vs ADF
    plot_d_vs_adf(df_grid, OUTPUT_DIR / "d_vs_adf.png")

    # Cumulative returns: baseline vs best vs d=0.5
    comparison = {}
    if "d=1.00" in detailed_results:
        comparison["Baseline (d=1.0)"] = detailed_results["d=1.00"]
    comparison[f"Best (d={best_d:.2f})"] = detailed_results[f"d={best_d:.2f}"]
    if "d=0.50" in detailed_results:
        comparison["d=0.50"] = detailed_results["d=0.50"]

    plot_cumulative_returns(comparison, OUTPUT_DIR / "cumulative_returns_comparison.png")
    plot_drawdowns(comparison, OUTPUT_DIR / "drawdown_comparison.png")

    # Monthly heatmap for best d
    best_dr = detailed_results[f"d={best_d:.2f}"]["results"]["daily_returns"]
    plot_monthly_heatmap(best_dr, OUTPUT_DIR / "monthly_heatmap_best.png",
                         title=f"Best d={best_d:.2f}")

    # --- Phase 3: Sensitivity analysis ---
    logger.info("=== Phase 3: Sensitivity Analysis ===")
    sens_df = run_sensitivity_analysis(cfg, df_exec, args.start_date, best_d)
    sens_df.to_csv(OUTPUT_DIR / "sensitivity_analysis.csv", index=False)
    logger.info("Sensitivity analysis saved to %s", OUTPUT_DIR / "sensitivity_analysis.csv")

    # --- Summary table ---
    summary_data = {
        "Configuration": ["Baseline (d=1.0)", f"Best (d={best_d:.2f})", "d=0.50"],
        "Sharpe (net)": [
            baseline_sharpe,
            best_sharpe,
            float(df_grid[df_grid["d"] == 0.5]["Sharpe_net"].values[0])
            if (df_grid["d"] == 0.5).any()
            else float("nan"),
        ],
        "AR (net %)": [
            float(df_grid[df_grid["d"] == 1.0]["AR_net"].values[0]) * 100,
            float(best_row["AR_net"]) * 100,
            float(df_grid[df_grid["d"] == 0.5]["AR_net"].values[0]) * 100
            if (df_grid["d"] == 0.5).any()
            else float("nan"),
        ],
        "Vol (%)": [
            float(df_grid[df_grid["d"] == 1.0]["Vol_net"].values[0]) * 100,
            float(best_row["Vol_net"]) * 100,
            float(df_grid[df_grid["d"] == 0.5]["Vol_net"].values[0]) * 100
            if (df_grid["d"] == 0.5).any()
            else float("nan"),
        ],
        "MDD (%)": [
            float(df_grid[df_grid["d"] == 1.0]["MDD"].values[0]) * 100,
            float(best_row["MDD"]) * 100,
            float(df_grid[df_grid["d"] == 0.5]["MDD"].values[0]) * 100
            if (df_grid["d"] == 0.5).any()
            else float("nan"),
        ],
        "Turnover": [
            float(df_grid[df_grid["d"] == 1.0]["Turnover"].values[0]),
            float(best_row["Turnover"]),
            float(df_grid[df_grid["d"] == 0.5]["Turnover"].values[0])
            if (df_grid["d"] == 0.5).any()
            else float("nan"),
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("FRACTIONAL DIFFERENTIATION EXPERIMENT — SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSharpe Improvement: {sharpe_improvement:+.4f}")
    print(f"Best d: {best_d:.2f}")
    print(f"ADF p-value (mean): {best_row['adf_pval_mean']:.4f}")
    print(f"Hurst (mean): {best_row['hurst_mean']:.4f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
