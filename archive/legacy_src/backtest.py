"""Backtest comparison: runs multiple strategy configurations and outputs results.

Uses ``backtest.runner.run_backtest_with_config`` (domain layer) directly,
bypassing the high-level ``LeadLagStrategy`` wrapper.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest_config import (
    create_timestamped_output_dir,
    DEFAULT_START_DATE,
    LOGIC_DIFF_BASELINE_PARAMS,
    STRATEGY_DEFAULTS,
)
from config import N_JP_ASSETS
from data_loader import download_data, preprocess_data
from domain.models.types import StrategyConfig
from backtest.runner import run_backtest_with_config
from performance import calculate_metrics


def _build_config(params: dict) -> StrategyConfig:
    """Build StrategyConfig from a parameter dict."""
    return StrategyConfig(
        k=params["K"],
        lambda_reg=params["lambda_reg"],
        q=params["q"],
        weight_mode=params["weight_mode"],
        dispersion_filter=params["dispersion_filter"],
        dispersion_metric=params.get("dispersion_metric", "long_short_mean_gap"),
        v3_mode=params["v3_mode"],
        ewma_half_life=params.get("ewma_half_life"),
        lambda_lw=params.get("lambda_lw", 0.0),
        lw_target=params.get("lw_target", "identity"),
        corr_window=params.get("corr_window", 60),
        include_v4_prior=params.get("include_v4_prior", False),
        signal_mode=params.get("signal_mode", "baseline"),
        gap_open_coef=params.get("gap_open_coef", 1.0),
        topix_beta_coef=params.get("topix_beta_coef", 1.20),
        beta_window=params.get("beta_window", 60),
        gamma=params.get("gamma", 0.5),
    )


def main():
    print("Step 1: Fetching and preprocessing data...")
    data = download_data(beta_window=60)
    df_exec = preprocess_data(data, beta_window=60)

    output_dir = create_timestamped_output_dir("backtest_compare_ewma")
    start = DEFAULT_START_DATE

    all_results = {}
    all_metrics = {}

    base_legacy = {
        **dict(LOGIC_DIFF_BASELINE_PARAMS),
        "signal_mode": "baseline",
        "gap_open_coef": STRATEGY_DEFAULTS["gap_open_coef"],
    }

    base_current = dict(STRATEGY_DEFAULTS)

    model_compare_configs = [
        {"label": "Legacy Baseline (baseline)", "prefix": "baseline_", "params": base_legacy},
        {
            "label": "Production Version (gap_residual, K=6, v4-v6)",
            "prefix": "production_",
            "params": base_current,
        },
    ]

    for cfg in model_compare_configs:
        label = cfg["label"]
        print(f"\n=== Running: {label} ===")
        config = _build_config(cfg["params"])
        res = run_backtest_with_config(df_exec, config, start_date=start)
        m = calculate_metrics(res["daily_return"])
        all_results[label] = res
        all_metrics[label] = m

    # Print comparison table
    print("\n" + "=" * 90)
    header = f"{'Metric':<20}"
    for label in all_metrics:
        header += f" {label:>22}"
    print(header)
    print("=" * 90)
    for k in ["AR", "RISK", "R/R", "MDD", "Total Return"]:
        row = f"{k:<20}"
        for label in all_metrics:
            v = all_metrics[label][k]
            if k == "R/R":
                row += f" {v:>22.2f}"
            else:
                row += f" {v*100:>21.2f}%"
        print(row)
    print("=" * 90)

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_path = os.path.join(output_dir, "comparison_metrics_ewma.csv")
    metrics_df.to_csv(metrics_path, encoding="utf-8-sig")
    print(f"Metrics saved to {metrics_path}")

    # Compile daily signal statistics
    combined_daily = None
    for (label, res_df), cfg in zip(all_results.items(), model_compare_configs):
        prefix = cfg["prefix"]

        if combined_daily is None:
            combined_daily = res_df[["gap_mean", "gap_std"]].copy()

        cols_to_extract = [
            "daily_return",
            "signal_mean",
            "signal_std",
            "weight_concentration",
            "active_count",
        ]

        # Output individual daily returns
        res_df.to_csv(
            os.path.join(output_dir, f"{prefix}daily_return.csv"), encoding="utf-8-sig"
        )

        renamed = (
            res_df[cols_to_extract]
            .rename(columns={"daily_return": "return"})
            .add_prefix(prefix)
        )
        combined_daily = combined_daily.join(renamed, how="outer")

    stats_path = os.path.join(output_dir, "01_signal_statistics.csv")
    combined_daily.to_csv(stats_path, encoding="utf-8-sig")
    print(f"Daily signal statistics saved to {stats_path}")

    # Monthly returns summary
    combined_monthly = None
    for (label, res_df), cfg in zip(all_results.items(), model_compare_configs):
        prefix = cfg["prefix"]
        res_df.index = pd.to_datetime(res_df.index)
        monthly_ret = res_df.groupby(res_df.index.strftime("%Y-%m"))[
            "daily_return"
        ].apply(lambda x: (1 + x).prod() - 1)
        monthly_ret.name = f"{prefix}return"

        if combined_monthly is None:
            combined_monthly = pd.DataFrame(monthly_ret)
        else:
            combined_monthly = combined_monthly.join(monthly_ret, how="outer")

    monthly_stats_path = os.path.join(output_dir, "monthly_return_summary.csv")
    combined_monthly.to_csv(monthly_stats_path, encoding="utf-8-sig")
    print(f"Monthly return summary saved to {monthly_stats_path}")

    # --- Chart ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for (label, res), color in zip(all_results.items(), colors):
        W = (1 + res["daily_return"]).cumprod()
        axes[0].plot(W.index, W.values, label=label, linewidth=1.1, color=color)
        dd = (W / W.cummax()) - 1
        axes[1].fill_between(
            dd.index, dd.values, 0, alpha=0.2, color=color, label=label
        )

    axes[0].set_title("Model Comparison: Cumulative Return (2015–Present)", fontsize=13)
    axes[0].set_ylabel("Cumulative Wealth")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Drawdown Comparison", fontsize=13)
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nChart saved to {chart_path}")


if __name__ == "__main__":
    main()
