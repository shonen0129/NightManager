import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backtest_config import (
    DEFAULT_START_DATE,
    STRATEGY_DEFAULTS,
    create_timestamped_output_dir,
)
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy


def _run_case(df_exec: pd.DataFrame, params: dict, start_date: str):
    strategy = LeadLagStrategy(df_exec=df_exec, **params)
    result = strategy.run_backtest(start_date=start_date)
    metrics = calculate_metrics(result["daily_return"])
    return result, metrics


def _save_chart(compare_daily: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for col in compare_daily.columns:
        wealth = (1.0 + compare_daily[col]).cumprod()
        axes[0].plot(wealth.index, wealth.values, label=col, linewidth=1.2)
        drawdown = (wealth / wealth.cummax()) - 1.0
        axes[1].plot(drawdown.index, drawdown.values, label=col, linewidth=1.0)

    axes[0].set_title("Dispersion Metric Comparison: sigma vs long_short_mean_gap")
    axes[0].set_ylabel("Cumulative Wealth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    print("[1/5] Loading and preprocessing data...")
    data = download_data()
    df_exec = preprocess_data(data)

    output_dir = create_timestamped_output_dir("dispersion_metric_compare")

    sigma_params = dict(STRATEGY_DEFAULTS)
    sigma_params["dispersion_metric"] = "sigma"

    d_metric_params = dict(STRATEGY_DEFAULTS)
    d_metric_params["dispersion_metric"] = "long_short_mean_gap"

    print("[2/5] Running current dispersion metric (sigma)...")
    sigma_res, sigma_metrics = _run_case(
        df_exec,
        sigma_params,
        DEFAULT_START_DATE,
    )

    print("[3/5] Running D_t dispersion metric (long-short mean gap)...")
    d_res, d_metrics = _run_case(
        df_exec,
        d_metric_params,
        DEFAULT_START_DATE,
    )

    print("[4/5] Saving csv outputs...")
    metrics_df = pd.DataFrame(
        [
            {"model": "current_sigma", **sigma_metrics},
            {"model": "dispersion_D_t", **d_metrics},
            {
                "model": "delta(D_t-sigma)",
                **{
                    k: d_metrics[k] - sigma_metrics[k]
                    for k in ["AR", "RISK", "R/R", "MDD", "Total Return"]
                },
            },
        ]
    )
    metrics_path = os.path.join(output_dir, "dispersion_metric_compare_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    compare_daily = pd.DataFrame(
        {
            "current_sigma": sigma_res["daily_return"],
            "dispersion_D_t": d_res["daily_return"],
        }
    ).dropna()
    daily_path = os.path.join(output_dir, "dispersion_metric_compare_daily_return.csv")
    compare_daily.to_csv(daily_path, encoding="utf-8-sig")

    indicator_path = os.path.join(output_dir, "dispersion_indicator_compare.csv")
    indicator_df = pd.DataFrame(
        {
            "sigma_s": sigma_res["sigma_s"],
            "sigma_indicator": sigma_res["dispersion_indicator"],
            "D_t_indicator": d_res["dispersion_indicator"],
            "scale_sigma": sigma_res["scale"],
            "scale_D_t": d_res["scale"],
        }
    ).dropna()
    indicator_df.to_csv(indicator_path, encoding="utf-8-sig")

    print("[5/5] Saving comparison chart...")
    chart_path = os.path.join(output_dir, "dispersion_metric_compare.png")
    _save_chart(compare_daily, chart_path)

    print("\n=== Metrics Comparison ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {daily_path}")
    print(f"Saved: {indicator_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
