import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backtest_config import (
    DEFAULT_START_DATE,
    LOGIC_DIFF_BASELINE_PARAMS,
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

    axes[0].set_title("STRATEGY_DEFAULTS vs LOGIC_DIFF_BASELINE_PARAMS")
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

    output_dir = create_timestamped_output_dir("strategy_defaults_vs_baseline")

    baseline_params = dict(LOGIC_DIFF_BASELINE_PARAMS)
    baseline_params["gap_open_coef"] = STRATEGY_DEFAULTS["gap_open_coef"]

    defaults_params = dict(STRATEGY_DEFAULTS)

    print("[2/5] Running LOGIC_DIFF_BASELINE_PARAMS...")
    baseline_res, baseline_metrics = _run_case(
        df_exec,
        baseline_params,
        DEFAULT_START_DATE,
    )

    print("[3/5] Running STRATEGY_DEFAULTS...")
    defaults_res, defaults_metrics = _run_case(
        df_exec,
        defaults_params,
        DEFAULT_START_DATE,
    )

    print("[4/5] Saving csv outputs...")
    metrics_df = pd.DataFrame(
        [
            {"model": "LOGIC_DIFF_BASELINE_PARAMS", **baseline_metrics},
            {"model": "STRATEGY_DEFAULTS", **defaults_metrics},
            {
                "model": "delta(defaults-baseline)",
                **{
                    k: defaults_metrics[k] - baseline_metrics[k]
                    for k in ["AR", "RISK", "R/R", "MDD", "Total Return"]
                },
            },
        ]
    )
    metrics_path = os.path.join(
        output_dir,
        "strategy_defaults_vs_baseline_metrics.csv",
    )
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    compare_daily = pd.DataFrame(
        {
            "LOGIC_DIFF_BASELINE_PARAMS": baseline_res["daily_return"],
            "STRATEGY_DEFAULTS": defaults_res["daily_return"],
        }
    ).dropna()
    daily_path = os.path.join(
        output_dir,
        "strategy_defaults_vs_baseline_daily_return.csv",
    )
    compare_daily.to_csv(daily_path, encoding="utf-8-sig")

    print("[5/5] Saving comparison chart...")
    chart_path = os.path.join(output_dir, "strategy_defaults_vs_baseline.png")
    _save_chart(compare_daily, chart_path)

    print("\n=== Metrics Comparison ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {daily_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
