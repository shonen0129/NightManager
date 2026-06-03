import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from backtest_config import MULTI_WINDOW_CONFIG, create_timestamped_output_dir
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy
from backtest_ensemble import run_backtest_multi_window


def run_baseline(df_exec):
    base_params = MULTI_WINDOW_CONFIG["strategy_params"]
    strategy = LeadLagStrategy(
        df_exec=df_exec,
        K=base_params["K"],
        lambda_reg=base_params["lambda_reg"],
        q=base_params["q"],
        weight_mode=base_params["weight_mode"],
        dispersion_filter=base_params["dispersion_filter"],
        v3_mode=base_params["v3_mode"],
        ewma_half_life=base_params["ewma_half_life"],
        lambda_lw=base_params["lambda_lw"],
        lw_target=base_params["lw_target"],
        corr_window=MULTI_WINDOW_CONFIG["baseline_window"],
    )
    result = strategy.run_backtest(start_date=MULTI_WINDOW_CONFIG["start_date"])
    return result, calculate_metrics(result["daily_return"])


def run_three_way_performance_weighted(df_exec):
    base_params = MULTI_WINDOW_CONFIG["strategy_params"]
    strategy = LeadLagStrategy(
        df_exec=df_exec,
        K=base_params["K"],
        lambda_reg=base_params["lambda_reg"],
        q=base_params["q"],
        weight_mode=base_params["weight_mode"],
        dispersion_filter=base_params["dispersion_filter"],
        v3_mode=base_params["v3_mode"],
        ewma_half_life=base_params["ewma_half_life"],
        lambda_lw=base_params["lambda_lw"],
        lw_target=base_params["lw_target"],
        corr_window=MULTI_WINDOW_CONFIG["baseline_window"],
    )

    result = run_backtest_multi_window(
        strategy=strategy,
        start_date=MULTI_WINDOW_CONFIG["start_date"],
        windows=MULTI_WINDOW_CONFIG["ensemble_windows"],
        combine_mode=MULTI_WINDOW_CONFIG["combine_mode"],
        performance_lookback=MULTI_WINDOW_CONFIG["performance_lookback"],
    )
    return result, calculate_metrics(result["daily_return"])


def make_plot(baseline_result, ensemble_result, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    baseline_window = MULTI_WINDOW_CONFIG["baseline_window"]
    windows_text = ",".join(str(w) for w in MULTI_WINDOW_CONFIG["ensemble_windows"])
    combine_mode = MULTI_WINDOW_CONFIG["combine_mode"]

    baseline_wealth = (1 + baseline_result["daily_return"]).cumprod()
    ensemble_wealth = (1 + ensemble_result["daily_return"]).cumprod()

    axes[0].plot(
        baseline_wealth.index,
        baseline_wealth.values,
        label=f"Baseline (L={baseline_window})",
        linewidth=1.2,
        color="tab:blue",
    )
    axes[0].plot(
        ensemble_wealth.index,
        ensemble_wealth.values,
        label=f"Ensemble (L={windows_text} / {combine_mode})",
        linewidth=1.2,
        color="tab:orange",
    )
    axes[0].set_title("Baseline vs Multi-window Ensemble")
    axes[0].set_ylabel("Cumulative Wealth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    baseline_dd = baseline_wealth / baseline_wealth.cummax() - 1.0
    ensemble_dd = ensemble_wealth / ensemble_wealth.cummax() - 1.0

    axes[1].plot(
        baseline_dd.index,
        baseline_dd.values,
        label="Baseline DD",
        linewidth=1.0,
        color="tab:blue",
    )
    axes[1].plot(
        ensemble_dd.index,
        ensemble_dd.values,
        label="Ensemble DD",
        linewidth=1.0,
        color="tab:orange",
    )
    axes[1].fill_between(
        baseline_dd.index,
        baseline_dd.values,
        0,
        alpha=0.15,
        color="tab:blue",
    )
    axes[1].fill_between(
        ensemble_dd.index,
        ensemble_dd.values,
        0,
        alpha=0.15,
        color="tab:orange",
    )
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    print("Step 1: loading data...")
    data = download_data()
    df_exec = preprocess_data(data)

    outputs = {}

    print(
        "Step 2: running baseline " f"(L={MULTI_WINDOW_CONFIG['baseline_window']})..."
    )
    baseline_result, baseline_metrics = run_baseline(df_exec)
    outputs["baseline_L60"] = baseline_metrics

    window_text = ",".join(str(w) for w in MULTI_WINDOW_CONFIG["ensemble_windows"])
    print(
        "Step 3: running multi-window ensemble "
        f"(L={window_text}, {MULTI_WINDOW_CONFIG['combine_mode']})..."
    )
    ensemble_result, ensemble_metrics = run_three_way_performance_weighted(df_exec)
    outputs["ensemble_perf_weighted_20_60_120"] = ensemble_metrics

    comparison = pd.DataFrame(outputs).T
    base = comparison.loc["baseline_L60"]
    for col in ["AR", "RISK", "R/R", "MDD", "Total Return"]:
        comparison[f"delta_vs_baseline_{col}"] = comparison[col] - base[col]

    output_dir = create_timestamped_output_dir("multi_window_compare")

    output_path = os.path.join(output_dir, "comparison_metrics_multi_window.csv")
    comparison.to_csv(output_path, encoding="utf-8-sig")

    chart_path = os.path.join(output_dir, "comparison_multi_window_3way.png")
    make_plot(baseline_result, ensemble_result, chart_path)

    print("\n=== Multi-window comparison ===")
    print(comparison[["AR", "RISK", "R/R", "MDD", "Total Return"]].to_string())

    print("\n=== Delta vs baseline_L60 ===")
    print(
        comparison[
            [
                "delta_vs_baseline_AR",
                "delta_vs_baseline_RISK",
                "delta_vs_baseline_R/R",
                "delta_vs_baseline_MDD",
                "delta_vs_baseline_Total Return",
            ]
        ].to_string()
    )

    baseline_wealth = (1 + baseline_result["daily_return"]).cumprod().iloc[-1]
    ensemble_wealth = (1 + ensemble_result["daily_return"]).cumprod().iloc[-1]
    print("\nFinal wealth check:")
    print(
        f"baseline(L={MULTI_WINDOW_CONFIG['baseline_window']}): "
        f"{baseline_wealth:.4f}"
    )
    print(f"ensemble(L={window_text}): {ensemble_wealth:.4f}")

    print(f"\nSaved: {output_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
