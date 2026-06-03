import os
import sys

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


def main() -> None:
    print("[1/4] Loading and preprocessing data...")
    data = download_data()
    df_exec = preprocess_data(data)

    start_date = DEFAULT_START_DATE

    current_params = dict(STRATEGY_DEFAULTS)
    current_params["signal_mode"] = "gap_residual"
    current_params["include_v4_prior"] = False
    current_params["K"] = 3

    k4_v4_params = dict(current_params)
    k4_v4_params["include_v4_prior"] = True
    k4_v4_params["K"] = 4

    print("[2/4] Running current (gap_residual, K=3, v4=False)...")
    current_res, current_metrics = _run_case(df_exec, current_params, start_date)

    print("[3/4] Running expanded (gap_residual, K=4, v4=True)...")
    k4_v4_res, k4_v4_metrics = _run_case(df_exec, k4_v4_params, start_date)

    print("[4/4] Saving outputs...")
    output_dir = create_timestamped_output_dir("gap_residual_k4_v4_compare")

    metrics_df = pd.DataFrame(
        [
            {"model": "current_gap_residual_K3_v4False", **current_metrics},
            {"model": "expanded_gap_residual_K4_v4True", **k4_v4_metrics},
        ]
    )
    metrics_df["delta_expanded_minus_current"] = [
        "",
        "",
    ]

    delta_row = {
        "model": "delta(expanded-current)",
    }
    for col in ["AR", "RISK", "R/R", "MDD", "Total Return"]:
        delta_row[col] = k4_v4_metrics[col] - current_metrics[col]
    metrics_df = pd.concat([metrics_df, pd.DataFrame([delta_row])], ignore_index=True)

    metrics_path = os.path.join(output_dir, "gap_residual_k4_v4_compare_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    daily_df = pd.DataFrame(
        {
            "current_gap_residual_K3_v4False": current_res["daily_return"],
            "expanded_gap_residual_K4_v4True": k4_v4_res["daily_return"],
        }
    ).dropna()
    daily_path = os.path.join(output_dir, "gap_residual_k4_v4_compare_daily_return.csv")
    daily_df.to_csv(daily_path, encoding="utf-8-sig")

    print("\n=== Metrics Comparison ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {daily_path}")


if __name__ == "__main__":
    main()
