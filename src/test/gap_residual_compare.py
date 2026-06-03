import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backtest_config import GAP_RESIDUAL_COMPARE_CONFIG, create_timestamped_output_dir
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy


def _run_case(df_exec, params, start_date):
    strategy = LeadLagStrategy(df_exec, **params)
    result = strategy.run_backtest(start_date=start_date)
    metrics = calculate_metrics(result["daily_return"])
    return result, metrics


def main():
    print("[1/4] Loading and preprocessing data...")
    data = download_data()
    df_exec = preprocess_data(data)

    cfg = GAP_RESIDUAL_COMPARE_CONFIG
    start_date = cfg["start_date"]
    c_coef = cfg["gap_open_coef"]

    base_params = dict(cfg["baseline_params"])
    base_params["signal_mode"] = "baseline"
    base_params["gap_open_coef"] = c_coef

    residual_params = dict(base_params)
    residual_params["signal_mode"] = "gap_residual"

    print("[2/4] Running baseline backtest...")
    baseline_res, baseline_metrics = _run_case(df_exec, base_params, start_date)

    print("[3/4] Running gap-residual backtest...")
    residual_res, residual_metrics = _run_case(df_exec, residual_params, start_date)

    print("[4/4] Saving comparison outputs...")
    output_dir = create_timestamped_output_dir("gap_residual_compare")

    metrics_df = pd.DataFrame(
        [
            {"model": "baseline", **baseline_metrics},
            {
                "model": f"gap_residual_c={c_coef}",
                **residual_metrics,
            },
        ]
    )
    metrics_path = os.path.join(output_dir, "gap_residual_compare_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    compare_daily = pd.DataFrame(
        {
            "baseline": baseline_res["daily_return"],
            f"gap_residual_c_{str(c_coef).replace('.', '_')}": residual_res[
                "daily_return"
            ],
        }
    ).dropna()
    daily_path = os.path.join(output_dir, "gap_residual_compare_daily_return.csv")
    compare_daily.to_csv(daily_path, encoding="utf-8-sig")

    print("\n=== Metrics Comparison ===")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {daily_path}")


if __name__ == "__main__":
    main()
