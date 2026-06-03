import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backtest_config import (
    GAP_RESIDUAL_COMPARE_CONFIG,
    SIGNIFICANCE_CONFIG,
    create_timestamped_output_dir,
)
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy


def _run_case(df_exec, params, start_date):
    strategy = LeadLagStrategy(df_exec, **params)
    result = strategy.run_backtest(start_date=start_date)
    return result["daily_return"].copy()


def _slice_period(series, start_date, end_date):
    out = series[series.index >= start_date]
    if end_date is not None:
        out = out[out.index <= end_date]
    return out


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

    print("[2/4] Running baseline and gap-residual backtests...")
    baseline = _run_case(df_exec, base_params, start_date)
    gap_residual = _run_case(df_exec, residual_params, start_date)

    print("[3/4] Calculating period-split metrics...")
    periods = SIGNIFICANCE_CONFIG["periods"]
    rows = []
    for label, st, ed in periods:
        b = _slice_period(baseline, st, ed)
        g = _slice_period(gap_residual, st, ed)

        b_metrics = calculate_metrics(b)
        g_metrics = calculate_metrics(g)

        rows.append(
            {
                "period": label,
                "model": "baseline",
                "samples": len(b),
                **b_metrics,
            }
        )
        rows.append(
            {
                "period": label,
                "model": f"gap_residual_c={c_coef}",
                "samples": len(g),
                **g_metrics,
            }
        )

    period_df = pd.DataFrame(rows)

    print("[4/4] Saving outputs...")
    output_dir = create_timestamped_output_dir("gap_residual_period_split")

    out_path = os.path.join(output_dir, "gap_residual_period_split.csv")
    period_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== Gap Residual Period Split ===")
    print(period_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
