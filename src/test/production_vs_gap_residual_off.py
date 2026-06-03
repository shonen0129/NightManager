"""Compare production-equivalent settings vs gap_residual disabled.

- Case A: STRATEGY_DEFAULTS as-is (typically signal_mode='gap_residual')
- Case B: identical params but signal_mode='baseline' (i.e., gap_residual off)

Outputs a timestamped folder under results/.
"""

import os
import sys
from typing import Dict, Tuple

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
from backtest.runner import run_backtest_with_config
from domain.models.types import StrategyConfig
from performance import calculate_metrics


def _build_config_from_defaults(overrides: Dict) -> StrategyConfig:
    params = dict(STRATEGY_DEFAULTS)
    params.update(overrides)

    return StrategyConfig(
        k=params["K"],
        lambda_reg=params["lambda_reg"],
        q=params["q"],
        weight_mode=params["weight_mode"],
        dispersion_filter=params["dispersion_filter"],
        dispersion_metric=params.get("dispersion_metric", "long_short_mean_gap"),
        v3_mode=params["v3_mode"],
        ewma_half_life=params["ewma_half_life"],
        lambda_lw=params["lambda_lw"],
        lw_target=params["lw_target"],
        corr_window=params["corr_window"],
        include_v4_prior=params["include_v4_prior"],
        signal_mode=params["signal_mode"],
        gap_open_coef=params["gap_open_coef"],
        gamma=params.get("gamma", 0.5),
    )


def _run_case(
    df_exec: pd.DataFrame,
    label: str,
    config: StrategyConfig,
    start_date: str,
) -> Tuple[pd.DataFrame, Dict]:
    print(f"\n=== Running: {label} ===")
    results = run_backtest_with_config(df_exec, config, start_date)
    metrics = calculate_metrics(results["daily_return"])
    return results, metrics


def main() -> None:
    print("[1/5] Loading and preprocessing data...")
    data = download_data()
    df_exec = preprocess_data(data)

    output_dir = create_timestamped_output_dir("production_vs_gap_residual_off")

    prod_label = "production_equivalent (signal_mode=gap_residual)"
    off_label = "gap_residual_off (signal_mode=baseline)"

    prod_config = _build_config_from_defaults({})
    off_config = _build_config_from_defaults({"signal_mode": "baseline"})

    print("[2/5] Running production-equivalent...")
    prod_res, prod_metrics = _run_case(
        df_exec, prod_label, prod_config, DEFAULT_START_DATE
    )

    print("[3/5] Running gap_residual OFF...")
    off_res, off_metrics = _run_case(df_exec, off_label, off_config, DEFAULT_START_DATE)

    print("[4/5] Saving csv outputs...")
    metrics_df = pd.DataFrame(
        [
            {"model": prod_label, **prod_metrics},
            {"model": off_label, **off_metrics},
            {
                "model": "delta(prod - off)",
                **{
                    k: prod_metrics[k] - off_metrics[k]
                    for k in ["AR", "RISK", "R/R", "MDD", "Total Return"]
                },
            },
        ]
    )
    metrics_path = os.path.join(
        output_dir, "production_vs_gap_residual_off_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    prod_res.to_csv(
        os.path.join(output_dir, "production_daily_results.csv"),
        encoding="utf-8-sig",
    )
    off_res.to_csv(
        os.path.join(output_dir, "gap_residual_off_daily_results.csv"),
        encoding="utf-8-sig",
    )

    compare_daily = pd.DataFrame(
        {
            "production_equivalent": prod_res["daily_return"],
            "gap_residual_off": off_res["daily_return"],
        }
    ).dropna()
    daily_path = os.path.join(
        output_dir, "production_vs_gap_residual_off_daily_return.csv"
    )
    compare_daily.to_csv(daily_path, encoding="utf-8-sig")

    summary_path = os.path.join(output_dir, "00_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Production vs gap_residual OFF\n")
        f.write(f"start_date: {DEFAULT_START_DATE}\n")
        f.write("\n[STRATEGY_DEFAULTS]\n")
        for k, v in STRATEGY_DEFAULTS.items():
            f.write(f"{k}: {v}\n")
        f.write("\n[Config Overrides]\n")
        f.write("production_equivalent: (none)\n")
        f.write("gap_residual_off: signal_mode=baseline\n")
        f.write("\n[Metrics]\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")

    print("[5/5] Done")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {daily_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
