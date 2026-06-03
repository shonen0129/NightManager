import os
import pandas as pd

from backtest_config import EWMA_STATIC_GRID_CONFIG, create_timestamped_output_dir
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy


def main():
    print("Step 1: loading data...")
    data = download_data()
    df_exec = preprocess_data(data)

    configs = [
        {
            "label": "Baseline Static (Signal+Filter)",
            "ewma_half_life": None,
        }
    ] + [
        {
            "label": f"Static (Signal+Filter, EWMA{h})",
            "ewma_half_life": h,
        }
        for h in EWMA_STATIC_GRID_CONFIG["half_lives"]
    ]

    all_metrics = {}
    for cfg in configs:
        print(f"Running: {cfg['label']}")
        strat = LeadLagStrategy(
            df_exec,
            weight_mode=EWMA_STATIC_GRID_CONFIG["weight_mode"],
            dispersion_filter=EWMA_STATIC_GRID_CONFIG["dispersion_filter"],
            v3_mode=EWMA_STATIC_GRID_CONFIG["v3_mode"],
            ewma_half_life=cfg["ewma_half_life"],
        )
        res = strat.run_backtest(start_date=EWMA_STATIC_GRID_CONFIG["start_date"])
        all_metrics[cfg["label"]] = calculate_metrics(res["daily_return"])

    metrics_df = pd.DataFrame(all_metrics).T
    base = metrics_df.loc["Baseline Static (Signal+Filter)"]
    for col in ["AR", "RISK", "R/R", "MDD", "Total Return"]:
        metrics_df[f"delta_vs_base_{col}"] = metrics_df[col] - base[col]

    out_dir = create_timestamped_output_dir("ewma_static_grid_compare")
    out_path = os.path.join(out_dir, "static_signal_filter_ewma_grid.csv")
    metrics_df.to_csv(out_path, encoding="utf-8-sig")

    print("\n=== Comparison (raw) ===")
    print(metrics_df[["AR", "RISK", "R/R", "MDD", "Total Return"]].to_string())
    print("\n=== Delta vs Baseline ===")
    print(
        metrics_df[
            [
                "delta_vs_base_AR",
                "delta_vs_base_RISK",
                "delta_vs_base_R/R",
                "delta_vs_base_MDD",
                "delta_vs_base_Total Return",
            ]
        ].to_string()
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
