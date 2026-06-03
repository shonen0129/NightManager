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
from strategy import LeadLagStrategy


def _plot_monthly_returns(monthly_returns: pd.Series, output_path: str) -> None:
    colors = ["#1b9e77" if x >= 0 else "#d95f02" for x in monthly_returns.values]

    plt.figure(figsize=(16, 6))
    plt.bar(monthly_returns.index, monthly_returns.values * 100.0, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("STRATEGY_DEFAULTS Monthly Return")
    plt.ylabel("Monthly Return (%)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    print("[1/4] Loading and preprocessing data...")
    data = download_data()
    df_exec = preprocess_data(data)

    print("[2/4] Running STRATEGY_DEFAULTS backtest...")
    strategy = LeadLagStrategy(df_exec=df_exec, **dict(STRATEGY_DEFAULTS))
    result = strategy.run_backtest(start_date=DEFAULT_START_DATE)
    daily_return = result["daily_return"].dropna()

    print("[3/4] Aggregating monthly returns...")
    monthly_returns = (1.0 + daily_return).resample("ME").prod() - 1.0
    monthly_df = monthly_returns.to_frame(name="monthly_return")
    monthly_df.index.name = "month_end"

    output_dir = create_timestamped_output_dir("strategy_defaults_monthly_return")
    csv_path = os.path.join(output_dir, "strategy_defaults_monthly_return.csv")
    png_path = os.path.join(output_dir, "strategy_defaults_monthly_return.png")
    monthly_df.to_csv(csv_path, encoding="utf-8-sig")

    print("[4/4] Saving monthly return chart...")
    _plot_monthly_returns(monthly_returns, png_path)

    positive_ratio = float((monthly_returns > 0).mean())
    print(f"Positive month ratio: {positive_ratio:.2%}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
