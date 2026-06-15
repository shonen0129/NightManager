import os
from copy import deepcopy

import pandas as pd

from backtest_config import (
    create_timestamped_output_dir,
    LOGIC_DIFF_BASELINE_PARAMS,
    LOGIC_DIFF_INCREMENTAL_STEPS,
    LOGIC_DIFF_SINGLE_FACTOR_CHANGES,
    SIGNIFICANCE_CONFIG,
)
from data_loader import download_data, preprocess_data
from strategy import LeadLagStrategy
from performance import calculate_metrics


TRADING_DAYS_PER_YEAR = SIGNIFICANCE_CONFIG["trading_days"]
START_DATE = SIGNIFICANCE_CONFIG["start_date"]


def calc_cagr(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    wealth = (1.0 + daily_returns).cumprod()
    total_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if total_years <= 0:
        return float("nan")
    return wealth.iloc[-1] ** (1.0 / total_years) - 1.0


def run_case(df_exec: pd.DataFrame, label: str, params: dict) -> dict:
    strat = LeadLagStrategy(df_exec=df_exec, **params)
    result = strat.run_backtest(start_date=START_DATE)
    metrics = calculate_metrics(result["daily_return"])
    cagr = calc_cagr(result["daily_return"])
    return {
        "case": label,
        "samples": int(len(result)),
        "CAGR": cagr,
        "AR": metrics["AR"],
        "RISK": metrics["RISK"],
        "R/R": metrics["R/R"],
        "MDD": metrics["MDD"],
        "Total Return": metrics["Total Return"],
    }


def main() -> None:
    print("[1/4] Loading data...")
    data = download_data()
    df_exec = preprocess_data(data)

    base = deepcopy(LOGIC_DIFF_BASELINE_PARAMS)

    # Incremental ladder from original baseline to current policy design.
    cases = []

    p0 = deepcopy(base)
    cases.append(("S0 Baseline(Original)", p0))

    current = deepcopy(p0)
    for step in LOGIC_DIFF_INCREMENTAL_STEPS:
        current = deepcopy(current)
        current.update(step["update"])
        cases.append((step["label"], current))

    print("[2/4] Running incremental backtests...")
    rows = []
    for idx, (label, params) in enumerate(cases, start=1):
        print(f"  - [{idx}/{len(cases)}] {label}")
        rows.append(run_case(df_exec, label, params))

    print("[3/4] Building impact table...")
    out = pd.DataFrame(rows)
    out["Delta_CAGR_vs_prev"] = out["CAGR"].diff()
    out["Delta_CAGR_vs_baseline"] = out["CAGR"] - out.loc[0, "CAGR"]
    out["Delta_CAGR_vs_prev_bps"] = out["Delta_CAGR_vs_prev"] * 10000
    out["Delta_CAGR_vs_baseline_bps"] = out["Delta_CAGR_vs_baseline"] * 10000

    # One-factor-at-a-time impacts from baseline (order-independent view).
    single_rows = []
    for change in LOGIC_DIFF_SINGLE_FACTOR_CHANGES:
        params = deepcopy(base)
        params.update(change["update"])
        single_rows.append(run_case(df_exec, change["label"], params))

    single_df = pd.DataFrame(single_rows)
    single_df["Delta_CAGR_vs_baseline"] = single_df["CAGR"] - out.loc[0, "CAGR"]
    single_df["Delta_CAGR_vs_baseline_bps"] = (
        single_df["Delta_CAGR_vs_baseline"] * 10000
    )

    output_dir = create_timestamped_output_dir("logic_cagr_impact")
    out_csv = os.path.join(output_dir, "logic_cagr_impact.csv")
    single_csv = os.path.join(output_dir, "logic_cagr_single_factor_impact.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    single_df.to_csv(single_csv, index=False, encoding="utf-8-sig")

    display_cols = [
        "case",
        "CAGR",
        "Delta_CAGR_vs_prev",
        "Delta_CAGR_vs_baseline",
        "R/R",
        "MDD",
        "Total Return",
    ]
    print("\n=== Incremental CAGR Impact ===")
    print(out[display_cols].to_string(index=False))
    print("\n=== Single-Factor CAGR Impact (vs Baseline) ===")
    print(
        single_df[
            [
                "case",
                "CAGR",
                "Delta_CAGR_vs_baseline",
                "R/R",
                "MDD",
                "Total Return",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {single_csv}")
    print("[4/4] Done.")


if __name__ == "__main__":
    main()
