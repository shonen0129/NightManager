import itertools
import os
from copy import deepcopy

import pandas as pd

from backtest_config import (
    create_timestamped_output_dir,
    LOGIC_DIFF_EXHAUSTIVE_BASE_PARAMS,
    LOGIC_DIFF_EXHAUSTIVE_OPTIONS,
    SIGNIFICANCE_CONFIG,
)
from data_loader import download_data, preprocess_data
from performance import calculate_metrics
from strategy import LeadLagStrategy


TRADING_DAYS_PER_YEAR = SIGNIFICANCE_CONFIG["trading_days"]
START_DATE = SIGNIFICANCE_CONFIG["start_date"]


def calc_cagr(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    wealth = (1.0 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return wealth.iloc[-1] ** (1.0 / years) - 1.0


def run_case(df_exec: pd.DataFrame, params: dict) -> dict:
    strat = LeadLagStrategy(df_exec=df_exec, **params)
    res = strat.run_backtest(start_date=START_DATE)
    metrics = calculate_metrics(res["daily_return"])

    out = {
        "samples": int(len(res)),
        "CAGR": calc_cagr(res["daily_return"]),
        "AR": metrics["AR"],
        "RISK": metrics["RISK"],
        "R/R": metrics["R/R"],
        "MDD": metrics["MDD"],
        "Total Return": metrics["Total Return"],
    }
    out.update(params)
    return out


def main() -> None:
    print("[1/4] Loading data...")
    data = download_data()
    df_exec = preprocess_data(data)

    base = deepcopy(LOGIC_DIFF_EXHAUSTIVE_BASE_PARAMS)

    # All combinations of the logic-diff switches.
    ewma_options = LOGIC_DIFF_EXHAUSTIVE_OPTIONS["ewma_half_life"]
    weight_options = LOGIC_DIFF_EXHAUSTIVE_OPTIONS["weight_mode"]
    filter_options = LOGIC_DIFF_EXHAUSTIVE_OPTIONS["dispersion_filter"]
    two_stage_options = LOGIC_DIFF_EXHAUSTIVE_OPTIONS["two_stage"]
    v4_options = LOGIC_DIFF_EXHAUSTIVE_OPTIONS["include_v4_prior"]
    k_options = LOGIC_DIFF_EXHAUSTIVE_OPTIONS["K"]

    combos = list(
        itertools.product(
            ewma_options,
            weight_options,
            filter_options,
            two_stage_options,
            v4_options,
            k_options,
        )
    )

    print(f"[2/4] Running exhaustive grid: {len(combos)} cases...")
    rows = []
    for i, (ewma, wmode, dfilter, two_stage, v4, k) in enumerate(combos, start=1):
        params = {
            **base,
            "K": k,
            "weight_mode": wmode,
            "dispersion_filter": dfilter,
            "ewma_half_life": ewma,
            "include_v4_prior": v4,
            "lambda_lw": 0.5 if two_stage else 0.0,
            "lw_target": "equicorrelation" if two_stage else "identity",
        }

        print(
            f"  - [{i:02d}/{len(combos)}] "
            f"K={k}, ewma={ewma}, w={wmode}, filter={dfilter}, "
            f"2stage={two_stage}, v4={v4}"
        )
        row = run_case(df_exec, params)
        row["two_stage"] = two_stage
        rows.append(row)

    result_df = pd.DataFrame(rows)

    print("[3/4] Ranking results...")
    by_cagr = result_df.sort_values("CAGR", ascending=False).reset_index(drop=True)
    by_rr = result_df.sort_values("R/R", ascending=False).reset_index(drop=True)

    output_dir = create_timestamped_output_dir("logic_combo_exhaustive")

    all_csv = os.path.join(output_dir, "logic_combo_exhaustive_all.csv")
    top_cagr_csv = os.path.join(output_dir, "logic_combo_exhaustive_top10_cagr.csv")
    top_rr_csv = os.path.join(output_dir, "logic_combo_exhaustive_top10_rr.csv")

    result_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    by_cagr.head(10).to_csv(top_cagr_csv, index=False, encoding="utf-8-sig")
    by_rr.head(10).to_csv(top_rr_csv, index=False, encoding="utf-8-sig")

    print("\n=== Best by CAGR ===")
    print(by_cagr.head(10).to_string(index=False))
    print("\n=== Best by R/R ===")
    print(by_rr.head(10).to_string(index=False))

    print("[4/4] Saved files:")
    print(f"  - {all_csv}")
    print(f"  - {top_cagr_csv}")
    print(f"  - {top_rr_csv}")


if __name__ == "__main__":
    main()
