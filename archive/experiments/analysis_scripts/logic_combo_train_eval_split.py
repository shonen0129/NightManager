import itertools
import os
import sys
from copy import deepcopy

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
PERIODS = SIGNIFICANCE_CONFIG["periods"]
TRAIN_LABEL = "2015-2019"
EVAL_LABELS = ["2020-2022", "2023-now"]


def calc_cagr(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return float("nan")
    wealth = (1.0 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return wealth.iloc[-1] ** (1.0 / years) - 1.0


def summarize_period(daily_returns: pd.Series) -> dict:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {
            "samples": 0,
            "CAGR": float("nan"),
            "AR": float("nan"),
            "RISK": float("nan"),
            "R/R": float("nan"),
            "MDD": float("nan"),
            "Total Return": float("nan"),
        }

    m = calculate_metrics(daily_returns)
    return {
        "samples": int(len(daily_returns)),
        "CAGR": calc_cagr(daily_returns),
        "AR": m["AR"],
        "RISK": m["RISK"],
        "R/R": m["R/R"],
        "MDD": m["MDD"],
        "Total Return": m["Total Return"],
    }


def period_slice(series: pd.Series, label: str) -> pd.Series:
    period_map = {k: (st, ed) for k, st, ed in PERIODS}
    if label not in period_map:
        raise ValueError(f"Unknown period label: {label}")

    st, ed = period_map[label]
    out = series[series.index >= st]
    if ed is not None:
        out = out[out.index <= ed]
    return out


def run_case(df_exec: pd.DataFrame, params: dict) -> tuple[pd.Series, dict]:
    strategy = LeadLagStrategy(df_exec=df_exec, **params)
    result = strategy.run_backtest(start_date=PERIODS[0][1])
    daily = result["daily_return"].copy()

    train_metrics = summarize_period(period_slice(daily, TRAIN_LABEL))
    eval_1_metrics = summarize_period(period_slice(daily, EVAL_LABELS[0]))
    eval_2_metrics = summarize_period(period_slice(daily, EVAL_LABELS[1]))
    eval_all = daily[daily.index >= PERIODS[1][1]]
    eval_all_metrics = summarize_period(eval_all)

    row = {
        "train_period": TRAIN_LABEL,
        "eval_period_1": EVAL_LABELS[0],
        "eval_period_2": EVAL_LABELS[1],
        "train_samples": train_metrics["samples"],
        "train_cagr": train_metrics["CAGR"],
        "train_rr": train_metrics["R/R"],
        "train_mdd": train_metrics["MDD"],
        "eval1_samples": eval_1_metrics["samples"],
        "eval1_cagr": eval_1_metrics["CAGR"],
        "eval1_rr": eval_1_metrics["R/R"],
        "eval1_mdd": eval_1_metrics["MDD"],
        "eval2_samples": eval_2_metrics["samples"],
        "eval2_cagr": eval_2_metrics["CAGR"],
        "eval2_rr": eval_2_metrics["R/R"],
        "eval2_mdd": eval_2_metrics["MDD"],
        "eval_all_samples": eval_all_metrics["samples"],
        "eval_all_cagr": eval_all_metrics["CAGR"],
        "eval_all_rr": eval_all_metrics["R/R"],
        "eval_all_mdd": eval_all_metrics["MDD"],
    }
    return daily, row


def main() -> None:
    print("[1/4] Loading data...")
    data = download_data()
    df_exec = preprocess_data(data)

    base = deepcopy(LOGIC_DIFF_EXHAUSTIVE_BASE_PARAMS)
    combos = list(
        itertools.product(
            LOGIC_DIFF_EXHAUSTIVE_OPTIONS["ewma_half_life"],
            LOGIC_DIFF_EXHAUSTIVE_OPTIONS["weight_mode"],
            LOGIC_DIFF_EXHAUSTIVE_OPTIONS["dispersion_filter"],
            LOGIC_DIFF_EXHAUSTIVE_OPTIONS["two_stage"],
            LOGIC_DIFF_EXHAUSTIVE_OPTIONS["include_v4_prior"],
            LOGIC_DIFF_EXHAUSTIVE_OPTIONS["K"],
        )
    )

    print(f"[2/4] Running train/eval split grid: {len(combos)} cases...")
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

        _, row = run_case(df_exec, params)
        row["two_stage"] = two_stage
        row.update(params)
        rows.append(row)

    result_df = pd.DataFrame(rows)

    print("[3/4] Selecting by train-period only and preparing OOS report...")
    top_train_cagr = result_df.sort_values("train_cagr", ascending=False).head(10)
    top_train_rr = result_df.sort_values("train_rr", ascending=False).head(10)

    best_cagr = top_train_cagr.iloc[0]
    best_rr = top_train_rr.iloc[0]

    summary = pd.DataFrame(
        [
            {
                "selection_rule": "best_train_cagr",
                "train_cagr": best_cagr["train_cagr"],
                "train_rr": best_cagr["train_rr"],
                "eval1_cagr": best_cagr["eval1_cagr"],
                "eval1_rr": best_cagr["eval1_rr"],
                "eval2_cagr": best_cagr["eval2_cagr"],
                "eval2_rr": best_cagr["eval2_rr"],
                "eval_all_cagr": best_cagr["eval_all_cagr"],
                "eval_all_rr": best_cagr["eval_all_rr"],
            },
            {
                "selection_rule": "best_train_rr",
                "train_cagr": best_rr["train_cagr"],
                "train_rr": best_rr["train_rr"],
                "eval1_cagr": best_rr["eval1_cagr"],
                "eval1_rr": best_rr["eval1_rr"],
                "eval2_cagr": best_rr["eval2_cagr"],
                "eval2_rr": best_rr["eval2_rr"],
                "eval_all_cagr": best_rr["eval_all_cagr"],
                "eval_all_rr": best_rr["eval_all_rr"],
            },
        ]
    )

    print("[4/4] Saving files...")
    output_dir = create_timestamped_output_dir("logic_combo_train_eval_split")

    all_path = os.path.join(output_dir, "logic_combo_train_eval_split_all.csv")
    top_cagr_path = os.path.join(
        output_dir, "logic_combo_train_eval_split_top10_train_cagr.csv"
    )
    top_rr_path = os.path.join(
        output_dir, "logic_combo_train_eval_split_top10_train_rr.csv"
    )
    summary_path = os.path.join(output_dir, "logic_combo_train_eval_split_summary.csv")

    result_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    top_train_cagr.to_csv(top_cagr_path, index=False, encoding="utf-8-sig")
    top_train_rr.to_csv(top_rr_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== Train/Eval Split Summary ===")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(all_path)
    print(top_cagr_path)
    print(top_rr_path)
    print(summary_path)


if __name__ == "__main__":
    main()
