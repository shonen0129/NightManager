import os
from statistics import NormalDist

import numpy as np
import pandas as pd

from backtest_config import LOGIC_DIFF_BASELINE_PARAMS, SIGNIFICANCE_CONFIG
from data_loader import download_data, preprocess_data
from strategy import LeadLagStrategy


TRADING_DAYS = SIGNIFICANCE_CONFIG["trading_days"]
START_DATE = SIGNIFICANCE_CONFIG["start_date"]
PERIODS = SIGNIFICANCE_CONFIG["periods"]


def to_none_if_nan(x):
    return None if pd.isna(x) else x


def annualized_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {
            "CAGR": np.nan,
            "AR": np.nan,
            "RISK": np.nan,
            "SR": np.nan,
            "MDD": np.nan,
        }

    wealth = (1.0 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan

    ar = ret.mean() * TRADING_DAYS
    risk = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sr = ar / risk if risk > 0 else np.nan

    dd = wealth / wealth.cummax() - 1.0
    mdd = dd.min()
    return {"CAGR": cagr, "AR": ar, "RISK": risk, "SR": sr, "MDD": mdd}


def psr(
    sr_daily: float, sr_star_daily: float, n: int, skew: float, kurt: float
) -> float:
    # Bailey-style PSR approximation.
    denom_sq = 1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2)
    if denom_sq <= 0 or n <= 2:
        return np.nan
    z = (sr_daily - sr_star_daily) * np.sqrt((n - 1.0) / denom_sq)
    return NormalDist().cdf(z)


def moving_block_indices(t: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, t, size=(t // block) + 2)
    idx = []
    for s in starts:
        idx.extend(((s + np.arange(block)) % t).tolist())
        if len(idx) >= t:
            break
    return np.array(idx[:t], dtype=int)


def bootstrap_cagr_diff(
    a: pd.Series,
    b: pd.Series,
    n_boot: int = SIGNIFICANCE_CONFIG["bootstrap_samples"],
    block: int = SIGNIFICANCE_CONFIG["bootstrap_block"],
    seed: int = SIGNIFICANCE_CONFIG["bootstrap_seed"],
) -> dict:
    aligned = pd.concat([a, b], axis=1).dropna()
    aligned.columns = ["a", "b"]
    t = len(aligned)
    rng = np.random.default_rng(seed)

    obs = (
        annualized_metrics(aligned["a"])["CAGR"]
        - annualized_metrics(aligned["b"])["CAGR"]
    )

    diffs = np.empty(n_boot, dtype=float)
    arr = aligned.values
    for i in range(n_boot):
        ix = moving_block_indices(t, block, rng)
        sa = pd.Series(arr[ix, 0])
        sb = pd.Series(arr[ix, 1])
        diffs[i] = annualized_metrics(sa)["CAGR"] - annualized_metrics(sb)["CAGR"]

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_two = min(1.0, p_two)

    return {
        "observed_diff": obs,
        "ci_2_5": lo,
        "ci_97_5": hi,
        "prob_diff_gt_0": float(np.mean(diffs > 0)),
        "p_value_two_sided": p_two,
    }


def reality_check_max_mean(
    excess_df: pd.DataFrame,
    n_boot: int = SIGNIFICANCE_CONFIG["bootstrap_samples"],
    block: int = SIGNIFICANCE_CONFIG["bootstrap_block"],
    seed: int = SIGNIFICANCE_CONFIG["reality_check_seed"],
) -> dict:
    # White-style max-mean bootstrap reality check (simplified).
    x = excess_df.dropna().values
    t, m = x.shape

    obs_means = x.mean(axis=0)
    obs_max = float(np.max(obs_means))

    centered = x - obs_means[None, :]
    rng = np.random.default_rng(seed)
    boot_max = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        ix = moving_block_indices(t, block, rng)
        bs = centered[ix, :].mean(axis=0)
        boot_max[i] = np.max(bs)

    p = float(np.mean(boot_max >= obs_max))
    best_col = excess_df.columns[int(np.argmax(obs_means))]
    return {
        "observed_max_mean_excess_daily": obs_max,
        "p_value": p,
        "best_candidate": best_col,
    }


def build_strategy_from_row(df_exec: pd.DataFrame, row: pd.Series) -> LeadLagStrategy:
    return LeadLagStrategy(
        df_exec=df_exec,
        K=int(row["K"]),
        lambda_reg=float(row["lambda_reg"]),
        q=float(row["q"]),
        weight_mode=str(row["weight_mode"]),
        dispersion_filter=bool(row["dispersion_filter"]),
        v3_mode=str(row["v3_mode"]),
        ewma_half_life=to_none_if_nan(row["ewma_half_life"]),
        lambda_lw=float(row["lambda_lw"]),
        lw_target=str(row["lw_target"]),
        corr_window=int(row["corr_window"]),
        include_v4_prior=bool(row["include_v4_prior"]),
    )


def main() -> None:
    print("[1/5] Loading data...")
    data = download_data()
    df_exec = preprocess_data(data)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    top_path = os.path.join(results_dir, "logic_combo_exhaustive_top10_cagr.csv")
    top_df = pd.read_csv(top_path)

    print("[2/5] Running top-10 strategies for daily return series...")
    ret_map = {}
    metric_rows = []
    for i, row in top_df.iterrows():
        name = f"Top{i+1}"
        strat = build_strategy_from_row(df_exec, row)
        back = strat.run_backtest(start_date=START_DATE)
        ret = back["daily_return"].copy()
        ret_map[name] = ret

        m = annualized_metrics(ret)
        metric_rows.append({"name": name, **m})

    metrics_df = pd.DataFrame(metric_rows).sort_values("CAGR", ascending=False)

    print("[3/5] Period-split robustness...")
    period_rows = []
    for _, r in metrics_df.iterrows():
        name = r["name"]
        series = ret_map[name]
        for label, st, ed in PERIODS:
            sub = series[series.index >= st]
            if ed is not None:
                sub = sub[sub.index <= ed]
            mm = annualized_metrics(sub)
            period_rows.append(
                {"name": name, "period": label, **mm, "samples": len(sub)}
            )
    period_df = pd.DataFrame(period_rows)

    print("[4/5] Bootstrap CAGR-diff + Reality Check + Deflated Sharpe-like stats...")
    # Best vs second-best by CAGR.
    ordered = metrics_df["name"].tolist()
    best = ordered[0]
    second = ordered[1]
    boot = bootstrap_cagr_diff(ret_map[best], ret_map[second])

    # Baseline strategy for reality check benchmark.
    baseline = LeadLagStrategy(
        df_exec=df_exec,
        **LOGIC_DIFF_BASELINE_PARAMS,
    ).run_backtest(start_date=START_DATE)["daily_return"]

    excess = pd.DataFrame({k: v for k, v in ret_map.items()}).sub(baseline, axis=0)
    rc = reality_check_max_mean(excess)

    # Deflated Sharpe-like: PSR against multiple-testing threshold.
    n_trials = len(ret_map)
    dsr_rows = []
    for name, series in ret_map.items():
        s = series.dropna()
        n = len(s)
        mu = s.mean()
        sd = s.std(ddof=1)
        if sd <= 0 or n <= 2:
            dsr_rows.append(
                {
                    "name": name,
                    "PSR_sr_gt_0": np.nan,
                    "DSR_like": np.nan,
                    "SR_ann": np.nan,
                }
            )
            continue

        sr_d = mu / sd
        skew = s.skew()
        kurt = s.kurtosis() + 3.0

        psr0 = psr(sr_d, 0.0, n, skew, kurt)
        sr_star_multi = NormalDist().inv_cdf(1.0 - 1.0 / n_trials) / np.sqrt(
            max(n - 1, 1)
        )
        dsr_like = psr(sr_d, sr_star_multi, n, skew, kurt)

        dsr_rows.append(
            {
                "name": name,
                "SR_ann": sr_d * np.sqrt(TRADING_DAYS),
                "PSR_sr_gt_0": psr0,
                "DSR_like": dsr_like,
                "sr_star_daily_multi": sr_star_multi,
            }
        )
    dsr_df = pd.DataFrame(dsr_rows).sort_values("DSR_like", ascending=False)

    summary = pd.DataFrame(
        [
            {
                "best_by_cagr": best,
                "second_by_cagr": second,
                **boot,
                **rc,
            }
        ]
    )

    print("[5/5] Saving outputs...")
    out_metrics = os.path.join(results_dir, "significance_top10_metrics.csv")
    out_period = os.path.join(results_dir, "significance_period_split.csv")
    out_dsr = os.path.join(results_dir, "significance_dsr_like.csv")
    out_summary = os.path.join(results_dir, "significance_summary.csv")

    metrics_df.to_csv(out_metrics, index=False, encoding="utf-8-sig")
    period_df.to_csv(out_period, index=False, encoding="utf-8-sig")
    dsr_df.to_csv(out_dsr, index=False, encoding="utf-8-sig")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print("\n=== Significance Summary ===")
    print(summary.to_string(index=False))
    print("\n=== DSR-like Top 5 ===")
    print(dsr_df.head(5).to_string(index=False))
    print("\nSaved:")
    print(out_metrics)
    print(out_period)
    print(out_dsr)
    print(out_summary)


if __name__ == "__main__":
    main()
