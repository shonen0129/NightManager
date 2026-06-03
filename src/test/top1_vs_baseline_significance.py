import numpy as np
import pandas as pd

from backtest_config import (
    LOGIC_DIFF_BASELINE_PARAMS,
    LOGIC_DIFF_TOP1_PARAMS,
    TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG,
)
from data_loader import download_data, preprocess_data
from strategy import LeadLagStrategy


TRADING_DAYS = TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG["trading_days"]
START_DATE = TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG["start_date"]


def moving_block_idx(t: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, t, size=(t // block) + 3)
    out = []
    for s in starts:
        out.extend(((s + np.arange(block)) % t).tolist())
        if len(out) >= t:
            break
    return np.array(out[:t], dtype=int)


def cagr(arr: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + arr)
    years = len(arr) / TRADING_DAYS
    return wealth[-1] ** (1.0 / years) - 1.0


def main() -> None:
    data = download_data()
    df_exec = preprocess_data(data)

    baseline = LeadLagStrategy(
        df_exec=df_exec,
        **LOGIC_DIFF_BASELINE_PARAMS,
    ).run_backtest(start_date=START_DATE)["daily_return"]

    top1 = LeadLagStrategy(
        df_exec=df_exec,
        **LOGIC_DIFF_TOP1_PARAMS,
    ).run_backtest(
        start_date=START_DATE
    )["daily_return"]

    aligned = pd.concat([top1, baseline], axis=1).dropna()
    aligned.columns = ["top1", "base"]
    excess = aligned["top1"] - aligned["base"]

    # HAC t-stat (Newey-West, lag=10)
    x = excess.values
    t = len(x)
    lags = TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG["hac_lags"]
    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / t
    var_hac = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gam = np.dot(xc[lag:], xc[:-lag]) / t
        var_hac += 2.0 * w * gam
    se_mean = np.sqrt(var_hac / t)
    t_hac = float(x.mean() / se_mean) if se_mean > 0 else np.nan

    # Moving block bootstrap
    rng = np.random.default_rng(TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG["bootstrap_seed"])
    b = TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG["bootstrap_samples"]
    block = TOP1_VS_BASELINE_SIGNIFICANCE_CONFIG["bootstrap_block"]
    arr = aligned.values
    ann_excess = np.empty(b, dtype=float)
    cagr_diff = np.empty(b, dtype=float)

    for i in range(b):
        ix = moving_block_idx(t, block, rng)
        s_top = arr[ix, 0]
        s_base = arr[ix, 1]
        ann_excess[i] = (s_top - s_base).mean() * TRADING_DAYS
        cagr_diff[i] = cagr(s_top) - cagr(s_base)

    obs_ann_excess = x.mean() * TRADING_DAYS
    obs_cagr_diff = cagr(aligned["top1"].values) - cagr(aligned["base"].values)

    ci_ann = np.percentile(ann_excess, [2.5, 97.5])
    ci_cagr = np.percentile(cagr_diff, [2.5, 97.5])
    p_ann = min(1.0, 2 * min(np.mean(ann_excess <= 0), np.mean(ann_excess >= 0)))
    p_cagr = min(1.0, 2 * min(np.mean(cagr_diff <= 0), np.mean(cagr_diff >= 0)))

    print(f"T={t}")
    print(f"obs_ann_excess={obs_ann_excess:.6f}")
    print(f"obs_cagr_diff={obs_cagr_diff:.6f}")
    print(f"hac_t_stat={t_hac:.4f}")
    print(
        "boot_ann_ci_2.5="
        f"{ci_ann[0]:.6f}, boot_ann_ci_97.5={ci_ann[1]:.6f}, p_two={p_ann:.4f}"
    )
    print(
        "boot_cagr_ci_2.5="
        f"{ci_cagr[0]:.6f}, boot_cagr_ci_97.5={ci_cagr[1]:.6f}, p_two={p_cagr:.4f}"
    )


if __name__ == "__main__":
    main()
