"""Scaling Strategy Comparison: Current vs Hybrid-Kelly vs Multi-stage

Runs a single backtest with dispersion_filter=False (scale=1.0 always),
then applies different scaling strategies POST-HOC to compare performance.

Strategies compared:
  A) Current 3-stage filter  (P10→0, P10-P25→0.5, P25+→1.0)
  B) 5-stage filter          (P10→0, P10-P20→0.3, P20-P35→0.6, P35-P60→0.8, P60+→1.0)
  C) 7-stage filter          (finer gradient)
  D) Hybrid Kelly            (current filter base + Kelly adjustment for α=1 zone)
  E) Continuous Kelly        (pure sigmoid mapping from D_t to scale)
  F) No filter               (always 1.0)
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import STRATEGY_DEFAULTS, N_US_ASSETS, N_JP_ASSETS
from data_loader import download_data, preprocess_data
from domain.models.types import StrategyConfig
from domain.signals import lead_lag as signals
from performance import calculate_metrics

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "kelly_analysis")
HISTORY_WINDOW = 60
OOS_SPLIT_DATE = "2020-01-01"


# ============================================================
# Scaling strategy implementations
# ============================================================

def scale_current_3stage(indicator: float, history: np.ndarray) -> float:
    """Current production: 3-stage discrete filter."""
    if len(history) < HISTORY_WINDOW:
        return 1.0
    hist = history[-HISTORY_WINDOW:]
    p10 = np.percentile(hist, 10)
    p25 = np.percentile(hist, 25)
    if indicator < p10:
        return 0.0
    elif indicator < p25:
        return 0.5
    return 1.0


def scale_5stage(indicator: float, history: np.ndarray) -> float:
    """5-stage discrete filter with finer gradation."""
    if len(history) < HISTORY_WINDOW:
        return 1.0
    hist = history[-HISTORY_WINDOW:]
    p10 = np.percentile(hist, 10)
    p20 = np.percentile(hist, 20)
    p35 = np.percentile(hist, 35)
    p60 = np.percentile(hist, 60)
    if indicator < p10:
        return 0.0
    elif indicator < p20:
        return 0.3
    elif indicator < p35:
        return 0.6
    elif indicator < p60:
        return 0.8
    return 1.0


def scale_7stage(indicator: float, history: np.ndarray) -> float:
    """7-stage discrete filter with even finer gradation."""
    if len(history) < HISTORY_WINDOW:
        return 1.0
    hist = history[-HISTORY_WINDOW:]
    p5 = np.percentile(hist, 5)
    p10 = np.percentile(hist, 10)
    p20 = np.percentile(hist, 20)
    p35 = np.percentile(hist, 35)
    p50 = np.percentile(hist, 50)
    p70 = np.percentile(hist, 70)
    if indicator < p5:
        return 0.0
    elif indicator < p10:
        return 0.15
    elif indicator < p20:
        return 0.35
    elif indicator < p35:
        return 0.55
    elif indicator < p50:
        return 0.75
    elif indicator < p70:
        return 0.90
    return 1.0


def scale_hybrid_kelly(
    indicator: float,
    history: np.ndarray,
    return_history: np.ndarray,
    kelly_fraction: float = 0.5,
) -> float:
    """Hybrid: current 3-stage base + Kelly adjustment in the α=1.0 zone.

    For D_t < P10: 0.0 (same as current)
    For P10 <= D_t < P25: 0.5 (same as current)
    For D_t >= P25: scale by Kelly-derived factor (0.5 to 1.2)
    """
    if len(history) < HISTORY_WINDOW:
        return 1.0
    hist = history[-HISTORY_WINDOW:]
    p10 = np.percentile(hist, 10)
    p25 = np.percentile(hist, 25)

    if indicator < p10:
        return 0.0
    elif indicator < p25:
        return 0.5

    # In the α=1.0 zone, apply Kelly refinement
    if len(return_history) < HISTORY_WINDOW:
        return 1.0

    # Compute conditional win rate for high-dispersion days
    ret_arr = return_history[-HISTORY_WINDOW:]
    hist_arr = hist

    # Median split: use indicator's position relative to history
    pctl = float(np.searchsorted(np.sort(hist_arr), indicator) / len(hist_arr))

    # Estimate Kelly parameters from recent high-dispersion returns
    mask_high = hist_arr >= np.percentile(hist_arr, 25)  # same cutoff
    if np.sum(mask_high) < 10:
        return 1.0

    recent_ret = ret_arr[mask_high[-len(ret_arr):]] if len(ret_arr) == len(hist_arr) else ret_arr
    if len(recent_ret) < 10:
        return 1.0

    mu = np.mean(recent_ret)
    var = np.var(recent_ret, ddof=1)

    if var < 1e-16 or mu <= 0:
        return 0.7  # conservative default when signal is unclear

    # Kelly fraction (with fractional Kelly)
    f_kelly = kelly_fraction * mu / var

    # Map f_kelly into [0.5, 1.2] range
    # Typical f_kelly values are 50-120 (because daily mu ~ 0.005, var ~ 1e-4)
    # Normalize to a usable range
    f_normalized = np.clip(f_kelly / 100.0, 0.5, 1.2)

    # Further modulate by D_t's percentile rank (higher D_t → higher scale)
    # This gives a smooth gradient from 0.5 to 1.2
    pctl_in_high = (pctl - 0.25) / 0.75  # 0 at P25, 1 at P100
    scale = 0.5 + 0.7 * pctl_in_high * (f_normalized / 1.2)

    return float(np.clip(scale, 0.5, 1.2))


def scale_continuous_sigmoid(
    indicator: float,
    history: np.ndarray,
) -> float:
    """Continuous sigmoid mapping from D_t percentile to scale [0, 1.0]."""
    if len(history) < HISTORY_WINDOW:
        return 1.0
    hist = history[-HISTORY_WINDOW:]
    # Percentile rank
    pctl = float(np.searchsorted(np.sort(hist), indicator) / len(hist))
    # Sigmoid centered at 0.2 (skewed toward aggressive trading)
    # σ(x) = 1 / (1 + exp(-k*(x - x0)))
    k = 10.0  # steepness
    x0 = 0.15  # center point (below P20)
    scale = 1.0 / (1.0 + np.exp(-k * (pctl - x0)))
    return float(np.clip(scale, 0.0, 1.0))


def scale_no_filter(indicator: float, history: np.ndarray) -> float:
    """No filter: always full exposure."""
    return 1.0


# ============================================================
# Backtest with unscaled returns
# ============================================================

def run_unscaled_backtest(
    df_exec: pd.DataFrame,
    config: StrategyConfig,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """Run backtest with dispersion_filter=False to get raw unscaled returns.

    Returns DataFrame with:
      - unscaled_return: return with scale=1.0
      - dispersion_indicator: D_t for each day
      - All original columns
    """
    all_cc_cols = [
        c for c in df_exec.columns if c.startswith("us_cc_") or c.startswith("jp_cc_")
    ]
    jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
    jp_close_sig_cols = [c for c in df_exec.columns if c.startswith("jp_close_sig_")]
    jp_open_trade_cols = [c for c in df_exec.columns if c.startswith("jp_open_trade_")]

    all_returns = df_exec[all_cc_cols].values
    date_index = df_exec.index.values
    jp_oc = df_exec[jp_oc_cols].values if jp_oc_cols else None
    jp_close_sig = df_exec[jp_close_sig_cols].values if jp_close_sig_cols else None
    jp_open_trade = df_exec[jp_open_trade_cols].values if jp_open_trade_cols else None

    n_u = N_US_ASSETS
    n_j = N_JP_ASSETS

    c_full = signals.compute_baseline_correlation(
        all_returns, date_index, config.ewma_half_life,
    )
    v0_static = signals.build_v3_static(n_u, n_j, config.include_v4_prior)
    base_vectors = signals.build_base_vectors(n_u, n_j)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    start_idx = max(
        df_exec.index.searchsorted(pd.to_datetime(start_date)),
        config.corr_window,
    )

    gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    jp_gap = df_exec[gap_cols].values if len(gap_cols) == n_j else None

    results = []
    dispersion_history = []

    # Pre-fill dispersion history
    def _compute_dispersion_at(index: int) -> float:
        sig_result_hist = signals.compute_signal(
            all_returns, index, n_u, config.corr_window, c_full,
            v0_static, v1, v2, config.k, config.lambda_reg,
            config.lambda_lw, config.lw_target, config.ewma_half_life,
            v3_dynamic=(config.v3_mode == "dynamic"),
            gap_open_coef=config.gap_open_coef,
        )
        gap_hist = (
            np.nan_to_num(jp_gap[index], nan=0.0) if jp_gap is not None
            else np.zeros(n_j)
        )
        if config.signal_mode == "gap_residual":
            sig_result_hist["signal"] = (
                sig_result_hist["r_hat_jp_cc"] - config.gap_open_coef * gap_hist
            )
        signal_hist = np.asarray(sig_result_hist["signal"], dtype=float)
        return signals.compute_dispersion_indicator(
            signal_hist, config.q, n_j, config.dispersion_metric,
        )

    history_start = max(0, start_idx - 60)
    for hist_i in range(history_start, start_idx):
        dispersion_history.append(_compute_dispersion_at(hist_i))

    for i in range(start_idx, len(df_exec)):
        t_trade = df_exec.index[i]

        sig_result = signals.compute_signal(
            all_returns, i, n_u, config.corr_window, c_full,
            v0_static, v1, v2, config.k, config.lambda_reg,
            config.lambda_lw, config.lw_target, config.ewma_half_life,
            v3_dynamic=(config.v3_mode == "dynamic"),
            gap_open_coef=config.gap_open_coef,
        )

        gap_t1 = (
            np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None
            else np.zeros(n_j)
        )

        if config.signal_mode == "gap_residual":
            sig_result["signal"] = (
                sig_result["r_hat_jp_cc"] - config.gap_open_coef * gap_t1
            )

        signal = np.asarray(sig_result["signal"], dtype=float)
        sigma_s = float(sig_result["sigma_s"])

        dispersion_ind = signals.compute_dispersion_indicator(
            signal, config.q, n_j, config.dispersion_metric,
        )

        # Build weights (NO scaling applied)
        weights = signals.build_weights(signal, config.q, n_j, config.weight_mode)

        # Compute UNSCALED return
        r_oc_t1 = (
            np.nan_to_num(jp_oc[i], nan=0.0, copy=True) if jp_oc is not None
            else np.zeros(n_j)
        )
        unscaled_return = float(np.sum(weights * r_oc_t1))

        dispersion_history.append(dispersion_ind)

        results.append({
            "trade_date": t_trade,
            "unscaled_return": unscaled_return,
            "dispersion_indicator": dispersion_ind,
            "sigma_s": sigma_s,
            "signal_std": float(np.std(signal)),
        })

    return pd.DataFrame(results).set_index("trade_date")


# ============================================================
# Apply scaling strategies post-hoc
# ============================================================

def apply_scaling_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all scaling strategies and compute scaled returns."""
    n = len(df)
    disp = df["dispersion_indicator"].values
    unscaled = df["unscaled_return"].values

    # Build arrays for each strategy
    strategies = {
        "A_current_3stage": np.zeros(n),
        "B_5stage": np.zeros(n),
        "C_7stage": np.zeros(n),
        "D_hybrid_kelly": np.zeros(n),
        "E_continuous_sigmoid": np.zeros(n),
        "F_no_filter": np.zeros(n),
    }

    scale_arrays = {k: np.zeros(n) for k in strategies}
    disp_history = []
    ret_history = []

    for i in range(n):
        hist_arr = np.array(disp_history)
        ret_arr = np.array(ret_history)

        s_a = scale_current_3stage(disp[i], hist_arr)
        s_b = scale_5stage(disp[i], hist_arr)
        s_c = scale_7stage(disp[i], hist_arr)
        s_d = scale_hybrid_kelly(disp[i], hist_arr, ret_arr, kelly_fraction=0.5)
        s_e = scale_continuous_sigmoid(disp[i], hist_arr)
        s_f = scale_no_filter(disp[i], hist_arr)

        scale_arrays["A_current_3stage"][i] = s_a
        scale_arrays["B_5stage"][i] = s_b
        scale_arrays["C_7stage"][i] = s_c
        scale_arrays["D_hybrid_kelly"][i] = s_d
        scale_arrays["E_continuous_sigmoid"][i] = s_e
        scale_arrays["F_no_filter"][i] = s_f

        strategies["A_current_3stage"][i] = unscaled[i] * s_a
        strategies["B_5stage"][i] = unscaled[i] * s_b
        strategies["C_7stage"][i] = unscaled[i] * s_c
        strategies["D_hybrid_kelly"][i] = unscaled[i] * s_d
        strategies["E_continuous_sigmoid"][i] = unscaled[i] * s_e
        strategies["F_no_filter"][i] = unscaled[i] * s_f

        disp_history.append(disp[i])
        # For Kelly: track the unscaled returns
        ret_history.append(unscaled[i])

    result_df = df.copy()
    for name, returns in strategies.items():
        result_df[f"ret_{name}"] = returns
        result_df[f"scale_{name}"] = scale_arrays[name]

    return result_df


# ============================================================
# Performance evaluation
# ============================================================

def evaluate_all(
    df: pd.DataFrame,
    label: str = "Full Period",
) -> pd.DataFrame:
    """Evaluate all strategies and return a comparison table."""
    strategy_cols = [c for c in df.columns if c.startswith("ret_")]
    rows = []
    for col in strategy_cols:
        name = col.replace("ret_", "")
        metrics = calculate_metrics(df[col])
        scale_col = f"scale_{name}"
        avg_scale = df[scale_col].mean() if scale_col in df.columns else 1.0
        active_pct = (df[scale_col] > 0).mean() if scale_col in df.columns else 1.0

        rows.append({
            "Strategy": name,
            "Period": label,
            "AR (%)": metrics.get("AR", 0) * 100,
            "Risk (%)": metrics.get("RISK", 0) * 100,
            "R/R": metrics.get("R/R", 0),
            "Sharpe": metrics.get("Sharpe", 0),
            "MDD (%)": metrics.get("MDD", 0) * 100,
            "Total Return (%)": metrics.get("Total Return", 0) * 100,
            "Avg Scale": avg_scale,
            "Active Days (%)": active_pct * 100,
        })
    return pd.DataFrame(rows)


def evaluate_oos(df: pd.DataFrame, split_date: str = OOS_SPLIT_DATE) -> pd.DataFrame:
    """Evaluate strategies on in-sample and out-of-sample periods."""
    train = df[df.index < split_date]
    test = df[df.index >= split_date]

    results = []
    if len(train) > 50:
        results.append(evaluate_all(train, f"IS (~{split_date})"))
    if len(test) > 50:
        results.append(evaluate_all(test, f"OOS ({split_date}~)"))
    results.append(evaluate_all(df, "Full Period"))

    return pd.concat(results, ignore_index=True)


# ============================================================
# Visualization
# ============================================================

def plot_cumulative_comparison(df: pd.DataFrame, output_path: str):
    """Plot cumulative return curves for all strategies."""
    strategy_cols = [c for c in df.columns if c.startswith("ret_")]
    labels = {
        "ret_A_current_3stage": "A: Current 3-stage",
        "ret_B_5stage": "B: 5-stage",
        "ret_C_7stage": "C: 7-stage",
        "ret_D_hybrid_kelly": "D: Hybrid Kelly",
        "ret_E_continuous_sigmoid": "E: Sigmoid",
        "ret_F_no_filter": "F: No filter",
    }
    colors = {
        "ret_A_current_3stage": "#e74c3c",
        "ret_B_5stage": "#3498db",
        "ret_C_7stage": "#2ecc71",
        "ret_D_hybrid_kelly": "#9b59b6",
        "ret_E_continuous_sigmoid": "#f39c12",
        "ret_F_no_filter": "#95a5a6",
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})

    # Cumulative returns
    ax = axes[0]
    for col in strategy_cols:
        cum = (1 + df[col]).cumprod()
        ax.plot(df.index, cum, label=labels.get(col, col), color=colors.get(col, "gray"), linewidth=1.5)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.set_title("Scaling Strategy Comparison: Cumulative Returns", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.axvline(pd.to_datetime(OOS_SPLIT_DATE), color="black", linestyle="--", alpha=0.5, label="OOS split")

    # Drawdowns for top strategies
    ax = axes[1]
    for col in ["ret_A_current_3stage", "ret_B_5stage", "ret_D_hybrid_kelly"]:
        cum = (1 + df[col]).cumprod()
        dd = cum / cum.cummax() - 1.0
        ax.fill_between(df.index, dd, 0, alpha=0.3, color=colors.get(col, "gray"), label=labels.get(col, col))
    ax.set_ylabel("Drawdown", fontsize=12)
    ax.set_title("Drawdown Comparison (Top 3)", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scale_distribution(df: pd.DataFrame, output_path: str):
    """Compare scale distributions across strategies."""
    scale_cols = [c for c in df.columns if c.startswith("scale_")]
    labels = {
        "scale_A_current_3stage": "A: Current",
        "scale_B_5stage": "B: 5-stage",
        "scale_C_7stage": "C: 7-stage",
        "scale_D_hybrid_kelly": "D: Hybrid",
        "scale_E_continuous_sigmoid": "E: Sigmoid",
        "scale_F_no_filter": "F: No filter",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for idx, col in enumerate(scale_cols):
        if idx >= len(axes):
            break
        ax = axes[idx]
        vals = df[col].values
        ax.hist(vals, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
        ax.set_title(labels.get(col, col), fontsize=11, fontweight="bold")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Count")
        ax.axvline(np.mean(vals), color="red", linestyle="--", alpha=0.7, label=f"mean={np.mean(vals):.3f}")
        ax.legend(fontsize=8)

    plt.suptitle("Scale Factor Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rolling_comparison(df: pd.DataFrame, output_path: str, window: int = 120):
    """Rolling Sharpe ratio comparison."""
    strategy_cols = ["ret_A_current_3stage", "ret_B_5stage", "ret_D_hybrid_kelly"]
    labels = {"ret_A_current_3stage": "A: Current", "ret_B_5stage": "B: 5-stage", "ret_D_hybrid_kelly": "D: Hybrid"}
    colors = {"ret_A_current_3stage": "#e74c3c", "ret_B_5stage": "#3498db", "ret_D_hybrid_kelly": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(16, 6))
    for col in strategy_cols:
        rolling_mean = df[col].rolling(window).mean()
        rolling_std = df[col].rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(245)).dropna()
        ax.plot(rolling_sharpe.index, rolling_sharpe, label=labels[col], color=colors[col], linewidth=1.2, alpha=0.8)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(pd.to_datetime(OOS_SPLIT_DATE), color="black", linestyle="--", alpha=0.5)
    ax.set_title(f"Rolling {window}-day Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SCALING STRATEGY COMPARISON BACKTEST")
    print("=" * 70)

    # Step 1: Load data & run unscaled backtest
    print("\n[1] Loading market data...")
    data = download_data()
    df_exec = preprocess_data(data)
    print(f"  Preprocessed: {len(df_exec)} trading days")

    config = StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=False,  # We'll apply scaling post-hoc
        dispersion_metric=STRATEGY_DEFAULTS.get("dispersion_metric", "long_short_mean_gap"),
        v3_mode=STRATEGY_DEFAULTS["v3_mode"],
        ewma_half_life=STRATEGY_DEFAULTS["ewma_half_life"],
        lambda_lw=STRATEGY_DEFAULTS["lambda_lw"],
        lw_target=STRATEGY_DEFAULTS["lw_target"],
        corr_window=STRATEGY_DEFAULTS["corr_window"],
        include_v4_prior=STRATEGY_DEFAULTS["include_v4_prior"],
        signal_mode=STRATEGY_DEFAULTS["signal_mode"],
        gap_open_coef=STRATEGY_DEFAULTS["gap_open_coef"],
        gamma=STRATEGY_DEFAULTS.get("gamma", 0.5),
    )

    print("\n[2] Running unscaled backtest (this may take a minute)...")
    df_raw = run_unscaled_backtest(df_exec, config, start_date="2015-01-01")
    print(f"  Backtest output: {len(df_raw)} days")
    print(f"  Mean unscaled return: {df_raw['unscaled_return'].mean():.6f}")
    print(f"  Win rate (unscaled):  {(df_raw['unscaled_return'] > 0).mean():.4f}")

    # Step 2: Apply scaling strategies
    print("\n[3] Applying scaling strategies post-hoc...")
    df_scaled = apply_scaling_strategies(df_raw)

    # Step 3: Evaluate performance
    print("\n[4] Evaluating performance...")
    eval_table = evaluate_oos(df_scaled, OOS_SPLIT_DATE)

    # Print results
    for period in eval_table["Period"].unique():
        subset = eval_table[eval_table["Period"] == period]
        print(f"\n{'=' * 70}")
        print(f"  {period}")
        print(f"{'=' * 70}")
        print(
            subset[["Strategy", "AR (%)", "Risk (%)", "R/R", "Sharpe", "MDD (%)", "Avg Scale", "Active Days (%)"]].to_string(
                index=False, float_format=lambda x: f"{x:.2f}"
            )
        )

    # Step 4: Determine the winner
    oos_data = eval_table[eval_table["Period"].str.startswith("OOS")]
    if len(oos_data) > 0:
        print(f"\n{'=' * 70}")
        print("  OOS Winner Analysis")
        print(f"{'=' * 70}")
        best_rr = oos_data.loc[oos_data["R/R"].idxmax()]
        best_sharpe = oos_data.loc[oos_data["Sharpe"].idxmax()]
        best_ar = oos_data.loc[oos_data["AR (%)"].idxmax()]
        print(f"  Best R/R:    {best_rr['Strategy']} ({best_rr['R/R']:.2f})")
        print(f"  Best Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe']:.4f})")
        print(f"  Best AR:     {best_ar['Strategy']} ({best_ar['AR (%)']:.2f}%)")

    # Step 5: Visualizations
    print("\n[5] Generating charts...")
    plot_cumulative_comparison(df_scaled, os.path.join(OUTPUT_DIR, "comparison_cumulative.png"))
    plot_scale_distribution(df_scaled, os.path.join(OUTPUT_DIR, "comparison_scale_dist.png"))
    plot_rolling_comparison(df_scaled, os.path.join(OUTPUT_DIR, "comparison_rolling_sharpe.png"))

    # Step 6: Save detailed results
    eval_table.to_csv(os.path.join(OUTPUT_DIR, "comparison_metrics.csv"), index=False)
    df_scaled.to_csv(os.path.join(OUTPUT_DIR, "comparison_daily.csv"))
    print(f"\n  All results saved to: {OUTPUT_DIR}")

    print(f"\n{'=' * 70}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
