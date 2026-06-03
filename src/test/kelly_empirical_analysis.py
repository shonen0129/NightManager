"""Kelly Criterion Empirical Analysis

Analyze the relationship between signal dispersion metrics and win rate
from backtest results, to evaluate feasibility of Kelly-based position sizing.

Outputs:
  - Bin-wise win rate / payoff / Kelly fraction tables
  - Rolling temporal stability analysis
  - Out-of-sample validation
  - All results saved to results/kelly_analysis/
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Configuration ----------
RESULTS_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "results",
    "production",
    "bt_manual_refresh",
    "daily_results.csv",
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "results",
    "kelly_analysis",
)

N_BINS = 10
ROLLING_WINDOW = 120  # days for rolling win-rate
OOS_SPLIT_DATE = "2020-01-01"


# ---------- Helpers ----------
def kelly_binary(p: float, b: float) -> float:
    """Binary Kelly: f* = (p*b - q) / b."""
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (p * b - q) / b
    return max(0.0, f)


def kelly_continuous(mu: float, sigma2: float) -> float:
    """Continuous Kelly: f* = mu / sigma^2."""
    if sigma2 <= 1e-16:
        return 0.0
    return mu / sigma2


# ---------- Main Analysis ----------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, parse_dates=["trade_date"])
    df = df.set_index("trade_date").sort_index()
    # Filter out days where scale=0 (no trade) for cleaner analysis
    # But keep them in a separate column for reference
    df["traded"] = df["scale"] > 0
    df["win"] = df["daily_return"] > 0
    return df


def analyze_bins(
    df: pd.DataFrame,
    metric_col: str,
    label: str,
) -> pd.DataFrame:
    """Bin-wise analysis of a metric vs win rate / Kelly."""
    active = df[df["traded"]].copy()
    if len(active) < 50:
        print(f"  Warning: only {len(active)} active days for {label}")
        return pd.DataFrame()

    active["bin"] = pd.qcut(active[metric_col], q=N_BINS, duplicates="drop")

    rows = []
    for bin_label, grp in active.groupby("bin", observed=True):
        n = len(grp)
        wins = grp[grp["daily_return"] > 0]
        losses = grp[grp["daily_return"] <= 0]
        p = len(wins) / n if n > 0 else 0
        avg_win = float(wins["daily_return"].mean()) if len(wins) > 0 else 0
        avg_loss = float(abs(losses["daily_return"].mean())) if len(losses) > 0 else 0
        b = avg_win / avg_loss if avg_loss > 0 else 0
        f_binary = kelly_binary(p, b)

        mu = float(grp["daily_return"].mean())
        sigma2 = float(grp["daily_return"].var(ddof=1))
        f_cont = kelly_continuous(mu, sigma2)

        rows.append(
            {
                "bin": str(bin_label),
                "n": n,
                "win_rate": p,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "payoff_ratio": b,
                "kelly_binary": f_binary,
                "mean_ret": mu,
                "std_ret": float(np.sqrt(sigma2)) if sigma2 > 0 else 0,
                "kelly_continuous": f_cont,
                "metric_median": float(grp[metric_col].median()),
            }
        )

    return pd.DataFrame(rows)


def analyze_logistic_fit(
    df: pd.DataFrame,
    metric_col: str,
    label: str,
) -> dict:
    """Fit logistic regression: P(win) = σ(a + b * metric)."""
    active = df[df["traded"]].copy()
    if len(active) < 50:
        return {}

    x = active[metric_col].values.astype(float)
    y = active["win"].values.astype(float)

    # Standardize x
    x_mean, x_std = np.mean(x), np.std(x)
    if x_std < 1e-12:
        return {}
    x_norm = (x - x_mean) / x_std

    # Simple logistic via scipy
    from scipy.optimize import minimize

    def neg_log_lik(params):
        a, b = params
        z = a + b * x_norm
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(neg_log_lik, [0.0, 0.0], method="Nelder-Mead")
    a, b = res.x

    # Spearman rank correlation as a simpler metric
    rho, pval = stats.spearmanr(x, y)

    return {
        "intercept": a,
        "slope_normalized": b,
        "x_mean": x_mean,
        "x_std": x_std,
        "spearman_rho": rho,
        "spearman_pval": pval,
    }


def rolling_analysis(
    df: pd.DataFrame,
    metric_col: str,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Rolling window analysis of conditional win rate."""
    active = df[df["traded"]].copy().reset_index()
    if len(active) < window:
        return pd.DataFrame()

    # Split metric into high/low by rolling median
    rows = []
    for end in range(window, len(active)):
        start = end - window
        chunk = active.iloc[start:end]
        med = chunk[metric_col].median()

        high = chunk[chunk[metric_col] >= med]
        low = chunk[chunk[metric_col] < med]

        wr_high = high["win"].mean() if len(high) > 0 else np.nan
        wr_low = low["win"].mean() if len(low) > 0 else np.nan
        ret_high = high["daily_return"].mean() if len(high) > 0 else np.nan
        ret_low = low["daily_return"].mean() if len(low) > 0 else np.nan

        rows.append(
            {
                "trade_date": active.iloc[end]["trade_date"],
                "wr_high": wr_high,
                "wr_low": wr_low,
                "wr_spread": wr_high - wr_low if not np.isnan(wr_high) else np.nan,
                "ret_high": ret_high,
                "ret_low": ret_low,
                "ret_spread": ret_high - ret_low
                if not np.isnan(ret_high)
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def oos_validation(
    df: pd.DataFrame,
    metric_col: str,
    split_date: str = OOS_SPLIT_DATE,
) -> dict:
    """Train Kelly model on pre-split data, evaluate on post-split."""
    active = df[df["traded"]].copy()

    train = active[active.index < split_date]
    test = active[active.index >= split_date]

    if len(train) < 100 or len(test) < 100:
        return {"error": "insufficient data for OOS split"}

    # Learn bin-wise Kelly from training set
    train["bin"] = pd.qcut(train[metric_col], q=5, duplicates="drop")
    bin_edges = train.groupby("bin", observed=True)[metric_col].agg(["min", "max"])

    # Map test data to training bins
    trained_kelly = {}
    for bin_label, grp in train.groupby("bin", observed=True):
        p = grp["win"].mean()
        wins = grp[grp["daily_return"] > 0]
        losses = grp[grp["daily_return"] <= 0]
        avg_win = wins["daily_return"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["daily_return"].mean()) if len(losses) > 0 else 0
        b = avg_win / avg_loss if avg_loss > 0 else 0

        trained_kelly[bin_label] = {
            "p": p,
            "b": b,
            "f_binary": kelly_binary(p, b),
            "low": grp[metric_col].min(),
            "high": grp[metric_col].max(),
        }

    # Evaluate on test set
    test_results = []
    for _, row in test.iterrows():
        val = row[metric_col]
        # Find best matching bin
        matched = None
        for bl, info in trained_kelly.items():
            if info["low"] <= val <= info["high"]:
                matched = bl
                break
        if matched is None:
            # Use closest edge
            if val < min(v["low"] for v in trained_kelly.values()):
                matched = min(trained_kelly, key=lambda k: trained_kelly[k]["low"])
            else:
                matched = max(trained_kelly, key=lambda k: trained_kelly[k]["high"])

        kelly_f = trained_kelly[matched]["f_binary"]
        test_results.append(
            {
                "date": row.name if hasattr(row, "name") else None,
                "return": row["daily_return"],
                "kelly_f": kelly_f,
                "current_scale": row["scale"],
            }
        )

    test_df = pd.DataFrame(test_results)

    # Compare: Current dispersion filter vs Kelly sizing
    ret_current = (test_df["return"] * test_df["current_scale"]).sum()
    ret_kelly = (test_df["return"] * test_df["kelly_f"]).sum()
    ret_half_kelly = (test_df["return"] * test_df["kelly_f"] * 0.5).sum()

    n_test = len(test_df)
    days_per_year = 245

    ar_current = ret_current * days_per_year / n_test
    ar_kelly = ret_kelly * days_per_year / n_test
    ar_half_kelly = ret_half_kelly * days_per_year / n_test

    risk_current = (test_df["return"] * test_df["current_scale"]).std() * np.sqrt(
        days_per_year
    )
    risk_kelly = (test_df["return"] * test_df["kelly_f"]).std() * np.sqrt(
        days_per_year
    )
    risk_half_kelly = (
        (test_df["return"] * test_df["kelly_f"] * 0.5).std() * np.sqrt(days_per_year)
    )

    return {
        "train_period": f"start ~ {split_date}",
        "test_period": f"{split_date} ~ end",
        "train_n": len(train),
        "test_n": n_test,
        "trained_bins": {
            str(k): {
                "p": v["p"],
                "payoff": v["b"],
                "kelly_f": v["f_binary"],
            }
            for k, v in trained_kelly.items()
        },
        "oos_AR_current_filter": ar_current,
        "oos_AR_kelly": ar_kelly,
        "oos_AR_half_kelly": ar_half_kelly,
        "oos_RISK_current_filter": risk_current,
        "oos_RISK_kelly": risk_kelly,
        "oos_RISK_half_kelly": risk_half_kelly,
        "oos_RR_current": ar_current / risk_current if risk_current > 0 else np.nan,
        "oos_RR_kelly": ar_kelly / risk_kelly if risk_kelly > 0 else np.nan,
        "oos_RR_half_kelly": ar_half_kelly / risk_half_kelly
        if risk_half_kelly > 0
        else np.nan,
    }


def plot_bin_analysis(
    bin_df: pd.DataFrame,
    metric_label: str,
    output_path: str,
):
    """Generate 2x2 plot of bin analysis results."""
    if len(bin_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Kelly Analysis: {metric_label}", fontsize=14, fontweight="bold")

    x = range(len(bin_df))
    xlabels = [f"Q{i+1}" for i in x]

    # Win rate
    ax = axes[0, 0]
    colors = ["#2ecc71" if v > 0.5 else "#e74c3c" for v in bin_df["win_rate"]]
    ax.bar(x, bin_df["win_rate"], color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Win Rate by Quantile")
    ax.set_ylabel("Win Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(0.3, 0.9)

    # Kelly fraction (binary)
    ax = axes[0, 1]
    ax.bar(x, bin_df["kelly_binary"], color="#3498db", alpha=0.8, edgecolor="white")
    ax.set_title("Kelly Fraction (Binary) by Quantile")
    ax.set_ylabel("f*")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)

    # Mean return
    ax = axes[1, 0]
    colors_ret = ["#2ecc71" if v > 0 else "#e74c3c" for v in bin_df["mean_ret"]]
    ax.bar(x, bin_df["mean_ret"] * 100, color=colors_ret, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Mean Daily Return (%) by Quantile")
    ax.set_ylabel("Mean Return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)

    # Payoff ratio
    ax = axes[1, 1]
    ax.bar(
        x, bin_df["payoff_ratio"], color="#9b59b6", alpha=0.8, edgecolor="white"
    )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Payoff Ratio (Avg Win / Avg Loss) by Quantile")
    ax.set_ylabel("Payoff Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rolling_stability(
    rolling_df: pd.DataFrame,
    metric_label: str,
    output_path: str,
):
    """Plot rolling win-rate spread (high vs low dispersion)."""
    if len(rolling_df) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Temporal Stability: {metric_label} (window={ROLLING_WINDOW}d)",
        fontsize=14,
        fontweight="bold",
    )

    dates = pd.to_datetime(rolling_df["trade_date"])

    # Win rate: high vs low
    ax = axes[0]
    ax.plot(dates, rolling_df["wr_high"], label="High dispersion", color="#2ecc71", alpha=0.7)
    ax.plot(dates, rolling_df["wr_low"], label="Low dispersion", color="#e74c3c", alpha=0.7)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.fill_between(
        dates,
        rolling_df["wr_high"],
        rolling_df["wr_low"],
        alpha=0.15,
        color="blue",
    )
    ax.set_ylabel("Rolling Win Rate")
    ax.legend(loc="upper left")
    ax.set_title("Rolling Win Rate: High vs Low Dispersion")

    # Win rate spread
    ax = axes[1]
    spread = rolling_df["wr_spread"]
    ax.fill_between(
        dates,
        spread,
        0,
        where=spread > 0,
        alpha=0.4,
        color="#2ecc71",
        label="High disp wins more",
    )
    ax.fill_between(
        dates,
        spread,
        0,
        where=spread <= 0,
        alpha=0.4,
        color="#e74c3c",
        label="Low disp wins more",
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Win Rate Spread (High - Low)")
    ax.set_title("Win Rate Spread Over Time")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("KELLY CRITERION EMPIRICAL ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"  Total days: {len(df)}")
    print(f"  Active (traded) days: {df['traded'].sum()}")
    print(f"  Date range: {df.index.min()} ~ {df.index.max()}")
    print(f"  Overall win rate: {df[df['traded']]['win'].mean():.4f}")
    print(f"  Overall mean return: {df[df['traded']]['daily_return'].mean():.6f}")

    # ---- Dispersion Indicator (D_t) ----
    print("\n" + "=" * 70)
    print("[2] Analysis: dispersion_indicator (D_t)")
    print("=" * 70)

    bin_dt = analyze_bins(df, "dispersion_indicator", "D_t")
    print("\n  Bin-wise results:")
    print(
        bin_dt[
            [
                "bin",
                "n",
                "win_rate",
                "payoff_ratio",
                "kelly_binary",
                "mean_ret",
                "kelly_continuous",
            ]
        ].to_string(index=False)
    )

    logistic_dt = analyze_logistic_fit(df, "dispersion_indicator", "D_t")
    print(f"\n  Logistic fit (normalized):")
    print(f"    Spearman ρ(D_t, win): {logistic_dt.get('spearman_rho', 'N/A'):.4f}")
    print(f"    p-value:              {logistic_dt.get('spearman_pval', 'N/A'):.4e}")
    print(f"    Slope (normalized):   {logistic_dt.get('slope_normalized', 'N/A'):.4f}")

    plot_bin_analysis(bin_dt, "dispersion_indicator (D_t)", os.path.join(OUTPUT_DIR, "bins_dt.png"))

    # ---- Sigma_s ----
    print("\n" + "=" * 70)
    print("[3] Analysis: sigma_s")
    print("=" * 70)

    bin_sigma = analyze_bins(df, "sigma_s", "sigma_s")
    print("\n  Bin-wise results:")
    print(
        bin_sigma[
            [
                "bin",
                "n",
                "win_rate",
                "payoff_ratio",
                "kelly_binary",
                "mean_ret",
                "kelly_continuous",
            ]
        ].to_string(index=False)
    )

    logistic_sigma = analyze_logistic_fit(df, "sigma_s", "sigma_s")
    print(f"\n  Logistic fit (normalized):")
    print(f"    Spearman ρ(σ_s, win): {logistic_sigma.get('spearman_rho', 'N/A'):.4f}")
    print(f"    p-value:              {logistic_sigma.get('spearman_pval', 'N/A'):.4e}")

    plot_bin_analysis(bin_sigma, "sigma_s", os.path.join(OUTPUT_DIR, "bins_sigma_s.png"))

    # ---- signal_std ----
    print("\n" + "=" * 70)
    print("[4] Analysis: signal_std")
    print("=" * 70)

    bin_sigstd = analyze_bins(df, "signal_std", "signal_std")
    print("\n  Bin-wise results:")
    print(
        bin_sigstd[
            [
                "bin",
                "n",
                "win_rate",
                "payoff_ratio",
                "kelly_binary",
                "mean_ret",
                "kelly_continuous",
            ]
        ].to_string(index=False)
    )

    logistic_sigstd = analyze_logistic_fit(df, "signal_std", "signal_std")
    print(f"\n  Logistic fit (normalized):")
    print(
        f"    Spearman ρ(signal_std, win): {logistic_sigstd.get('spearman_rho', 'N/A'):.4f}"
    )
    print(f"    p-value:                    {logistic_sigstd.get('spearman_pval', 'N/A'):.4e}")

    plot_bin_analysis(bin_sigstd, "signal_std", os.path.join(OUTPUT_DIR, "bins_signal_std.png"))

    # ---- Rolling Stability ----
    print("\n" + "=" * 70)
    print("[5] Temporal Stability (Rolling Window)")
    print("=" * 70)

    for metric in ["dispersion_indicator", "sigma_s"]:
        print(f"\n  Rolling analysis: {metric}")
        rolling = rolling_analysis(df, metric, ROLLING_WINDOW)
        if len(rolling) > 0:
            spread_mean = rolling["wr_spread"].mean()
            spread_std = rolling["wr_spread"].std()
            pct_positive = (rolling["wr_spread"] > 0).mean()
            print(f"    Mean WR spread (high-low): {spread_mean:.4f}")
            print(f"    Std WR spread:             {spread_std:.4f}")
            print(f"    % of time high > low:      {pct_positive:.1%}")
            plot_rolling_stability(
                rolling,
                metric,
                os.path.join(OUTPUT_DIR, f"rolling_{metric}.png"),
            )

    # ---- Out-of-Sample Validation ----
    print("\n" + "=" * 70)
    print("[6] Out-of-Sample Validation")
    print("=" * 70)

    for metric in ["dispersion_indicator", "sigma_s", "signal_std"]:
        print(f"\n  OOS validation: {metric} (split={OOS_SPLIT_DATE})")
        oos = oos_validation(df, metric, OOS_SPLIT_DATE)
        if "error" in oos:
            print(f"    Error: {oos['error']}")
            continue

        print(f"    Train: {oos['train_n']} days, Test: {oos['test_n']} days")
        print(f"    OOS AR (current filter): {oos['oos_AR_current_filter']:.2%}")
        print(f"    OOS AR (full Kelly):     {oos['oos_AR_kelly']:.2%}")
        print(f"    OOS AR (half Kelly):     {oos['oos_AR_half_kelly']:.2%}")
        print(f"    OOS RISK (current):      {oos['oos_RISK_current_filter']:.2%}")
        print(f"    OOS RISK (kelly):        {oos['oos_RISK_kelly']:.2%}")
        print(f"    OOS R/R (current):       {oos['oos_RR_current']:.2f}")
        print(f"    OOS R/R (kelly):         {oos['oos_RR_kelly']:.2f}")
        print(f"    OOS R/R (half kelly):    {oos['oos_RR_half_kelly']:.2f}")

        print(f"    Trained bins:")
        for bname, binfo in oos["trained_bins"].items():
            print(
                f"      {bname}: p={binfo['p']:.3f}, payoff={binfo['payoff']:.3f}, f*={binfo['kelly_f']:.4f}"
            )

    # ---- Save all tables ----
    bin_dt.to_csv(os.path.join(OUTPUT_DIR, "bins_dt.csv"), index=False)
    bin_sigma.to_csv(os.path.join(OUTPUT_DIR, "bins_sigma_s.csv"), index=False)
    bin_sigstd.to_csv(os.path.join(OUTPUT_DIR, "bins_signal_std.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
