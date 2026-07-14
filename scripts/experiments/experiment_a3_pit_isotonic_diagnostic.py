#!/usr/bin/env python
"""A3: PITビニング isotonic診断スクリプト.

RuleD動的グロス制御の妥当性を検証する:
  1. バックテストから日次リターンを取得
  2. ローリング252日Sharpe（IR proxy）を計算（shift(1)でリーク防止）
  3. PIT percentileを計算（strictly historical）
  4. 翌日実現リターンとの関係をisotonic regressionで可視化
  5. 現行3分位（Low=0.75x, Mid/High=1.0x）が妥当かを判定

Usage:
  python3 scripts/experiments/experiment_a3_pit_isotonic_diagnostic.py \
    --output-dir reports/sprint_a3_pit_isotonic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245


def compute_rolling_sharpe(daily_returns: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling Sharpe ratio (IR proxy) with shift(1) for leak prevention.

    IR_t = mean(r[t-window:t]) / std(r[t-window:t]) * sqrt(252)
    Shifted by 1 so IR_t uses data up to t-1 only.
    """
    rolling_mean = daily_returns.rolling(window, min_periods=window // 2).mean()
    rolling_std = daily_returns.rolling(window, min_periods=window // 2).std(ddof=1)
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(TRADING_DAYS)
    return rolling_sharpe.shift(1)


def compute_pit_percentiles(ir_series: np.ndarray, window: int = 252) -> np.ndarray:
    """Compute PIT percentile of each day's IR using strictly historical window.

    p_t = rank of ir[t] within ir[t-window : t] (exclusive of t), expressed as percentile.
    Returns NaN for the first `window` entries.
    """
    T = len(ir_series)
    percentiles = np.full(T, np.nan)
    for t in range(window, T):
        hist = ir_series[t - window : t]
        valid = hist[np.isfinite(hist)]
        if len(valid) < window // 2:
            continue
        rank = np.searchsorted(np.sort(valid), ir_series[t], side="right")
        percentiles[t] = (rank / len(valid)) * 100.0
    return percentiles


def load_daily_returns_from_backtest(
    gap_dir: Path, df_exec: pd.DataFrame, cfg: dict, start_date: str
) -> pd.Series:
    """Run baseline backtest and extract daily net returns."""
    from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
    from leadlag.execution.backtester import BacktestEngine

    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    model._start_date = start_date
    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=start_date, slippage_bps=5.0,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    return results["daily_returns"]


def main():
    parser = argparse.ArgumentParser(description="A3: PIT Isotonic Diagnostic")
    parser.add_argument("--output-dir", default="reports/sprint_a3_pit_isotonic")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--window", type=int, default=252, help="PIT rolling window")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    from leadlag.data.cache import load_df_exec_from_local_cache
    from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
    from leadlag.execution.backtester import BacktestEngine

    # 1. Run baseline backtest
    logger.info("Loading df_exec and running baseline backtest...")
    df_exec = load_df_exec_from_local_cache()
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg = yaml.safe_load(f)

    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    model._start_date = args.start_date
    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=args.start_date, slippage_bps=5.0,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    daily_returns = results["daily_returns"]
    daily_returns.index = pd.to_datetime(daily_returns.index)
    logger.info("Backtest: %d daily returns", len(daily_returns))

    # 2. Compute rolling Sharpe (IR proxy)
    ir_series = compute_rolling_sharpe(daily_returns, window=args.window)
    ir_valid = ir_series.dropna()
    logger.info("IR series: %d valid entries (after %d-day warmup)",
                len(ir_valid), args.window)

    if len(ir_valid) < 100:
        logger.error("Insufficient IR data (%d rows). Need >= 100.", len(ir_valid))
        return

    # 3. Compute PIT percentiles
    ir_values = ir_series.values
    pit_pct = compute_pit_percentiles(ir_values, window=args.window)

    # 4. Align PIT percentile with next-day returns
    df = pd.DataFrame({
        "ir": ir_values,
        "pit_percentile": pit_pct,
        "daily_return": daily_returns.values,
    }, index=daily_returns.index)

    # PIT percentile at date t predicts return at date t+1
    df["next_return"] = df["daily_return"].shift(-1)
    merged = df.dropna(subset=["pit_percentile", "next_return"])
    logger.info("Merged %d aligned records", len(merged))

    if len(merged) < 100:
        logger.error("Insufficient data for isotonic regression (%d rows). Need >= 100.", len(merged))
        return

    # 5. Isotonic regression: next_return ~ pit_percentile
    x = merged["pit_percentile"].values
    y = merged["next_return"].values
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    y_iso = iso.fit_transform(x, y)

    # Check monotonicity direction
    corr, pval = spearmanr(x, y)
    logger.info("Spearman corr(pit_pct, next_return) = %.4f (p=%.4f)", corr, pval)

    # 6. Bin analysis: 10 deciles
    decile_bins = np.percentile(x, np.arange(0, 101, 10))
    bin_indices = np.digitize(x, decile_bins[1:-1])
    bin_stats = []
    for b in range(10):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_stats.append({
                "decile": b + 1,
                "n": int(mask.sum()),
                "mean_pit_pct": float(np.mean(x[mask])),
                "mean_next_return": float(np.mean(y[mask])),
                "std_next_return": float(np.std(y[mask], ddof=1)),
                "sharpe_contrib": float(np.mean(y[mask]) / max(np.std(y[mask], ddof=1), 1e-8) * np.sqrt(TRADING_DAYS)),
            })
    bin_df = pd.DataFrame(bin_stats)
    logger.info("\n%s", bin_df.to_string(index=False))

    # 7. Current 3-tertile analysis
    tertile_low = np.percentile(x, 33.3333)
    tertile_high = np.percentile(x, 66.6667)
    low_mask = x <= tertile_low
    mid_mask = (x > tertile_low) & (x < tertile_high)
    high_mask = x >= tertile_high

    tertile_stats = pd.DataFrame([
        {"bin": "Low", "n": int(low_mask.sum()), "mean_ret": float(np.mean(y[low_mask])),
         "sharpe_contrib": float(np.mean(y[low_mask]) / max(np.std(y[low_mask], ddof=1), 1e-8) * np.sqrt(TRADING_DAYS))},
        {"bin": "Mid", "n": int(mid_mask.sum()), "mean_ret": float(np.mean(y[mid_mask])),
         "sharpe_contrib": float(np.mean(y[mid_mask]) / max(np.std(y[mid_mask], ddof=1), 1e-8) * np.sqrt(TRADING_DAYS))},
        {"bin": "High", "n": int(high_mask.sum()), "mean_ret": float(np.mean(y[high_mask])),
         "sharpe_contrib": float(np.mean(y[high_mask]) / max(np.std(y[high_mask], ddof=1), 1e-8) * np.sqrt(TRADING_DAYS))},
    ])
    logger.info("\nTertile analysis (current RuleD bins):\n%s", tertile_stats.to_string(index=False))

    # 8. Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(x, y, alpha=0.15, s=8, color="steelblue", label="daily obs")
    sorted_idx = np.argsort(x)
    ax1.plot(x[sorted_idx], y_iso[sorted_idx], color="red", linewidth=2, label="isotonic fit")
    ax1.axvline(tertile_low, color="orange", linestyle="--", alpha=0.7, label="33.3% (Low/Mid)")
    ax1.axvline(tertile_high, color="green", linestyle="--", alpha=0.7, label="66.7% (Mid/High)")
    ax1.set_xlabel("PIT Percentile of Ex-Ante IR")
    ax1.set_ylabel("Next-Day Net Return")
    ax1.set_title("Isotonic Regression: Next-Day Return vs PIT IR Percentile")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(bin_df["decile"], bin_df["mean_next_return"], color="steelblue", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("PIT Percentile Decile")
    ax2.set_ylabel("Mean Next-Day Net Return")
    ax2.set_title("Decile Analysis: Mean Return by PIT Bin")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "pit_isotonic_diagnostic.png", dpi=150)
    logger.info("Saved plot to %s/pit_isotonic_diagnostic.png", out_dir)

    # 9. Save data
    merged.to_csv(out_dir / "pit_percentile_vs_return.csv", index=False)
    bin_df.to_csv(out_dir / "decile_stats.csv", index=False)
    tertile_stats.to_csv(out_dir / "tertile_stats.csv", index=False)

    # 10. Verdict
    low_sharpe = tertile_stats.loc[tertile_stats["bin"] == "Low", "sharpe_contrib"].values[0]
    mid_sharpe = tertile_stats.loc[tertile_stats["bin"] == "Mid", "sharpe_contrib"].values[0]
    high_sharpe = tertile_stats.loc[tertile_stats["bin"] == "High", "sharpe_contrib"].values[0]

    verdict_lines = []
    verdict_lines.append(f"# A3 PIT Isotonic Diagnostic Report\n")
    verdict_lines.append(f"## Data: {len(merged)} aligned observations\n")
    verdict_lines.append(f"## Spearman corr(pit_pct, next_return) = {corr:.4f} (p={pval:.4f})\n")
    verdict_lines.append(f"\n### Tertile Sharpe Contributions\n")
    verdict_lines.append(tertile_stats.to_string(index=False))
    verdict_lines.append(f"\n### Decile Analysis\n")
    verdict_lines.append(bin_df.to_string(index=False))

    if low_sharpe < mid_sharpe and low_sharpe < high_sharpe:
        verdict_lines.append(f"\n### Verdict: SUPPORTED\n")
        verdict_lines.append("Low tertile has lower Sharpe contribution → 0.75x multiplier is justified.\n")
        verdict_lines.append("Current 3-tertile RuleD is reasonable. No change needed.\n")
    else:
        verdict_lines.append(f"\n### Verdict: NEEDS REVIEW\n")
        verdict_lines.append("Low tertile does NOT have lower Sharpe contribution.\n")
        verdict_lines.append("Consider: (1) maintaining current RuleD (simpler), or (2) logistic g(p) fit.\n")
        if abs(corr) < 0.05:
            verdict_lines.append("Spearman correlation is negligible → PIT percentile has no predictive power for next-day return.\n")
            verdict_lines.append("Recommendation: Keep current RuleD unchanged (no evidence for change).\n")

    report_text = "\n".join(verdict_lines)
    (out_dir / "pit_isotonic_report.md").write_text(report_text)
    logger.info("Report saved to %s/pit_isotonic_report.md", out_dir)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
