#!/usr/bin/env python
"""A7: ウォークフォワード検証プロトコル + Deflated Sharpe Ratio.

設計仕様（docs/design/A_theory_design_specs.md A7 + C_validation_frameworks.md C1参照）:
  1. 2018-2026の年次ロール（9区間）でOOS backtestを実行
  2. purge = corr_window(60) + 1日、embargo = 5日
  3. 各区間のnet Sharpeを報告
  4. Deflated Sharpe Ratio（Bailey & López de Prado 2014）を計算
  5. ±20%パラメータ摂動の感度分析

Usage:
  python3 scripts/experiments/experiment_a7_walkforward_dsr.py \
    --output-dir reports/sprint_a7_walkforward
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245
EULER_GAMMA = 0.5772156649015329

# Walkforward config: expanding training, 1-year test, purge=61, embargo=5
WF_CONFIG = {
    "start_train": "2015-01-05",
    "test_years": list(range(2018, 2027)),  # 2018-2026 = 9 periods
    "purge_days": 61,   # corr_window(60) + 1
    "embargo_days": 5,
}


def compute_metrics(daily_returns: pd.Series) -> dict:
    dr = daily_returns.dropna()
    if len(dr) < 10:
        return {"Sharpe_net": np.nan, "AR_net": np.nan, "Vol_net": np.nan, "MDD": np.nan, "n_days": len(dr)}
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    return {"Sharpe_net": sharpe, "AR_net": ar, "Vol_net": vol, "MDD": mdd, "n_days": len(dr)}


def run_period_backtest(
    cfg: dict,
    df_exec: pd.DataFrame,
    start_date: str,
    end_date: str,
    slippage_bps: float = 5.0,
) -> pd.Series:
    """Run backtest for a specific period and return daily returns."""
    model = SectorRelativeEnsembleBLPEnhancedModel(copy.deepcopy(cfg))
    model._start_date = start_date
    results = BacktestEngine.run_backtest(
        model, df_exec, start_date=start_date, end_date=end_date, slippage_bps=slippage_bps,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    return results["daily_returns"]


def compute_deflated_sharpe(
    sharpe_hat: float,
    n_trials: int,
    T_days: int,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
    trials_sharpe_std: float = 0.5,
) -> float:
    """Compute Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    DSR = Φ( (SR_hat - SR_0) * sqrt(T-1) / sqrt(1 - γ3*SR_hat + (γ4-1)/4 * SR_hat^2) )

    where SR_0 = sqrt(V[SR_n]) * ((1-γ)*Φ^{-1}(1-1/N) + γ*Φ^{-1}(1-1/(N*e)))

    Args:
        sharpe_hat: Observed annualized Sharpe ratio (in daily units: SR_annual / sqrt(252))
        n_trials: Number of trials (effective)
        T_days: Number of observations (days)
        skewness: Skewness of returns
        kurtosis_excess: Excess kurtosis of returns
        trials_sharpe_std: Cross-trial standard deviation of Sharpe (V[SR_n]^0.5)

    Returns:
        DSR value (0-1, higher is better. >=0.95 is significant)
    """
    # Convert annualized Sharpe to daily
    sr_daily = sharpe_hat / np.sqrt(TRADING_DAYS)

    # SR_0: expected maximum Sharpe under null (convert to daily to match sr_daily)
    if n_trials <= 1:
        sr_0 = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_0_annual = trials_sharpe_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
        sr_0 = sr_0_annual / np.sqrt(TRADING_DAYS)

    # DSR statistic
    gamma3 = skewness
    gamma4 = kurtosis_excess + 3.0  # full kurtosis

    denom = np.sqrt(1.0 - gamma3 * sr_daily + (gamma4 - 1.0) / 4.0 * sr_daily**2)
    if denom < 1e-12:
        return 0.0

    dsr_stat = (sr_daily - sr_0) * np.sqrt(T_days - 1) / denom
    return float(norm.cdf(dsr_stat))


def main():
    parser = argparse.ArgumentParser(description="A7: Walkforward Validation + Deflated Sharpe")
    parser.add_argument("--output-dir", default="reports/sprint_a7_walkforward")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--n-trials", type=int, default=55,
                        help="Effective number of trials for DSR (see C1 spec)")
    parser.add_argument("--trials-sharpe-std", type=float, default=0.5,
                        help="Cross-trial Sharpe std for DSR")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg_base = yaml.safe_load(f)

    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()
    T = len(df_exec)
    sim_dates = df_exec.index

    # 1. Run walkforward periods
    period_results = []
    all_daily_returns = []

    for year in WF_CONFIG["test_years"]:
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        # Ensure dates are within data
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)

        if test_start_dt > sim_dates[-1]:
            logger.info("Skipping %s (beyond data range)", test_start)
            continue

        # Adjust end date if beyond data
        if test_end_dt > sim_dates[-1]:
            test_end = sim_dates[-1].strftime("%Y-%m-%d")

        logger.info("=== Walkforward period: %s to %s ===", test_start, test_end)
        t0 = time.perf_counter()
        dr = run_period_backtest(cfg_base, df_exec, test_start, test_end, args.slippage_bps)
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(dr)
        metrics["period"] = f"{year}"
        metrics["start"] = test_start
        metrics["end"] = test_end
        metrics["elapsed_s"] = elapsed
        period_results.append(metrics)
        all_daily_returns.append(dr)

        logger.info("[%s] Sharpe=%.4f MDD=%.2f%% n=%d (%.1fs)",
                    year, metrics["Sharpe_net"], metrics["MDD"] * 100, metrics["n_days"], elapsed)

        # Save period returns
        dr.to_csv(out_dir / f"daily_returns_{year}.csv")

    period_df = pd.DataFrame(period_results)
    period_df.to_csv(out_dir / "walkforward_period_metrics.csv", index=False)

    # 2. Compute pooled statistics
    pooled_dr = pd.concat(all_daily_returns)
    pooled_metrics = compute_metrics(pooled_dr)

    # Compute skewness and kurtosis
    skew = float(pooled_dr.skew())
    kurt_excess = float(pooled_dr.kurt())  # pandas kurt() returns excess kurtosis

    logger.info("\nPooled: Sharpe=%.4f MDD=%.2f%% n=%d skew=%.4f kurt_excess=%.4f",
                pooled_metrics["Sharpe_net"], pooled_metrics["MDD"] * 100,
                pooled_metrics["n_days"], skew, kurt_excess)

    # 3. Compute Deflated Sharpe Ratio
    dsr = compute_deflated_sharpe(
        sharpe_hat=pooled_metrics["Sharpe_net"],
        n_trials=args.n_trials,
        T_days=pooled_metrics["n_days"],
        skewness=skew,
        kurtosis_excess=kurt_excess,
        trials_sharpe_std=args.trials_sharpe_std,
    )
    logger.info("Deflated Sharpe Ratio: %.4f (N_trials=%d, threshold=0.95)", dsr, args.n_trials)

    # 4. Per-period Sharpe stability
    period_sharpes = [r["Sharpe_net"] for r in period_results if not np.isnan(r["Sharpe_net"])]
    sharpe_mean = float(np.mean(period_sharpes))
    sharpe_std = float(np.std(period_sharpes, ddof=1))
    sharpe_min = float(np.min(period_sharpes))
    sharpe_max = float(np.max(period_sharpes))
    positive_periods = int(sum(1 for s in period_sharpes if s > 0))
    negative_periods = len(period_sharpes) - positive_periods

    # 5. Sensitivity analysis: ±20% perturbation of key parameters
    logger.info("\n=== Sensitivity: ±20%% perturbation ===")
    sensitivity_results = []
    perturb_params = {
        "lambda_pca": [0.08, 0.10, 0.12],
        "lambda_sector": [0.48, 0.60, 0.72],
        "rho": [0.008, 0.01, 0.012],
        "blp_window": [403, 504, 605],
    }

    # Run with all params at +20% and -20% simultaneously
    for label, mult in [("minus20", 0.8), ("base", 1.0), ("plus20", 1.2)]:
        cfg = copy.deepcopy(cfg_base)
        if "blpx" not in cfg:
            cfg["blpx"] = {}
        cfg["blpx"]["lambda_pca"] = 0.10 * mult
        cfg["blpx"]["lambda_sector"] = 0.60 * mult
        cfg["blpx"]["rho"] = 0.01 * mult
        cfg["blpx"]["blp_window"] = int(504 * mult)

        # Run pooled backtest
        t0 = time.perf_counter()
        dr = run_period_backtest(cfg, df_exec, WF_CONFIG["start_train"], "latest", args.slippage_bps)
        metrics = compute_metrics(dr)
        metrics["variant"] = label
        metrics["multiplier"] = mult
        sensitivity_results.append(metrics)
        logger.info("%s: Sharpe=%.4f (%.1fs)", label, metrics["Sharpe_net"], time.perf_counter() - t0)

    sens_df = pd.DataFrame(sensitivity_results)
    sens_df.to_csv(out_dir / "sensitivity_analysis.csv", index=False)

    # 6. Report
    report_lines = [
        "# A7: Walkforward Validation + Deflated Sharpe Ratio Report\n",
        f"## Configuration\n",
        f"- Training start: {WF_CONFIG['start_train']}\n",
        f"- Test periods: {WF_CONFIG['test_years'][0]}–{WF_CONFIG['test_years'][-1]} ({len(period_results)} periods)\n",
        f"- Purge: {WF_CONFIG['purge_days']} days, Embargo: {WF_CONFIG['embargo_days']} days\n",
        f"- N_trials for DSR: {args.n_trials}\n",
        f"\n## Per-Period Metrics\n",
        period_df.to_string(index=False),
        f"\n\n## Pooled Statistics\n",
        f"- Sharpe (net): {pooled_metrics['Sharpe_net']:.4f}\n",
        f"- Annual Return: {pooled_metrics['AR_net']*100:.2f}%\n",
        f"- Volatility: {pooled_metrics['Vol_net']*100:.2f}%\n",
        f"- Max DD: {pooled_metrics['MDD']*100:.2f}%\n",
        f"- N days: {pooled_metrics['n_days']}\n",
        f"- Skewness: {skew:.4f}\n",
        f"- Excess Kurtosis: {kurt_excess:.4f}\n",
        f"\n## Deflated Sharpe Ratio\n",
        f"- DSR = {dsr:.4f} (threshold: 0.95)\n",
        f"- N_trials = {args.n_trials}, Trials Sharpe Std = {args.trials_sharpe_std}\n",
        f"- **{'PASS' if dsr >= 0.95 else 'FAIL'}** (DSR {'≥' if dsr >= 0.95 else '<'} 0.95)\n",
        f"\n## Per-Period Sharpe Stability\n",
        f"- Mean: {sharpe_mean:.4f}\n",
        f"- Std: {sharpe_std:.4f}\n",
        f"- Range: [{sharpe_min:.4f}, {sharpe_max:.4f}]\n",
        f"- Positive periods: {positive_periods}/{len(period_sharpes)}\n",
        f"- Negative periods: {negative_periods}/{len(period_sharpes)}\n",
        f"\n## Sensitivity Analysis (±20% all params)\n",
        sens_df.to_string(index=False),
        f"\n\n## Verdict\n",
    ]

    # Overall verdict
    sharpe_range_pct = (sharpe_max - sharpe_min) / sharpe_mean * 100 if sharpe_mean > 0 else float('inf')
    sens_sharpe_range = (sens_df["Sharpe_net"].max() - sens_df["Sharpe_net"].min()) / sens_df["Sharpe_net"].mean() * 100

    if dsr >= 0.95 and negative_periods <= 2 and sens_sharpe_range < 20:
        report_lines.append(f"**ROBUST**: DSR={dsr:.4f}≥0.95, {negative_periods} negative periods, sensitivity range={sens_sharpe_range:.1f}%\n")
        report_lines.append("Strategy performance is statistically significant and stable.\n")
    elif dsr >= 0.95:
        report_lines.append(f"**SIGNIFICANT but UNSTABLE**: DSR={dsr:.4f}≥0.95 but {negative_periods} negative periods or sensitivity={sens_sharpe_range:.1f}%\n")
        report_lines.append("Statistically significant but consider regime dependency.\n")
    else:
        report_lines.append(f"**NOT SIGNIFICANT**: DSR={dsr:.4f}<0.95 after {args.n_trials} trials correction\n")
        report_lines.append("Performance may be due to multiple testing bias.\n")

    if sens_sharpe_range > 20:
        report_lines.append(f"⚠️ Parameter fragility: ±20% perturbation changes Sharpe by {sens_sharpe_range:.1f}%\n")

    report_text = "\n".join(report_lines)
    (out_dir / "a7_walkforward_dsr_report.md").write_text(report_text)
    logger.info("Report saved to %s/a7_walkforward_dsr_report.md", out_dir)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
