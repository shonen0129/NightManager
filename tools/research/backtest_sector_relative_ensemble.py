#!/usr/bin/env python
"""Backtesting script for Sector Relative Ensemble (PCA-Ensemble) Model.

Loads config, runs simulation, generates metrics, diagnostics, plots, and audits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.reporting.metrics import calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sector Relative Ensemble Backtest Suite")
    parser.add_argument("--config", default="configs/production.yaml", help="Path to config file")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--end-date", default="latest", help="Backtest end date")
    parser.add_argument("--train-end-date", default="2019-12-31", help="Train period end date")
    parser.add_argument("--oos-start-date", default="2020-01-01", help="OOS period start date")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage bps (one-way)")
    parser.add_argument("--output-dir", default="results/sector_relative_ensemble/", help="Output directory")

    # Overrides for PCA-Ensemble parameters
    parser.add_argument("--normalization", default="zscore", choices=["zscore", "identity", "rank_normalize"])
    parser.add_argument("--production-signal-weight", type=float, default=0.5)
    parser.add_argument("--residual-signal-weight", type=float, default=0.5)
    parser.add_argument("--long-short-frac", type=float, default=0.3)
    parser.add_argument("--n-assets", type=int, default=17)
    parser.add_argument("--weight-mode", default="signal", choices=["signal", "equal"])

    # Flags
    parser.add_argument("--run-audit", type=str, default="true", help="Run audit checks ('true'/'false')")
    parser.add_argument("--save-signals", type=str, default="true", help="Save signals CSV")
    parser.add_argument("--save-weights", type=str, default="true", help="Save weights CSV")
    parser.add_argument("--save-plots", type=str, default="true", help="Generate and save diagnostic plots")

    return parser.parse_args()


def calculate_subperiod_metrics(
    daily_ret: pd.Series,
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> dict[str, dict]:
    """Calculate metrics for Train, OOS, and Full subperiods."""
    train_ret = daily_ret.loc[:train_end]
    oos_ret = daily_ret.loc[oos_start:]

    metrics = {
        "train": calculate_metrics(train_ret),
        "oos": calculate_metrics(oos_ret),
        "full": calculate_metrics(daily_ret),
    }
    return metrics


def calculate_top_drawdown_periods(daily_ret: pd.Series, n: int = 5) -> pd.DataFrame:
    """Identify the top N drawdown periods."""
    wealth = (1.0 + daily_ret).cumprod()
    running_max = wealth.cummax()
    (wealth / running_max) - 1.0

    periods = []
    in_drawdown = False
    peak_date = wealth.index[0]
    peak_val = wealth.iloc[0]
    valley_date = wealth.index[0]
    valley_val = wealth.iloc[0]
    start_date = wealth.index[0]

    for date, val in wealth.items():
        if val < peak_val:
            if not in_drawdown:
                in_drawdown = True
                start_date = date
            if val < valley_val:
                valley_date = date
                valley_val = val
        else:
            if in_drawdown:
                periods.append({
                    "Start Date": start_date.strftime("%Y-%m-%d"),
                    "Peak Date": peak_date.strftime("%Y-%m-%d"),
                    "Valley Date": valley_date.strftime("%Y-%m-%d"),
                    "Recovery Date": date.strftime("%Y-%m-%d"),
                    "Max Drawdown": (valley_val / peak_val) - 1.0,
                })
                in_drawdown = False
            peak_date = date
            peak_val = val
            valley_date = date
            valley_val = val

    if in_drawdown:
        periods.append({
            "Start Date": start_date.strftime("%Y-%m-%d"),
            "Peak Date": peak_date.strftime("%Y-%m-%d"),
            "Valley Date": valley_date.strftime("%Y-%m-%d"),
            "Recovery Date": "Not recovered",
            "Max Drawdown": (valley_val / peak_val) - 1.0,
        })

    df = pd.DataFrame(periods)
    if not df.empty:
        df = df.sort_values(by="Max Drawdown").head(n)
    return df


from leadlag.execution.backtester import BacktestEngine

def generate_slippage_sensitivity(
    model: SectorRelativeEnsembleModel,
    df_exec: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Calculate Sharpe, AR, and MDD across multiple slippage options."""
    slips = [0.0, 2.5, 5.0, 7.5, 10.0]
    records = []

    for slip in slips:
        res = BacktestEngine.run_backtest(model, df_exec, start_date=start_date, end_date=end_date, slippage_bps=slip)
        m = calculate_metrics(res["daily_returns"])
        records.append({
            "Slippage (bps)": slip,
            "AR": m.get("AR", 0.0),
            "Sharpe": m.get("Sharpe", 0.0),
            "MDD": m.get("MDD", 0.0),
        })
    return pd.DataFrame(records)


def main():
    args = parse_arguments()

    # 1. Load config file
    config_path = ROOT / args.config
    if config_path.exists():
        logger.info(f"Loading YAML config from: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config path {config_path} not found. Running with defaults.")
        cfg = {}

    # Override options with command line arguments if provided
    if "ensemble" not in cfg:
        cfg["ensemble"] = {}
    cfg["ensemble"]["raw_pca_weight"] = args.production_signal_weight
    cfg["ensemble"]["residual_pca_weight"] = args.residual_signal_weight
    cfg["ensemble"]["normalization"] = args.normalization

    if "portfolio" not in cfg:
        cfg["portfolio"] = {}
    cfg["portfolio"]["long_short_frac"] = args.long_short_frac
    cfg["portfolio"]["n_assets"] = args.n_assets
    cfg["portfolio"]["weight_mode"] = args.weight_mode

    if "costs" not in cfg:
        cfg["costs"] = {}
    cfg["costs"]["slippage_bps_per_side"] = args.slippage_bps

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir.startswith("results") else ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Download and Preprocess Data
    logger.info("Downloading market data...")
    raw_data = download_data(beta_window=60)
    logger.info("Preprocessing market data...")
    df_exec = preprocess_data(raw_data, beta_window=60)

    # Compute TOPIX returns
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    # 3. Instantiate PCA-Ensemble model and run generic BacktestEngine
    model = SectorRelativeEnsembleModel(cfg)
    from leadlag.execution.backtester import BacktestEngine
    results = BacktestEngine.run_backtest(model, df_exec, start_date=args.start_date, end_date=args.end_date, slippage_bps=args.slippage_bps)

    # 4. Save CSV Output Files
    # Summary Metrics
    t_end_dt = pd.to_datetime(args.train_end_date)
    o_start_dt = pd.to_datetime(args.oos_start_date)
    sub_metrics = calculate_subperiod_metrics(results["daily_returns"], t_end_dt, o_start_dt)

    pd.DataFrame([sub_metrics["train"]]).to_csv(out_dir / "metrics_summary_train.csv", index=False)
    pd.DataFrame([sub_metrics["oos"]]).to_csv(out_dir / "metrics_summary_oos.csv", index=False)
    pd.DataFrame([sub_metrics["full"]]).to_csv(out_dir / "metrics_summary_full.csv", index=False)

    # Daily results
    results["daily_returns_gross"].to_csv(out_dir / "daily_gross_returns.csv", header=["gross_return"])
    results["daily_costs"].to_csv(out_dir / "daily_costs.csv", header=["cost"])
    results["daily_returns"].to_csv(out_dir / "daily_net_returns.csv", header=["net_return"])
    results["equity_curve"].to_csv(out_dir / "daily_equity_curve.csv", header=["equity"])
    results["drawdown"].to_csv(out_dir / "daily_drawdown.csv", header=["drawdown"])
    results["daily_turnover"].to_csv(out_dir / "daily_turnover.csv", header=["turnover"])

    # Signals / weights
    if args.save_signals == "true":
        results["signals"].to_csv(out_dir / "signals.csv")
        results["normalized_signals"].to_csv(out_dir / "normalized_signals.csv")
    if args.save_weights == "true":
        results["weights"].to_csv(out_dir / "weights.csv")

    # Positions (signs of weights)
    positions_df = results["weights"].apply(np.sign)
    positions_df.to_csv(out_dir / "positions.csv")

    # 5. Diagnostics
    logger.info("Computing diagnostics...")

    # flattened signal arrays for correlation
    raw_pca_flat = results["raw_pca_signals"].values.flatten()
    residual_pca_flat = results["residual_pca_signals"].values.flatten()

    raw_pca_p3_corr = np.corrcoef(raw_pca_flat, residual_pca_flat)[0, 1]
    raw_pca_p3_rank_corr, _ = spearmanr(raw_pca_flat, residual_pca_flat)
    raw_pca_p3_sign_agree = np.mean(np.sign(raw_pca_flat) == np.sign(residual_pca_flat))

    pd.DataFrame([{"correlation": raw_pca_p3_corr}]).to_csv(out_dir / "component_signal_correlation.csv", index=False)
    pd.DataFrame([{"rank_correlation": raw_pca_p3_rank_corr}]).to_csv(out_dir / "component_rank_correlation.csv", index=False)
    pd.DataFrame([{"sign_agreement": raw_pca_p3_sign_agree}]).to_csv(out_dir / "component_signal_agreement.csv", index=False)

    top_dd = calculate_top_drawdown_periods(results["daily_returns"], 5)
    top_dd.to_csv(out_dir / "top_drawdown_periods.csv", index=False)

    slip_sens = generate_slippage_sensitivity(model, df_exec, args.start_date, args.end_date)
    slip_sens.to_csv(out_dir / "slippage_sensitivity.csv", index=False)

    # 6. Plots
    if args.save_plots == "true":
        logger.info("Generating plots...")

        # Equity curve plot
        plt.figure(figsize=(10, 5))
        plt.plot(results["equity_curve"], label="Sector Relative Ensemble (Net)", color="navy")
        plt.title("SRE Cumulative Return")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png", dpi=150)
        plt.close()

        # Drawdown plot
        plt.figure(figsize=(10, 5))
        plt.fill_between(results["drawdown"].index, results["drawdown"].values, 0, color="crimson", alpha=0.3)
        plt.plot(results["drawdown"], color="crimson")
        plt.title("SRE Drawdown Profile")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "drawdown.png", dpi=150)
        plt.close()

        # Rolling Sharpe plot (250d)
        roll_sharpe = []
        dates = results["daily_returns"].index
        for idx in range(250, len(dates)):
            slice_ret = results["daily_returns"].iloc[idx - 250 : idx]
            m = calculate_metrics(slice_ret)
            roll_sharpe.append(m.get("Sharpe", np.nan))

        plt.figure(figsize=(10, 5))
        plt.plot(dates[250:], roll_sharpe, color="darkgreen")
        plt.title("SRE 250-Day Rolling Sharpe Ratio")
        plt.xlabel("Date")
        plt.ylabel("Sharpe")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_sharpe.png", dpi=150)
        plt.close()

        # Rolling IC (250d)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(columns=lambda c: c.replace("jp_oc_", ""))

        # Calculate daily IC
        daily_ics = []
        for date in results["signals"].index:
            sig_t = results["signals"].loc[date].values
            r_t = y_jp_oc_df.loc[date].values
            if np.std(sig_t) > 0 and np.std(r_t) > 0:
                ic_val, _ = spearmanr(sig_t, r_t)
                daily_ics.append(ic_val)
            else:
                daily_ics.append(0.0)

        daily_ics_series = pd.Series(daily_ics, index=results["signals"].index)
        roll_ic_series = daily_ics_series.rolling(250).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(roll_ic_series, color="purple")
        plt.title("SRE 250-Day Rolling Information Coefficient (IC)")
        plt.xlabel("Date")
        plt.ylabel("Rolling IC")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_ic.png", dpi=150)
        plt.close()

        # Turnover plot
        plt.figure(figsize=(10, 5))
        plt.plot(results["daily_turnover"].rolling(20).mean(), color="teal", label="20-Day SMA")
        plt.plot(results["daily_turnover"], color="teal", alpha=0.3)
        plt.title("SRE Daily Portfolio Turnover")
        plt.xlabel("Date")
        plt.ylabel("Turnover")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "turnover.png", dpi=150)
        plt.close()

        # Signal Heatmap (100 days slice)
        plt.figure(figsize=(12, 6))
        sns.heatmap(results["signals"].iloc[-100:].T, cmap="RdYlBu_r", cbar=True)
        plt.title("SRE Daily Signals Heatmap (Last 100 Days)")
        plt.tight_layout()
        plt.savefig(out_dir / "signal_heatmap.png", dpi=150)
        plt.close()

    # 7. Audits
    if args.run_audit == "true":
        logger.info("Running audits...")
        from leadlag.compliance.auditor import ComplianceAuditor
        audit_res = ComplianceAuditor.run_audit(model, df_exec, results, out_dir)
        logger.info(f"Audit results written. All checks passed: {audit_res['all_passed']}")

    # 8. Markdown report generation (final_report.md)
    logger.info("Creating markdown report...")
    report_content = f"""# Sector Relative Ensemble (PCA-Ensemble) Backtest Report

This report summarizes the historical performance and safety checks of the PCA-Ensemble production model.

## Model Summary
- **Name**: sector_relative_ensemble
- **Display Name**: Sector Relative Ensemble
- **Short Name**: PCA-Ensemble
- **Slippage**: {args.slippage_bps} bps per side (one-way)

## Performance Metrics

| Period | Annualized Return (AR) | Risk (Vol) | R/R Ratio | Sharpe Ratio | Max Drawdown (MDD) |
|---|---|---|---|---|---|
| **Train** (2015-01-05 to 2019-12-31) | {sub_metrics["train"].get("AR", 0.0)*100:.2f}% | {sub_metrics["train"].get("RISK", 0.0)*100:.2f}% | {sub_metrics["train"].get("R/R", 0.0):.4f} | {sub_metrics["train"].get("Sharpe", 0.0):.4f} | {sub_metrics["train"].get("MDD", 0.0)*100:.2f}% |
| **OOS** (2020-01-01 to Present) | {sub_metrics["oos"].get("AR", 0.0)*100:.2f}% | {sub_metrics["oos"].get("RISK", 0.0)*100:.2f}% | {sub_metrics["oos"].get("R/R", 0.0):.4f} | {sub_metrics["oos"].get("Sharpe", 0.0):.4f} | {sub_metrics["oos"].get("MDD", 0.0)*100:.2f}% |
| **Full** (2015-01-05 to Present) | {sub_metrics["full"].get("AR", 0.0)*100:.2f}% | {sub_metrics["full"].get("RISK", 0.0)*100:.2f}% | {sub_metrics["full"].get("R/R", 0.0):.4f} | {sub_metrics["full"].get("Sharpe", 0.0):.4f} | {sub_metrics["full"].get("MDD", 0.0)*100:.2f}% |

## Diagnostics
- **Raw-PCA vs Residual-PCA Signal Correlation**: {raw_pca_p3_corr:.4f}
- **Raw-PCA vs Residual-PCA Rank Correlation**: {raw_pca_p3_rank_corr:.4f}
- **Raw-PCA vs Residual-PCA Sign Agreement**: {raw_pca_p3_sign_agree*100:.2f}%

### Worst Drawdown Periods
{top_dd.to_markdown(index=False)}

### Slippage Sensitivity
{slip_sens.to_markdown(index=False)}

## Safety Audit Summary
- **Audit Directory**: `results/sector_relative_ensemble/audit/`
- **Pass Status**: {"All Passed" if audit_res["all_passed"] else "Warning / Failures Found"}
"""
    with open(out_dir / "final_report.md", "w") as f:
        f.write(report_content)

    logger.info("Backtest execution finished successfully.")


if __name__ == "__main__":
    main()
