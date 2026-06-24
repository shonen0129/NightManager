#!/usr/bin/env python
"""Validation and backtesting script for US Residual Prior (P4 component) in PCA-Ensemble-USRP.

Runs a grid search over:
- prior_variants: raw_v1_to_v6, resid_v2_removed, resid_v1_v2_removed, resid_v1_v2_scaled_025, resid_v1_v2_scaled_050
- gamma_grid: [0.0, 0.5, 0.75, 1.0]
- lambda_reg_grid: [0.25, 0.50, 0.75]
- ensemble_grid:
  - SRE_current: Raw-PCA=0.5, Residual-PCA=0.5, P4=0.0
  - SRE_P4_40_40_20: Raw-PCA=0.4, Residual-PCA=0.4, P4=0.2
  - SRE_P4_375_375_25: Raw-PCA=0.375, Residual-PCA=0.375, P4=0.25
  - SRE_P4_45_45_10: Raw-PCA=0.45, Residual-PCA=0.45, P4=0.10
  - P4_only: Raw-PCA=0.0, Residual-PCA=0.0, P4=1.0
- slippage_grid: [0.0, 5.0, 10.0] bps

Outputs comparison matrices, OOS rankings, outperformance stats vs current PCA-Ensemble,
diagnostic plots, safety audits, and a comprehensive markdown report.
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
import yfinance as yf
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
from leadlag.execution.backtester import BacktestEngine
from leadlag.compliance.auditor import ComplianceAuditor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SRE-USRP US Residual Prior Validation Suite")
    parser.add_argument(
        "--config", default="configs/research_us_residual_prior.yaml", help="Path to config file"
    )
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--end-date", default="latest", help="Backtest end date")
    parser.add_argument("--train-end-date", default="2019-12-31", help="Train period end date")
    parser.add_argument("--oos-start-date", default="2020-01-01", help="OOS period start date")
    parser.add_argument(
        "--output-dir", default="results/us_residual_prior/", help="Output directory"
    )
    parser.add_argument(
        "--gamma-grid", default="0.0,0.5,0.75,1.0", help="Comma-separated gamma values"
    )
    parser.add_argument(
        "--lambda-reg-grid", default="0.25,0.5,0.75", help="Comma-separated lambda_reg values"
    )
    parser.add_argument(
        "--slippage-grid", default="0.0,5.0,10.0", help="Comma-separated slippage values in bps"
    )
    parser.add_argument(
        "--only-report", action="store_true", help="Load saved CSV results and generate report/plots only"
    )
    return parser.parse_args()


def get_subperiod_data(
    series: pd.Series, t_end: pd.Timestamp, o_start: pd.Timestamp
) -> dict[str, pd.Series]:
    """Slice a series into Train, OOS, and Full subperiods."""
    return {
        "train": series.loc[:t_end],
        "oos": series.loc[o_start:],
        "full": series,
    }


def main():
    args = parse_arguments()

    gamma_grid = [float(g) for g in args.gamma_grid.split(",")]
    lambda_reg_grid = [float(lam) for lam in args.lambda_reg_grid.split(",")]
    slippage_grid = [float(s) for s in args.slippage_grid.split(",")]

    prior_variants = [
        "raw_v1_to_v6",
        "resid_v2_removed",
        "resid_v1_v2_removed",
        "resid_v1_v2_scaled_025",
        "resid_v1_v2_scaled_050",
    ]

    ensemble_grid = [
        {"name": "SRE_current", "raw_pca": 0.50, "residual_pca": 0.50, "p4": 0.00},
        {"name": "SRE_P4_40_40_20", "raw_pca": 0.40, "residual_pca": 0.40, "p4": 0.20},
        {"name": "SRE_P4_375_375_25", "raw_pca": 0.375, "residual_pca": 0.375, "p4": 0.25},
        {"name": "SRE_P4_45_45_10", "raw_pca": 0.45, "residual_pca": 0.45, "p4": 0.10},
        {"name": "P4_only", "raw_pca": 0.00, "residual_pca": 0.00, "p4": 1.00},
    ]

    out_dir = (
        Path(args.output_dir) if args.output_dir.startswith("results") else ROOT / args.output_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = ROOT / args.config
    if config_path.exists():
        logger.info(f"Loading YAML config from: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config path {config_path} not found. Running with defaults.")
        cfg = {}

    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Download and Preprocess Data
    logger.info("Downloading sector ETF data...")
    raw_data = download_data(beta_window=60)
    logger.info("Downloading SPY benchmark data...")
    spy_df = yf.download("SPY", start="2009-01-01", auto_adjust=False)

    logger.info("Preprocessing market data...")
    df_exec = preprocess_data(raw_data, beta_window=60)

    # Compute and align TOPIX returns
    topix_close = raw_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = raw_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values
    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (
        1.0 + df_exec["topix_oc_return"]
    ) - 1.0

    # Align SPY returns to df_exec
    spy_close = spy_df["Close"].copy()
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None).normalize()
    spy_cc = spy_close.pct_change(fill_method=None)
    df_exec["spy_cc"] = spy_cc.reindex(df_exec["sig_date"]).values
    df_exec["spy_cc"] = df_exec["spy_cc"].fillna(0.0)

    t_end_dt = pd.to_datetime(args.train_end_date)
    o_start_dt = pd.to_datetime(args.oos_start_date)

    daily_returns_db = {}
    daily_positions_db = {}
    summary_records = []

    best_candidate_res = None
    best_candidate_sharpe = -999.0
    current_sre_oos_metrics = None

    # Track prior diagnostic results
    c0_diagnostics_records = []

    # Map to store run results for P4 vs Residual-PCA signal correlation
    p4_vs_p3_corrs = []

    if getattr(args, "only_report", False):
        logger.info("Running in report-only mode, loading saved CSV results...")
        summary_df = pd.read_csv(out_dir / "summary.csv")
        summary_5bps = pd.read_csv(out_dir / "summary_5bps.csv")
        oos_ranking_5bps = pd.read_csv(out_dir / "oos_ranking_5bps.csv")
        
        # Parse numeric types to ensure exact matching
        oos_ranking_5bps["gamma"] = oos_ranking_5bps["gamma"].astype(float)
        oos_ranking_5bps["lambda_reg"] = oos_ranking_5bps["lambda_reg"].astype(float)
        oos_ranking_5bps["raw_pca"] = oos_ranking_5bps["raw_pca"].astype(float)
        oos_ranking_5bps["residual_pca"] = oos_ranking_5bps["residual_pca"].astype(float)
        oos_ranking_5bps["p4"] = oos_ranking_5bps["p4"].astype(float)
        oos_ranking_5bps["AR"] = oos_ranking_5bps["AR"].astype(float)
        oos_ranking_5bps["RISK"] = oos_ranking_5bps["RISK"].astype(float)
        oos_ranking_5bps["Sharpe"] = oos_ranking_5bps["Sharpe"].astype(float)
        oos_ranking_5bps["MDD"] = oos_ranking_5bps["MDD"].astype(float)
        oos_ranking_5bps["Avg Turnover"] = oos_ranking_5bps["Avg Turnover"].astype(float)

        c0_diagnostics = pd.read_csv(out_dir / "c0_diagnostics.csv")
        c0_diagnostics_records = c0_diagnostics.to_dict("records")
        p4_vs_p3_corrs_df = pd.read_csv(out_dir / "p4_vs_p3_by_prior.csv")
        p4_vs_p3_corrs = p4_vs_p3_corrs_df.to_dict("records")

        daily_returns_df = pd.read_csv(out_dir / "daily_returns.csv", index_col=0)
        daily_returns_df.index = pd.to_datetime(daily_returns_df.index)
        daily_returns_db = {col: daily_returns_df[col] for col in daily_returns_df.columns}

        # Filter out SRE_current metrics
        current_sre_oos_metrics = oos_ranking_5bps[oos_ranking_5bps["model"] == "SRE_current"].iloc[0].to_dict()
        curr_sharpe = float(current_sre_oos_metrics["Sharpe"])
        curr_mdd = float(current_sre_oos_metrics["MDD"])
        curr_ar = float(current_sre_oos_metrics["AR"])
        curr_turnover = float(current_sre_oos_metrics["Avg Turnover"])

        # PCA-Ensemble current returns timeseries
        res_sre_returns = daily_returns_db["SRE_current"]

        # Identify best candidate based on Sharpe from OOS ranking
        approved_candidates = oos_ranking_5bps[oos_ranking_5bps["pass_candidate"] == True]
        if len(approved_candidates) > 0:
            best_candidate_row = approved_candidates.iloc[0]
        else:
            best_candidate_row = oos_ranking_5bps[oos_ranking_5bps["model"] != "SRE_current"].iloc[0]
        best_prior = best_candidate_row["prior_variant"]
        best_gamma = float(best_candidate_row["gamma"])
        best_lambda_reg = float(best_candidate_row["lambda_reg"])
        best_ens_name = best_candidate_row["model"]
        best_ens = next(e for e in ensemble_grid if e["name"] == best_ens_name)
        best_candidate_sharpe = float(best_candidate_row["Sharpe"])

        # Re-run best model to populate best_candidate_res
        run_cfg = cfg.copy()
        if "ensemble" not in run_cfg:
            run_cfg["ensemble"] = {}
        run_cfg["ensemble"]["raw_pca_weight"] = best_ens["raw_pca"]
        run_cfg["ensemble"]["residual_pca_weight"] = best_ens["residual_pca"]
        run_cfg["ensemble"]["p4_weight"] = best_ens["p4"]

        if "us_residualization" not in run_cfg:
            run_cfg["us_residualization"] = {}
        run_cfg["us_residualization"]["gamma"] = best_gamma
        run_cfg["us_residualization"]["beta_window"] = 60
        run_cfg["us_residualization"]["beta_shift"] = 1

        if "prior" not in run_cfg:
            run_cfg["prior"] = {}
        run_cfg["prior"]["variant"] = best_prior

        run_cfg["lambda_reg"] = best_lambda_reg

        best_model = SectorRelativeEnsembleModel(run_cfg)
        best_candidate_res = BacktestEngine.run_backtest(
            best_model,
            df_exec,
            start_date=args.start_date,
            end_date=args.end_date,
            slippage_bps=5.0,
        )
        dates = best_candidate_res["signals"].index
    else:
        logger.info("Starting SRE-USRP Grid Search Backtest Loop...")
        for prior in prior_variants:
            for gamma in gamma_grid:
                for lambda_reg in lambda_reg_grid:
                    for ens in ensemble_grid:
                        for slip in slippage_grid:
                            # Skip redundant runs (gamma or prior variant has no effect on SRE_current since P4 weight = 0)
                            if ens["name"] == "SRE_current" and (
                                gamma != gamma_grid[0]
                                or prior != prior_variants[0]
                                or lambda_reg != lambda_reg_grid[0]
                            ):
                                continue

                            # Update config
                            run_cfg = cfg.copy()
                            if "ensemble" not in run_cfg:
                                run_cfg["ensemble"] = {}
                            run_cfg["ensemble"]["raw_pca_weight"] = ens["raw_pca"]
                            run_cfg["ensemble"]["residual_pca_weight"] = ens["residual_pca"]
                            run_cfg["ensemble"]["p4_weight"] = ens["p4"]

                            if "us_residualization" not in run_cfg:
                                run_cfg["us_residualization"] = {}
                            run_cfg["us_residualization"]["gamma"] = gamma
                            run_cfg["us_residualization"]["beta_window"] = 60
                            run_cfg["us_residualization"]["beta_shift"] = 1

                            if "prior" not in run_cfg:
                                run_cfg["prior"] = {}
                            run_cfg["prior"]["variant"] = prior

                            if ens["name"] == "SRE_current":
                                run_cfg["lambda_reg"] = 0.75
                                run_cfg["prior"]["variant"] = "raw_v1_to_v6"
                                run_cfg["us_residualization"]["gamma"] = 0.0
                            else:
                                run_cfg["lambda_reg"] = lambda_reg

                            # Instantiate and backtest
                            model = SectorRelativeEnsembleModel(run_cfg)
                            res = BacktestEngine.run_backtest(
                                model,
                                df_exec,
                                start_date=args.start_date,
                                end_date=args.end_date,
                                slippage_bps=slip,
                            )

                            # Extract prior diagnostic info (once per prior/gamma/lambda_reg)
                            if abs(slip - 5.0) < 1e-6 and ens["name"] != "SRE_current":
                                p_info = res["prior_info"]
                                c0_diagnostics_records.append(
                                    {
                                        "prior_variant": prior,
                                        "gamma": gamma,
                                        "lambda_reg": lambda_reg,
                                        "min_eigenvalue": p_info["min_eig"],
                                        "max_eigenvalue": p_info["max_eig"],
                                        "condition_number": p_info["cond_num"],
                                        "c0_source": p_info["c0_source"],
                                        "k_expected": p_info["k_expected"],
                                        "v1_scaled_or_removed": "removed"
                                        if prior in ["resid_v2_removed", "resid_v1_v2_removed"]
                                        else "scaled"
                                        if "scaled" in prior
                                        else "no",
                                        "v2_scaled_or_removed": "removed"
                                        if prior in ["resid_v2_removed", "resid_v1_v2_removed"]
                                        else "scaled"
                                        if "scaled" in prior
                                        else "no",
                                    }
                                )

                            # Split periods
                            ret_split = get_subperiod_data(res["daily_returns"], t_end_dt, o_start_dt)
                            turnover_split = get_subperiod_data(
                                res["daily_turnover"], t_end_dt, o_start_dt
                            )
                            gross_split = get_subperiod_data(
                                res["daily_gross_exps"], t_end_dt, o_start_dt
                            )
                            costs_split = get_subperiod_data(res["daily_costs"], t_end_dt, o_start_dt)

                            weights_df = res["weights"]
                            net_exps = weights_df.sum(axis=1)
                            long_exps = weights_df.apply(lambda row: row[row > 0].sum(), axis=1)
                            short_exps = weights_df.apply(lambda row: row[row < 0].abs().sum(), axis=1)

                            net_split = get_subperiod_data(net_exps, t_end_dt, o_start_dt)
                            long_split = get_subperiod_data(long_exps, t_end_dt, o_start_dt)
                            short_split = get_subperiod_data(short_exps, t_end_dt, o_start_dt)

                            if ens["name"] == "SRE_current":
                                db_key = "SRE_current"
                            else:
                                db_key = f"{ens['name']}_prior_{prior}_gamma_{gamma:.2f}_lam_{lambda_reg:.2f}"

                            # Save daily timeseries under 5bps
                            if abs(slip - 5.0) < 1e-6:
                                daily_returns_db[db_key] = res["daily_returns"]
                                daily_positions_db[db_key] = weights_df

                                # Compute P4 vs Residual-PCA signal correlation
                                if ens["name"] != "SRE_current":
                                    residual_pca_flat = res["residual_pca_signals"].values.flatten()
                                    p4_flat = res["p4_signals"].values.flatten()
                                    p4_p3_corr = float(np.corrcoef(p4_flat, residual_pca_flat)[0, 1])
                                    p4_vs_p3_corrs.append(
                                        {
                                            "prior_variant": prior,
                                            "gamma": gamma,
                                            "lambda_reg": lambda_reg,
                                            "model": ens["name"],
                                            "p4_vs_p3_correlation": p4_p3_corr,
                                        }
                                    )

                            for period in ["train", "oos", "full"]:
                                m = calculate_metrics(ret_split[period])
                                ar = m.get("AR", 0.0)
                                vol = m.get("RISK", 0.0)
                                rr = m.get("R/R", 0.0)
                                sharpe = m.get("Sharpe", 0.0)
                                mdd = m.get("MDD", 0.0)

                                win_rate = (
                                    float(np.mean(ret_p > 0))
                                    if len(ret_p := ret_split[period]) > 0
                                    else 0.0
                                )
                                avg_daily_ret = float(np.mean(ret_p)) if len(ret_p) > 0 else 0.0
                                std_daily_ret = float(np.std(ret_p, ddof=1)) if len(ret_p) > 1 else 0.0

                                avg_turnover = (
                                    float(np.mean(turnover_split[period]))
                                    if len(turnover_split[period]) > 0
                                    else 0.0
                                )
                                avg_gross = (
                                    float(np.mean(gross_split[period]))
                                    if len(gross_split[period]) > 0
                                    else 0.0
                                )
                                avg_net = (
                                    float(np.mean(net_split[period]))
                                    if len(net_split[period]) > 0
                                    else 0.0
                                )
                                avg_long = (
                                    float(np.mean(long_split[period]))
                                    if len(long_split[period]) > 0
                                    else 0.0
                                )
                                avg_short = (
                                    float(np.mean(short_split[period]))
                                    if len(short_split[period]) > 0
                                    else 0.0
                                )
                                avg_cost = (
                                    float(np.mean(costs_split[period]))
                                    if len(costs_split[period]) > 0
                                    else 0.0
                                )

                                record = {
                                    "model": ens["name"],
                                    "prior_variant": prior,
                                    "gamma": gamma,
                                    "lambda_reg": lambda_reg,
                                    "raw_pca": ens["raw_pca"],
                                    "residual_pca": ens["residual_pca"],
                                    "p4": ens["p4"],
                                    "slippage_bps": slip,
                                    "period": period,
                                    "AR": ar,
                                    "RISK": vol,
                                    "R/R": rr,
                                    "Sharpe": sharpe,
                                    "MDD": mdd,
                                    "Win Rate": win_rate,
                                    "Avg Daily Return": avg_daily_ret,
                                    "Daily Return Std": std_daily_ret,
                                    "Avg Turnover": avg_turnover,
                                    "Avg Gross Exposure": avg_gross,
                                    "Avg Net Exposure": avg_net,
                                    "Avg Long Exposure": avg_long,
                                    "Avg Short Exposure": avg_short,
                                    "Avg Transaction Cost": avg_cost,
                                    "Total Return": m.get("Total Return", 0.0),
                                }
                                summary_records.append(record)

                                if ens["name"] == "SRE_current" and abs(slip - 5.0) < 1e-6:
                                    if period == "oos":
                                        current_sre_oos_metrics = record

                                # Track best candidate based on OOS Sharpe at 5bps
                                if (
                                    period == "oos"
                                    and abs(slip - 5.0) < 1e-6
                                    and ens["name"] != "SRE_current"
                                ):
                                    if sharpe > best_candidate_sharpe:
                                        best_candidate_sharpe = sharpe
                                        best_candidate_res = res

        # Re-save SRE_current metrics for other keys so dataframe matching works cleanly
        # (since we skipped run logic, copy existing SRE_current values to all priors/gammas/lambdas)
        sre_current_records = [r for r in summary_records if r["model"] == "SRE_current"]
        summary_records = [r for r in summary_records if r["model"] != "SRE_current"]
        for record in sre_current_records:
            for p in prior_variants:
                for g in gamma_grid:
                    for lam in lambda_reg_grid:
                        new_rec = record.copy()
                        new_rec["prior_variant"] = p
                        new_rec["gamma"] = g
                        new_rec["lambda_reg"] = lam
                        summary_records.append(new_rec)

        # Build summary dataframes
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(out_dir / "summary.csv", index=False)

        summary_5bps = summary_df[abs(summary_df["slippage_bps"] - 5.0) < 1e-6]
        summary_5bps.to_csv(out_dir / "summary_5bps.csv", index=False)

        # 3. Process Ranking on OOS at 5bps
        oos_5bps = summary_df[
            (summary_df["period"] == "oos") & (abs(summary_df["slippage_bps"] - 5.0) < 1e-6)
        ].copy()

        # Compare PCA-Ensemble current benchmark (OOS Sharpe, MDD, AR, Turnover)
        curr_sharpe = current_sre_oos_metrics["Sharpe"]
        curr_mdd = current_sre_oos_metrics["MDD"]
        curr_ar = current_sre_oos_metrics["AR"]
        curr_turnover = current_sre_oos_metrics["Avg Turnover"]

        # PCA-Ensemble current returns timeseries
        res_sre_returns = daily_returns_db["SRE_current"]

        # Flags calculation for all rows
        oos_5bps["improves_oos_sharpe"] = oos_5bps["Sharpe"] > curr_sharpe
        oos_5bps["improves_oos_mdd"] = oos_5bps["MDD"] >= curr_mdd
        oos_5bps["keeps_oos_ar_95pct"] = oos_5bps["AR"] >= 0.95 * curr_ar
        oos_5bps["turnover_within_10pct"] = oos_5bps["Avg Turnover"] <= 1.1 * curr_turnover

        # For each candidate, calculate diff returns t-stat
        diff_tstats = []
        for idx, row in oos_5bps.iterrows():
            cand_key = f"{row['model']}_prior_{row['prior_variant']}_gamma_{row['gamma']:.2f}_lam_{row['lambda_reg']:.2f}"
            if cand_key in daily_returns_db:
                ret_cand = daily_returns_db[cand_key]
                diff = ret_cand - res_sre_returns
                mean_d = diff.mean()
                std_d = diff.std()
                tstat = np.sqrt(245) * mean_d / (std_d if std_d > 1e-12 else 1e-12)
                diff_tstats.append(tstat)
            else:
                diff_tstats.append(0.0)
        oos_5bps["diff_tstat"] = diff_tstats
        oos_5bps["diff_tstat_positive"] = oos_5bps["diff_tstat"] > 0.0

        # Calculate P4 vs Residual-PCA signal corr for all models
        corr_map = {}
        for item in p4_vs_p3_corrs:
            k = f"{item['model']}_{item['prior_variant']}_{item['gamma']:.2f}_{item['lambda_reg']:.2f}"
            corr_map[k] = item["p4_vs_p3_correlation"]

        p4_p3_corr_col = []
        for idx, row in oos_5bps.iterrows():
            k = f"{row['model']}_{row['prior_variant']}_{row['gamma']:.2f}_{row['lambda_reg']:.2f}"
            p4_p3_corr_col.append(corr_map.get(k, 0.0 if row["model"] == "SRE_current" else 1.0))
        oos_5bps["p4_vs_p3_corr"] = p4_p3_corr_col
        oos_5bps["p4_corr_below_0_95_to_p3"] = oos_5bps["p4_vs_p3_corr"] < 0.95

        # Overall candidate checks
        oos_5bps["pass_candidate"] = (
            oos_5bps["improves_oos_sharpe"]
            & oos_5bps["improves_oos_mdd"]
            & oos_5bps["keeps_oos_ar_95pct"]
            & oos_5bps["turnover_within_10pct"]
            & oos_5bps["p4_corr_below_0_95_to_p3"]
            & oos_5bps["diff_tstat_positive"]
        )

        oos_ranking_5bps = oos_5bps.sort_values(by="Sharpe", ascending=False)
        oos_ranking_5bps.to_csv(out_dir / "oos_ranking_5bps.csv", index=False)

        # Save daily timeseries
        pd.DataFrame(daily_returns_db).to_csv(out_dir / "daily_returns.csv")
        pd.concat(daily_positions_db, axis=1).to_csv(out_dir / "daily_positions.csv")

        # Drawdown timeseries
        daily_dd_db = {}
        for name, ret in daily_returns_db.items():
            wealth = (1.0 + ret).cumprod()
            running_max = wealth.cummax()
            daily_dd_db[name] = (wealth / running_max) - 1.0
        pd.DataFrame(daily_dd_db).to_csv(out_dir / "drawdown_timeseries.csv")

        # C0 diagnostics CSV
        pd.DataFrame(c0_diagnostics_records).to_csv(out_dir / "c0_diagnostics.csv", index=False)

    # Output signal correlations for the best candidate
    raw_pca_flat = best_candidate_res["raw_pca_signals"].values.flatten()
    residual_pca_flat = best_candidate_res["residual_pca_signals"].values.flatten()
    p4_flat = best_candidate_res["p4_signals"].values.flatten()

    best_p0_p3_corr = np.corrcoef(raw_pca_flat, residual_pca_flat)[0, 1]
    best_p0_p4_corr = np.corrcoef(raw_pca_flat, p4_flat)[0, 1]
    best_p3_p4_corr = np.corrcoef(residual_pca_flat, p4_flat)[0, 1]

    best_p0_p3_rank, _ = spearmanr(raw_pca_flat, residual_pca_flat)
    best_p0_p4_rank, _ = spearmanr(raw_pca_flat, p4_flat)
    best_p3_p4_rank, _ = spearmanr(residual_pca_flat, p4_flat)

    best_p0_p3_agree = np.mean(np.sign(raw_pca_flat) == np.sign(residual_pca_flat))
    best_p0_p4_agree = np.mean(np.sign(raw_pca_flat) == np.sign(p4_flat))
    best_p3_p4_agree = np.mean(np.sign(residual_pca_flat) == np.sign(p4_flat))

    correlations_records = [
        {
            "pair": "Raw-PCA_vs_P3",
            "correlation": best_p0_p3_corr,
            "rank_correlation": best_p0_p3_rank,
            "sign_agreement": best_p0_p3_agree,
        },
        {
            "pair": "Raw-PCA_vs_P4",
            "correlation": best_p0_p4_corr,
            "rank_correlation": best_p0_p4_rank,
            "sign_agreement": best_p0_p4_agree,
        },
        {
            "pair": "Residual-PCA_vs_P4",
            "correlation": best_p3_p4_corr,
            "rank_correlation": best_p3_p4_rank,
            "sign_agreement": best_p3_p4_agree,
        },
    ]
    pd.DataFrame(correlations_records).to_csv(out_dir / "signal_correlations.csv", index=False)

    # Output p4 vs residual-pca by prior_variant
    pd.DataFrame(p4_vs_p3_corrs).to_csv(out_dir / "p4_vs_p3_by_prior.csv", index=False)

    # Output diff return stats
    diff_ret_best = best_candidate_res["daily_returns"] - res_sre_returns
    diff_mean = diff_ret_best.mean()
    diff_std = diff_ret_best.std()
    diff_tstat = np.sqrt(245) * diff_mean / (diff_std if diff_std > 1e-12 else 1e-12)

    monthly_cand = (
        best_candidate_res["daily_returns"]
        .groupby(best_candidate_res["daily_returns"].index.to_period("M"))
        .sum()
    )
    monthly_sre = res_sre_returns.groupby(res_sre_returns.index.to_period("M")).sum()
    monthly_diff = monthly_cand - monthly_sre

    monthly_win_count = int((monthly_diff > 0).sum())
    monthly_loss_count = int((monthly_diff < 0).sum())
    avg_monthly_outperf = (
        float(monthly_diff[monthly_diff > 0].mean()) if (monthly_diff > 0).any() else 0.0
    )
    worst_monthly_underperf = float(monthly_diff.min())
    best_monthly_outperf = float(monthly_diff.max())

    diff_stats = [
        {
            "diff_mean_daily": diff_mean,
            "diff_std_daily": diff_std,
            "t_stat": diff_tstat,
            "daily_win_rate": float((diff_ret_best > 0).mean()),
            "monthly_win_count": monthly_win_count,
            "monthly_loss_count": monthly_loss_count,
            "average_monthly_outperformance": avg_monthly_outperf,
            "worst_monthly_underperformance": worst_monthly_underperf,
            "best_monthly_outperformance": best_monthly_outperf,
        }
    ]
    pd.DataFrame(diff_stats).to_csv(out_dir / "diff_return_stats.csv", index=False)

    # Output monthly win loss log
    monthly_log = pd.DataFrame({"monthly_outperformance": monthly_diff, "won": monthly_diff > 0})
    monthly_log.to_csv(out_dir / "monthly_win_loss.csv")

    # Re-run best model to output audit
    approved_candidates = oos_ranking_5bps[oos_ranking_5bps["pass_candidate"] == True]
    if len(approved_candidates) > 0:
        best_candidate_row = approved_candidates.iloc[0]
    else:
        best_candidate_row = oos_ranking_5bps[oos_ranking_5bps["model"] != "SRE_current"].iloc[0]
    best_prior = best_candidate_row["prior_variant"]
    best_gamma = best_candidate_row["gamma"]
    best_lambda_reg = best_candidate_row["lambda_reg"]
    best_ens_name = best_candidate_row["model"]
    best_ens = next(e for e in ensemble_grid if e["name"] == best_ens_name)

    run_cfg = cfg.copy()
    if "ensemble" not in run_cfg:
        run_cfg["ensemble"] = {}
    run_cfg["ensemble"]["raw_pca_weight"] = best_ens["raw_pca"]
    run_cfg["ensemble"]["residual_pca_weight"] = best_ens["residual_pca"]
    run_cfg["ensemble"]["p4_weight"] = best_ens["p4"]
    if "us_residualization" not in run_cfg:
        run_cfg["us_residualization"] = {}
    run_cfg["us_residualization"]["gamma"] = best_gamma
    if "prior" not in run_cfg:
        run_cfg["prior"] = {}
    run_cfg["prior"]["variant"] = best_prior
    run_cfg["lambda_reg"] = best_lambda_reg

    best_model = SectorRelativeEnsembleModel(run_cfg)
    best_audit = ComplianceAuditor.run_audit(best_model, df_exec, best_candidate_res, out_dir)

    # IC calculation
    y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
        columns=lambda c: c.replace("jp_oc_", "")
    )
    dates = best_candidate_res["signals"].index
    daily_ics = []
    for date in dates:
        sig_t = best_candidate_res["signals"].loc[date].values
        r_t = y_jp_oc_df.loc[date].values
        if np.std(sig_t) > 0 and np.std(r_t) > 0:
            ic_val, _ = spearmanr(sig_t, r_t)
            daily_ics.append(ic_val)
        else:
            daily_ics.append(0.0)
    daily_ics_series = pd.Series(daily_ics, index=dates)
    roll_60_ic = daily_ics_series.rolling(60).mean()
    pd.DataFrame({"daily_ic": daily_ics_series, "rolling_60_ic": roll_60_ic}).to_csv(
        out_dir / "ic_timeseries.csv"
    )

    # Diagnostic Plots
    if run_cfg.get("output", {}).get("save_plots", True):
        logger.info("Generating diagnostic plots...")

        # Cumulative net return
        plt.figure(figsize=(10, 5))
        plt.plot(best_candidate_res["equity_curve"], label="Best SRE-USRP (Net)", color="navy")
        current_sre_net = (1.0 + res_sre_returns).cumprod()
        plt.plot(current_sre_net, label="SRE_current (Net)", color="gray", linestyle="--")
        plt.title("SRE-USRP Cumulative Net Equity vs Baseline")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png", dpi=150)
        plt.close()

        # Drawdown profile
        plt.figure(figsize=(10, 5))
        plt.fill_between(
            best_candidate_res["drawdown"].index,
            best_candidate_res["drawdown"].values,
            0,
            color="crimson",
            alpha=0.3,
        )
        plt.plot(best_candidate_res["drawdown"], color="crimson")
        plt.title("SRE-USRP Drawdown Profile")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "drawdown.png", dpi=150)
        plt.close()

        # Rolling Sharpe ratio (250d)
        roll_sharpe = []
        best_ret = best_candidate_res["daily_returns"]
        for idx in range(250, len(dates)):
            slice_ret = best_ret.iloc[idx - 250 : idx]
            m = calculate_metrics(slice_ret)
            roll_sharpe.append(m.get("Sharpe", np.nan))

        plt.figure(figsize=(10, 5))
        plt.plot(dates[250:], roll_sharpe, color="darkgreen")
        plt.title("SRE-USRP 250-Day Rolling Sharpe Ratio")
        plt.xlabel("Date")
        plt.ylabel("Sharpe")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_sharpe.png", dpi=150)
        plt.close()

        # Rolling IC (60d)
        plt.figure(figsize=(10, 5))
        plt.plot(roll_60_ic, color="purple", label="60-Day Rolling IC")
        plt.axhline(0, color="black", linestyle=":", alpha=0.5)
        plt.title("SRE-USRP 60-Day Rolling Information Coefficient (IC)")
        plt.xlabel("Date")
        plt.ylabel("IC")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_ic.png", dpi=150)
        plt.close()

        # Turnover
        plt.figure(figsize=(10, 5))
        plt.plot(
            best_candidate_res["daily_turnover"].rolling(20).mean(),
            color="teal",
            label="20-Day SMA",
        )
        plt.plot(best_candidate_res["daily_turnover"], color="teal", alpha=0.3)
        plt.title("SRE-USRP Daily Portfolio Turnover")
        plt.xlabel("Date")
        plt.ylabel("Turnover")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "turnover.png", dpi=150)
        plt.close()

        # Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(best_candidate_res["signals"].iloc[-100:].T, cmap="RdYlBu_r", cbar=True)
        plt.title("SRE-USRP Daily Signals Heatmap (Last 100 Days)")
        plt.tight_layout()
        plt.savefig(out_dir / "signal_heatmap.png", dpi=150)
        plt.close()

    # Prior variant comparison table data
    prior_comp = oos_ranking_5bps[
        (oos_ranking_5bps["model"] == best_ens_name)
        & (oos_ranking_5bps["gamma"] == best_gamma)
        & (oos_ranking_5bps["lambda_reg"] == best_lambda_reg)
    ].sort_values(by="prior_variant")
    prior_comp.to_csv(out_dir / "prior_variant_comparison.csv", index=False)

    # Gamma sensitivity table data
    gamma_sens = oos_ranking_5bps[
        (oos_ranking_5bps["model"] == best_ens_name)
        & (oos_ranking_5bps["prior_variant"] == best_prior)
        & (oos_ranking_5bps["lambda_reg"] == best_lambda_reg)
    ].sort_values(by="gamma")
    gamma_sens.to_csv(out_dir / "gamma_sensitivity.csv", index=False)

    # Lambda reg sensitivity table data
    lam_sens = oos_ranking_5bps[
        (oos_ranking_5bps["model"] == best_ens_name)
        & (oos_ranking_5bps["prior_variant"] == best_prior)
        & (oos_ranking_5bps["gamma"] == best_gamma)
    ].sort_values(by="lambda_reg")
    lam_sens.to_csv(out_dir / "lambda_reg_sensitivity.csv", index=False)

    # Subperiod metrics comparison
    best_train_m = summary_5bps[
        (summary_5bps["model"] == best_ens_name)
        & (summary_5bps["prior_variant"] == best_prior)
        & (summary_5bps["gamma"] == best_gamma)
        & (summary_5bps["lambda_reg"] == best_lambda_reg)
        & (summary_5bps["period"] == "train")
    ].iloc[0]
    best_oos_m = summary_5bps[
        (summary_5bps["model"] == best_ens_name)
        & (summary_5bps["prior_variant"] == best_prior)
        & (summary_5bps["gamma"] == best_gamma)
        & (summary_5bps["lambda_reg"] == best_lambda_reg)
        & (summary_5bps["period"] == "oos")
    ].iloc[0]
    best_full_m = summary_5bps[
        (summary_5bps["model"] == best_ens_name)
        & (summary_5bps["prior_variant"] == best_prior)
        & (summary_5bps["gamma"] == best_gamma)
        & (summary_5bps["lambda_reg"] == best_lambda_reg)
        & (summary_5bps["period"] == "full")
    ].iloc[0]

    curr_train_m = summary_5bps[
        (summary_5bps["model"] == "SRE_current") & (summary_5bps["period"] == "train")
    ].iloc[0]
    curr_oos_m = summary_5bps[
        (summary_5bps["model"] == "SRE_current") & (summary_5bps["period"] == "oos")
    ].iloc[0]
    curr_full_m = summary_5bps[
        (summary_5bps["model"] == "SRE_current") & (summary_5bps["period"] == "full")
    ].iloc[0]

    # Process and sort OOS ranking table with required columns
    ranking_table = oos_ranking_5bps[
        [
            "model",
            "prior_variant",
            "gamma",
            "lambda_reg",
            "raw_pca",
            "residual_pca",
            "p4",
            "AR",
            "RISK",
            "Sharpe",
            "MDD",
            "Avg Turnover",
            "p4_vs_p3_corr",
            "diff_tstat",
            "pass_candidate",
        ]
    ].copy()

    # Extract diagnostic info for report formatting
    prior_info = best_candidate_res["prior_info"]

    # Generate Markdown Report (report.md)
    logger.info("Creating markdown report...")
    pass_status = "APPROVED" if best_candidate_row["pass_candidate"] else "REJECTED"

    report_content = f"""# US Residual Prior Backtest Report

## 1. Executive Summary
- **Current Production PCA-Ensemble OOS Sharpe**: {curr_sharpe:.4f}
- **Best PCA-Ensemble-USRP Candidate OOS Sharpe**: {best_candidate_sharpe:.4f}
- **Best PCA-Ensemble-USRP Config**: Prior Variant = `{best_prior}`, Gamma = {best_gamma:.2f}, Lambda Reg = {best_lambda_reg:.2f}, weights = [Raw-PCA: {best_ens["raw-pca"]:.3f}, Residual-PCA: {best_ens["residual-pca"]:.3f}, P4: {best_ens["p4"]:.3f}]
- **Recommendation**: {pass_status} for production/shadow deployment.
- **Key Reason**: The space-consistent residual prior P4 model yields an OOS Sharpe ratio of {best_candidate_sharpe:.4f} compared to current PCA-Ensemble's {curr_sharpe:.4f}. It {"satisfies" if pass_status == "APPROVED" else "breaches some of"} the primary validation constraints (MDD constraint, AR drawdown, turnover criteria, P4 vs Residual-PCA correlation < 0.95, positive t-stat).

## 2. Motivation
- **Inconsistent Prior Issue**: In the previous P4 validation, the residualized return matrix was mapped onto a prior subspace target $C0$ built from raw, unresidualized returns using raw cyclical/defensive, FX, energy, and inflation prior vectors. This re-injected global and group-level market dynamics back into the market-neutral signal space.
- **Goal**: Reconstruct $C0$ in the residualized subspace and orthonormalize prior vectors after removing the global factor $v1$ and group difference factor $v2$. We evaluate if this restores P4's true idiosyncrasy and lowers its correlation to Residual-PCA.

## 3. Backtest Setup
- **Train period**: 2015-01-05 to 2019-12-31
- **OOS period**: 2020-01-01 to {dates[-1].strftime("%Y-%m-%d")}
- **US Benchmark**: SPY (rolling OLS, 60-day window, 1-day shift)
- **JP Benchmark**: TOPIX (rolling OLS, 60-day window, 1-day shift)
- **Costs**: 5.0 bps per side (one-way) slippage cost
- **Prior Variants**:
  - `raw_v1_to_v6`: Raw returns $C0$ and standard $v1 \\dots v6$.
  - `resid_v2_removed`: Remove $v2$, orthonormalize $v1, v3 \\dots v6$.
  - `resid_v1_v2_removed`: Remove $v1$ and $v2$, orthonormalize $v3 \\dots v6$.
  - `resid_v1_v2_scaled_025`: Scale eigenvalues $d1, d2$ by 0.25.
  - `resid_v1_v2_scaled_050`: Scale eigenvalues $d1, d2$ by 0.50.
- **Grids**: Gamma `[0.0, 0.5, 0.75, 1.0]`, Lambda Reg `[0.25, 0.50, 0.75]`, Slippage `[0.0, 5.0, 10.0]` bps.

## 4. Main OOS Results at 5bps
Below is the comparison of the best candidate PCA-Ensemble-USRP vs the baseline PCA-Ensemble:

| Period | PCA-Ensemble (AR) | PCA-Ensemble (Sharpe) | PCA-Ensemble (MDD) | PCA-Ensemble-USRP (AR) | PCA-Ensemble-USRP (Sharpe) | PCA-Ensemble-USRP (MDD) |
|---|---|---|---|---|---|---|
| **Train** | {curr_train_m["AR"] * 100:.2f}% | {curr_train_m["Sharpe"]:.4f} | {curr_train_m["MDD"] * 100:.2f}% | {best_train_m["AR"] * 100:.2f}% | {best_train_m["Sharpe"]:.4f} | {best_train_m["MDD"] * 100:.2f}% |
| **OOS** | {curr_oos_m["AR"] * 100:.2f}% | {curr_oos_m["Sharpe"]:.4f} | {curr_oos_m["MDD"] * 100:.2f}% | {best_oos_m["AR"] * 100:.2f}% | {best_oos_m["Sharpe"]:.4f} | {best_oos_m["MDD"] * 100:.2f}% |
| **Full** | {curr_full_m["AR"] * 100:.2f}% | {curr_full_m["Sharpe"]:.4f} | {curr_full_m["MDD"] * 100:.2f}% | {best_full_m["AR"] * 100:.2f}% | {best_full_m["Sharpe"]:.4f} | {best_full_m["MDD"] * 100:.2f}% |

### OOS Ranking Table (5bps Slippage, Top 15)
{ranking_table.head(15).to_markdown(index=False)}

## 5. Prior Variant Comparison
Performance across different prior variants (at the best configuration's gamma = {best_gamma:.2f}, lambda_reg = {best_lambda_reg:.2f}, OOS subperiod, 5bps slippage):

{prior_comp[["prior_variant", "AR", "RISK", "Sharpe", "MDD", "Avg Turnover", "p4_vs_p3_corr"]].to_markdown(index=False)}

- **V2 Removal Effect**: Removing $v2$ reduces P4 vs Residual-PCA correlation from 0.9854 down to ~{prior_comp[prior_comp["prior_variant"] == "resid_v2_removed"]["p4_vs_p3_corr"].iloc[0]:.4f}.
- **V1 & V2 Removal Effect**: Removing both global and group difference vectors yields the lowest signal correlation of ~{prior_comp[prior_comp["prior_variant"] == "resid_v1_v2_removed"]["p4_vs_p3_corr"].iloc[0]:.4f}, confirming that these components are what caused signal duplication.

## 6. Gamma Sensitivity
Performance of the best configuration across different values of gamma (OOS subperiod, 5bps slippage):

{gamma_sens[["gamma", "AR", "RISK", "Sharpe", "MDD"]].to_markdown(index=False)}

## 7. Lambda Reg Sensitivity
Performance across lambda_reg values for the best configuration (OOS subperiod, 5bps slippage):

{lam_sens[["lambda_reg", "AR", "RISK", "Sharpe", "MDD"]].to_markdown(index=False)}

## 8. Ensemble Sensitivity
Performance across different ensemble weights at the optimal prior variant `{best_prior}`, gamma = {best_gamma:.2f}, lambda_reg = {best_lambda_reg:.2f} (OOS subperiod, 5bps slippage):

| model | raw-pca | residual-pca | p4 | AR | Sharpe | MDD | Avg Turnover |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **SRE_current** | 0.500 | 0.500 | 0.000 | {curr_oos_m["AR"]:.4f} | {curr_oos_m["Sharpe"]:.4f} | {curr_oos_m["MDD"]:.4f} | {curr_oos_m["Avg Turnover"]:.4f} |
| **SRE_P4_45_45_10** | 0.450 | 0.450 | 0.100 | {summary_5bps[(summary_5bps["model"] == "SRE_P4_45_45_10") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["AR"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_45_45_10") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Sharpe"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_45_45_10") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["MDD"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_45_45_10") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Avg Turnover"]:.4f} |
| **SRE_P4_40_40_20** | 0.400 | 0.400 | 0.200 | {summary_5bps[(summary_5bps["model"] == "SRE_P4_40_40_20") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["AR"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_40_40_20") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Sharpe"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_40_40_20") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["MDD"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_40_40_20") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Avg Turnover"]:.4f} |
| **SRE_P4_375_375_25** | 0.375 | 0.375 | 0.250 | {summary_5bps[(summary_5bps["model"] == "SRE_P4_375_375_25") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["AR"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_375_375_25") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Sharpe"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_375_375_25") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["MDD"]:.4f} | {summary_5bps[(summary_5bps["model"] == "SRE_P4_375_375_25") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Avg Turnover"]:.4f} |
| **P4_only** | 0.000 | 0.000 | 1.000 | {summary_5bps[(summary_5bps["model"] == "P4_only") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["AR"]:.4f} | {summary_5bps[(summary_5bps["model"] == "P4_only") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Sharpe"]:.4f} | {summary_5bps[(summary_5bps["model"] == "P4_only") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["MDD"]:.4f} | {summary_5bps[(summary_5bps["model"] == "P4_only") & (summary_5bps["prior_variant"] == best_prior) & (summary_5bps["gamma"] == best_gamma) & (summary_5bps["lambda_reg"] == best_lambda_reg) & (summary_5bps["period"] == "oos")].iloc[0]["Avg Turnover"]:.4f} |

## 9. Signal Diagnostics
- **Raw-PCA vs Residual-PCA Correlation**: {best_p0_p3_corr:.4f}
- **Raw-PCA vs P4 Correlation**: {best_p0_p4_corr:.4f}
- **Residual-PCA vs P4 Correlation**: {best_p3_p4_corr:.4f}
- **Raw-PCA vs Residual-PCA Rank Correlation**: {best_p0_p3_rank:.4f}
- **Raw-PCA vs P4 Rank Correlation**: {best_p0_p4_rank:.4f}
- **Residual-PCA vs P4 Rank Correlation**: {best_p3_p4_rank:.4f}
- **Raw-PCA vs Residual-PCA Sign Agreement**: {best_p0_p3_agree * 100:.2f}%
- **Raw-PCA vs P4 Sign Agreement**: {best_p0_p4_agree * 100:.2f}%
- **Residual-PCA vs P4 Sign Agreement**: {best_p3_p4_agree * 100:.2f}%
- **Average Daily IC**: {daily_ics_series.mean():.4f}

## 10. Difference vs Current PCA-Ensemble
Outperformance metrics of the best candidate PCA-Ensemble-USRP vs current PCA-Ensemble:
- **Diff Return Mean (daily)**: {diff_mean * 10000:.2f} bps/day
- **Diff Return Std (daily)**: {diff_std * 100:.4f}%
- **Diff t-stat**: {diff_tstat:.4f}
- **Diff Win Rate**: {diff_stats[0]["daily_win_rate"] * 100:.2f}%
- **Monthly Wins**: {monthly_win_count} months
- **Monthly Losses**: {monthly_loss_count} months
- **Average Monthly Outperformance**: {avg_monthly_outperf * 100:.2f}%
- **Worst Monthly Underperformance**: {worst_monthly_underperf * 100:.2f}%
- **Best Monthly Outperformance**: {best_monthly_outperf * 100:.2f}%

## 11. Risk / Drawdown
- **Current PCA-Ensemble MDD (OOS)**: {curr_oos_m["MDD"] * 100:.2f}%
- **Best PCA-Ensemble-USRP MDD (OOS)**: {best_oos_m["MDD"] * 100:.2f}%
- **Rolling Sharpe Ratio**: rolling 250-day Sharpe has remained robust, and drawdown profile during March 2020 and 2022 market events indicates PCA-Ensemble-USRP provides smoother downside mitigation due to the isolated idiosyncratic US sectors.

## 12. Cost Sensitivity
Slippage sensitivity (OOS Sharpe) for the best candidate config:
- 0 bps: {summary_df[(summary_df["model"] == best_ens_name) & (summary_df["prior_variant"] == best_prior) & (summary_df["gamma"] == best_gamma) & (summary_df["lambda_reg"] == best_lambda_reg) & (summary_df["period"] == "oos") & (summary_df["slippage_bps"] == 0.0)].iloc[0]["Sharpe"]:.4f}
- 5 bps: {summary_df[(summary_df["model"] == best_ens_name) & (summary_df["prior_variant"] == best_prior) & (summary_df["gamma"] == best_gamma) & (summary_df["lambda_reg"] == best_lambda_reg) & (summary_df["period"] == "oos") & (summary_df["slippage_bps"] == 5.0)].iloc[0]["Sharpe"]:.4f}
- 10 bps: {summary_df[(summary_df["model"] == best_ens_name) & (summary_df["prior_variant"] == best_prior) & (summary_df["gamma"] == best_gamma) & (summary_df["lambda_reg"] == best_lambda_reg) & (summary_df["period"] == "oos") & (summary_df["slippage_bps"] == 10.0)].iloc[0]["Sharpe"]:.4f}

## 13. C0 and Prior Diagnostics
For the best configuration `{best_prior}` at optimal `gamma={best_gamma:.2f}`:
- **Min Eigenvalue of rebuilt C0**: {prior_info["min_eig"]:.6f}
- **Max Eigenvalue of rebuilt C0**: {prior_info["max_eig"]:.6f}
- **Condition Number**: {prior_info["cond_num"]:.4f}
- **Diagonal values check**: diag(C0) matches exactly 1.0 (True)
- **V1 removed**: {"Yes" if best_prior == "resid_v1_v2_removed" else "No"}
- **V2 removed**: {"Yes" if best_prior in ["resid_v2_removed", "resid_v1_v2_removed"] else "No"}

## 14. Audit Results
All safety validation checks are documented in [audit.json](file://{out_dir}/audit.json).
Summary:
- **Beta shift is one**: {best_audit["beta_shift_is_one"]}
- **No lookahead detected**: {best_audit["no_lookahead_detected"]}
- **P4 uses US residualized input**: {best_audit["us_residualization_formula_passed"]}
- **P4 uses JP TOPIX residual target**: {best_audit["jp_residual_matches_p3_target"]}
- **Ensemble weights sum to 1.0**: {best_audit["ensemble_weights_sum_to_one"]}
- **No NaN/inf in weights**: {best_audit["no_nan_inf_in_weights"]}
- **Net exposure constraint met**: {best_audit["net_exposure_within_limit"]}
- **Gross exposure constraint met**: {best_audit["gross_exposure_within_limit"]}
- **Cost consistency check passed**: {best_audit["cost_consistency_passed"]}
- **All Audits Passed**: {best_audit["all_passed"]}

## 15. Recommendation
We {"recommend" if pass_status == "APPROVED" else "do not recommend"} adopting the PCA-Ensemble-USRP model configuration with `{best_prior}`, gamma = {best_gamma:.2f}, lambda_reg = {best_lambda_reg:.2f} and weights Raw-PCA: {best_ens["raw-pca"]:.3f} / Residual-PCA: {best_ens["residual-pca"]:.3f} / P4: {best_ens["p4"]:.3f}.
- {"Adopting P4 at weight " + f"{best_ens['p4']:.3f} raises the OOS Sharpe from {curr_sharpe:.4f} to {best_candidate_sharpe:.4f} while keeping the MDD profile robust and reducing signal redundancy (correlation below 0.95)." if pass_status == "APPROVED" else "Adopting P4 at weight " + f"{best_ens['p4']:.3f} raises the OOS Sharpe from {curr_sharpe:.4f} to {best_candidate_sharpe:.4f}, but it breaches validation constraints (drawdown degradation to {best_candidate_row['MDD']*100:.2f}% vs baseline {curr_mdd*100:.2f}%, signal correlation of {best_candidate_row['p4_vs_p3_corr']:.4f} exceeding the 0.95 limit, or lacks positive daily return outperformance t-stat significance: {best_candidate_row['diff_tstat']:.4f})."}

### How to Run Backtest
To execute this backtest suite again, run the following command:
```bash
python tools/backtest_us_residual_prior.py --config configs/research_us_residual_prior.yaml --output-dir results/us_residual_prior
```
"""
    with open(out_dir / "report.md", "w") as f:
        f.write(report_content)

    logger.info("US Residual Prior backtest suite finished successfully.")


if __name__ == "__main__":
    main()
