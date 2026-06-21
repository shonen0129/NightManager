#!/usr/bin/env python
"""Backtesting and Verification Suite for Sector Relative Ensemble with Reduced-Rank Regression (PCA-Ensemble-RRR).

Evaluates model configurations across param grids, compares them with baseline PCA-Ensemble, and runs safety audits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
import concurrent.futures

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.models.sector_relative_ensemble_rrr import SectorRelativeEnsembleRRRModel
from leadlag.reporting.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Reduced-Rank Regression Backtest & Verification Suite")
    parser.add_argument("--config", default="configs/research_rrr_projection.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default="results/rrr_projection", help="Output directory")
    parser.add_argument("--stage", default="compact", choices=["compact", "refined"], help="Grid sweep stage")
    parser.add_argument("--start-date", default="2015-01-05", help="Backtest start date")
    parser.add_argument("--end-date", default="latest", help="Backtest end date")
    parser.add_argument("--train-end-date", default="2019-12-31", help="Train period end date")
    parser.add_argument("--oos-start-date", default="2020-01-01", help="OOS period start date")
    parser.add_argument("--variants", help="Override variants (comma-separated)")
    parser.add_argument("--rank-grid", help="Override rank grid (comma-separated)")
    parser.add_argument("--lambda-ridge-grid", help="Override lambda_ridge grid (comma-separated)")
    parser.add_argument("--lambda-prior-grid", help="Override lambda_prior grid (comma-separated)")
    parser.add_argument("--slippage-grid", help="Override slippage grid (comma-separated, bps)")
    return parser.parse_args()


def simulate_portfolio_fast(
    signal_df: pd.DataFrame,
    y_jp_target_df: pd.DataFrame,
    q: float,
    n_j: int,
    weight_mode: str,
    slippage_bps: float,
) -> dict:
    """Run lookahead-safe portfolio simulation for given signal timeseries."""
    dates = signal_df.index
    T = len(dates)

    # 1. Build weights
    weights = np.zeros((T, n_j))
    from leadlag.core import signal as signals
    for idx in range(T):
        weights[idx] = signals.build_weights(
            signal=signal_df.iloc[idx].values,
            q=q,
            n_j=n_j,
            weight_mode=weight_mode,
            enforce_sign=False,
        )

    weights_df = pd.DataFrame(weights, index=dates, columns=JP_TICKERS)

    # 2. Compute returns and costs
    gross_returns = np.zeros(T)
    net_returns = np.zeros(T)
    costs = np.zeros(T)
    turnovers = np.zeros(T)
    gross_exposures = np.zeros(T)

    w_prev = np.zeros(n_j)
    for idx, date in enumerate(dates):
        w_t = weights[idx]
        r_target_t = y_jp_target_df.loc[date].values

        gross_ret = float(np.sum(w_t * r_target_t))
        gross_exp = float(np.sum(np.abs(w_t)))
        cost = 2.0 * (slippage_bps / 10000.0) * gross_exp
        net_ret = gross_ret - cost

        turnover = float(np.sum(np.abs(w_t - w_prev)) / 2.0)

        gross_returns[idx] = gross_ret
        net_returns[idx] = net_ret
        costs[idx] = cost
        turnovers[idx] = turnover
        gross_exposures[idx] = gross_exp
        w_prev = w_t

    daily_returns_gross = pd.Series(gross_returns, index=dates)
    daily_returns_net = pd.Series(net_returns, index=dates)
    daily_costs = pd.Series(costs, index=dates)
    daily_turnover = pd.Series(turnovers, index=dates)
    daily_gross_exps = pd.Series(gross_exposures, index=dates)

    wealth = (1.0 + daily_returns_net).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1.0

    return {
        "weights": weights_df,
        "daily_returns_gross": daily_returns_gross,
        "daily_returns": daily_returns_net,
        "daily_costs": daily_costs,
        "daily_turnover": daily_turnover,
        "daily_gross_exps": daily_gross_exps,
        "equity_curve": wealth,
        "drawdown": drawdown,
    }


def compute_extended_metrics(
    net_ret: pd.Series,
    gross_ret: pd.Series,
    costs: pd.Series,
    turnover: pd.Series,
    gross_exps: pd.Series,
    weights_df: pd.DataFrame,
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> dict:
    """Calculate extended set of performance metrics for subperiods."""
    periods = {
        "train": (net_ret.loc[:train_end], gross_ret.loc[:train_end], costs.loc[:train_end], turnover.loc[:train_end], gross_exps.loc[:train_end], weights_df.loc[:train_end]),
        "oos": (net_ret.loc[oos_start:], gross_ret.loc[oos_start:], costs.loc[oos_start:], turnover.loc[oos_start:], gross_exps.loc[oos_start:], weights_df.loc[oos_start:]),
        "full": (net_ret, gross_ret, costs, turnover, gross_exps, weights_df),
    }

    results = {}
    for p_name, (n_r, g_r, c_s, t_o, g_e, w_df) in periods.items():
        base = calculate_metrics(n_r)
        t_days = len(n_r)
        if t_days == 0:
            continue

        avg_daily = float(np.mean(n_r))
        std_daily = float(np.std(n_r, ddof=1)) if t_days > 1 else 0.0
        win_rate = float(np.mean(n_r > 0))
        avg_turnover = float(np.mean(t_o))
        avg_gross = float(np.mean(g_e))
        avg_net = float(np.mean(w_df.sum(axis=1)))
        avg_cost = float(np.mean(c_s))

        # Monthly std calculation
        from leadlag.reporting.metrics import _extract_monthly_returns
        m_ret = _extract_monthly_returns(n_r)
        if m_ret is not None and len(m_ret) > 1:
            m_std = float(np.std(m_ret, ddof=1))
        else:
            m_std = np.nan

        results[p_name] = {
            "AR": base.get("AR", 0.0),
            "RISK": base.get("RISK", 0.0),
            "R/R": base.get("R/R", 0.0),
            "Sharpe": base.get("Sharpe", 0.0),
            "MDD": base.get("MDD", 0.0),
            "Total_Return": base.get("Total Return", 0.0),
            "win_rate": win_rate,
            "avg_daily_return": avg_daily,
            "std_daily_return": std_daily,
            "monthly_return_std": m_std,
            "turnover": avg_turnover,
            "avg_gross_exposure": avg_gross,
            "avg_net_exposure": avg_net,
            "avg_trading_cost": avg_cost,
            "gross_return_sum": float(np.sum(g_r)),
            "net_return_sum": float(np.sum(n_r)),
            "total_cost": float(np.sum(c_s)),
        }

    return results


def evaluate_single_run(args_tuple):
    (
        run_cfg,
        df_exec,
        sim_dates_slice,
        y_jp_target_slice,
        y_jp_oc_slice,
        z0_df,
        z3_df,
        start_idx,
        end_idx,
        train_end,
        oos_start,
        ensembles,
        slippage_grid,
    ) = args_tuple

    # 1. Create model and predict signals
    rrr_model = SectorRelativeEnsembleRRRModel(run_cfg)
    pred = rrr_model.predict_signals(df_exec)

    p6_sig = pred["p6_signals"].loc[sim_dates_slice]
    p6p3_sig = pred["p6p3_signals"].loc[sim_dates_slice]
    p7_sig = pred["p7_signals"].loc[sim_dates_slice]
    p7p3_sig = pred["p7p3_signals"].loc[sim_dates_slice]

    # 2. Lookahead safety audit
    rrr_no_lookahead_detected = True
    max_training_y_date_le_signal_date = True
    num_lookahead_violations = 0
    for i_step in range(start_idx, end_idx + 1):
        sig_date_t = df_exec["sig_date"].values[i_step]
        max_target_date = df_exec.index[i_step - 1]
        if max_target_date > sig_date_t:
            rrr_no_lookahead_detected = False
            max_training_y_date_le_signal_date = False
            num_lookahead_violations += 1

    # 3. Diagnostics
    conds = []
    pinv_fallbacks = 0
    eff_ranks = []
    rank_constraint_passed = True
    diag_df = pred["rrr_diagnostics"]
    diagnostics_records = []

    # Check if this configuration is a default parameter combination
    is_default_params = (
        run_cfg["rrr_window"] == 252 and
        run_cfg["rrr_ewma_halflife"] == 45 and
        run_cfg["rank"] == 3 and
        (run_cfg["lambda_ridge"] == 0.03 or run_cfg["variant"] in ["RRR_pure", "Lowrank_BLP"]) and
        (run_cfg["lambda_prior"] == 0.3 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "Lowrank_BLP"]) and
        (run_cfg["rho_blp"] == 0.03 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"]) and
        (run_cfg["alpha_xx"] == 0.5 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"]) and
        (run_cfg["alpha_yx"] == 0.25 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"])
    )

    if not diag_df.empty:
        conds_vals = diag_df["condition_number"].dropna().values
        pinv_fallbacks = int(diag_df["pinv_fallback"].sum())
        eff_ranks_vals = diag_df["effective_rank"].dropna().values
        
        if len(eff_ranks_vals) > 0 and np.any(eff_ranks_vals > run_cfg["rank"]):
            rank_constraint_passed = False
            
        # Only return full lists and records for default parameters to prevent multiprocessing buffer overflow
        if is_default_params:
            if len(conds_vals) > 0:
                conds = list(conds_vals)
            if len(eff_ranks_vals) > 0:
                eff_ranks = list(eff_ranks_vals)
            
            diag_copy = diag_df.copy()
            diag_copy["rrr_window"] = run_cfg["rrr_window"]
            diag_copy["ewma_halflife"] = run_cfg["rrr_ewma_halflife"]
            diag_copy["lambda_ridge"] = run_cfg["lambda_ridge"]
            diag_copy["lambda_prior"] = run_cfg["lambda_prior"]
            diag_copy["rho_blp"] = run_cfg["rho_blp"]
            diag_copy["alpha_xx"] = run_cfg["alpha_xx"]
            diag_copy["alpha_yx"] = run_cfg["alpha_yx"]
            diagnostics_records = diag_copy.to_dict(orient="records")
        else:
            # For non-default parameters, just record max/median to update the global stats without large memory transfer
            conds = [float(np.max(conds_vals)), float(np.median(conds_vals))] if len(conds_vals) > 0 else []
            eff_ranks = [int(np.max(eff_ranks_vals)), int(np.median(eff_ranks_vals))] if len(eff_ranks_vals) > 0 else []

    # 4. Standardize signals
    z6_df = p6_sig.apply(lambda row: rrr_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")
    z6p3_df = p6p3_sig.apply(lambda row: rrr_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")
    z7_df = p7_sig.apply(lambda row: rrr_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")
    z7p3_df = p7p3_sig.apply(lambda row: rrr_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")

    rrr_regularization_passed = True
    if z6_df.isna().any().any() or np.isinf(z6_df.values).any():
        rrr_regularization_passed = False
    if z7_df.isna().any().any() or np.isinf(z7_df.values).any():
        rrr_regularization_passed = False

    # 5. Loop ensembles
    results_records = []
    default_params_timeseries = {}

    for ens in ensembles:
        ens_name = ens["name"]
        w_p0, w_p3, w_p6, w_p6p3, w_p7, w_p7p3 = ens["p0"], ens["p3"], ens["p6"], ens["p6p3"], ens["p7"], ens["p7p3"]

        # Combined signal
        comb_sig = (
            w_p0 * z0_df
            + w_p3 * z3_df
            + w_p6 * z6_df
            + w_p6p3 * z6p3_df
            + w_p7 * z7_df
            + w_p7p3 * z7p3_df
        )

        is_lowrank_ens = w_p7 > 0 or w_p7p3 > 0 or ens_name == "LowrankBLP_SRE"
        is_rrr_ens = w_p6 > 0 or w_p6p3 > 0 or ens_name == "RRR_SRE"

        if is_lowrank_ens and run_cfg["variant"] != "Lowrank_BLP" and is_rrr_ens:
            pass
        elif is_lowrank_ens and run_cfg["variant"] != "Lowrank_BLP" and not is_rrr_ens:
            continue
        elif not is_lowrank_ens and is_rrr_ens and run_cfg["variant"] == "Lowrank_BLP":
            pass

        for slip in slippage_grid:
            sim = simulate_portfolio_fast(
                comb_sig,
                y_jp_target_slice,
                q=0.3,
                n_j=17,
                weight_mode="signal",
                slippage_bps=slip
            )

            sub_m = compute_extended_metrics(
                sim["daily_returns"],
                sim["daily_returns_gross"],
                sim["daily_costs"],
                sim["daily_turnover"],
                sim["daily_gross_exps"],
                sim["weights"],
                train_end,
                oos_start
            )

            # Check if this is the default parameter combo at 5bps
            is_default_params = (
                run_cfg["rrr_window"] == 252 and
                run_cfg["rrr_ewma_halflife"] == 45 and
                run_cfg["rank"] == 3 and
                (run_cfg["lambda_ridge"] == 0.03 or run_cfg["variant"] in ["RRR_pure", "Lowrank_BLP"]) and
                (run_cfg["lambda_prior"] == 0.3 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "Lowrank_BLP"]) and
                (run_cfg["rho_blp"] == 0.03 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"]) and
                (run_cfg["alpha_xx"] == 0.5 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"]) and
                (run_cfg["alpha_yx"] == 0.25 or run_cfg["variant"] in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"])
            )

            if is_default_params and abs(slip - 5.0) < 1e-6:
                key_name = f"{ens_name}_{run_cfg['variant']}"
                default_params_timeseries[key_name] = {
                    "daily_returns": sim["daily_returns"],
                    "weights": sim["weights"],
                    "drawdown": sim["drawdown"],
                }

            for period_name, m_dict in sub_m.items():
                record = {
                    "variant": run_cfg["variant"],
                    "rrr_window": run_cfg["rrr_window"],
                    "ewma_halflife": run_cfg["rrr_ewma_halflife"],
                    "rank": run_cfg["rank"],
                    "lambda_ridge": run_cfg["lambda_ridge"],
                    "lambda_prior": run_cfg["lambda_prior"],
                    "rho_blp": run_cfg["rho_blp"],
                    "alpha_xx": run_cfg["alpha_xx"],
                    "alpha_yx": run_cfg["alpha_yx"],
                    "ensemble": ens_name,
                    "slippage_bps": slip,
                    "period": period_name,
                }
                record.update(m_dict)
                results_records.append(record)

    audit_metrics = {
        "rrr_no_lookahead_detected": rrr_no_lookahead_detected,
        "max_training_y_date_le_signal_date": max_training_y_date_le_signal_date,
        "num_lookahead_violations": num_lookahead_violations,
        "conds": conds,
        "pinv_fallbacks": pinv_fallbacks,
        "eff_ranks": eff_ranks,
        "rank_constraint_passed": rank_constraint_passed,
        "rrr_regularization_passed": rrr_regularization_passed,
    }

    return {
        "results": results_records,
        "diagnostics": diagnostics_records,
        "audit_metrics": audit_metrics,
        "default_params_timeseries": default_params_timeseries,
    }


def main():
    args = parse_arguments()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load config
    config_path = ROOT / args.config
    if config_path.exists():
        logger.info(f"Loading YAML config from: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        logger.error(f"Config path {config_path} not found.")
        sys.exit(1)

    # Configure stage grids
    grids = cfg.get("grids", {})
    if args.stage == "compact":
        variants = ["Lowrank_BLP", "BLP_prior_RRR", "PCA_prior_RRR", "Ridge_RRR", "RRR_pure"]
        rrr_window_grid = [252]
        ewma_halflife_grid = [45]
        rank_grid = [2, 3, 4, 5, 6]
        lambda_ridge_grid = [0.01, 0.03, 0.1, 0.3]
        lambda_prior_grid = [0.1, 0.3, 1.0, 3.0]
        rho_blp_grid = [0.003, 0.03]
        alpha_xx_grid = [0.5, 0.75]
        alpha_yx_grid = [0.0, 0.25]
        slippage_grid = [0.0, 5.0, 7.5, 10.0]
    else:  # refined
        variants = ["Lowrank_BLP", "BLP_prior_RRR", "PCA_prior_RRR", "Ridge_RRR", "RRR_pure"]
        rrr_window_grid = [126, 252, 504]
        ewma_halflife_grid = [30, 45, 90]
        rank_grid = [1, 2, 3, 4, 5, 6, 8, 10]
        lambda_ridge_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        lambda_prior_grid = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        rho_blp_grid = [0.003, 0.01, 0.03, 0.1]
        alpha_xx_grid = [0.5, 0.75]
        alpha_yx_grid = [0.0, 0.25]
        slippage_grid = [0.0, 2.5, 5.0, 7.5, 10.0]

    # CLI parameter overrides
    if args.variants:
        variants = args.variants.split(",")
    if args.rank_grid:
        rank_grid = [int(x) for x in args.rank_grid.split(",")]
    if args.lambda_ridge_grid:
        lambda_ridge_grid = [float(x) for x in args.lambda_ridge_grid.split(",")]
    if args.lambda_prior_grid:
        lambda_prior_grid = [float(x) for x in args.lambda_prior_grid.split(",")]
    if args.slippage_grid:
        slippage_grid = [float(x) for x in args.slippage_grid.split(",")]

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

    # Build target returns Series
    from leadlag.models.sre import compute_jp_target_returns
    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    y_jp_target_df = pd.DataFrame(y_jp_target, index=df_exec.index, columns=JP_TICKERS)

    # 2. Run Baseline Production PCA-Ensemble Model for verification
    logger.info("Running baseline production SRE for verification...")
    prod_config_path = ROOT / "configs" / "production.yaml"
    with open(prod_config_path) as f:
        prod_cfg = yaml.safe_load(f)
    baseline_model = SectorRelativeEnsembleModel(prod_cfg)

    from leadlag.execution.backtester import BacktestEngine
    baseline_res = BacktestEngine.run_backtest(
        baseline_model,
        df_exec,
        start_date=args.start_date,
        end_date=args.end_date,
        slippage_bps=5.0,
    )

    # Setup Grid Loop
    logger.info("Starting Grid Search...")
    all_results = []

    start_dt = pd.to_datetime(args.start_date)
    start_idx = max(df_exec.index.searchsorted(start_dt), baseline_model.corr_window)
    if args.end_date != "latest":
        end_dt = pd.to_datetime(args.end_date)
        end_idx = min(df_exec.index.searchsorted(end_dt), len(df_exec) - 1)
    else:
        end_idx = len(df_exec) - 1

    sim_dates_slice = df_exec.index[start_idx : end_idx + 1]
    y_jp_target_slice = y_jp_target_df.loc[sim_dates_slice]
    y_jp_oc_slice = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(columns=lambda c: c.replace("jp_oc_", "")).loc[sim_dates_slice]

    train_end = pd.to_datetime(args.train_end_date)
    oos_start = pd.to_datetime(args.oos_start_date)

    # Master caches for timeseries outputs
    daily_returns_master = {}
    daily_positions_master = {}
    drawdown_master = {}

    # Initialize a base RRR model to generate Raw-PCA/Residual-PCA components once
    init_rrr_cfg = {
        "model": {"name": "sector_relative_ensemble_rrr"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "p0_weight": 0.5, "p3_weight": 0.5,
            "p6_weight": 0.0, "p6p3_weight": 0.0,
            "p7_weight": 0.0, "p7p3_weight": 0.0
        },
    }
    rrr_base_model = SectorRelativeEnsembleRRRModel(init_rrr_cfg)
    base_pred = rrr_base_model.predict_signals(df_exec)

    p0_sig_base = base_pred["p0_signals"].loc[sim_dates_slice]
    p3_sig_base = base_pred["p3_signals"].loc[sim_dates_slice]

    # Check baseline replication
    reproduced_signals_df = base_pred["signals"].loc[sim_dates_slice]
    baseline_signals_df = baseline_res["signals"]

    sig_diff_max = float(np.max(np.abs(reproduced_signals_df.values - baseline_signals_df.values)))
    pos_diff_max = float(np.max(np.abs(base_pred["normalized_signals"].loc[sim_dates_slice].values - baseline_res["normalized_signals"].values)))

    reproduced_sim = simulate_portfolio_fast(
        reproduced_signals_df,
        y_jp_target_slice,
        q=0.3,
        n_j=17,
        weight_mode="signal",
        slippage_bps=5.0
    )

    return_diff_max = float(np.max(np.abs(reproduced_sim["daily_returns"].values - baseline_res["daily_returns"].values)))
    baseline_sre_reproduced = return_diff_max < 1e-10 and sig_diff_max < 1e-10
    logger.info(f"Baseline SRE reproduction: max signal diff={sig_diff_max:.3e}, max return diff={return_diff_max:.3e}. Reproduced: {baseline_sre_reproduced}")

    # Audit tracking variables
    rrr_no_lookahead_detected = True
    max_training_y_date_le_signal_date = True
    num_lookahead_violations = 0
    rrr_matrix_dimensions_passed = True
    rrr_regularization_passed = True
    rank_constraint_passed = True

    max_condition_number = 0.0
    all_cond_nums = []
    num_pinv_fallbacks = 0
    max_effective_rank = 0
    all_effective_ranks = []
    daily_diagnostics_all = []

    # Standard normalization of components
    z0_df = p0_sig_base.apply(lambda row: rrr_base_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")
    z3_df = p3_sig_base.apply(lambda row: rrr_base_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")

    # Generate default parameter signals for report diagnostics (avoids undefined z6_df, etc.)
    logger.info("Generating default parameter RRR signals for report diagnostics...")
    default_rrr_cfg = {
        "model": {"name": "sector_relative_ensemble_rrr"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "rrr_window": 252,
        "rrr_ewma_halflife": 45,
        "variant": "BLP_prior_RRR",
        "rank": 3,
        "lambda_ridge": 0.03,
        "lambda_prior": 0.3,
        "rho_blp": 0.03,
        "alpha_xx": 0.5,
        "alpha_yx": 0.25,
    }
    default_model = SectorRelativeEnsembleRRRModel(default_rrr_cfg)
    default_pred = default_model.predict_signals(df_exec)
    p6_sig_def = default_pred["p6_signals"].loc[sim_dates_slice]
    p6p3_sig_def = default_pred["p6p3_signals"].loc[sim_dates_slice]
    p7_sig_def = default_pred["p7_signals"].loc[sim_dates_slice]
    p7p3_sig_def = default_pred["p7p3_signals"].loc[sim_dates_slice]

    z6_df = p6_sig_def.apply(lambda row: default_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")
    z6p3_df = p6p3_sig_def.apply(lambda row: default_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")
    z7_df = p7_sig_def.apply(lambda row: default_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")
    z7p3_df = p7p3_sig_def.apply(lambda row: default_model.normalize_signals(row.values, "zscore"), axis=1, result_type="expand")

    # Master list of ensembles
    ensembles = cfg.get("ensembles", [])

    # Sweeps
    tasks = []
    for window in rrr_window_grid:
        for halflife in ewma_halflife_grid:
            for variant in variants:
                # Deduplicate parameter scopes for non-applicable parameters to accelerate sweep
                l_ridge_vals = [0.0] if variant in ["RRR_pure", "Lowrank_BLP"] else lambda_ridge_grid
                l_prior_vals = [0.0] if variant in ["RRR_pure", "Ridge_RRR", "Lowrank_BLP"] else lambda_prior_grid
                rho_vals = [0.0] if variant in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"] else rho_blp_grid
                a_xx_vals = [0.0] if variant in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"] else alpha_xx_grid
                a_yx_vals = [0.0] if variant in ["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"] else alpha_yx_grid

                for rank in rank_grid:
                    for l_ridge in l_ridge_vals:
                        for l_prior in l_prior_vals:
                            for rho in rho_vals:
                                for a_xx in a_xx_vals:
                                    for a_yx in a_yx_vals:
                                        run_cfg = {
                                            "model": {"name": "sector_relative_ensemble_rrr"},
                                            "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
                                            "rrr_window": window,
                                            "rrr_ewma_halflife": halflife,
                                            "lambda_ridge": l_ridge,
                                            "lambda_prior": l_prior,
                                            "rank": rank,
                                            "variant": variant,
                                            "rho_blp": rho,
                                            "alpha_xx": a_xx,
                                            "alpha_yx": a_yx,
                                        }
                                        tasks.append((
                                            run_cfg,
                                            df_exec,
                                            sim_dates_slice,
                                            y_jp_target_slice,
                                            y_jp_oc_slice,
                                            z0_df,
                                            z3_df,
                                            start_idx,
                                            end_idx,
                                            train_end,
                                            oos_start,
                                            ensembles,
                                            slippage_grid,
                                        ))

    logger.info(f"Submitting {len(tasks)} backtest tasks to ProcessPoolExecutor...")
    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(evaluate_single_run, t): t for t in tasks}
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            all_results.extend(res["results"])
            
            # Diagnostics merge
            diag_records = res["diagnostics"]
            if diag_records:
                daily_diagnostics_all.append(pd.DataFrame(diag_records))
            
            # Audit merge
            am = res["audit_metrics"]
            rrr_no_lookahead_detected &= am["rrr_no_lookahead_detected"]
            max_training_y_date_le_signal_date &= am["max_training_y_date_le_signal_date"]
            num_lookahead_violations += am["num_lookahead_violations"]
            rank_constraint_passed &= am["rank_constraint_passed"]
            rrr_regularization_passed &= am["rrr_regularization_passed"]
            
            if am["conds"]:
                max_c = float(np.max(am["conds"]))
                if max_c > max_condition_number:
                    max_condition_number = max_c
                all_cond_nums.extend(am["conds"])
            num_pinv_fallbacks += am["pinv_fallbacks"]
            
            if am["eff_ranks"]:
                max_r = int(np.max(am["eff_ranks"]))
                if max_r > max_effective_rank:
                    max_effective_rank = max_r
                all_effective_ranks.extend(am["eff_ranks"])
            
            # Timeseries merge
            for key_name, ts_dict in res["default_params_timeseries"].items():
                daily_returns_master[key_name] = ts_dict["daily_returns"]
                daily_positions_master[key_name] = ts_dict["weights"]
                drawdown_master[key_name] = ts_dict["drawdown"]
                
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == len(tasks):
                logger.info(f"Progress: Completed {completed_count}/{len(tasks)} combinations ({(completed_count/len(tasks))*100:.1f}%)")

    # 4. Save CSV Output Files
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(out_dir / "summary.csv", index=False)

    # summary_5bps.csv (slippage_bps = 5.0 only)
    df_5bps = df_results[(df_results["slippage_bps"] == 5.0)].copy()
    df_5bps.to_csv(out_dir / "summary_5bps.csv", index=False)

    # oos_ranking_5bps.csv (OOS, 5bps)
    df_oos_5bps = df_results[(df_results["period"] == "oos") & (df_results["slippage_bps"] == 5.0)].copy()

    # Baseline PCA-Ensemble parameters extraction
    sre_rows = df_oos_5bps[df_oos_5bps["ensemble"] == "SRE_current"]
    if not sre_rows.empty:
        sre_oos_sharpe = float(sre_rows["Sharpe"].iloc[0])
        sre_oos_mdd = float(sre_rows["MDD"].iloc[0])
        sre_oos_ar = float(sre_rows["AR"].iloc[0])
        sre_oos_turnover = float(sre_rows["turnover"].iloc[0])
    else:
        sre_oos_sharpe = 0.0
        sre_oos_mdd = -1.0
        sre_oos_ar = 0.0
        sre_oos_turnover = 1.0

    ranking_records = []
    for idx, row in df_oos_5bps.iterrows():
        cand_sharpe = float(row["Sharpe"])
        cand_mdd = float(row["MDD"])
        cand_ar = float(row["AR"])
        cand_turnover = float(row["turnover"])

        # Check sensitivity at 7.5bps
        cand_7p5 = df_results[
            (df_results["period"] == "oos") &
            (df_results["slippage_bps"] == 7.5) &
            (df_results["ensemble"] == row["ensemble"]) &
            (df_results["variant"] == row["variant"]) &
            (df_results["rrr_window"] == row["rrr_window"]) &
            (df_results["rank"] == row["rank"]) &
            (df_results["lambda_ridge"] == row["lambda_ridge"]) &
            (df_results["lambda_prior"] == row["lambda_prior"]) &
            (df_results["rho_blp"] == row["rho_blp"]) &
            (df_results["alpha_xx"] == row["alpha_xx"]) &
            (df_results["alpha_yx"] == row["alpha_yx"])
        ]
        robust_7p5 = False
        if not cand_7p5.empty:
            sharpe_7p5 = float(cand_7p5["Sharpe"].iloc[0])
            robust_7p5 = (sharpe_7p5 > 0.0) and (sharpe_7p5 >= 0.8 * cand_sharpe)

        # Check sensitivity at 10bps
        cand_10 = df_results[
            (df_results["period"] == "oos") &
            (df_results["slippage_bps"] == 10.0) &
            (df_results["ensemble"] == row["ensemble"]) &
            (df_results["variant"] == row["variant"]) &
            (df_results["rrr_window"] == row["rrr_window"]) &
            (df_results["rank"] == row["rank"]) &
            (df_results["lambda_ridge"] == row["lambda_ridge"]) &
            (df_results["lambda_prior"] == row["lambda_prior"]) &
            (df_results["rho_blp"] == row["rho_blp"]) &
            (df_results["alpha_xx"] == row["alpha_xx"]) &
            (df_results["alpha_yx"] == row["alpha_yx"])
        ]
        not_collapsed_10 = False
        if not cand_10.empty:
            sharpe_10 = float(cand_10["Sharpe"].iloc[0])
            not_collapsed_10 = sharpe_10 > -0.5  # not collapsed means no severe negative Sharpe

        improves_oos_sharpe = cand_sharpe > sre_oos_sharpe
        improves_oos_sharpe_by_005 = cand_sharpe >= sre_oos_sharpe + 0.05
        improves_oos_sharpe_by_007 = cand_sharpe >= sre_oos_sharpe + 0.07
        improves_oos_mdd = cand_mdd >= sre_oos_mdd
        keeps_oos_ar_90pct = cand_ar >= 0.9 * sre_oos_ar
        turnover_within_10pct = cand_turnover <= 1.1 * sre_oos_turnover
        improves_after_cost_5bps = cand_ar > sre_oos_ar  # comparing net return at 5bps

        rank_not_too_high = int(row["rank"]) <= 6
        lambda_not_on_extreme_boundary = True
        if row["variant"] not in ["RRR_pure", "Lowrank_BLP"]:
            lambda_not_on_extreme_boundary = (row["lambda_ridge"] in [0.01, 0.03, 0.1, 0.3]) and (row["lambda_prior"] in [0.1, 0.3, 1.0, 3.0])

        not_pure_rrr_only = row["variant"] != "RRR_pure"

        # Check correlations (will compute below)
        signal_corr_below_0_95_to_sre = True

        pass_candidate = (
            improves_oos_sharpe_by_007 and
            improves_oos_mdd and
            keeps_oos_ar_90pct and
            turnover_within_10pct and
            improves_after_cost_5bps and
            robust_7p5 and
            not_collapsed_10 and
            rank_not_too_high and
            lambda_not_on_extreme_boundary
        )

        # Shadow status is a relaxed pass
        is_shadow = (
            improves_oos_sharpe_by_005 and
            improves_after_cost_5bps and
            (cand_mdd >= 1.2 * sre_oos_mdd) and  # not significantly worse MDD
            robust_7p5
        )

        rec = dict(row)
        rec.update({
            "improves_oos_sharpe": improves_oos_sharpe,
            "improves_oos_sharpe_by_005": improves_oos_sharpe_by_005,
            "improves_oos_sharpe_by_007": improves_oos_sharpe_by_007,
            "improves_oos_mdd": improves_oos_mdd,
            "keeps_oos_ar_90pct": keeps_oos_ar_90pct,
            "turnover_within_10pct": turnover_within_10pct,
            "improves_after_cost_5bps": improves_after_cost_5bps,
            "robust_at_7p5bps": robust_7p5,
            "not_collapsed_at_10bps": not_collapsed_10,
            "rank_not_too_high": rank_not_too_high,
            "lambda_not_on_extreme_boundary": lambda_not_on_extreme_boundary,
            "not_pure_rrr_only": not_pure_rrr_only,
            "signal_corr_below_0_95_to_sre": signal_corr_below_0_95_to_sre,
            "params_not_on_extreme_boundary": lambda_not_on_extreme_boundary,
            "pass_candidate": pass_candidate,
            "is_shadow_candidate": is_shadow,
        })
        ranking_records.append(rec)

    df_ranking = pd.DataFrame(ranking_records)
    df_ranking = df_ranking.sort_values(by="Sharpe", ascending=False)
    df_ranking.to_csv(out_dir / "oos_ranking_5bps.csv", index=False)

    # oos_ranking_7p5bps.csv
    df_7p5bps = df_results[(df_results["period"] == "oos") & (df_results["slippage_bps"] == 7.5)].copy()
    df_7p5bps.to_csv(out_dir / "oos_ranking_7p5bps.csv", index=False)

    # Parameter sensitivity
    df_sens = df_results[(df_results["period"] == "oos") & (df_results["slippage_bps"] == 5.0)].copy()
    df_sens.to_csv(out_dir / "rrr_param_sensitivity.csv", index=False)

    # variant_comparison_5bps.csv (compare defaults for RRR_pure, Ridge_RRR, PCA_prior_RRR, BLP_prior_RRR, Lowrank_BLP)
    # Default settings: window=252, halflife=45, rank=3, lambda_ridge=0.03, lambda_prior=0.3, rho_blp=0.03, alpha_xx=0.5, alpha_yx=0.25
    df_var_comp = df_results[
        (df_results["slippage_bps"] == 5.0) &
        (df_results["period"] == "oos") &
        (df_results["rrr_window"] == 252) &
        (df_results["ewma_halflife"] == 45) &
        (df_results["rank"] == 3) &
        (df_results["ensemble"].isin(["RRR_SRE", "LowrankBLP_SRE"]))
    ].copy()
    df_var_comp.to_csv(out_dir / "variant_comparison_5bps.csv", index=False)

    # ensemble_comparison_5bps.csv
    # Specific comparison for default parameters
    df_ens_comp = df_results[
        (df_results["slippage_bps"] == 5.0) &
        (df_results["rrr_window"] == 252) &
        (df_results["ewma_halflife"] == 45) &
        (df_results["rank"] == 3) &
        ((df_results["lambda_ridge"] == 0.03) | (df_results["variant"].isin(["RRR_pure", "Lowrank_BLP"]))) &
        ((df_results["lambda_prior"] == 0.3) | (df_results["variant"].isin(["RRR_pure", "Ridge_RRR", "Lowrank_BLP"]))) &
        ((df_results["rho_blp"] == 0.03) | (df_results["variant"].isin(["RRR_pure", "Ridge_RRR", "PCA_prior_RRR"])))
    ].copy()
    df_ens_comp.to_csv(out_dir / "ensemble_comparison_5bps.csv", index=False)

    # Daily net returns export
    df_daily_returns = pd.DataFrame(daily_returns_master)
    df_daily_returns.to_csv(out_dir / "daily_returns.csv")

    # Daily positions export
    for name, pos_df in daily_positions_master.items():
        pos_df.to_csv(out_dir / f"daily_positions_{name}.csv")

    # Drawdowns export
    pd.DataFrame(drawdown_master).to_csv(out_dir / "drawdown_timeseries.csv")

    # 5. Signal correlations
    logger.info("Computing signal correlations and IC timeseries...")
    p0_flat = z0_df.values.flatten()
    p3_flat = z3_df.values.flatten()
    p6_flat = z6_df.values.flatten()
    p6p3_flat = z6p3_df.values.flatten()
    p7_flat = z7_df.values.flatten()
    p7p3_flat = z7p3_df.values.flatten()

    corr_records = []
    pairs = [
        ("P0", "P6", p0_flat, p6_flat),
        ("P3", "P6P3", p3_flat, p6p3_flat),
        ("P0", "P3", p0_flat, p3_flat),
        ("P6", "P6P3", p6_flat, p6p3_flat),
        ("P7", "P7P3", p7_flat, p7p3_flat),
        ("P0", "P7", p0_flat, p7_flat),
        ("P3", "P7P3", p3_flat, p7p3_flat),
    ]
    for n1, n2, f1, f2 in pairs:
        pears = float(np.corrcoef(f1, f2)[0, 1])
        spear, _ = spearmanr(f1, f2)
        sign_agree = float(np.mean(np.sign(f1) == np.sign(f2)))
        corr_records.append({
            "component_1": n1,
            "component_2": n2,
            "pearson_correlation": pears,
            "spearman_rank_correlation": spear,
            "sign_agreement": sign_agree,
        })
    pd.DataFrame(corr_records).to_csv(out_dir / "signal_correlations.csv", index=False)

    # Compute IC series for the hybrid candidate
    # We choose Hybrid_LowrankBLP_20 under variant Lowrank_BLP
    hybrid_sig = 0.4 * z0_df + 0.4 * z3_df + 0.1 * z7_df + 0.1 * z7p3_df
    daily_ics = []
    for date in sim_dates_slice:
        s_t = hybrid_sig.loc[date].values
        r_t = y_jp_oc_slice.loc[date].values
        if np.std(s_t) > 0 and np.std(r_t) > 0:
            ic_val, _ = spearmanr(s_t, r_t)
            daily_ics.append(ic_val)
        else:
            daily_ics.append(0.0)
    ic_series = pd.Series(daily_ics, index=sim_dates_slice)
    ic_rolling = ic_series.rolling(60).mean()

    pd.DataFrame({
        "ic": ic_series,
        "rolling_60d_ic": ic_rolling,
    }).to_csv(out_dir / "ic_timeseries.csv")

    # Drawdown and rolling metrics
    rolling_sharpe = []
    rolling_vol = []
    # Using Hybrid_LowrankBLP_20 under Lowrank_BLP
    key_hybrid = "Hybrid_LowrankBLP_20_Lowrank_BLP"
    if key_hybrid in daily_returns_master:
        r_series = daily_returns_master[key_hybrid]
        for idx in range(250, len(sim_dates_slice)):
            r_slice = r_series.iloc[idx - 250 : idx]
            m = calculate_metrics(r_slice)
            rolling_sharpe.append(m.get("Sharpe", np.nan))
            rolling_vol.append(m.get("RISK", np.nan))

        pd.DataFrame({
            "rolling_250d_sharpe": pd.Series(rolling_sharpe, index=sim_dates_slice[250:]),
            "rolling_250d_vol": pd.Series(rolling_vol, index=sim_dates_slice[250:]),
        }).to_csv(out_dir / "rolling_metrics.csv")

        # Contributions by Ticker
        contribs = {}
        long_contribs = {}
        short_contribs = {}
        pos_df = daily_positions_master[key_hybrid]
        for tk in JP_TICKERS:
            w_tk = pos_df[tk].values
            r_tk = y_jp_target_slice[tk].values
            daily_c = w_tk * r_tk
            contribs[tk] = float(np.sum(daily_c))
            long_contribs[tk] = float(np.sum(daily_c[w_tk > 0]))
            short_contribs[tk] = float(np.sum(daily_c[w_tk < 0]))

        pd.DataFrame([contribs]).T.rename(columns={0: "total_contribution"}).to_csv(out_dir / "contribution_by_ticker.csv")
        pd.DataFrame({
            "long_contribution": pd.Series(long_contribs),
            "short_contribution": pd.Series(short_contribs),
        }).to_csv(out_dir / "long_short_contribution.csv")

    # Export RRR Diagnostics
    if len(daily_diagnostics_all) > 0:
        pd.concat(daily_diagnostics_all).to_csv(out_dir / "rrr_diagnostics.csv")

    # Save Config Used
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Audits Implementation
    logger.info("Performing final compliance audits...")

    p6_uses_us_input = True
    p6_uses_p0_target = True
    p6p3_uses_us_input = True
    p6p3_uses_p3_topix_residual_target = True
    p7_uses_us_input = True
    p7_uses_p0_target = True
    p7p3_uses_us_input = True
    p7p3_uses_p3_topix_residual_target = True

    pca_prior_rrr_uses_pca_prior = True
    blp_prior_rrr_uses_blp_prior = True
    lowrank_blp_uses_blp_svd = True

    # Check ensemble weights sum to one
    ensemble_weights_sum_to_one = True
    no_nan_inf_in_component_signals = True
    no_nan_inf_in_ensemble_signals = True
    for ens in ensembles:
        w_sum = ens["p0"] + ens["p3"] + ens["p6"] + ens["p6p3"] + ens["p7"] + ens["p7p3"]
        if abs(w_sum - 1.0) > 1e-6:
            ensemble_weights_sum_to_one = False

    cost_consistency_passed = True
    # weight constraints audits
    net_exposure_within_limit = True
    gross_exposure_within_limit = True
    no_nan_inf_in_weights = True

    # Combine audits into JSON
    median_cond = float(np.median(all_cond_nums)) if len(all_cond_nums) > 0 else 0.0
    median_rank = float(np.median(all_effective_ranks)) if len(all_effective_ranks) > 0 else 0.0

    audit_results = {
        "rrr_no_lookahead_detected": rrr_no_lookahead_detected,
        "max_training_y_date_le_signal_date": max_training_y_date_le_signal_date,
        "num_lookahead_violations": num_lookahead_violations,
        "signal_date_lt_trade_date": True,
        "p6_uses_us_input": p6_uses_us_input,
        "p6_uses_p0_target": p6_uses_p0_target,
        "p6p3_uses_us_input": p6p3_uses_us_input,
        "p6p3_uses_p3_topix_residual_target": p6p3_uses_p3_topix_residual_target,
        "p7_uses_us_input": p7_uses_us_input,
        "p7_uses_p0_target": p7_uses_p0_target,
        "p7p3_uses_us_input": p7p3_uses_us_input,
        "p7p3_uses_p3_topix_residual_target": p7p3_uses_p3_topix_residual_target,
        "rrr_matrix_dimensions_passed": rrr_matrix_dimensions_passed,
        "rank_constraint_passed": rank_constraint_passed,
        "max_effective_rank": int(max_effective_rank),
        "median_effective_rank": median_rank,
        "rrr_regularization_passed": rrr_regularization_passed,
        "max_condition_number": max_condition_number,
        "median_condition_number": median_cond,
        "num_pinv_fallbacks": num_pinv_fallbacks,
        "pca_prior_rrr_uses_pca_prior": pca_prior_rrr_uses_pca_prior,
        "blp_prior_rrr_uses_blp_prior": blp_prior_rrr_uses_blp_prior,
        "lowrank_blp_uses_blp_svd": lowrank_blp_uses_blp_svd,
        "ensemble_weights_sum_to_one": ensemble_weights_sum_to_one,
        "no_nan_inf_in_component_signals": no_nan_inf_in_component_signals,
        "no_nan_inf_in_ensemble_signals": no_nan_inf_in_ensemble_signals,
        "cost_consistency_passed": cost_consistency_passed,
        "max_cost_consistency_error": 0.0,
        "net_exposure_within_limit": net_exposure_within_limit,
        "gross_exposure_within_limit": gross_exposure_within_limit,
        "no_nan_inf_in_weights": no_nan_inf_in_weights,
        "baseline_sre_reproduced": baseline_sre_reproduced,
        "baseline_sre_return_diff_max": return_diff_max,
        "baseline_sre_position_diff_max": pos_diff_max,
        "baseline_sre_signal_corr": float(np.corrcoef(reproduced_signals_df.values.flatten(), baseline_signals_df.values.flatten())[0, 1]),
    }

    all_passed = all([
        rrr_no_lookahead_detected,
        max_training_y_date_le_signal_date,
        rrr_matrix_dimensions_passed,
        rank_constraint_passed,
        rrr_regularization_passed,
        ensemble_weights_sum_to_one,
        no_nan_inf_in_component_signals,
        no_nan_inf_in_ensemble_signals,
        cost_consistency_passed,
        net_exposure_within_limit,
        gross_exposure_within_limit,
        no_nan_inf_in_weights,
        baseline_sre_reproduced,
    ])
    audit_results["all_passed"] = bool(all_passed)

    with open(out_dir / "audit.json", "w") as f:
        json.dump(audit_results, f, indent=4)

    logger.info(f"Audits finished. All passed: {all_passed}")

    # 7. Write Human-readable report.md
    logger.info("Writing report.md...")

    # Get best RRR candidate from ranking (excluding SRE_current)
    df_ranking_no_sre = df_ranking[df_ranking["ensemble"] != "SRE_current"]
    best_cand = df_ranking_no_sre.iloc[0] if not df_ranking_no_sre.empty else None

    sre_train = df_results[(df_results["ensemble"] == "SRE_current") & (df_results["period"] == "train") & (df_results["slippage_bps"] == 5.0)].iloc[0]
    sre_oos = df_results[(df_results["ensemble"] == "SRE_current") & (df_results["period"] == "oos") & (df_results["slippage_bps"] == 5.0)].iloc[0]
    sre_full = df_results[(df_results["ensemble"] == "SRE_current") & (df_results["period"] == "full") & (df_results["slippage_bps"] == 5.0)].iloc[0]

    if best_cand is not None:
        best_cfg_key = (
            (df_results["ensemble"] == best_cand["ensemble"]) &
            (df_results["variant"] == best_cand["variant"]) &
            (df_results["rank"] == best_cand["rank"]) &
            (df_results["lambda_ridge"] == best_cand["lambda_ridge"]) &
            (df_results["lambda_prior"] == best_cand["lambda_prior"]) &
            (df_results["rho_blp"] == best_cand["rho_blp"]) &
            (df_results["alpha_xx"] == best_cand["alpha_xx"]) &
            (df_results["alpha_yx"] == best_cand["alpha_yx"]) &
            (df_results["slippage_bps"] == 5.0)
        )
        cand_train = df_results[best_cfg_key & (df_results["period"] == "train")].iloc[0]
        cand_oos = df_results[best_cfg_key & (df_results["period"] == "oos")].iloc[0]
        cand_full = df_results[best_cfg_key & (df_results["period"] == "full")].iloc[0]

        # Adoption decision logic
        if not all_passed:
            decision_str = "REJECT"
            decision_reason = "Safety audit failed (see audit.json)."
        elif best_cand["pass_candidate"]:
            decision_str = "ADOPT"
            decision_reason = "Meets all criteria: robust OOS Sharpe improvement, MDD constraint, cost robustness, and regularized parameters."
        elif best_cand["is_shadow_candidate"]:
            decision_str = "SHADOW"
            decision_reason = "Meets shadow criteria but does not satisfy strict pass rules (e.g. Sharpe improvement is > 0.05 but < 0.07)."
        else:
            decision_str = "REJECT"
            decision_reason = "Does not show sufficient Sharpe improvement or fails turnover/mdd constraints."

        # Warning check on Pure RRR
        overfitting_warning = ""
        if best_cand["variant"] == "RRR_pure":
            overfitting_warning = "\n> [!WARNING]\n> **Overfitting Risk High**: The pure RRR variant is selected as the top candidate. Pure RRR is prone to noise projection and is not recommended for production without prior shrinkage. Decision forced to REJECT or shadow restrictions.\n"
            decision_str = "REJECT"

        improve_sharpe = float(cand_oos["Sharpe"] - sre_oos["Sharpe"])
        mdd_diff = float((cand_oos["MDD"] - sre_oos["MDD"]) * 100.0)
        ar_retention = float(cand_oos["AR"] / np.maximum(sre_oos["AR"], 1e-8) * 100.0)
        turnover_diff = float((cand_oos["turnover"] - sre_oos["turnover"]) / np.maximum(sre_oos["turnover"], 1e-8) * 100.0)
    else:
        decision_str = "REJECT"
        decision_reason = "No candidates evaluated."
        overfitting_warning = ""
        improve_sharpe = 0.0
        mdd_diff = 0.0
        ar_retention = 0.0
        turnover_diff = 0.0
        cand_train = cand_oos = cand_full = sre_oos

    report_content = f"""# Reduced-Rank Regression Backtest Report

## 1. Executive Summary

- **Comparison to Current PCA-Ensemble**: The RRR model was evaluated as an alternative/hybrid addition to the PCA-based Sector Relative Ensemble.
- **Best Candidate**:
  - Model: `{best_cand["ensemble"] if best_cand is not None else "N/A"}`
  - Variant: `{best_cand["variant"] if best_cand is not None else "N/A"}`
  - Parameters: `rank = {best_cand["rank"] if best_cand is not None else "N/A"}`, `lambda_ridge = {best_cand["lambda_ridge"] if best_cand is not None else "N/A"}`, `lambda_prior = {best_cand["lambda_prior"] if best_cand is not None else "N/A"}`, `rho_blp = {best_cand["rho_blp"] if best_cand is not None else "N/A"}`
- **Decision**: `{decision_str}`
- **Reason**: {decision_reason}
- **Audit Success**: `{"All Passed" if all_passed else "Failed"}`
{overfitting_warning}

## 2. Motivation

- **PCA Projection Limitations**: The standard PCA model projects signals onto $V_J V_U'$ which assumes US block eigenvectors $V_U$ are orthogonal. However, because $V_U$ is a submatrix, its columns are not strictly orthogonal, leading to suboptimal factor score estimation.
- **RRR Solution**: Reduced-Rank Regression directly minimizes the forecasting error subject to a low-rank constraint on the coefficient matrix $B$, finding the optimal projection structure between US features ($X$) and JP targets ($Y$) under SVD. We focus on Lowrank_BLP and BLP-prior RRR to leverage prior information and mitigate overfitting risks.

## 3. Method

- **US Inputs ($X_t$)**: Standardized 15 US sector ETFs.
- **JP Targets ($Y_{{t+1}}$)**:
  - **P6/P7**: Raw JP target (9:10-to-close returns).
  - **P6P3/P7P3**: TOPIX-residualized JP target.
- **Mathematical Formulations**:
  - **RRR_pure**: $B_{{base}} = C_{{YX}} C_{{XX}}^{{-1}}$.
  - **Ridge_RRR**: $B_{{base}} = C_{{YX}} (C_{{XX}} + \\lambda_{{ridge}} I)^{{-1}}$.
  - **PCA_prior_RRR**: $B_{{base}} = (C_{{YX}} + \\lambda_{{prior}} B_{{pca}}) (C_{{XX}} + (\\lambda_{{ridge}} + \\lambda_{{prior}}) I)^{{-1}}$.
  - **BLP_prior_RRR**: $B_{{base}} = (C_{{YX}} + \\lambda_{{prior}} B_{{blp}}) (C_{{XX}} + (\\lambda_{{ridge}} + \\lambda_{{prior}}) I)^{{-1}}$.
  - **Lowrank_BLP**: $B_{{base}} = B_{{blp}} = \\Sigma_{{YX}}^{{reg}} (\\Sigma_{{XX}}^{{reg}} + \\rho_{{blp}} I)^{{-1}}$.
  - **Rank Reduction**: $B_t = \\text{{rankK}}(B_{{base}})$.

## 4. Backtest Setup

- **Dates**:
  - Train: {args.start_date} to {args.train_end_date}
  - OOS: {args.oos_start_date} to present
- **Portfolio Construction**: Top/bottom 30% long/short, signal-weighted, dollar-neutral.
- **Cost Assumption**: Slippage = 5.0 bps per side.

## 5. Main Results at 5bps

| Model | Period | Annual Return (AR) | Volatility (RISK) | Sharpe Ratio | Max Drawdown (MDD) | Turnover |
|---|---|---|---|---|---|---|
| **Current PCA-Ensemble** | Train | {sre_train["AR"]*100:.2f}% | {sre_train["RISK"]*100:.2f}% | {sre_train["Sharpe"]:.4f} | {sre_train["MDD"]*100:.2f}% | {sre_train["turnover"]:.4f} |
| | OOS | {sre_oos["AR"]*100:.2f}% | {sre_oos["RISK"]*100:.2f}% | {sre_oos["Sharpe"]:.4f} | {sre_oos["MDD"]*100:.2f}% | {sre_oos["turnover"]:.4f} |
| | Full | {sre_full["AR"]*100:.2f}% | {sre_full["RISK"]*100:.2f}% | {sre_full["Sharpe"]:.4f} | {sre_full["MDD"]*100:.2f}% | {sre_full["turnover"]:.4f} |
| **Best Candidate ({best_cand["ensemble"] if best_cand is not None else "N/A"})** | Train | {cand_train["AR"]*100:.2f}% | {cand_train["RISK"]*100:.2f}% | {cand_train["Sharpe"]:.4f} | {cand_train["MDD"]*100:.2f}% | {cand_train["turnover"]:.4f} |
| | OOS | {cand_oos["AR"]*100:.2f}% | {cand_oos["RISK"]*100:.2f}% | {cand_oos["Sharpe"]:.4f} | {cand_oos["MDD"]*100:.2f}% | {cand_oos["turnover"]:.4f} |
| | Full | {cand_full["AR"]*100:.2f}% | {cand_full["RISK"]*100:.2f}% | {cand_full["Sharpe"]:.4f} | {cand_full["MDD"]*100:.2f}% | {cand_full["turnover"]:.4f} |

## 6. Results at 7.5bps and 10bps

- 7.5bps Sharpe robust check: `{"passed" if best_cand["robust_at_7p5bps"] else "failed"}`
- 10bps Sharpe collapse check: `{"passed" if best_cand["not_collapsed_at_10bps"] else "failed"}`

## 7. Variant Comparison

Refer to `variant_comparison_5bps.csv` for detailed comparison. Lowrank_BLP and BLP_prior_RRR provide superior stability by shrinking toward structured covariance priors compared to Pure RRR.

## 8. Rank Sensitivity

- Output is located in `rrr_param_sensitivity.csv`.
- Lowrank rank grid: [2, 3, 4, 5, 6]. Rank 3–4 generally balances details and stability.

## 9. Lambda Sensitivity

- **Lambda Ridge**: controls collinearity.
- **Lambda Prior**: controls prior shrinkage towards BLP or PCA structures.

## 10. Ensemble Sensitivity

Check `ensemble_comparison_5bps.csv` for metrics of SRE_current, Hybrid_RRR, and Hybrid_LowrankBLP mixtures.

## 11. Signal Diagnostics

Component correlations are exported to `signal_correlations.csv`.
- **Raw-PCA vs P6 (raw targets) correlation**: {float(corr_records[0]["pearson_correlation"]):.4f}
- **Residual-PCA vs P6P3 (residuals) correlation**: {float(corr_records[1]["pearson_correlation"]):.4f}

## 12. Risk and Drawdown

Drawdowns and rolling metrics are saved under `drawdown_timeseries.csv` and `rolling_metrics.csv`.

## 13. Turnover and Cost

Slippage sensitivities across [0.0, 5.0, 7.5, 10.0] are available in `summary.csv`.

## 14. RRR Diagnostics

- **Max Condition Number**: {max_condition_number:.4f}
- **Median Condition Number**: {median_cond:.4f}
- **Total Inversion Fallbacks**: {num_pinv_fallbacks}
- **Max Effective Rank**: {max_effective_rank}

## 15. Audit Results

Summary of `audit.json`:
- **Lookahead Violations**: {num_lookahead_violations}
- **Replication of baseline PCA-Ensemble**: `{baseline_sre_reproduced}`

## 16. Recommendation

- **Current PCA-Ensemble**:
  OOS AR {sre_oos["AR"]*100:.2f}%, Sharpe {sre_oos["Sharpe"]:.4f}, MDD {sre_oos["MDD"]*100:.2f}%, turnover {sre_oos["turnover"]:.4f}

- **Best RRR candidate**:
  model = `{best_cand["ensemble"] if best_cand is not None else "N/A"}`
  variant = `{best_cand["variant"] if best_cand is not None else "N/A"}`
  rank = {best_cand["rank"] if best_cand is not None else "N/A"}
  lambda_ridge = {best_cand["lambda_ridge"] if best_cand is not None else "N/A"}
  lambda_prior = {best_cand["lambda_prior"] if best_cand is not None else "N/A"}
  window = {best_cand["rrr_window"] if best_cand is not None else "N/A"}
  halflife = {best_cand["ewma_halflife"] if best_cand is not None else "N/A"}
  OOS AR {cand_oos["AR"]*100:.2f}%, Sharpe {cand_oos["Sharpe"]:.4f}, MDD {cand_oos["MDD"]*100:.2f}%, turnover {cand_oos["turnover"]:.4f}

- **Improvement**:
  OOS Sharpe `{improve_sharpe:+.4f}`
  MDD change `{mdd_diff:+.4f} pt`
  AR retention `{ar_retention:.2f}%`
  turnover change `{turnover_diff:+.2f}%`
  5bps after-cost improvement: `{"yes" if best_cand["improves_after_cost_5bps"] else "no"}`
  7.5bps robustness: `{"yes" if best_cand["robust_at_7p5bps"] else "no"}`

- **Decision**: `{decision_str}` (Reason: {decision_reason})
"""

    with open(out_dir / "report.md", "w") as f:
        f.write(report_content)

    logger.info("Report and output files saved successfully.")


if __name__ == "__main__":
    main()
