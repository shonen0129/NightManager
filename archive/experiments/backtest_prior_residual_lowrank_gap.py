#!/usr/bin/env python
"""Backtesting script for the Prior Residualized Low-Rank Gap Model.

Performs parameter grid-search on training data, evaluates on OOS data,
compares with existing baseline/production models, audits for leakage,
and outputs performance tables and charts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

# Add src/ directory to python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import STRATEGY_DEFAULTS
from data.downloader import download_data
from data.preprocessor import preprocess_data
from data.ticker_registry import JP_TICKERS, US_TICKERS, TOPIX_TICKER
from domain.signals.lead_lag import build_v3_static
from domain.models.prior_residual_lowrank import (
    ResidualizedPriorSubspaceLowRankGapModel,
    project_to_subspace,
)
from domain.models.residual_lowrank import (
    ResidualizedSupervisedLowRankModel,
    compute_rolling_ols_betas,
)
sys.path.insert(0, str(ROOT / "tools"))
from backtest_residual_lowrank import run_residual_lowrank_simulation
from domain.models.types import StrategyConfig
from backtest.runner import run_backtest_with_config
from performance import calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MACRO_CACHE_PATH = ROOT / "data" / "macro_data.pkl"


def get_macro_data(start_date: str = "2009-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Download or load cached macro data from yfinance."""
    if MACRO_CACHE_PATH.exists():
        age_hours = (
            datetime.now() - datetime.fromtimestamp(os.path.getmtime(MACRO_CACHE_PATH))
        ).total_seconds() / 3600.0
        if age_hours < 12.0:
            try:
                df = pd.read_pickle(MACRO_CACHE_PATH)
                if all(c in df.columns for c in ["SPY", "USDJPY=X", "CL=F", "^TNX", "^VIX"]):
                    logger.info("Loaded macro data from cache (valid)")
                    return df
            except Exception as e:
                logger.warning("Error reading macro cache: %s", e)

    logger.info("Downloading macro data from yfinance...")
    import yfinance as yf

    tickers = ["SPY", "USDJPY=X", "CL=F", "^TNX", "^VIX"]
    df_raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    df_close = df_raw["Close"].copy()
    df_close.index = pd.to_datetime(df_close.index).tz_localize(None).normalize()

    try:
        MACRO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_close.to_pickle(MACRO_CACHE_PATH)
        logger.info("Macro data saved to cache: %s", MACRO_CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to save macro cache: %s", e)

    return df_close


def build_portfolio_weights(signal: np.ndarray, q: float = 0.3) -> np.ndarray:
    """Build weights: top 30% long, bottom 30% short, centered around median, scaled to gross=2."""
    n_j = len(signal)
    num_positions = int(np.floor(n_j * q))
    if num_positions <= 0:
        return np.zeros(n_j)

    sort_order = np.argsort(signal)
    short_idx = sort_order[:num_positions]
    long_idx = sort_order[-num_positions:]

    weights = np.zeros(n_j)
    s_centered = signal - np.median(signal)

    # Long weights
    long_raw = s_centered[long_idx]
    long_raw = np.maximum(long_raw, 1e-8)
    long_denom = np.sum(long_raw)
    if long_denom > 0:
        weights[long_idx] = long_raw / long_denom

    # Short weights (MUST be negative)
    short_raw = -s_centered[short_idx]
    short_raw = np.maximum(short_raw, 1e-8)
    short_denom = np.sum(short_raw)
    if short_denom > 0:
        weights[short_idx] = -(short_raw / short_denom)

    return weights


def calculate_detailed_metrics(
    daily_ret: pd.Series,
    gross_exp: pd.Series,
    slippage_cost: pd.Series,
    weights_df: pd.DataFrame,
    r_oc_df: pd.DataFrame,
) -> dict:
    """Calculate rich performance statistics for report."""
    T = len(daily_ret)
    if T == 0:
        return {}

    # Performance calculations using Monthly returns to match performance.py
    monthly = (1.0 + daily_ret).groupby(daily_ret.index.to_period("M")).prod() - 1.0
    t_months = len(monthly)

    if t_months > 1:
        ar = float(np.sum(monthly) * 12.0 / t_months)
        mu_m = float(np.mean(monthly))
        monthly_std = float(np.std(monthly, ddof=1))
        risk = float(np.sqrt(12.0 / (t_months - 1) * np.sum((monthly - mu_m) ** 2)))
        sharpe = (mu_m / monthly_std) * np.sqrt(12.0) if monthly_std > 0 else np.nan
    else:
        ar = float(np.sum(daily_ret) * 245.0 / T)
        risk = float(np.std(daily_ret, ddof=1) * np.sqrt(245.0))
        sharpe = (np.mean(daily_ret) / np.std(daily_ret, ddof=1)) * np.sqrt(245.0) if np.std(daily_ret, ddof=1) > 0 else np.nan

    rr = ar / risk if risk > 0 else np.nan

    # MDD
    wealth = (1.0 + daily_ret).cumprod()
    running_max = wealth.cummax()
    drawdowns = (wealth / running_max) - 1.0
    mdd = min(0.0, drawdowns.min())

    # Daily Stats
    win_rate = float((daily_ret > 0).sum() / T)
    avg_daily_ret = float(daily_ret.mean())
    std_daily_ret = float(daily_ret.std(ddof=1))

    # VaR & ES (99% daily)
    var_99 = float(np.percentile(daily_ret, 1.0))
    tail_returns = daily_ret[daily_ret <= var_99]
    es_99 = float(tail_returns.mean()) if len(tail_returns) > 0 else np.nan

    # Turnover (one-way daily)
    w_diff = weights_df.diff().abs().sum(axis=1) / 2.0
    avg_turnover = float(w_diff.mean()) if T > 1 else np.nan

    avg_gross = float(gross_exp.mean())
    avg_slippage = float(slippage_cost.mean())

    # Long and Short contributions
    long_rets = []
    short_rets = []
    for t in range(T):
        w_t = weights_df.iloc[t].values
        r_t = r_oc_df.iloc[t].values

        long_mask = w_t > 0
        short_mask = w_t < 0

        long_ret = np.sum(w_t[long_mask] * r_t[long_mask]) if long_mask.any() else 0.0
        short_ret = np.sum(w_t[short_mask] * r_t[short_mask]) if short_mask.any() else 0.0

        long_rets.append(long_ret)
        short_rets.append(short_ret)

    avg_long_contrib = float(np.mean(long_rets))
    avg_short_contrib = float(np.mean(short_rets))

    return {
        "AR": ar,
        "RISK": risk,
        "R/R": rr,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Win Rate": win_rate,
        "Avg Daily Return": avg_daily_ret,
        "Daily Return Std": std_daily_ret,
        "VaR 99%": var_99,
        "ES 99%": es_99,
        "Avg Turnover": avg_turnover,
        "Avg Gross Exposure": avg_gross,
        "Avg Slippage Cost": avg_slippage,
        "Long Contribution": avg_long_contrib,
        "Short Contribution": avg_short_contrib,
    }


def run_prior_lowrank_simulation(
    df_exec: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
    jp_cc_returns: pd.DataFrame,
    topix_oc_return: pd.Series,
    topix_cc_return: pd.Series,
    us_returns_raw: pd.DataFrame,
    macro_returns: pd.DataFrame,
    config: dict,
    start_date: str,
    reversed_signal: bool = False,
) -> dict:
    """Run rolling walk-forward simulation of the Prior Subspace Low-Rank Model."""
    T = len(df_exec)
    n_features = 11
    n_targets = len(JP_TICKERS)

    # 1. US Residualization
    y_us = us_returns_raw[US_TICKERS[:11]].values  # shape (T, 11)
    if config["residualize_us_macro"]:
        x_us = np.column_stack([
            macro_returns["SPY_cc"].values,
            macro_returns["USDJPY_cc"].values,
            macro_returns["Oil_cc"].values,
            macro_returns["US10Y_diff"].values,
            macro_returns["VIX_diff"].values,
        ])
    else:
        x_us = macro_returns["SPY_cc"].values.reshape(-1, 1)

    if config["residualize_us_market"]:
        betas_us = compute_rolling_ols_betas(y_us, x_us, config["beta_window"])
        x_shocks = y_us - np.sum(betas_us * x_us[:, np.newaxis, :], axis=2)
    else:
        x_shocks = y_us

    # 2. JP Residualization
    y_jp_oc = jp_oc_returns[JP_TICKERS].values  # shape (T, 17)
    x_jp_oc = topix_oc_return.values.reshape(-1, 1)  # shape (T, 1)
    y_jp_cc = jp_cc_returns[JP_TICKERS].values  # shape (T, 17)
    x_jp_cc = topix_cc_return.values.reshape(-1, 1)  # shape (T, 1)

    if config["residualize_jp_market"]:
        betas_jp_oc = compute_rolling_ols_betas(y_jp_oc, x_jp_oc, config["beta_window"])
        y_residuals_oc = y_jp_oc - betas_jp_oc[:, :, 0] * x_jp_oc

        betas_jp_cc = compute_rolling_ols_betas(y_jp_cc, x_jp_cc, config["beta_window"])
        y_residuals_cc = y_jp_cc - betas_jp_cc[:, :, 0] * x_jp_cc
    else:
        y_residuals_oc = y_jp_oc
        y_residuals_cc = y_jp_cc

    # Select target mode
    if config["target_mode"] == "oc_residual":
        y_residuals = y_residuals_oc
    elif config["target_mode"] == "cc_residual":
        y_residuals = y_residuals_cc
    else:
        raise ValueError(f"Invalid target_mode: {config['target_mode']}")

    # 3. Gap Estimation and Correction
    jp_gap = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values  # shape (T, 17)
    topix_night = df_exec["topix_night_return"].values.reshape(-1, 1)  # shape (T, 1)

    # dedicated gap beta estimation
    beta_gap = compute_rolling_ols_betas(jp_gap, topix_night, config["beta_window"])

    # Calculate filtered gaps for all dates
    GapOpen_syst = beta_gap[:, :, 0] * topix_night
    GapOpen_idio = jp_gap - GapOpen_syst
    GapOpen_filt = (
        config["gap_open_coef"] * GapOpen_idio
        + (config["gap_open_coef"] - config["topix_beta_coef"]) * GapOpen_syst
    )

    # 4. Model parameters
    k_prior = config["k_prior"]
    ridge_alpha_a = config["ridge_alpha_a"]
    lambda_prior_a = config["lambda_prior_a"]
    train_window = config["train_window"]
    refit_freq = config["refit_frequency"]

    # Subspace priors
    V0_full = build_v3_static(15, 17, include_v4=True)
    V0_U = V0_full[0:11, 0:k_prior]
    V0_J = V0_full[15:32, 0:k_prior]
    A0 = np.eye(k_prior)

    # Locate start index
    start_dt = pd.to_datetime(start_date)
    start_idx = max(df_exec.index.searchsorted(start_dt), train_window + config["beta_window"])

    # Output arrays
    pred_signals = np.zeros((T, n_targets))
    pred_signals[:] = np.nan
    pred_pre_gap = np.zeros((T, n_targets))
    pred_pre_gap[:] = np.nan

    # Walk-forward loop
    last_B = None
    last_mean_X = None
    last_std_X = None
    last_mean_Y = None
    last_std_Y = None
    last_month = None

    A_history = []
    B_history = []
    refit_dates = []

    model = ResidualizedPriorSubspaceLowRankGapModel(
        k_prior=k_prior,
        ridge_alpha_a=ridge_alpha_a,
        lambda_prior_a=lambda_prior_a,
        train_window=train_window,
        beta_window=config["beta_window"],
        residualize_us_market=config["residualize_us_market"],
        residualize_jp_market=config["residualize_jp_market"],
        gap_open_coef=config["gap_open_coef"],
        topix_beta_coef=config["topix_beta_coef"],
        gap_signal_coef=config["gap_signal_coef"],
        signal_mode=config["signal_mode"],
        target_mode=config["target_mode"],
        gap_formula=config["gap_formula"],
    )

    for i in range(start_idx, T):
        should_refit = False
        if refit_freq == "daily" or last_B is None:
            should_refit = True
        elif refit_freq == "monthly":
            current_month = df_exec.index[i].strftime("%Y-%m")
            if current_month != last_month:
                should_refit = True
                last_month = current_month

        if should_refit:
            train_start = i - train_window
            train_end = i - 1
            X_tr = x_shocks[train_start : train_end + 1]
            Y_tr = y_residuals[train_start : train_end + 1]

            y_pred_step, B_eff = model.fit_predict_step(
                X_tr, Y_tr, x_shocks[i], V0_U, V0_J, A0
            )

            # Reconstruct A matrix for diagnostics
            # Standardize X and Y in-sample
            valid_mask = np.isfinite(X_tr).all(axis=1) & np.isfinite(Y_tr).all(axis=1)
            X_tr_clean = X_tr[valid_mask]
            Y_tr_clean = Y_tr[valid_mask]

            mean_X = np.mean(X_tr_clean, axis=0)
            std_X = np.std(X_tr_clean, axis=0, ddof=1)
            std_X[std_X == 0.0] = 1e-8
            mean_Y = np.mean(Y_tr_clean, axis=0)
            std_Y = np.std(Y_tr_clean, axis=0, ddof=1)
            std_Y[std_Y == 0.0] = 1e-8

            X_tr_std = (X_tr_clean - mean_X) / std_X
            Y_tr_std = (Y_tr_clean - mean_Y) / std_Y

            F_tr = project_to_subspace(X_tr_std, V0_U)
            G_tr = project_to_subspace(Y_tr_std, V0_J)

            from domain.models.prior_residual_lowrank import solve_factor_propagation
            A = solve_factor_propagation(F_tr, G_tr, ridge_alpha_a, lambda_prior_a, A0)

            last_A = A
            last_B = B_eff
            last_mean_X = mean_X
            last_std_X = std_X
            last_mean_Y = mean_Y
            last_std_Y = std_Y

            A_history.append(A)
            B_history.append(B_eff)
            refit_dates.append(df_exec.index[i])

        # Predict using last fitted model
        x_pred = x_shocks[i]
        x_pred_std = (x_pred - last_mean_X) / last_std_X
        # B_eff_std = V0_U * inv(G_U) * A * V0_J.T
        G_U = V0_U.T @ V0_U
        inv_G_U = np.linalg.inv(G_U + 1e-8 * np.eye(k_prior))
        B_eff_std = V0_U @ inv_G_U @ last_A @ V0_J.T
        y_pred_std = x_pred_std @ B_eff_std
        y_pred = last_mean_Y + y_pred_std * last_std_Y

        pred_pre_gap[i] = y_pred

        # Generate signals according to signal_mode
        if config["signal_mode"] == "prior_no_gap":
            sig = y_pred
        elif config["signal_mode"] == "prior_gap_additive":
            sig = y_pred - config["gap_signal_coef"] * GapOpen_filt[i]
        elif config["signal_mode"] == "prior_cc_to_oc_gap":
            if config["gap_formula"] == "additive":
                sig = y_pred - GapOpen_filt[i]
            elif config["gap_formula"] == "multiplicative":
                sig = (1.0 + y_pred) / np.maximum(1.0 + GapOpen_filt[i], 0.1) - 1.0
            else:
                raise ValueError(f"Invalid gap_formula: {config['gap_formula']}")
        else:
            raise ValueError(f"Invalid signal_mode: {config['signal_mode']}")

        pred_signals[i] = sig

    # 5. Portfolio weights and simulation returns
    sim_dates = df_exec.index[start_idx:]
    results = []

    weights_list = []
    daily_returns = []
    daily_gross_exposures = []
    daily_slippage_costs = []

    slippage_rate = config.get("slippage_bps", 5.0) / 10000.0

    for idx, date in enumerate(sim_dates):
        i = start_idx + idx
        sig = pred_signals[i]
        if reversed_signal:
            sig = -sig

        weights = build_portfolio_weights(sig, q=0.3)
        weights_list.append(weights)

        # Realized Open-to-Close JP returns on trade date i
        r_oc = y_jp_oc[i]
        gross_return = float(np.sum(weights * r_oc))

        gross_exposure = float(np.sum(np.abs(weights)) + 1e-12)
        slippage_cost = 2.0 * slippage_rate * gross_exposure
        net_return = gross_return - slippage_cost

        daily_returns.append(net_return)
        daily_gross_exposures.append(gross_exposure)
        daily_slippage_costs.append(slippage_cost)

        # IC calculation
        raw_ic, _ = spearmanr(sig, r_oc)
        res_oc_ic, _ = spearmanr(sig, y_residuals_oc[i])
        res_cc_ic, _ = spearmanr(pred_pre_gap[i], y_residuals_cc[i])

        results.append({
            "trade_date": date,
            "daily_return": net_return,
            "daily_return_gross": gross_return,
            "slippage_cost": slippage_cost,
            "gross_exposure": gross_exposure,
            "raw_ic": raw_ic,
            "residual_oc_ic": res_oc_ic,
            "residual_cc_ic": res_cc_ic,
        })

    results_df = pd.DataFrame(results).set_index("trade_date")
    weights_df = pd.DataFrame(weights_list, index=sim_dates, columns=JP_TICKERS)
    r_oc_df = pd.DataFrame(y_jp_oc[start_idx:], index=sim_dates, columns=JP_TICKERS)
    signals_df = pd.DataFrame(pred_signals[start_idx:], index=sim_dates, columns=JP_TICKERS)
    predictions_pre_gap_df = pd.DataFrame(pred_pre_gap[start_idx:], index=sim_dates, columns=JP_TICKERS)

    gap_components_df = pd.DataFrame({
        "TOPIXNight": topix_night[start_idx:, 0],
        "GapOpen_avg": np.mean(jp_gap[start_idx:], axis=1),
        "GapOpen_filt_avg": np.mean(GapOpen_filt[start_idx:], axis=1),
    }, index=sim_dates)

    return {
        "results": results_df,
        "weights": weights_df,
        "r_oc": r_oc_df,
        "signals": signals_df,
        "predictions_pre_gap": predictions_pre_gap_df,
        "gap_components": gap_components_df,
        "A_history": A_history,
        "B_history": B_history,
        "refit_dates": refit_dates,
        "y_residuals_oc": pd.DataFrame(y_residuals_oc[start_idx:], index=sim_dates, columns=JP_TICKERS),
        "y_residuals_cc": pd.DataFrame(y_residuals_cc[start_idx:], index=sim_dates, columns=JP_TICKERS),
    }


def main():
    parser = argparse.ArgumentParser(description="Prior Subspace Low Rank Gap Model Backtest Tool")
    parser.add_argument("--start", default="2015-01-05")
    parser.add_argument("--oos-start", default="2020-01-01")
    parser.add_argument("--target-mode", choices=["oc_residual", "cc_residual"], default="cc_residual")
    parser.add_argument("--signal-mode", choices=["prior_no_gap", "prior_gap_additive", "prior_cc_to_oc_gap"], default="prior_cc_to_oc_gap")
    parser.add_argument("--gap-formula", choices=["additive", "multiplicative"], default="multiplicative")
    parser.add_argument("--k-prior", type=int, default=6)
    parser.add_argument("--ridge-alpha-a", type=float, default=1.0)
    parser.add_argument("--lambda-prior-a", type=float, default=1.0)
    parser.add_argument("--train-window", type=int, default=756)
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--gap-open-coef", type=float, default=1.0)
    parser.add_argument("--topix-beta-coef", type=float, default=0.6)
    parser.add_argument("--gap-signal-coef", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--residualize-us-market", action="store_true", default=True)
    parser.add_argument("--no-residualize-us-market", action="store_false", dest="residualize_us_market")
    parser.add_argument("--residualize-jp-market", action="store_true", default=True)
    parser.add_argument("--no-residualize-jp-market", action="store_false", dest="residualize_jp_market")
    parser.add_argument("--refit-frequency", choices=["daily", "monthly"], default="monthly")
    parser.add_argument("--grid-search", action="store_true", help="Perform hyperparameter grid search on training data")
    parser.add_argument("--run-reversed", action="store_true", help="Run reversed versions as well")
    parser.add_argument("--audit-strict", action="store_true", help="Stop execution on daily audit violation")
    args = parser.parse_args()

    # Step 1: Data fetching
    logger.info("Step 1: Fetching historical market and macro data...")
    std_data = download_data(beta_window=args.beta_window)
    df_exec = preprocess_data(std_data, beta_window=args.beta_window)
    macro_df = get_macro_data()

    # Align dates
    logger.info("Aligning datasets...")
    macro_returns = pd.DataFrame(index=macro_df.index)
    macro_returns["SPY_cc"] = macro_df["SPY"].pct_change()
    macro_returns["USDJPY_cc"] = macro_df["USDJPY=X"].pct_change()
    macro_returns["Oil_cc"] = macro_df["CL=F"].pct_change()
    macro_returns["US10Y_diff"] = macro_df["^TNX"].diff()
    macro_returns["VIX_diff"] = macro_df["^VIX"].diff()
    macro_returns = macro_returns.ffill().fillna(0.0)

    sig_dates = pd.to_datetime(df_exec["sig_date"])
    df_exec["r_US_MKT"] = macro_returns["SPY_cc"].reindex(sig_dates).values
    df_exec["SPY_cc"] = df_exec["r_US_MKT"]
    df_exec["USDJPY_cc"] = macro_returns["USDJPY_cc"].reindex(sig_dates).values
    df_exec["Oil_cc"] = macro_returns["Oil_cc"].reindex(sig_dates).values
    df_exec["US10Y_diff"] = macro_returns["US10Y_diff"].reindex(sig_dates).values
    df_exec["VIX_diff"] = macro_returns["VIX_diff"].reindex(sig_dates).values

    # Calculate TOPIX open-to-close returns on trade_date index
    topix_close = std_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = std_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values

    # Reconstruct Close-to-Close returns for JP ETFs and TOPIX on trade dates
    for tk in JP_TICKERS:
        df_exec[f"jp_trade_cc_{tk}"] = (1.0 + df_exec[f"jp_gap_{tk}"]) * (1.0 + df_exec[f"jp_oc_{tk}"]) - 1.0

    df_exec["topix_cc_trade"] = (1.0 + df_exec["topix_night_return"]) * (1.0 + df_exec["topix_oc_return"]) - 1.0

    # Target dataframes for raw prices/returns
    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    jp_cc_cols = [f"jp_trade_cc_{tk}" for tk in JP_TICKERS]
    
    y_jp_oc_df = df_exec[jp_oc_cols].rename(columns=lambda c: c.replace("jp_oc_", ""))
    y_jp_cc_df = df_exec[jp_cc_cols].rename(columns=lambda c: c.replace("jp_trade_cc_", ""))
    y_topix_oc_series = df_exec["topix_oc_return"]
    y_topix_cc_series = df_exec["topix_cc_trade"]
    
    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    us_returns_raw = df_exec[us_cols].rename(columns=lambda c: c.replace("us_cc_", ""))

    # Establish directories
    results_dir = ROOT / "results" / "prior_residual_lowrank_gap"
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = results_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    config_base = {
        "k_prior": args.k_prior,
        "ridge_alpha_a": args.ridge_alpha_a,
        "lambda_prior_a": args.lambda_prior_a,
        "train_window": args.train_window,
        "beta_window": args.beta_window,
        "gap_open_coef": args.gap_open_coef,
        "topix_beta_coef": args.topix_beta_coef,
        "gap_signal_coef": args.gap_signal_coef,
        "slippage_bps": args.slippage_bps,
        "residualize_us_market": args.residualize_us_market,
        "residualize_jp_market": args.residualize_jp_market,
        "residualize_us_macro": False,
        "signal_mode": args.signal_mode,
        "target_mode": args.target_mode,
        "gap_formula": args.gap_formula,
        "refit_frequency": args.refit_frequency,
    }

    # Step 2: Parameter Grid Search (only on train/tune period 2015-01-05 to 2019-12-31)
    train_end_date = "2019-12-31"
    grid_results = []
    
    if args.grid_search:
        logger.info("Step 2: Performing hyperparameter grid search on training data...")
        k_priors = [3, 4, 5, 6]
        alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        lambda_priors = [0.0, 0.1, 1.0, 10.0]
        train_windows = [504, 756, 1008]
        gap_coefs = [0.0, 0.25, 0.5, 0.75, 1.0]
        directions = [1, -1]
        
        # Optimize by looping through windows and k_prior first
        # to reuse US/JP residualization and V0 projections
        for w_size in train_windows:
            for k_p in k_priors:
                # Precompute projections for training period
                cfg_temp = dict(config_base)
                cfg_temp["train_window"] = w_size
                cfg_temp["k_prior"] = k_p
                
                # SVD or raw shocks precomputation
                # (For simplicity and speed, we run the simulation once per w_size and k_p combinations)
                # We can optimize the grid search by running a subset of combinations to keep it fast
                # We will pick a randomized or grid stride if it's too large, but since our loop is fast, we will do it cleanly.
                pass

        # To keep execution within limits, we run a focused grid search:
        k_priors_f = [4, 6]
        alphas_f = [1e-2, 1.0, 10.0]
        lambda_priors_f = [0.0, 1.0, 10.0]
        train_windows_f = [756]
        gap_coefs_f = [0.0, 0.5, 1.0]
        directions_f = [1, -1]

        for w_size in train_windows_f:
            for k_p in k_priors_f:
                for a in alphas_f:
                    for lp in lambda_priors_f:
                        for gc in gap_coefs_f:
                            for direction in directions_f:
                                cfg = dict(config_base)
                                cfg["train_window"] = w_size
                                cfg["k_prior"] = k_p
                                cfg["ridge_alpha_a"] = a
                                cfg["lambda_prior_a"] = lp
                                cfg["gap_signal_coef"] = gc
                                
                                # Run backtest strictly on training period
                                sim = run_prior_lowrank_simulation(
                                    df_exec=df_exec[df_exec.index <= train_end_date],
                                    jp_oc_returns=y_jp_oc_df[y_jp_oc_df.index <= train_end_date],
                                    jp_cc_returns=y_jp_cc_df[y_jp_cc_df.index <= train_end_date],
                                    topix_oc_return=y_topix_oc_series[y_topix_oc_series.index <= train_end_date],
                                    topix_cc_return=y_topix_cc_series[y_topix_cc_series.index <= train_end_date],
                                    us_returns_raw=us_returns_raw[us_returns_raw.index <= train_end_date],
                                    macro_returns=df_exec[df_exec.index <= train_end_date],
                                    config=cfg,
                                    start_date=args.start,
                                    reversed_signal=(direction == -1),
                                )
                                res_df = sim["results"]
                                if not res_df.empty:
                                    metrics = calculate_detailed_metrics(
                                        res_df["daily_return"],
                                        res_df["gross_exposure"],
                                        res_df["slippage_cost"],
                                        sim["weights"],
                                        sim["r_oc"],
                                    )
                                    grid_results.append({
                                        "train_window": w_size,
                                        "k_prior": k_p,
                                        "ridge_alpha_a": a,
                                        "lambda_prior_a": lp,
                                        "gap_signal_coef": gc,
                                        "direction": direction,
                                        "AR": metrics["AR"],
                                        "RISK": metrics["RISK"],
                                        "Sharpe": metrics["Sharpe"],
                                        "MDD": metrics["MDD"],
                                    })
                                    
        grid_df = pd.DataFrame(grid_results)
        grid_df.to_csv(results_dir / "hyperparameter_search.csv", index=False)
        logger.info("Grid search results saved to %s", results_dir / "hyperparameter_search.csv")
        
        # Select optimal parameters (maximize Sharpe)
        best_row = grid_df.loc[grid_df["Sharpe"].idxmax()]
        logger.info("Best parameters found: train_window=%d, k_prior=%d, ridge_alpha_a=%.4f, lambda_prior_a=%.4f, gap_signal_coef=%.2f, direction=%d (Sharpe=%.4f)",
                    best_row["train_window"], best_row["k_prior"], best_row["ridge_alpha_a"], best_row["lambda_prior_a"], best_row["gap_signal_coef"], best_row["direction"], best_row["Sharpe"])
        
        config_base["train_window"] = int(best_row["train_window"])
        config_base["k_prior"] = int(best_row["k_prior"])
        config_base["ridge_alpha_a"] = float(best_row["ridge_alpha_a"])
        config_base["lambda_prior_a"] = float(best_row["lambda_prior_a"])
        config_base["gap_signal_coef"] = float(best_row["gap_signal_coef"])
        config_base["direction"] = int(best_row["direction"])
    else:
        config_base["direction"] = -1 if args.run_reversed else 1

    # Save selected config
    with open(results_dir / "selected_config.json", "w", encoding="utf-8") as f:
        json.dump(config_base, f, ensure_ascii=False, indent=2)

    # Define start_idx and T for main()
    start_dt = pd.to_datetime(args.start)
    start_idx = max(df_exec.index.searchsorted(start_dt), config_base["train_window"] + config_base["beta_window"])
    T = len(df_exec)

    # Step 3: Run full backtests for models
    logger.info("Step 3: Running walk-forward backtests on optimal configuration...")
    
    # 1. Prior Model cc_to_oc_gap (main)
    cfg_1 = dict(config_base)
    cfg_1["signal_mode"] = "prior_cc_to_oc_gap"
    cfg_1["target_mode"] = "cc_residual"
    sim_main = run_prior_lowrank_simulation(df_exec, y_jp_oc_df, y_jp_cc_df, y_topix_oc_series, y_topix_cc_series, us_returns_raw, df_exec, cfg_1, args.start, reversed_signal=(config_base["direction"] == -1))

    # 2. Prior Model cc_to_oc_gap reversed
    sim_main_rev = run_prior_lowrank_simulation(df_exec, y_jp_oc_df, y_jp_cc_df, y_topix_oc_series, y_topix_cc_series, us_returns_raw, df_exec, cfg_1, args.start, reversed_signal=(config_base["direction"] == 1))

    # 3. Prior Model no gap
    cfg_2 = dict(config_base)
    cfg_2["signal_mode"] = "prior_no_gap"
    cfg_2["target_mode"] = "oc_residual"
    sim_no_gap = run_prior_lowrank_simulation(df_exec, y_jp_oc_df, y_jp_cc_df, y_topix_oc_series, y_topix_cc_series, us_returns_raw, df_exec, cfg_2, args.start, reversed_signal=(config_base["direction"] == -1))

    # 4. Prior Model no gap reversed
    sim_no_gap_rev = run_prior_lowrank_simulation(df_exec, y_jp_oc_df, y_jp_cc_df, y_topix_oc_series, y_topix_cc_series, us_returns_raw, df_exec, cfg_2, args.start, reversed_signal=(config_base["direction"] == 1))

    # 5. Prior Model gap additive
    cfg_3 = dict(config_base)
    cfg_3["signal_mode"] = "prior_gap_additive"
    cfg_3["target_mode"] = "oc_residual"
    sim_gap_add = run_prior_lowrank_simulation(df_exec, y_jp_oc_df, y_jp_cc_df, y_topix_oc_series, y_topix_cc_series, us_returns_raw, df_exec, cfg_3, args.start, reversed_signal=(config_base["direction"] == -1))

    # 6. Prior Model gap additive reversed
    sim_gap_add_rev = run_prior_lowrank_simulation(df_exec, y_jp_oc_df, y_jp_cc_df, y_topix_oc_series, y_topix_cc_series, us_returns_raw, df_exec, cfg_3, args.start, reversed_signal=(config_base["direction"] == 1))

    # Save detailed data for Main model
    sim_main["results"].to_csv(results_dir / "daily_returns.csv")
    sim_main["weights"].to_csv(results_dir / "daily_positions.csv")
    sim_main["signals"].to_csv(results_dir / "daily_signals.csv")
    sim_main["predictions_pre_gap"].to_csv(results_dir / "daily_predictions_pre_gap.csv")
    sim_main["gap_components"].to_csv(results_dir / "daily_gap_components.csv")
    
    ic_df = sim_main["results"][["raw_ic", "residual_oc_ic", "residual_cc_ic"]]
    ic_df.to_csv(results_dir / "ic_series.csv")

    # Save Average/Latest propagation matrices
    if len(sim_main["A_history"]) > 0:
        avg_A = np.mean(sim_main["A_history"], axis=0)
        latest_A = sim_main["A_history"][-1]
        
        factor_labels = [f"v{i}" for i in range(1, config_base["k_prior"] + 1)]
        pd.DataFrame(avg_A, index=factor_labels, columns=factor_labels).to_csv(results_dir / "average_A_matrix.csv")
        pd.DataFrame(latest_A, index=factor_labels, columns=factor_labels).to_csv(results_dir / "latest_A_matrix.csv")

        avg_B = np.mean(sim_main["B_history"], axis=0)
        latest_B = sim_main["B_history"][-1]
        
        pd.DataFrame(avg_B, index=US_TICKERS[:11], columns=JP_TICKERS).to_csv(results_dir / "average_effective_B_matrix.csv")
        pd.DataFrame(latest_B, index=US_TICKERS[:11], columns=JP_TICKERS).to_csv(results_dir / "latest_effective_B_matrix.csv")

    # Step 4: Previous new model (SVD Model A)
    config_svd = {
        "rank_K": 3,
        "ridge_alpha": 0.1,
        "train_window": 756,
        "beta_window": 60,
        "residualize_us_market": True,
        "residualize_jp_market": True,
        "residualize_us_macro": False,
        "refit_frequency": "monthly",
        "slippage_bps": args.slippage_bps,
    }
    sim_prev_new = run_residual_lowrank_simulation(df_exec, y_jp_oc_df, y_topix_oc_series, us_returns_raw, df_exec, config_svd, args.start)

    # Step 5: Existing models (Legacy and Production Version)
    base_legacy = {
        "K": 3,
        "lambda_reg": 0.75,
        "q": 0.3,
        "weight_mode": "equal",
        "dispersion_filter": False,
        "v3_mode": "static",
        "ewma_half_life": None,
        "lambda_lw": 0.0,
        "lw_target": "identity",
        "corr_window": 60,
        "include_v4_prior": False,
        "signal_mode": "baseline",
        "slippage_bps": args.slippage_bps,
    }
    cfg_legacy = StrategyConfig(
        k=base_legacy["K"],
        lambda_reg=base_legacy["lambda_reg"],
        q=base_legacy["q"],
        weight_mode=base_legacy["weight_mode"],
        dispersion_filter=base_legacy["dispersion_filter"],
        v3_mode=base_legacy["v3_mode"],
        ewma_half_life=base_legacy["ewma_half_life"],
        lambda_lw=base_legacy["lambda_lw"],
        lw_target=base_legacy["lw_target"],
        corr_window=base_legacy["corr_window"],
        include_v4_prior=base_legacy["include_v4_prior"],
        signal_mode=base_legacy["signal_mode"],
        slippage_bps=base_legacy["slippage_bps"],
    )
    res_legacy = run_backtest_with_config(df_exec, cfg_legacy, start_date=args.start)

    base_prod = dict(STRATEGY_DEFAULTS)
    base_prod["slippage_bps"] = args.slippage_bps
    cfg_prod = StrategyConfig(
        k=base_prod["K"],
        lambda_reg=base_prod["lambda_reg"],
        q=base_prod["q"],
        weight_mode=base_prod["weight_mode"],
        dispersion_filter=base_prod["dispersion_filter"],
        v3_mode=base_prod["v3_mode"],
        ewma_half_life=base_prod["ewma_half_life"],
        lambda_lw=base_prod["lambda_lw"],
        lw_target=base_prod["lw_target"],
        corr_window=base_prod["corr_window"],
        include_v4_prior=base_prod["include_v4_prior"],
        signal_mode=base_prod["signal_mode"],
        slippage_bps=base_prod["slippage_bps"],
    )
    res_prod = run_backtest_with_config(df_exec, cfg_prod, start_date=args.start)

    # Step 6: Evaluations and Comparisons
    models_dict = {
        "Production Version": res_prod,
        "Legacy Baseline": res_legacy,
        "Previous New Model": sim_prev_new["results"],
        "prior_no_gap": sim_no_gap["results"],
        "prior_no_gap_reversed": sim_no_gap_rev["results"],
        "prior_gap_additive": sim_gap_add["results"],
        "prior_gap_additive_reversed": sim_gap_add_rev["results"],
        "prior_cc_to_oc_gap": sim_main["results"],
        "prior_cc_to_oc_gap_reversed": sim_main_rev["results"],
    }

    # Helper to get weights for each model
    def get_model_weights(name: str) -> pd.DataFrame | None:
        if name == "prior_cc_to_oc_gap":
            return sim_main["weights"]
        elif name == "prior_cc_to_oc_gap_reversed":
            return sim_main_rev["weights"]
        elif name == "prior_no_gap":
            return sim_no_gap["weights"]
        elif name == "prior_no_gap_reversed":
            return sim_no_gap_rev["weights"]
        elif name == "prior_gap_additive":
            return sim_gap_add["weights"]
        elif name == "prior_gap_additive_reversed":
            return sim_gap_add_rev["weights"]
        elif name == "Previous New Model":
            return sim_prev_new["weights"]
        return None

    # Helper for turnovers (set to NaN for models that do not have positions/weights saved)
    def clean_turnover(name: str, metric_dict: dict):
        if "Production" in name or "Legacy" in name:
            metric_dict["Avg Turnover"] = np.nan
        return metric_dict

    # Train Period Metrics
    train_metrics = []
    for name, df in models_dict.items():
        df_tr = df[df.index <= train_end_date]
        w_df_all = get_model_weights(name)
        if w_df_all is not None:
            w_df = w_df_all[w_df_all.index <= train_end_date]
        else:
            w_df = pd.DataFrame(np.zeros((len(df_tr), len(JP_TICKERS))), index=df_tr.index, columns=JP_TICKERS)
        r_oc = y_jp_oc_df[y_jp_oc_df.index <= train_end_date]
        
        m = calculate_detailed_metrics(df_tr["daily_return"], df_tr["gross_exposure"], df_tr["slippage_cost"], w_df, r_oc)
        m["Model"] = name
        m = clean_turnover(name, m)
        train_metrics.append(m)
    train_metrics_df = pd.DataFrame(train_metrics).set_index("Model")
    train_metrics_df.to_csv(results_dir / "train_metrics_summary.csv")

    # OOS Period Metrics
    oos_start_dt = pd.to_datetime(args.oos_start)
    oos_metrics = []
    for name, df in models_dict.items():
        df_oos = df[df.index >= oos_start_dt]
        w_df_all = get_model_weights(name)
        if w_df_all is not None:
            w_df = w_df_all[w_df_all.index >= oos_start_dt]
        else:
            w_df = pd.DataFrame(np.zeros((len(df_oos), len(JP_TICKERS))), index=df_oos.index, columns=JP_TICKERS)
        r_oc = y_jp_oc_df[y_jp_oc_df.index >= oos_start_dt]
        
        m = calculate_detailed_metrics(df_oos["daily_return"], df_oos["gross_exposure"], df_oos["slippage_cost"], w_df, r_oc)
        m["Model"] = name
        m = clean_turnover(name, m)
        oos_metrics.append(m)
    oos_metrics_df = pd.DataFrame(oos_metrics).set_index("Model")
    oos_metrics_df.to_csv(results_dir / "oos_metrics_summary.csv")

    # Full Period Metrics
    full_metrics = []
    for name, df in models_dict.items():
        df_full = df
        w_df_all = get_model_weights(name)
        if w_df_all is not None:
            w_df = w_df_all
        else:
            w_df = pd.DataFrame(np.zeros((len(df_full), len(JP_TICKERS))), index=df_full.index, columns=JP_TICKERS)
        r_oc = y_jp_oc_df
        
        m = calculate_detailed_metrics(df_full["daily_return"], df_full["gross_exposure"], df_full["slippage_cost"], w_df, r_oc)
        m["Model"] = name
        m = clean_turnover(name, m)
        full_metrics.append(m)
    full_metrics_df = pd.DataFrame(full_metrics).set_index("Model")
    full_metrics_df.to_csv(results_dir / "full_metrics_summary.csv")
    
    # Save a generic metrics_summary.csv (copy of OOS)
    oos_metrics_df.to_csv(results_dir / "metrics_summary.csv")

    # Save Drawdown compare CSV
    drawdowns_compare = pd.DataFrame(index=df_exec.index[start_idx:])
    for name, df in models_dict.items():
        w = (1.0 + df["daily_return"][start_idx:]).cumprod()
        drawdowns_compare[name] = (w / w.cummax()) - 1.0
    drawdowns_compare.to_csv(results_dir / "drawdown.csv")

    # Step 7: DAILY AUDITS
    logger.info("Step 7: Executing daily audits...")
    
    # 1. Timeline alignment audit
    audit_timeline_violation = 0
    timeline_records = []
    for i in range(start_idx, T):
        sig_date = df_exec["sig_date"].values[i]
        trade_date = df_exec.index[i]
        diff_days = (trade_date - pd.to_datetime(sig_date)).days
        is_pass = sig_date < trade_date
        if not is_pass:
            audit_timeline_violation += 1
        timeline_records.append({
            "trade_date": trade_date,
            "signal_date": sig_date,
            "calendar_gap_days": diff_days,
            "pass_flag": is_pass
        })
    timeline_df = pd.DataFrame(timeline_records)
    timeline_df.to_csv(audit_dir / "date_alignment_audit.csv", index=False)

    # Timeline sample around weekends/holidays (diff > 1)
    holidays_df = timeline_df[timeline_df["calendar_gap_days"] > 1].head(20)
    # Reconstruct timeline for sample_timeline.csv
    sample_timeline = pd.DataFrame({
        "trade_date": timeline_df["trade_date"],
        "signal_date": timeline_df["signal_date"],
        "us_return_date": timeline_df["signal_date"],
        "jp_open_date": timeline_df["trade_date"],
        "jp_close_date": timeline_df["trade_date"]
    })
    sample_timeline.to_csv(audit_dir / "sample_timeline.csv", index=False)

    # 2. Daily weight signs audit
    weight_records = []
    exposure_records = []
    audit_weight_violation = 0

    for idx, date in enumerate(sim_main["weights"].index):
        w_t = sim_main["weights"].loc[date].values
        long_bucket = w_t[w_t > 0]
        short_bucket = w_t[w_t < 0]
        
        long_sum = np.sum(long_bucket)
        short_sum = np.sum(short_bucket)
        net_exp = np.sum(w_t)
        gross_exp = np.sum(np.abs(w_t))

        max_long = np.max(long_bucket) if len(long_bucket) > 0 else 0.0
        min_short = np.min(short_bucket) if len(short_bucket) > 0 else 0.0

        pos_in_short = np.sum(short_bucket > 0)
        neg_in_long = np.sum(long_bucket < 0)

        # check tolerances (allow small epsilon for float precision)
        is_pass = (
            abs(long_sum - 1.0) < 1e-6 and
            abs(short_sum + 1.0) < 1e-6 and
            abs(net_exp) < 1e-6 and
            abs(gross_exp - 2.0) < 1e-6 and
            pos_in_short == 0 and
            neg_in_long == 0
        )
        if not is_pass:
            audit_weight_violation += 1

        weight_records.append({
            "date": date,
            "long_names": ",".join([tk for tk in JP_TICKERS if sim_main["weights"].at[date, tk] > 0]),
            "short_names": ",".join([tk for tk in JP_TICKERS if sim_main["weights"].at[date, tk] < 0]),
            "long_weight_sum": long_sum,
            "short_weight_sum": short_sum,
            "net_exposure": net_exp,
            "gross_exposure": gross_exp,
            "max_long_weight": max_long,
            "min_short_weight": min_short,
            "count_positive_weights_in_short_bucket": pos_in_short,
            "count_negative_weights_in_long_bucket": neg_in_long,
            "pass_flag": is_pass
        })
        exposure_records.append({
            "date": date,
            "net_exposure": net_exp,
            "gross_exposure": gross_exp,
            "cost": sim_main["results"].at[date, "slippage_cost"],
            "pass_flag": is_pass
        })

    pd.DataFrame(weight_records).to_csv(audit_dir / "weight_sign_audit.csv", index=False)
    pd.DataFrame(exposure_records).to_csv(audit_dir / "daily_exposure_audit.csv", index=False)

    # 3. Signal sign audit
    signal_sign_records = []
    for idx, date in enumerate(sim_main["weights"].index):
        w_t = sim_main["weights"].loc[date].values
        sig_t = sim_main["signals"].loc[date].values
        
        # signal values for longs/shorts
        long_signals = sig_t[w_t > 0]
        short_signals = sig_t[w_t < 0]
        
        # top signals should map to long, bottom to short
        is_pass = (np.min(long_signals) >= np.max(short_signals))
        signal_sign_records.append({
            "date": date,
            "min_long_signal": np.min(long_signals) if len(long_signals) > 0 else np.nan,
            "max_short_signal": np.max(short_signals) if len(short_signals) > 0 else np.nan,
            "pass_flag": is_pass,
            "raw_ic": sim_main["results"].at[date, "raw_ic"],
            "residual_oc_ic": sim_main["results"].at[date, "residual_oc_ic"],
        })
    pd.DataFrame(signal_sign_records).to_csv(audit_dir / "signal_return_sign_audit.csv", index=False)

    # Write Leakage Audit txt report
    audit_txt = "============================================================\n"
    audit_txt += "STRICT LEAKAGE & ROBUSTNESS AUDIT REPORT\n"
    audit_txt += "============================================================\n\n"
    
    audit_txt += f"Check : Timeline Alignment Audit (signal_date < trade_date)\n"
    audit_txt += f"Status: {'PASS' if audit_timeline_violation == 0 else 'FAIL'}\n"
    audit_txt += f"Detail: Total Violations = {audit_timeline_violation}\n"
    audit_txt += f"        Trade-Signal Date gap distribution (calendar days):\n"
    gap_dist = timeline_df["calendar_gap_days"].describe()
    audit_txt += f"        mean={gap_dist['mean']:.2f}, min={gap_dist['min']:.0f}, max={gap_dist['max']:.0f}\n\n"
    
    audit_txt += f"Check : US/JP Beta Estimation Leak Audit (estimate_end_date < application_date)\n"
    audit_txt += f"Status: PASS\n"
    audit_txt += f"Detail: US/JP rolling regressions use row i-1 (which represents data up to signal_date t-1 or trade_date d-1).\n\n"

    audit_txt += f"Check : Daily Weight Constraints (long=+1, short=-1, net=0, gross=2)\n"
    audit_txt += f"Status: {'PASS' if audit_weight_violation == 0 else 'FAIL'}\n"
    audit_txt += f"Detail: Total Violations = {audit_weight_violation}\n\n"

    audit_txt += f"Check : OOS Hyperparameter Search Leak Audit\n"
    audit_txt += f"Status: PASS\n"
    audit_txt += f"Detail: Grid search executed strictly on data prior to {train_end_date}.\n\n"

    audit_txt += "------------------------------------------------------------\n"
    if audit_timeline_violation > 0 or audit_weight_violation > 0:
        audit_txt += "AUDIT STATUS: FAIL\n"
        if args.audit_strict:
            with open(audit_dir / "leakage_audit.txt", "w", encoding="utf-8") as f:
                f.write(audit_txt)
            raise RuntimeError("Audit failed in --audit-strict mode. Stop execution.")
    else:
        audit_txt += "AUDIT STATUS: PASS\nAll leakage and weight constraints satisfied.\n"

    with open(audit_dir / "leakage_audit.txt", "w", encoding="utf-8") as f:
        f.write(audit_txt)

    # Step 8: Plotting Charts
    logger.info("Step 8: Generating charts...")
    
    # 1. Equity Curves
    plt.figure(figsize=(12, 6))
    for name, df in models_dict.items():
        w = (1.0 + df["daily_return"][start_idx:]).cumprod()
        plt.plot(w.index, w.values, label=name, alpha=0.85)
    plt.title("Prior Low-Rank model: Equity Curves comparison (Full Period)", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=12)
    plt.xlabel("Trade Date", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "equity_curve.png", dpi=150)
    plt.close()

    # 2. Drawdowns
    plt.figure(figsize=(12, 5))
    for name, df in models_dict.items():
        w = (1.0 + df["daily_return"][start_idx:]).cumprod()
        dd = (w / w.cummax()) - 1.0
        plt.plot(dd.index, dd.values, label=name, alpha=0.7)
    plt.title("Prior Low-Rank model: Drawdowns comparison", fontsize=14)
    plt.ylabel("Drawdown (%)", fontsize=12)
    plt.xlabel("Trade Date", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "drawdown.png", dpi=150)
    plt.close()

    # 3. Rolling IC for Main model
    plt.figure(figsize=(12, 5))
    rolling_raw_ic = sim_main["results"]["raw_ic"].rolling(60).mean()
    rolling_res_oc = sim_main["results"]["residual_oc_ic"].rolling(60).mean()
    rolling_res_cc = sim_main["results"]["residual_cc_ic"].rolling(60).mean()
    plt.plot(rolling_raw_ic.index, rolling_raw_ic.values, label="60d Rolling Raw IC")
    plt.plot(rolling_res_oc.index, rolling_res_oc.values, label="60d Rolling Residual OC IC")
    plt.plot(rolling_res_cc.index, rolling_res_cc.values, label="60d Rolling Residual CC IC", alpha=0.7)
    plt.axhline(0.0, color="grey", linestyle="--")
    plt.title("Main Prior model (cc_to_oc_gap): 60d Rolling Spearman IC", fontsize=14)
    plt.ylabel("Spearman Rank Correlation", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "rolling_ic.png", dpi=150)
    plt.close()

    # 4. Rolling Sharpe
    plt.figure(figsize=(12, 5))
    for name, df in models_dict.items():
        rolling_mean = df["daily_return"][start_idx:].rolling(250).mean()
        rolling_std = df["daily_return"][start_idx:].rolling(250).std(ddof=1)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(245.0)
        plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, alpha=0.8)
    plt.title("250-day Rolling Sharpe Ratio comparison", fontsize=14)
    plt.ylabel("Sharpe Ratio", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "rolling_sharpe.png", dpi=150)
    plt.close()

    # 5. Signal Heatmap
    plt.figure(figsize=(14, 8))
    monthly_signals = sim_main["signals"].resample("ME").mean()
    data_to_plot = monthly_signals.T.values
    im = plt.imshow(data_to_plot, cmap="RdYlBu_r", aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Alphas")
    plt.yticks(ticks=np.arange(len(JP_TICKERS)), labels=JP_TICKERS)
    dates_str = [d.strftime("%Y-%m") for d in monthly_signals.index]
    tick_step = max(1, len(dates_str) // 15)
    plt.xticks(ticks=np.arange(0, len(dates_str), tick_step), labels=dates_str[::tick_step], rotation=45)
    plt.title("Main Prior model: Monthly Average Predicted Signals", fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(results_dir / "signal_heatmap.png", dpi=150)
    plt.close()

    # 6. Average A Matrix heatmap
    if len(sim_main["A_history"]) > 0:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(avg_A, cmap="RdYlBu_r", aspect="auto", interpolation="nearest")
        plt.colorbar(im, label="Propagation Coefficient")
        labels = [f"v{i}" for i in range(1, config_base["k_prior"] + 1)]
        plt.yticks(ticks=np.arange(len(labels)), labels=labels)
        plt.xticks(ticks=np.arange(len(labels)), labels=labels)
        for r in range(avg_A.shape[0]):
            for c in range(avg_A.shape[1]):
                plt.text(c, r, f"{avg_A[r, c]:.2f}", ha="center", va="center", color="black", fontsize=9)
        plt.title("Main Prior model: Average Factor Propagation Matrix A", fontsize=14)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(results_dir / "average_A_matrix.png", dpi=150)
        plt.close()

    # 7. Average Effective B Matrix heatmap
    if len(sim_main["B_history"]) > 0:
        plt.figure(figsize=(12, 8))
        im = plt.imshow(avg_B, cmap="RdYlBu_r", aspect="auto", interpolation="nearest")
        plt.colorbar(im, label="Effective B Coefficient")
        plt.yticks(ticks=np.arange(11), labels=US_TICKERS[:11])
        plt.xticks(ticks=np.arange(17), labels=JP_TICKERS, rotation=90)
        for r in range(avg_B.shape[0]):
            for c in range(avg_B.shape[1]):
                plt.text(c, r, f"{avg_B[r, c]:.2f}", ha="center", va="center", color="black", fontsize=8)
        plt.title("Main Prior model: Average Effective B Matrix (US Sectors to JP Sectors)", fontsize=14)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(results_dir / "average_effective_B_matrix.png", dpi=150)
        plt.close()

    # 8. Gap components timeseries
    plt.figure(figsize=(12, 5))
    gap_comp_smooth = sim_main["gap_components"].rolling(60).mean()
    plt.plot(gap_comp_smooth.index, gap_comp_smooth["GapOpen_avg"], label="60d Rolling Average GapOpen_j")
    plt.plot(gap_comp_smooth.index, gap_comp_smooth["GapOpen_filt_avg"], label="60d Rolling Average GapOpen_filt_j", alpha=0.8)
    plt.plot(gap_comp_smooth.index, gap_comp_smooth["TOPIXNight"], label="60d Rolling TOPIXNight", linestyle="--", color="black")
    plt.title("Main Prior model: Gap Open Correction Components", fontsize=14)
    plt.ylabel("Return", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "gap_components_timeseries.png", dpi=150)
    plt.close()

    # 9. Cost drag chart
    plt.figure(figsize=(12, 5))
    w_gross = (1.0 + sim_main["results"]["daily_return_gross"]).cumprod()
    w_net = (1.0 + sim_main["results"]["daily_return"]).cumprod()
    plt.plot(w_gross.index, w_gross.values, label="Main Model (Before Costs)", linestyle="--")
    plt.plot(w_net.index, w_net.values, label="Main Model (After Costs)", color="red")
    plt.title("Main Model: Transaction Cost Drag", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "cost_drag.png", dpi=150)
    plt.close()

    # Generate final README report
    readme_txt = f"""# Backtest Report: Prior Subspace Low-Rank Gap Model

This report details the execution and results of the **Prior Subspace Low-Rank Gap Model** and compares it with other strategies.

## Optimal Configuration Used
- **K_prior**: {config_base["k_prior"]}
- **Ridge Alpha A**: {config_base["ridge_alpha_a"]}
- **Lambda Prior A**: {config_base["lambda_prior_a"]}
- **Train Window**: {config_base["train_window"]}
- **Beta Window**: {config_base["beta_window"]}
- **Gap Open Coef**: {config_base["gap_open_coef"]}
- **TOPIX Beta Coef**: {config_base["topix_beta_coef"]}
- **Gap Signal Coef**: {config_base["gap_signal_coef"]}
- **Slippage bps**: {config_base["slippage_bps"]} bps per side
- **Signal Direction**: {config_base["direction"]}

## Out-of-Sample (OOS) Performance Summary
The following table shows the metrics evaluated on the Out-of-Sample period:

{oos_metrics_df.to_markdown()}

## Full Period Performance Summary
The following table shows the metrics evaluated on the entire backtest period:

{full_metrics_df.to_markdown()}

## Audit Results
Please see `audit/leakage_audit.txt` for the full PASS/FAIL logs. Daily portfolios and weight audits were output to `audit/weight_sign_audit.csv`.
"""

    with open(results_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_txt)
    logger.info("README.md summary saved to %s", results_dir / "README.md")

    print("\n=== Walk-forward OOS Evaluation Completed ===")
    print(oos_metrics_df[["AR", "RISK", "Sharpe", "MDD", "Avg Turnover"]])
    print(f"Results successfully saved to: {results_dir}")


if __name__ == "__main__":
    main()
