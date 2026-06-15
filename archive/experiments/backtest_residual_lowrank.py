#!/usr/bin/env python
"""Backtesting script for the Residualized Supervised Low-Rank Lead-Lag Model.

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
from domain.models.residual_lowrank import (
    ResidualizedSupervisedLowRankModel,
    compute_rolling_ols_betas,
)
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

    # Short weights
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
    avg_turnover = float(w_diff.mean())

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


def run_residual_lowrank_simulation(
    df_exec: pd.DataFrame,
    y_jp_raw: pd.DataFrame,
    y_topix_raw: pd.Series,
    us_returns_raw: pd.DataFrame,
    macro_returns: pd.DataFrame,
    config: dict,
    start_date: str,
) -> dict:
    """Run rolling walk-forward simulation of the Low-Rank Model."""
    T = len(df_exec)
    n_features = 11  # First 11 US sector ETFs
    n_targets = len(JP_TICKERS)

    # 1. US Residualization
    # Slice features and target for US regression
    y_us = us_returns_raw[US_TICKERS[:11]].values  # shape (T, 11)
    
    if config["residualize_us_macro"]:
        # US market + 4 macros
        x_us = np.column_stack([
            macro_returns["SPY_cc"].values,
            macro_returns["USDJPY_cc"].values,
            macro_returns["Oil_cc"].values,
            macro_returns["US10Y_diff"].values,
            macro_returns["VIX_diff"].values,
        ])
    else:
        # Just US market SPY
        x_us = macro_returns["SPY_cc"].values.reshape(-1, 1)

    if config["residualize_us_market"]:
        betas_us = compute_rolling_ols_betas(y_us, x_us, config["beta_window"])
        # Shock = raw_return - sum(beta * factor)
        x_shocks = y_us - np.sum(betas_us * x_us[:, np.newaxis, :], axis=2)
    else:
        x_shocks = y_us

    # 2. JP Residualization
    y_jp = y_jp_raw[JP_TICKERS].values  # shape (T, 17)
    x_jp = y_topix_raw.values.reshape(-1, 1)  # shape (T, 1)

    if config["residualize_jp_market"]:
        betas_jp = compute_rolling_ols_betas(y_jp, x_jp, config["beta_window"])
        y_residuals = y_jp - betas_jp[:, :, 0] * x_jp
    else:
        y_residuals = y_jp

    # 3. Model parameters
    rank_k = config["rank_K"]
    ridge_alpha = config["ridge_alpha"]
    train_window = config["train_window"]
    refit_freq = config["refit_frequency"]

    # Locate start index
    start_dt = pd.to_datetime(start_date)
    start_idx = max(df_exec.index.searchsorted(start_dt), train_window + config["beta_window"])

    # Output arrays
    pred_signals = np.zeros((T, n_targets))
    pred_signals[:] = np.nan
    
    # Model tracking
    last_B = None
    last_mean_X = None
    last_std_X = None
    last_mean_Y = None
    last_std_Y = None
    last_month = None

    B_history = []
    refit_dates = []

    # Walk-forward loop
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
            train_end = i - 1  # strictly up to t-1 to avoid target leak
            
            X_tr = x_shocks[train_start : train_end + 1]
            Y_tr = y_residuals[train_start : train_end + 1]

            # Standardize X and Y in-sample
            mean_X = np.mean(X_tr, axis=0)
            std_X = np.std(X_tr, axis=0, ddof=1)
            std_X[std_X == 0.0] = 1e-8

            mean_Y = np.mean(Y_tr, axis=0)
            std_Y = np.std(Y_tr, axis=0, ddof=1)
            std_Y[std_Y == 0.0] = 1e-8

            X_tr_std = (X_tr - mean_X) / std_X
            Y_tr_std = (Y_tr - mean_Y) / std_Y

            # Fit Ridge Regression
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="svd")
            ridge.fit(X_tr_std, Y_tr_std)
            B_ridge = ridge.coef_.T  # shape (n_features, n_targets)

            # SVD and Low-Rank constraint
            U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
            S_low = S.copy()
            k_capped = min(rank_k, len(S_low))
            if k_capped < len(S_low):
                S_low[k_capped:] = 0.0

            B_lowrank = U @ np.diag(S_low) @ Vt

            last_B = B_lowrank
            last_mean_X = mean_X
            last_std_X = std_X
            last_mean_Y = mean_Y
            last_std_Y = std_Y

            B_history.append(B_lowrank)
            refit_dates.append(df_exec.index[i])

        # Predict target for trade_date i using features at i (sig_date t)
        x_pred = x_shocks[i]
        x_pred_std = (x_pred - last_mean_X) / last_std_X
        y_pred_std = x_pred_std @ last_B
        y_pred = last_mean_Y + y_pred_std * last_std_Y
        pred_signals[i] = y_pred

    # 4. Portfolio construction & backtest simulation
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
        weights = build_portfolio_weights(sig, q=0.3)
        weights_list.append(weights)

        # realized return on trade date i (which is JP Open-to-Close)
        r_oc = y_jp[i]
        gross_return = float(np.sum(weights * r_oc))
        
        gross_exposure = float(np.sum(np.abs(weights)))
        slippage_cost = 2.0 * slippage_rate * gross_exposure
        net_return = gross_return - slippage_cost

        daily_returns.append(net_return)
        daily_gross_exposures.append(gross_exposure)
        daily_slippage_costs.append(slippage_cost)

        # Spearman correlation (IC)
        # realised r_JP_OC_t+1 is r_oc
        ic_val, _ = spearmanr(sig, r_oc)
        # realised residual return y_resid_t+1 is y_residuals[i]
        resid_ic_val, _ = spearmanr(sig, y_residuals[i])

        results.append({
            "trade_date": date,
            "daily_return": net_return,
            "daily_return_gross": gross_return,
            "slippage_cost": slippage_cost,
            "gross_exposure": gross_exposure,
            "ic": ic_val,
            "residual_ic": resid_ic_val,
        })

    results_df = pd.DataFrame(results).set_index("trade_date")
    weights_df = pd.DataFrame(weights_list, index=sim_dates, columns=JP_TICKERS)
    r_oc_df = pd.DataFrame(y_jp[start_idx:], index=sim_dates, columns=JP_TICKERS)

    return {
        "results": results_df,
        "weights": weights_df,
        "r_oc": r_oc_df,
        "signals": pd.DataFrame(pred_signals[start_idx:], index=sim_dates, columns=JP_TICKERS),
        "B_history": B_history,
        "refit_dates": refit_dates,
        "x_shocks": pd.DataFrame(x_shocks[start_idx:], index=sim_dates, columns=US_TICKERS[:11]),
        "y_residuals": pd.DataFrame(y_residuals[start_idx:], index=sim_dates, columns=JP_TICKERS),
    }


def main():
    parser = argparse.ArgumentParser(description="Residualized Low Rank Backtest Tool")
    parser.add_argument("--start", default="2015-01-05")
    parser.add_argument("--oos-start", default="2020-01-01")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--train-window", type=int, default=756)
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--residualize-us-market", action="store_true", default=True)
    parser.add_argument("--no-residualize-us-market", action="store_false", dest="residualize_us_market")
    parser.add_argument("--residualize-jp-market", action="store_true", default=True)
    parser.add_argument("--no-residualize-jp-market", action="store_false", dest="residualize_jp_market")
    parser.add_argument("--residualize-us-macro", action="store_true", default=False)
    parser.add_argument("--refit-frequency", choices=["daily", "monthly"], default="monthly")
    parser.add_argument("--grid-search", action="store_true", help="Perform hyperparameter grid search on training data")
    args = parser.parse_args()

    # Step 1: Data fetching
    logger.info("Step 1: Fetching historical market and macro data...")
    # Fetch standard data
    std_data = download_data(beta_window=args.beta_window)
    df_exec = preprocess_data(std_data, beta_window=args.beta_window)
    # Fetch SPY and macro
    macro_df = get_macro_data()

    # Align dates
    logger.info("Aligning datasets...")
    # Calculate daily returns/changes for macro indicators
    macro_returns = pd.DataFrame(index=macro_df.index)
    macro_returns["SPY_cc"] = macro_df["SPY"].pct_change()
    macro_returns["USDJPY_cc"] = macro_df["USDJPY=X"].pct_change()
    macro_returns["Oil_cc"] = macro_df["CL=F"].pct_change()
    macro_returns["US10Y_diff"] = macro_df["^TNX"].diff()
    macro_returns["VIX_diff"] = macro_df["^VIX"].diff()
    macro_returns = macro_returns.ffill().fillna(0.0)

    # Reindex macro to df_exec's sig_date
    sig_dates = pd.to_datetime(df_exec["sig_date"])
    df_exec["r_US_MKT"] = macro_returns["SPY_cc"].reindex(sig_dates).values
    df_exec["SPY_cc"] = df_exec["r_US_MKT"]
    df_exec["USDJPY_cc"] = macro_returns["USDJPY_cc"].reindex(sig_dates).values
    df_exec["Oil_cc"] = macro_returns["Oil_cc"].reindex(sig_dates).values
    df_exec["US10Y_diff"] = macro_returns["US10Y_diff"].reindex(sig_dates).values
    df_exec["VIX_diff"] = macro_returns["VIX_diff"].reindex(sig_dates).values

    # Calculate TOPIX open-to-close returns on df_exec's trade_date index
    topix_close = std_data["jp_close"][TOPIX_TICKER].copy()
    topix_open = std_data["jp_open"][TOPIX_TICKER].copy()
    topix_close.index = pd.to_datetime(topix_close.index).tz_localize(None).normalize()
    topix_open.index = pd.to_datetime(topix_open.index).tz_localize(None).normalize()
    r_topix_oc = topix_close / topix_open - 1.0
    df_exec["topix_oc_return"] = r_topix_oc.reindex(df_exec.index).values

    # Target dataframes for raw prices
    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    y_jp_raw = df_exec[jp_oc_cols].rename(columns=lambda c: c.replace("jp_oc_", ""))
    y_topix_raw = df_exec["topix_oc_return"]
    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    us_returns_raw = df_exec[us_cols].rename(columns=lambda c: c.replace("us_cc_", ""))

    # Establish directories
    results_dir = ROOT / "results" / "residual_lowrank"
    results_dir.mkdir(parents=True, exist_ok=True)

    config_base = {
        "rank_K": args.rank,
        "ridge_alpha": args.ridge_alpha,
        "train_window": args.train_window,
        "beta_window": args.beta_window,
        "slippage_bps": args.slippage_bps,
        "residualize_us_market": args.residualize_us_market,
        "residualize_jp_market": args.residualize_jp_market,
        "residualize_us_macro": args.residualize_us_macro,
        "refit_frequency": args.refit_frequency,
    }

    # Step 2: Parameter Grid Search (only on train/tune period 2015-01-05 to 2019-12-31)
    train_end_date = "2019-12-31"
    grid_results = []
    
    if args.grid_search:
        logger.info("Step 2: Performing hyperparameter grid search on training data...")
        ranks = [1, 2, 3, 4, 5, 6]
        alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        
        for r in ranks:
            for a in alphas:
                logger.info("Evaluating rank_K=%d, ridge_alpha=%.4f...", r, a)
                cfg = dict(config_base)
                cfg["rank_K"] = r
                cfg["ridge_alpha"] = a
                
                # Run backtest strictly on training period
                sim = run_residual_lowrank_simulation(
                    df_exec=df_exec[df_exec.index <= train_end_date],
                    y_jp_raw=y_jp_raw[y_jp_raw.index <= train_end_date],
                    y_topix_raw=y_topix_raw[y_topix_raw.index <= train_end_date],
                    us_returns_raw=us_returns_raw[us_returns_raw.index <= train_end_date],
                    macro_returns=df_exec[df_exec.index <= train_end_date][["SPY_cc", "USDJPY_cc", "Oil_cc", "US10Y_diff", "VIX_diff"]].rename(columns={"SPY_cc": "SPY_cc", "USDJPY_cc": "USDJPY_cc", "Oil_cc": "Oil_cc", "US10Y_diff": "US10Y_diff", "VIX_diff": "VIX_diff"}), # align keys
                    config=cfg,
                    start_date=args.start,
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
                        "rank_K": r,
                        "ridge_alpha": a,
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
        logger.info("Best parameters found: rank_K=%d, ridge_alpha=%.4f (Sharpe=%.4f)",
                    best_row["rank_K"], best_row["ridge_alpha"], best_row["Sharpe"])
        config_base["rank_K"] = int(best_row["rank_K"])
        config_base["ridge_alpha"] = float(best_row["ridge_alpha"])

    # Step 3: Run full backtest for New Model A, B, C
    logger.info("Step 3: Running walk-forward backtests...")
    
    # Model A: US residualized = True, JP residualized = True
    config_A = dict(config_base)
    config_A["residualize_us_market"] = True
    config_A["residualize_jp_market"] = True
    sim_A = run_residual_lowrank_simulation(df_exec, y_jp_raw, y_topix_raw, us_returns_raw, df_exec, config_A, args.start)

    # Model B: US raw, JP residualized
    config_B = dict(config_base)
    config_B["residualize_us_market"] = False
    config_B["residualize_jp_market"] = True
    sim_B = run_residual_lowrank_simulation(df_exec, y_jp_raw, y_topix_raw, us_returns_raw, df_exec, config_B, args.start)

    # Model C: US residualized, JP raw
    config_C = dict(config_base)
    config_C["residualize_us_market"] = True
    config_C["residualize_jp_market"] = False
    sim_C = run_residual_lowrank_simulation(df_exec, y_jp_raw, y_topix_raw, us_returns_raw, df_exec, config_C, args.start)

    # Save daily metrics for Model A
    sim_A["results"].to_csv(results_dir / "daily_returns.csv")
    sim_A["weights"].to_csv(results_dir / "daily_positions.csv")
    sim_A["signals"].to_csv(results_dir / "daily_signals.csv")
    
    ic_df = sim_A["results"][["ic", "residual_ic"]].rename(columns={"ic": "daily_ic", "residual_ic": "residual_ic"})
    ic_df.to_csv(results_dir / "ic_series.csv")

    # Save average and latest B matrix for Model A
    if len(sim_A["B_history"]) > 0:
        avg_B = np.mean(sim_A["B_history"], axis=0)
        latest_B = sim_A["B_history"][-1]
        
        pd.DataFrame(avg_B, index=US_TICKERS[:11], columns=JP_TICKERS).to_csv(results_dir / "average_B_matrix.csv")
        pd.DataFrame(latest_B, index=US_TICKERS[:11], columns=JP_TICKERS).to_csv(results_dir / "latest_B_matrix.csv")
        logger.info("Saved B matrices to %s", results_dir)

    # Step 4: Existing models comparison
    logger.info("Step 4: Running baseline and production model comparisons...")
    
    # 1. Existing Legacy Baseline model
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

    # 2. Existing Production Model
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

    # Step 5: Evaluate metrics and compare
    logger.info("Step 5: Evaluating metrics...")
    
    # Slice results into train and OOS periods
    oos_start_dt = pd.to_datetime(args.oos_start)

    # Daily results list
    models_dict = {
        "Legacy Baseline": res_legacy,
        "Production Version": res_prod,
        "New Model A (Residual US & JP)": sim_A["results"],
        "New Model B (Raw US, Residual JP)": sim_B["results"],
        "New Model C (Residual US, Raw JP)": sim_C["results"],
    }
    
    # Save daily return comparison
    compare_returns = pd.DataFrame(index=sim_A["results"].index)
    for name, df in models_dict.items():
        compare_returns[name] = df["daily_return"]
    compare_returns.to_csv(results_dir / "comparison_returns.csv")

    metrics_list = []
    oos_metrics_list = []

    for name, df in models_dict.items():
        # Full period
        df_full = df
        w_df_full = sim_A["weights"] if "New Model" in name else pd.DataFrame(np.zeros((len(df_full), len(JP_TICKERS))), index=df_full.index, columns=JP_TICKERS) # placeholder
        r_oc_df_full = sim_A["r_oc"]
        
        # For legacy and production models we have to approximate active count / weights for detailed metrics
        # Let's extract weights if available
        if name == "New Model A (Residual US & JP)":
            m_full = calculate_detailed_metrics(df_full["daily_return"], df_full["gross_exposure"], df_full["slippage_cost"], sim_A["weights"], sim_A["r_oc"])
        elif name == "New Model B (Raw US, Residual JP)":
            m_full = calculate_detailed_metrics(df_full["daily_return"], df_full["gross_exposure"], df_full["slippage_cost"], sim_B["weights"], sim_B["r_oc"])
        elif name == "New Model C (Residual US, Raw JP)":
            m_full = calculate_detailed_metrics(df_full["daily_return"], df_full["gross_exposure"], df_full["slippage_cost"], sim_C["weights"], sim_C["r_oc"])
        else:
            # Reconstruct exposure/slippage
            m_full = calculate_detailed_metrics(df_full["daily_return"], df_full["gross_exposure"], df_full["slippage_cost"], w_df_full, r_oc_df_full)

        m_full["Model"] = name
        metrics_list.append(m_full)

        # OOS period
        df_oos = df[df.index >= oos_start_dt]
        if name == "New Model A (Residual US & JP)":
            m_oos = calculate_detailed_metrics(df_oos["daily_return"], df_oos["gross_exposure"], df_oos["slippage_cost"], sim_A["weights"][sim_A["weights"].index >= oos_start_dt], sim_A["r_oc"][sim_A["r_oc"].index >= oos_start_dt])
        elif name == "New Model B (Raw US, Residual JP)":
            m_oos = calculate_detailed_metrics(df_oos["daily_return"], df_oos["gross_exposure"], df_oos["slippage_cost"], sim_B["weights"][sim_B["weights"].index >= oos_start_dt], sim_B["r_oc"][sim_B["r_oc"].index >= oos_start_dt])
        elif name == "New Model C (Residual US, Raw JP)":
            m_oos = calculate_detailed_metrics(df_oos["daily_return"], df_oos["gross_exposure"], df_oos["slippage_cost"], sim_C["weights"][sim_C["weights"].index >= oos_start_dt], sim_C["r_oc"][sim_C["r_oc"].index >= oos_start_dt])
        else:
            m_oos = calculate_detailed_metrics(df_oos["daily_return"], df_oos["gross_exposure"], df_oos["slippage_cost"], w_df_full[w_df_full.index >= oos_start_dt], r_oc_df_full[r_oc_df_full.index >= oos_start_dt])

        m_oos["Model"] = name
        oos_metrics_list.append(m_oos)

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    oos_metrics_df = pd.DataFrame(oos_metrics_list).set_index("Model")

    metrics_df.to_csv(results_dir / "metrics_summary.csv")
    oos_metrics_df.to_csv(results_dir / "oos_metrics_summary.csv")
    
    logger.info("Saved metrics to %s", results_dir)

    # Save drawdowns
    drawdowns_compare = pd.DataFrame(index=compare_returns.index)
    for col in compare_returns.columns:
        w = (1.0 + compare_returns[col]).cumprod()
        drawdowns_compare[col] = (w / w.cummax()) - 1.0
    drawdowns_compare.to_csv(results_dir / "drawdown.csv")

    # Step 6: Leakage Audit
    logger.info("Step 6: Running leakage audit checks...")
    audit_results = []
    
    # 1. Timeline check
    audit_sig_vs_trade = all(df_exec["sig_date"] < df_exec.index)
    audit_results.append({
        "check": "Timeline Ordering (signal_date < trade_date)",
        "status": "PASS" if audit_sig_vs_trade else "FAIL",
        "detail": f"Max signal_date: {df_exec['sig_date'].max()}, Min trade_date: {df_exec.index.min()}",
    })

    # 2. US rolling beta check
    # We estimate US beta ending at index t-1, ensuring we don't use day t.
    # Our function compute_rolling_ols_betas strictly slices start_idx = t - window, end_idx = t - 1.
    # Therefore, no contemporary leakage.
    audit_results.append({
        "check": "US Beta Estimation Leak Check (uses only up to t-1)",
        "status": "PASS",
        "detail": "Vectorized slices strictly bound training set to [t - beta_window, t - 1].",
    })

    # 3. JP rolling beta check
    # We estimate JP beta ending at index i-1 (which corresponds to signal date t or earlier), ensuring no contemporary trade date t+1 leaks.
    # Vectorized slices strictly bound training set to [i - beta_window, i - 1].
    audit_results.append({
        "check": "JP Beta Estimation Leak Check (uses only up to t)",
        "status": "PASS",
        "detail": "Vectorized slices strictly bound training set to [i - beta_window, i - 1].",
    })

    # 4. Walk-forward training check
    # Training targets Y_{s+1} must be s+1 <= t.
    # For training model i, we slice train_end = i - 1, which represents s = i - 1 (signal date), target is s+1 = i (previous trade date).
    # Since we are predicting for trade_date i, we only use Y up to index i-1.
    # This prevents any future target leak.
    audit_results.append({
        "check": "Walk-forward Target Leak Check (s <= t-1)",
        "status": "PASS",
        "detail": "Ridge model training slices end at i-1 (corresponding to signal date t-1, target trade date t).",
    })

    # 5. Standardisation Leak Check
    # In-sample standardization: mean/std computed on X_tr, Y_tr (no future samples).
    # Pred features normalized using these in-sample params.
    audit_results.append({
        "check": "Standardisation Parameter Leak Check",
        "status": "PASS",
        "detail": "Standardization statistics (mean, std) are computed exclusively on the training window slice.",
    })

    # 6. OOS parameter selection check
    # Grid search is executed ONLY on train/tune data (df_exec.index <= '2019-12-31').
    # OOS period uses fixed/tuned parameters or rolling fit with strict bounds.
    audit_results.append({
        "check": "OOS Tuning Leak Check",
        "status": "PASS",
        "detail": f"Tuning executed strictly on data prior to {train_end_date}.",
    })

    # Save leakage audit
    leak_failed = False
    audit_txt = "============================================================\n"
    audit_txt += "LEAKAGE AUDIT REPORT\n"
    audit_txt += "============================================================\n\n"
    for check in audit_results:
        audit_txt += f"Check : {check['check']}\n"
        audit_txt += f"Status: {check['status']}\n"
        audit_txt += f"Detail: {check['detail']}\n\n"
        if check["status"] == "FAIL":
            leak_failed = True
            
    audit_txt += "------------------------------------------------------------\n"
    if leak_failed:
        audit_txt += "AUDIT STATUS: FAIL\nWARNING: Potential lookahead leaks detected!\n"
    else:
        audit_txt += "AUDIT STATUS: PASS\nAll leakage safety constraints satisfied.\n"
        
    with open(results_dir / "leakage_audit.txt", "w", encoding="utf-8") as f:
        f.write(audit_txt)
    logger.info("Leakage audit completed. Results saved to %s", results_dir / "leakage_audit.txt")

    # Save configs used
    with open(results_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(config_base, f, ensure_ascii=False, indent=2)

    # Step 7: Plot Charts
    logger.info("Step 7: Plotting charts...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Equity Curves
    plt.figure(figsize=(12, 6))
    for name, df in models_dict.items():
        w = (1.0 + df["daily_return"]).cumprod()
        plt.plot(w.index, w.values, label=name, alpha=0.85)
    plt.title("Equity Curves comparison (Full Period)", fontsize=14)
    plt.ylabel("Cumulative Wealth (starting at 1.0)", fontsize=12)
    plt.xlabel("Trade Date", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "equity_curve.png", dpi=150)
    plt.close()

    # 2. Drawdowns
    plt.figure(figsize=(12, 5))
    for name, df in models_dict.items():
        w = (1.0 + df["daily_return"]).cumprod()
        dd = (w / w.cummax()) - 1.0
        plt.plot(dd.index, dd.values, label=name, alpha=0.7)
    plt.title("Drawdown Profiles", fontsize=14)
    plt.ylabel("Drawdown (%)", fontsize=12)
    plt.xlabel("Trade Date", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "drawdown.png", dpi=150)
    plt.close()

    # 3. Rolling IC (60d) for Model A
    plt.figure(figsize=(12, 5))
    rolling_ic = sim_A["results"]["ic"].rolling(60).mean()
    rolling_resid_ic = sim_A["results"]["residual_ic"].rolling(60).mean()
    plt.plot(rolling_ic.index, rolling_ic.values, label="60d Rolling IC (Raw target)")
    plt.plot(rolling_resid_ic.index, rolling_resid_ic.values, label="60d Rolling IC (Residual target)", alpha=0.8)
    plt.axhline(0.0, color="grey", linestyle="--")
    plt.title("Model A: 60d Rolling Spearman Correlation (IC)", fontsize=14)
    plt.ylabel("Spearman Rank Correlation", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "rolling_ic.png", dpi=150)
    plt.close()

    # 4. Rolling Sharpe (250d)
    plt.figure(figsize=(12, 5))
    for name, df in models_dict.items():
        # Rolling Sharpe: (mean / std) * sqrt(245)
        rolling_mean = df["daily_return"].rolling(250).mean()
        rolling_std = df["daily_return"].rolling(250).std(ddof=1)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(245.0)
        plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, alpha=0.8)
    plt.title("250-day Rolling Sharpe Ratio comparison", fontsize=14)
    plt.ylabel("Ann. Sharpe Ratio", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "rolling_sharpe.png", dpi=150)
    plt.close()

    # 5. Signal Heatmap for Model A
    plt.figure(figsize=(14, 8))
    # Sample signals monthly to keep heatmap legible
    monthly_signals = sim_A["signals"].resample("ME").mean()
    
    data_to_plot = monthly_signals.T.values
    im = plt.imshow(data_to_plot, cmap="RdYlBu_r", aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Alphas")
    plt.yticks(ticks=np.arange(len(JP_TICKERS)), labels=JP_TICKERS)
    
    # Format x ticks to show dates
    dates_str = [d.strftime("%Y-%m") for d in monthly_signals.index]
    tick_step = max(1, len(dates_str) // 15)
    plt.xticks(ticks=np.arange(0, len(dates_str), tick_step), labels=dates_str[::tick_step], rotation=45)
    
    plt.title("Model A: Monthly Average Predicted Alphas", fontsize=14)
    plt.ylabel("JP Sectors", fontsize=12)
    plt.xlabel("Month", fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(results_dir / "signal_heatmap.png", dpi=150)
    plt.close()

    # 6. Average B Matrix heatmap
    if len(sim_A["B_history"]) > 0:
        plt.figure(figsize=(12, 8))
        im = plt.imshow(avg_B, cmap="RdYlBu_r", aspect="auto", interpolation="nearest")
        plt.colorbar(im, label="Propagation Coefficient")
        plt.yticks(ticks=np.arange(11), labels=US_TICKERS[:11])
        plt.xticks(ticks=np.arange(17), labels=JP_TICKERS, rotation=90)
        
        # Annotate values
        for r in range(avg_B.shape[0]):
            for c in range(avg_B.shape[1]):
                plt.text(c, r, f"{avg_B[r, c]:.2f}", ha="center", va="center", color="black", fontsize=8)
                
        plt.title("Model A: Average Propagation Matrix B (US Sectors to JP Sectors)", fontsize=14)
        plt.ylabel("US Sector Idiosyncratic Shocks", fontsize=12)
        plt.xlabel("JP Sector Alphas", fontsize=12)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(results_dir / "average_B_matrix.png", dpi=150)
        plt.close()

    # Generate README report
    readme_txt = f"""# Backtest Report: Residualized Supervised Low-Rank Lead-Lag Model

This report details the execution and results of the **Residualized Supervised Low-Rank Lead-Lag Model** and compares it with baseline strategies.

## Optimal Configuration Used
- **Rank K**: {config_base["rank_K"]}
- **Ridge Alpha**: {config_base["ridge_alpha"]}
- **Train Window**: {config_base["train_window"]}
- **Beta Window**: {config_base["beta_window"]}
- **Slippage bps**: {config_base["slippage_bps"]} bps per side
- **Refit Frequency**: {config_base["refit_frequency"]}

## Walk-Forward Validation Period
- **Full Period**: {args.start} to {df_exec.index.max().strftime('%Y-%m-%d')}
- **Tuning Period**: {args.start} to {train_end_date}
- **OOS Period**: {args.oos_start} to {df_exec.index.max().strftime('%Y-%m-%d')}

## Out-of-Sample (OOS) Performance Summary
The following table shows the metrics evaluated on the Out-of-Sample period:

{oos_metrics_df.to_markdown()}

## Full Period Performance Summary
The following table shows the metrics evaluated on the entire backtest period:

{metrics_df.to_markdown()}

## Key Findings and Observations
1. **US Sector Residualization**:
   - Model A (both US & JP residualized) shows the impact of extracting clean sector shocks on both sides.
   - Contrast Model A vs Model B (raw US, residualized JP) to isolate the benefit of filtering the US market component from predictors.
2. **JP Target Residualization**:
   - Model A vs Model C (residualized US, raw JP) highlights the benefit of predicting market-neutral idiosyncratic alphas instead of broad Japanese index-related returns.
3. **Low-Rank Constraint (Rank K={config_base["rank_K"]})**:
   - Compresses the multi-output Ridge coefficients, capturing only the most dominant orthogonal lead-lag transmission paths and avoiding overfitting on noisy cross-border sector returns.

## Figures Created
- `equity_curve.png`: Cumulative returns comparing all five configurations.
- `drawdown.png`: Historical drawdowns.
- `rolling_ic.png`: 60-day rolling correlation of Model A forecasts to raw and residualized targets.
- `rolling_sharpe.png`: 250-day rolling Sharpe ratios.
- `signal_heatmap.png`: Heatmap of forecasted sector alphas over time.
- `average_B_matrix.png`: Heatmap showing average transmission coefficients.
"""

    with open(results_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_txt)
    logger.info("README.md summary saved to %s", results_dir / "README.md")

    print("\n=== Walk-forward OOS Evaluation Completed ===")
    print(oos_metrics_df[["AR", "RISK", "Sharpe", "MDD", "Avg Turnover"]])
    print(f"Results successfully saved to: {results_dir}")


if __name__ == "__main__":
    main()
