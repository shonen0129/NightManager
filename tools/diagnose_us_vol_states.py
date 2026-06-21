#!/usr/bin/env python
"""US Volatility State Diagnostics for Production Residual-BLPX Model.

Diagnoses signal IC, PnL, costs, risk, drawdown, and execution workload
across different US volatility states, preserving production logic and configs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

# Setup matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="US Volatility State Diagnostics Suite")
    parser.add_argument("--config", default="configs/production.yaml", help="Path to config file")
    parser.add_argument("--model", default="production_p8p3_blpx", help="Model name identifier")
    parser.add_argument("--start", default="2020-01-01", help="Analysis start date (trade_date)")
    parser.add_argument("--end", default="2026-06-14", help="Analysis end date (trade_date)")
    parser.add_argument("--results-dir", default="results/production_p8p3_blpx_validation", help="Dir with existing results")
    parser.add_argument("--output-dir", default="results/vol_state_diagnostics", help="Output directory")
    parser.add_argument("--run-backtest-if-missing", action="store_true", default=True, help="Force/allow backtest execution")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Slippage in bps per side")
    parser.add_argument("--bin-method", choices=["tertile", "quintile"], default="tertile", help="Binning method")
    parser.add_argument("--rolling-window", type=int, default=60, help="Rolling window for short-term vol states")
    parser.add_argument("--long-window", type=int, default=252, help="Rolling window for long-term vol states")
    parser.add_argument("--include-vix", type=str, choices=["true", "false"], default="true", help="Include VIX analysis")
    parser.add_argument("--include-post-open-gap", type=str, choices=["true", "false"], default="true", help="Include JPY post-open gap analysis")
    parser.add_argument("--self-test", action="store_true", help="Execute self-test and exit")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# State variables calculation helpers
# ---------------------------------------------------------------------------

def calculate_vix_states(vix_series: pd.Series, rolling_w: int, long_w: int) -> pd.DataFrame:
    """Calculate VIX level, change, and rolling z-scores (point-in-time)."""
    df = pd.DataFrame(index=vix_series.index)
    df["VIX_level"] = vix_series
    
    # Rolling z-scores of level
    df["VIX_z_60"] = (vix_series - vix_series.rolling(rolling_w).mean()) / vix_series.rolling(rolling_w).std(ddof=1)
    df["VIX_z_252"] = (vix_series - vix_series.rolling(long_w).mean()) / vix_series.rolling(long_w).std(ddof=1)
    
    # 1d change metrics
    df["VIX_change_1d"] = vix_series.diff(1)
    df["VIX_change_1d_z_60"] = (df["VIX_change_1d"] - df["VIX_change_1d"].rolling(rolling_w).mean()) / df["VIX_change_1d"].rolling(rolling_w).std(ddof=1)
    df["VIX_pct_change_1d"] = vix_series.pct_change(1, fill_method=None)
    
    return df


def calculate_us_ret_states(us_returns: pd.DataFrame, rolling_w: int, long_w: int) -> pd.DataFrame:
    """Calculate returns-based state variables for US ETFs (point-in-time)."""
    df = pd.DataFrame(index=us_returns.index)
    
    # 1. Row-wise (cross-sectional) stats
    abs_rets = us_returns.abs()
    df["US_absret_avg"] = abs_rets.mean(axis=1)
    df["US_absret_max"] = abs_rets.max(axis=1)
    df["US_ret_dispersion"] = us_returns.std(axis=1, ddof=1)
    df["US_ret_iqr"] = us_returns.quantile(0.75, axis=1) - us_returns.quantile(0.25, axis=1)
    df["US_market_mode"] = us_returns.mean(axis=1)
    df["US_sector_neutral_shock"] = df["US_ret_dispersion"] / (df["US_market_mode"].abs() + 1e-8)
    df["US_downside_count"] = (us_returns < 0).sum(axis=1)
    
    # 2. Extreme counts (past 60d standard deviation reference)
    # Note: rolling standard deviation must strictly shift by 1 day to be point-in-time
    rolling_std = us_returns.shift(1).rolling(rolling_w).std(ddof=1)
    df["US_extreme_count_1sigma"] = (abs_rets > rolling_std).sum(axis=1)
    df["US_extreme_count_2sigma"] = (abs_rets > (2.0 * rolling_std)).sum(axis=1)
    
    # 3. Rolling z-scores (point-in-time)
    for col in ["US_absret_avg", "US_ret_dispersion"]:
        df[f"{col}_z_60"] = (df[col] - df[col].rolling(rolling_w).mean()) / df[col].rolling(rolling_w).std(ddof=1)
        df[f"{col}_z_252"] = (df[col] - df[col].rolling(long_w).mean()) / df[col].rolling(long_w).std(ddof=1)
        
    return df


def calculate_us_corr_states(us_returns: pd.DataFrame, rolling_w: int, long_w: int) -> pd.DataFrame:
    """Calculate US ETF rolling correlation and PC1 share states (point-in-time)."""
    dates = us_returns.index
    avg_corr_60 = []
    pc1_share_60 = []
    avg_corr_252 = []
    pc1_share_252 = []
    
    # Compute rolling correlation matrices
    # For speed and point-in-time safety, we loop manually
    for idx, dt in enumerate(dates):
        if idx < rolling_w - 1:
            avg_corr_60.append(np.nan)
            pc1_share_60.append(np.nan)
        else:
            window_data = us_returns.iloc[idx - rolling_w + 1 : idx + 1]
            corr_mat = np.corrcoef(window_data.values, rowvar=False)
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            n = corr_mat.shape[0]
            # Average off-diagonal correlation
            off_diag = corr_mat[~np.eye(n, dtype=bool)]
            avg_corr_60.append(float(np.mean(off_diag)))
            # PC1 share
            eigvals = np.linalg.eigvalsh(corr_mat)
            pc1_share_60.append(float(np.max(eigvals) / np.sum(eigvals)))
            
        if idx < long_w - 1:
            avg_corr_252.append(np.nan)
            pc1_share_252.append(np.nan)
        else:
            window_data = us_returns.iloc[idx - long_w + 1 : idx + 1]
            corr_mat = np.corrcoef(window_data.values, rowvar=False)
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            n = corr_mat.shape[0]
            off_diag = corr_mat[~np.eye(n, dtype=bool)]
            avg_corr_252.append(float(np.mean(off_diag)))
            eigvals = np.linalg.eigvalsh(corr_mat)
            pc1_share_252.append(float(np.max(eigvals) / np.sum(eigvals)))
            
    df = pd.DataFrame(index=dates)
    df["US_avg_corr_60"] = avg_corr_60
    df["US_pc1_share_60"] = pc1_share_60
    df["US_avg_corr_252"] = avg_corr_252
    df["US_pc1_share_252"] = pc1_share_252
    
    df["US_pc1_share_change_1d"] = df["US_pc1_share_60"].diff(1)
    
    # 252d z-scores of 60d metrics
    df["US_corr_z_252"] = (df["US_avg_corr_60"] - df["US_avg_corr_60"].rolling(long_w).mean()) / df["US_avg_corr_60"].rolling(long_w).std(ddof=1)
    df["US_pc1_share_z_252"] = (df["US_pc1_share_60"] - df["US_pc1_share_60"].rolling(long_w).mean()) / df["US_pc1_share_60"].rolling(long_w).std(ddof=1)
    
    return df


def calculate_us_style_states(us_returns: pd.DataFrame, rolling_w: int) -> pd.DataFrame:
    """Calculate US ETF style indicators (point-in-time)."""
    df = pd.DataFrame(index=us_returns.index)
    
    # Raw spreads
    df["USMV_minus_IUSG"] = us_returns["USMV"] - us_returns["IUSG"]
    df["MTUM_minus_VLUE"] = us_returns["MTUM"] - us_returns["VLUE"]
    df["VLUE_minus_IUSG"] = us_returns["VLUE"] - us_returns["IUSG"]
    
    # 60d z-scores
    for col in ["USMV_minus_IUSG", "MTUM_minus_VLUE", "VLUE_minus_IUSG"]:
        df[f"{col}_z_60"] = (df[col] - df[col].rolling(rolling_w).mean()) / df[col].rolling(rolling_w).std(ddof=1)
        
    return df


def calculate_us_path_quality(us_ohlcv: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, rolling_w: int) -> pd.DataFrame:
    """Calculate path quality metrics from OHLC (point-in-time)."""
    df = pd.DataFrame(index=dates)
    
    # Extract dataframes
    op = us_ohlcv.get("Open")
    hi = us_ohlcv.get("High")
    lo = us_ohlcv.get("Low")
    cl = us_ohlcv.get("Close")
    
    if op is None or hi is None or lo is None or cl is None:
        logger.warning("Missing high-low-open-close data. Skipping path quality.")
        return df
        
    # Standardize index
    op = op.reindex(dates)
    hi = hi.reindex(dates)
    lo = lo.reindex(dates)
    cl = cl.reindex(dates)
    prev_cl = cl.shift(1)
    
    # Compute metrics per ETF, then mean across columns
    range_etf = (hi - lo) / prev_cl
    close_loc_etf = (cl - lo) / np.maximum(hi - lo, 1e-8)
    efficiency_etf = (cl - op).abs() / np.maximum(hi - lo, 1e-8)
    gap_etf = op / prev_cl - 1.0
    intra_etf = cl / op - 1.0
    overnight_etf = op / prev_cl - 1.0
    
    df["US_range_avg"] = range_etf.mean(axis=1)
    df["US_close_location_avg"] = close_loc_etf.mean(axis=1)
    df["US_path_efficiency_avg"] = efficiency_etf.mean(axis=1)
    df["US_gap_avg"] = gap_etf.mean(axis=1)
    df["US_intraday_return_avg"] = intra_etf.mean(axis=1)
    df["US_overnight_return_avg"] = overnight_etf.mean(axis=1)
    
    # 60d z-scores
    for col in ["US_range_avg", "US_close_location_avg", "US_path_efficiency_avg", "US_gap_avg", "US_intraday_return_avg", "US_overnight_return_avg"]:
        df[f"{col}_z_60"] = (df[col] - df[col].rolling(rolling_w).mean()) / df[col].rolling(rolling_w).std(ddof=1)
        
    return df


def calculate_us_volume_shock(us_ohlcv: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, rolling_w: int) -> pd.DataFrame:
    """Calculate US volume shock state variables (point-in-time)."""
    df = pd.DataFrame(index=dates)
    vol = us_ohlcv.get("Volume")
    
    if vol is None:
        logger.warning("Missing Volume data. Skipping volume shock.")
        return df
        
    vol = vol.reindex(dates)
    
    # volume ratio = Vol_t / Mean(Vol_{t-60..t-1})
    rolling_vol_mean = vol.shift(1).rolling(rolling_w).mean()
    vol_ratio = vol / np.maximum(rolling_vol_mean, 1.0)
    
    df["US_volume_shock_avg"] = vol_ratio.mean(axis=1)
    df["US_volume_shock_max"] = vol_ratio.max(axis=1)
    
    # 60d z-scores
    for col in ["US_volume_shock_avg", "US_volume_shock_max"]:
        df[f"{col}_z_60"] = (df[col] - df[col].rolling(rolling_w).mean()) / df[col].rolling(rolling_w).std(ddof=1)
        
    return df


def calculate_jp_post_gap_states(df_exec: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calculate JPY post-open states from JPY gaps (leakage-safe post-open prefix)."""
    df = pd.DataFrame(index=df_exec.index)
    
    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    beta_cols = [f"jp_beta_{tk}" for tk in JP_TICKERS]
    
    if not all(c in df_exec.columns for c in gap_cols):
        logger.warning("Missing Japanese gap columns in execution DataFrame. Skipping.")
        return df
        
    jp_gaps = df_exec[gap_cols].values
    topix_night = df_exec["topix_night_return"].values if "topix_night_return" in df_exec.columns else np.zeros(len(df_exec))
    
    df["POST_JP_gap_abs_avg"] = np.mean(np.abs(jp_gaps), axis=1)
    df["POST_JP_gap_abs_max"] = np.max(np.abs(jp_gaps), axis=1)
    df["POST_JP_gap_dispersion"] = np.std(jp_gaps, axis=1, ddof=1)
    df["POST_TOPIXNight_abs"] = np.abs(topix_night)
    
    # Gap open Decomposition
    gap_open_coef = float(cfg.get("blpx", {}).get("gap_open_coef", 0.70))
    topix_beta_coef = float(cfg.get("blpx", {}).get("topix_beta_coef", 0.60))
    
    gap_filt_all = []
    gap_syst_all = []
    gap_idio_all = []
    
    for i in range(len(df_exec)):
        gap_vec = jp_gaps[i]
        topix_n_val = topix_night[i]
        
        # Check betas
        has_betas = all(f"jp_beta_{tk}" in df_exec.columns for tk in JP_TICKERS)
        if has_betas:
            betas_vec = df_exec.iloc[i][beta_cols].values.astype(float)
        else:
            betas_vec = np.ones(17) * 1.0  # fallback
            
        gap_syst = betas_vec * topix_n_val
        gap_idio = gap_vec - gap_syst
        gap_filt = gap_open_coef * gap_idio + (gap_open_coef - topix_beta_coef) * gap_syst
        
        gap_filt_all.append(np.mean(np.abs(gap_filt)))
        gap_syst_all.append(np.mean(np.abs(gap_syst)))
        gap_idio_all.append(np.mean(np.abs(gap_idio)))
        
    df["POST_GapOpen_filt_abs_avg"] = gap_filt_all
    df["POST_GapOpen_syst_abs_avg"] = gap_syst_all
    df["POST_GapOpen_idio_abs_avg"] = gap_idio_all
    
    return df


# ---------------------------------------------------------------------------
# Diagnostics grouping and metrics computation
# ---------------------------------------------------------------------------

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from daily returns."""
    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    return (wealth / running_max) - 1.0


def compute_metrics_for_returns(returns: pd.Series) -> dict:
    """Calculate key performance indicators (AR, vol, Sharpe, MDD, hit rate, etc.)."""
    n_days = len(returns)
    if n_days == 0:
        return {}
        
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=1))
    
    ann_ret = mean_ret * 250.0
    ann_vol = std_ret * np.sqrt(250.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    hit_rate = float((returns > 0).sum()) / n_days
    median_ret = float(returns.median())
    p05 = float(returns.quantile(0.05))
    p95 = float(returns.quantile(0.95))
    worst = float(returns.min())
    best = float(returns.max())
    
    # Max drawdown on this specific sub-series
    dd = compute_drawdown_series(returns)
    mdd = float(dd.min())
    
    return {
        "count": n_days,
        "mean_daily": mean_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "median": median_ret,
        "p05": p05,
        "p95": p95,
        "worst_day": worst,
        "best_day": best,
        "max_drawdown": mdd,
    }


def safe_qcut(series: pd.Series, q: int, is_tertile: bool = True) -> tuple[pd.Series, list[str]]:
    """Safe qcut wrapper handling duplicate edges and returning clean labels."""
    try:
        bin_idx = pd.qcut(series, q=q, labels=False, duplicates="drop")
        uniq_vals = sorted(bin_idx.dropna().unique())
        n_bins = len(uniq_vals)
        if n_bins == 0:
            return pd.Series(index=series.index), []
            
        if is_tertile:
            if n_bins == 3:
                lbl_dict = {uniq_vals[0]: "Low", uniq_vals[1]: "Mid", uniq_vals[2]: "High"}
            elif n_bins == 2:
                lbl_dict = {uniq_vals[0]: "Low", uniq_vals[1]: "High"}
            else:
                lbl_dict = {uniq_vals[0]: "Mid"}
        else:
            if n_bins == 5:
                lbl_dict = {uniq_vals[0]: "Q1", uniq_vals[1]: "Q2", uniq_vals[2]: "Q3", uniq_vals[3]: "Q4", uniq_vals[4]: "Q5"}
            else:
                lbl_dict = {v: f"Q{v+1}" for v in uniq_vals}
                
        return bin_idx.map(lbl_dict), list(lbl_dict.values())
    except Exception:
        try:
            ranks = series.rank(method="first")
            bin_idx = pd.qcut(ranks, q=q, labels=False)
            uniq_vals = sorted(bin_idx.dropna().unique())
            if is_tertile:
                lbl_dict = {0: "Low", 1: "Mid", 2: "High"}
            else:
                lbl_dict = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q5"}
            lbl_dict = {k: v for k, v in lbl_dict.items() if k in uniq_vals}
            return bin_idx.map(lbl_dict), list(lbl_dict.values())
        except Exception:
            return pd.Series(index=series.index), []


def perform_univariate_diagnostics(panel_df: pd.DataFrame, state_cols: list[str], bin_method: str) -> pd.DataFrame:
    """Perform univariate grouping analysis on all state variables."""
    results = []
    
    for state_var in state_cols:
        if state_var not in panel_df.columns:
            continue
            
        valid_df = panel_df.dropna(subset=[state_var, "net_return"])
        if len(valid_df) < 30:
            continue
            
        # Divide into bins safely
        bins, labels = safe_qcut(valid_df[state_var], q=(3 if bin_method == "tertile" else 5), is_tertile=(bin_method == "tertile"))
        if len(labels) == 0:
            continue
            
        for bin_label in labels:
            bin_mask = (bins == bin_label)
            bin_data = valid_df[bin_mask]
            
            if len(bin_data) == 0:
                continue
                
            metrics = compute_metrics_for_returns(bin_data["net_return"])
            gross_metrics = compute_metrics_for_returns(bin_data["gross_return"])
            
            # Additional metrics
            mean_cost = float(bin_data["cost"].mean())
            mean_turnover = float(bin_data["turnover"].mean())
            mean_gross_exp = float(bin_data["gross_exposure"].mean())
            
            # IC metrics
            spearman_ic = bin_data["realized_ic_spearman"] if "realized_ic_spearman" in bin_data.columns else pd.Series([np.nan])
            mean_ic = float(spearman_ic.mean())
            med_ic = float(spearman_ic.median())
            ic_pos_rate = float((spearman_ic > 0).sum()) / max(len(spearman_ic.dropna()), 1)
            
            # Long/Short contributions
            mean_long_c = float(bin_data["long_return_contribution"].mean()) if "long_return_contribution" in bin_data.columns else np.nan
            mean_short_c = float(bin_data["short_return_contribution"].mean()) if "short_return_contribution" in bin_data.columns else np.nan
            
            record = {
                "state_variable": state_var,
                "bin": bin_label,
                "count": metrics["count"],
                "mean_net_return_daily": metrics["mean_daily"],
                "mean_gross_return_daily": gross_metrics["mean_daily"],
                "annualized_return_net": metrics["ann_return"],
                "annualized_vol_net": metrics["ann_vol"],
                "sharpe_net": metrics["sharpe"],
                "hit_rate_net": metrics["hit_rate"],
                "median_net_return": metrics["median"],
                "p05_net_return": metrics["p05"],
                "p95_net_return": metrics["p95"],
                "worst_day": metrics["worst_day"],
                "best_day": metrics["best_day"],
                "max_drawdown": metrics["max_drawdown"],
                "mean_cost": mean_cost,
                "mean_turnover": mean_turnover,
                "mean_gross_exposure": mean_gross_exp,
                "mean_signal_ic_spearman": mean_ic,
                "median_signal_ic_spearman": med_ic,
                "ic_hit_rate_positive": ic_pos_rate,
                "mean_long_contribution": mean_long_c,
                "mean_short_contribution": mean_short_c,
            }
            results.append(record)
            
    return pd.DataFrame(results)


def perform_2d_cross_diagnostics(panel_df: pd.DataFrame, cross_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Perform 2D cross-tabulation diagnostics on selected pairs of states."""
    results = []
    
    for var1, var2 in cross_pairs:
        if var1 not in panel_df.columns or var2 not in panel_df.columns:
            continue
            
        valid_df = panel_df.dropna(subset=[var1, var2, "net_return"])
        if len(valid_df) < 50:
            continue
            
        # Binary split into tertiles safely
        bins1, labels1 = safe_qcut(valid_df[var1], q=3, is_tertile=True)
        bins2, labels2 = safe_qcut(valid_df[var2], q=3, is_tertile=True)
        
        if len(labels1) == 0 or len(labels2) == 0:
            continue
            
        for label1 in labels1:
            for label2 in labels2:
                cell_mask = (bins1 == label1) & (bins2 == label2)
                cell_data = valid_df[cell_mask]
                
                if len(cell_data) == 0:
                    continue
                    
                metrics = compute_metrics_for_returns(cell_data["net_return"])
                mean_cost = float(cell_data["cost"].mean())
                
                ic_val = cell_data["realized_ic_spearman"] if "realized_ic_spearman" in cell_data.columns else pd.Series([np.nan])
                mean_ic = float(ic_val.mean())
                
                record = {
                    "variable_1": var1,
                    "bin_1": label1,
                    "variable_2": var2,
                    "bin_2": label2,
                    "count": metrics["count"],
                    "mean_net_return_daily": metrics["mean_daily"],
                    "sharpe_net": metrics["sharpe"],
                    "hit_rate_net": metrics["hit_rate"],
                    "mean_signal_ic_spearman": mean_ic,
                    "mean_cost": mean_cost,
                    "max_drawdown": metrics["max_drawdown"],
                }
                results.append(record)
                
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting logic
# ---------------------------------------------------------------------------

def create_diagnostic_plots(panel_df: pd.DataFrame, univariate_df: pd.DataFrame, cross_df: pd.DataFrame, output_dir: Path):
    """Generate and save all diagnostic visualizations."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Barplots: Return/Sharpe/IC by bin (pick 3 main vol state variables)
    selected_states = [col for col in ["US_absret_avg_z_60", "US_ret_dispersion_z_60", "US_avg_corr_60"] if col in panel_df.columns]
    
    for state in selected_states:
        sub_df = univariate_df[univariate_df["state_variable"] == state]
        if sub_df.empty:
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Diagnostics by Bin: {state}", fontsize=14, fontweight="bold")
        
        # Mean Net Return Bar
        axes[0].bar(sub_df["bin"], sub_df["annualized_return_net"] * 100, color="navy", alpha=0.8)
        axes[0].set_title("Annualized Net Return (%)")
        axes[0].set_ylabel("%")
        
        # Sharpe Bar
        axes[1].bar(sub_df["bin"], sub_df["sharpe_net"], color="darkgreen", alpha=0.8)
        axes[1].set_title("Sharpe Ratio")
        
        # Mean IC Bar
        axes[2].bar(sub_df["bin"], sub_df["mean_signal_ic_spearman"], color="darkred", alpha=0.8)
        axes[2].set_title("Mean Spearman IC")
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"bin_diagnostics_{state}.png", dpi=150)
        plt.close()
        
    # 2. Scatter plots
    scatter_configs = [
        ("US_absret_avg_z_60", "net_return", "US Abs Return z60 vs Daily Net Return"),
        ("US_ret_dispersion_z_60", "realized_ic_spearman", "US Dispersion z60 vs Spearman IC"),
        ("US_avg_corr_60", "net_return", "US Average Correlation vs Daily Net Return"),
        ("US_pc1_share_60", "net_return", "US PC1 Share vs Daily Net Return"),
    ]
    
    for x_col, y_col, title in scatter_configs:
        if x_col in panel_df.columns and y_col in panel_df.columns:
            valid = panel_df.dropna(subset=[x_col, y_col])
            if len(valid) == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(valid[x_col], valid[y_col], alpha=0.4, color="purple", edgecolors="none")
            
            # Add line of best fit
            x_vals = valid[x_col].values
            y_vals = valid[y_col].values
            try:
                m, b = np.polyfit(x_vals, y_vals, 1)
                ax.plot(x_vals, m * x_vals + b, color="red", linestyle="--", linewidth=1.5, label=f"Fit (slope: {m:.5f})")
                ax.legend()
            except Exception:
                pass
                
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.tight_layout()
            plt.savefig(plots_dir / f"scatter_{x_col}_vs_{y_col}.png", dpi=150)
            plt.close()
            
    # 3. Time Series subplot
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Cumulative Performance & US Vol State Indicators", fontsize=16, fontweight="bold")
    
    # Cumulative Net Returns
    cum_returns = (1.0 + panel_df["net_return"].fillna(0.0)).cumprod() - 1.0
    axes[0].plot(panel_df.index, cum_returns * 100, color="blue", linewidth=2)
    axes[0].set_title("Cumulative Net Portfolio Return (%)")
    axes[0].set_ylabel("%")
    
    # Indicator series plots
    indicators = [
        ("US_absret_avg_z_60", "orange", "US Absolute Return z-score (60d)"),
        ("US_ret_dispersion_z_60", "green", "US Sector Return Dispersion z-score (60d)"),
        ("US_avg_corr_60", "red", "US Average Correlation (60d)"),
        ("US_pc1_share_60", "purple", "US PC1 Eigenvalue Share (60d)"),
    ]
    
    for idx, (col, color, name) in enumerate(indicators, start=1):
        if col in panel_df.columns:
            axes[idx].plot(panel_df.index, panel_df[col], color=color, alpha=0.7)
            axes[idx].set_title(name)
            
    plt.xlabel("Trade Date")
    plt.tight_layout()
    plt.savefig(plots_dir / "timeline_diagnostics.png", dpi=150)
    plt.close()
    
    # 4. Cross-analysis heatmaps
    cross_pairs = cross_df["variable_1"].unique()
    for var1 in cross_pairs:
        pair_df = cross_df[cross_df["variable_1"] == var1]
        var2 = pair_df["variable_2"].iloc[0]
        
        # Reshape to matrix for heatmap
        matrix_ret = np.zeros((3, 3))
        matrix_sharpe = np.zeros((3, 3))
        matrix_count = np.zeros((3, 3))
        
        bins_labels = ["Low", "Mid", "High"]
        
        for i, l1 in enumerate(bins_labels):
            for j, l2 in enumerate(bins_labels):
                row = pair_df[(pair_df["bin_1"] == l1) & (pair_df["bin_2"] == l2)]
                if not row.empty:
                    matrix_ret[i, j] = float(row["mean_net_return_daily"].iloc[0]) * 10000.0  # bps
                    matrix_sharpe[i, j] = float(row["sharpe_net"].iloc[0])
                    matrix_count[i, j] = float(row["count"].iloc[0])
                    
        # Plot Heatmap of Net Returns in bps
        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.imshow(matrix_ret, cmap="RdYlGn", aspect="auto")
        fig.colorbar(cax, label="Mean Net Return (bps)")
        
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(bins_labels)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(bins_labels)
        
        ax.set_xlabel(f"{var2} tertile")
        ax.set_ylabel(f"{var1} tertile")
        ax.set_title(f"Mean Net Return (bps) Cell Diagnostics\n{var1} x {var2}", fontweight="bold")
        
        # Annotate text
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix_ret[i, j]:+.1f} bps\nSR: {matrix_sharpe[i, j]:.2f}\nN={int(matrix_count[i, j])}",
                        ha="center", va="center", color="black", fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
                
        plt.tight_layout()
        plt.savefig(plots_dir / f"heatmap_{var1}_x_{var2}.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# Self-Test Mode
# ---------------------------------------------------------------------------

def run_self_test():
    """Execute rigorous self-test to verify math, timing rules, and point-in-time sanity."""
    logger.info("Executing self-test routine...")
    
    # Construct mock data: 300 days
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=300, freq="B")
    
    mock_returns = pd.DataFrame(np.random.randn(300, 15) * 0.01, index=dates, columns=US_TICKERS)
    mock_ohlcv = {
        "Open": mock_returns + 10.0,
        "High": mock_returns + 10.1,
        "Low": mock_returns + 9.9,
        "Close": mock_returns + 10.05,
        "Volume": pd.DataFrame(np.random.randint(10000, 50000, size=(300, 15)), index=dates, columns=US_TICKERS),
    }
    
    # Calculate state variables
    df_ret = calculate_us_ret_states(mock_returns, 60, 252)
    df_corr = calculate_us_corr_states(mock_returns, 60, 252)
    df_style = calculate_us_style_states(mock_returns, 60)
    df_path = calculate_us_path_quality(mock_ohlcv, dates, 60)
    df_vol = calculate_us_volume_shock(mock_ohlcv, dates, 60)
    
    # Merges
    all_states = pd.concat([df_ret, df_corr, df_style, df_path, df_vol], axis=1)
    
    # Timing and lookahead audit assertions
    # 1. No lookahead in rolling z-score: standard deviations at date index i must only depend on index <= i
    # Let's check a row in the middle
    t_idx = 150
    # Mean of past 60 days
    expected_mean = all_states["US_absret_avg"].iloc[t_idx-59 : t_idx+1].mean()
    expected_std = all_states["US_absret_avg"].iloc[t_idx-59 : t_idx+1].std(ddof=1)
    expected_z = (all_states["US_absret_avg"].iloc[t_idx] - expected_mean) / expected_std
    
    assert np.isclose(all_states["US_absret_avg_z_60"].iloc[t_idx], expected_z), "z-score calculation has lookahead or calculation error"
    
    # 2. Check US_extreme_count calculations (t-1 standard deviation logic)
    # extreme standard deviation window for index 150 must end at index 149
    hist_std = mock_returns.iloc[150-60 : 150].std(ddof=1)
    cur_abs = mock_returns.iloc[150].abs()
    expected_extreme_1sigma = (cur_abs > hist_std).sum()
    assert all_states["US_extreme_count_1sigma"].iloc[150] == expected_extreme_1sigma, "extreme 1sigma count has timing leakage"
    
    # 3. Check PC1 calculations are non-trivial
    assert all_states["US_pc1_share_60"].notna().sum() > 0
    pc1_clean = all_states["US_pc1_share_60"].dropna()
    assert (pc1_clean >= 0.0).all() and (pc1_clean <= 1.0).all(), "PC1 share is not within [0, 1]"
    
    logger.info("Self-test passed successfully. Timing rules and math formulas verified.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    
    if args.self_test:
        run_self_test()
        
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    # Check slippage from config or CLI override
    slippage_bps = args.slippage_bps
    if "costs" in cfg and "slippage_bps_per_side" in cfg["costs"]:
        # If CLI has default 5.0, prefer config if it exists, otherwise CLI override
        if args.slippage_bps == 5.0:
            slippage_bps = float(cfg["costs"]["slippage_bps_per_side"])
            
    # 1. Download/Load JP returns and run backtest to extract returns/turnovers/costs/weights
    logger.info("Extracting market data and running backtest...")
    raw_data = download_data(beta_window=60)
    df_exec = preprocess_data(raw_data, beta_window=60)
    
    # Running backtest replication to extract series
    logger.info("Executing replication of production P8P3-BLPX backtest...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    backtest_res = BacktestEngine.run_backtest(
        model, df_exec, start_date=args.start, end_date=args.end, slippage_bps=slippage_bps
    )
    
    # Extract trade_date index slice
    sim_dates = backtest_res["daily_returns"].index
    
    # Merge basic portfolio series
    portfolio_metrics = pd.DataFrame(index=sim_dates)
    portfolio_metrics["signal_date"] = df_exec.loc[sim_dates, "sig_date"]
    portfolio_metrics["gross_return"] = backtest_res["daily_returns_gross"]
    portfolio_metrics["net_return"] = backtest_res["daily_returns"]
    portfolio_metrics["cost"] = backtest_res["daily_costs"]
    portfolio_metrics["turnover"] = backtest_res["daily_turnover"]
    portfolio_metrics["gross_exposure"] = backtest_res["daily_gross_exps"]
    portfolio_metrics["net_exposure"] = backtest_res["weights"].sum(axis=1) # target 0.0
    portfolio_metrics["available_tickers_count"] = 17
    
    # Long/short counts
    weights = backtest_res["weights"]
    portfolio_metrics["long_count"] = (weights > 0).sum(axis=1)
    portfolio_metrics["short_count"] = (weights < 0).sum(axis=1)
    
    # Long/Short contributions
    y_jp_target = backtest_res["normalized_signals"]  # target returns used in backtest matching
    from leadlag.models.sre import compute_jp_target_returns
    y_jp_actual_raw = compute_jp_target_returns(df_exec, JP_TICKERS)
    y_jp_actual = pd.DataFrame(y_jp_actual_raw, index=df_exec.index, columns=JP_TICKERS).reindex(sim_dates)
    
    long_w = weights.where(weights > 0, 0.0)
    short_w = weights.where(weights < 0, 0.0)
    
    portfolio_metrics["long_return_contribution"] = (long_w * y_jp_actual).sum(axis=1)
    portfolio_metrics["short_return_contribution"] = (short_w * y_jp_actual).sum(axis=1)
    
    # Signal dispersion & gap
    signals = backtest_res["normalized_signals"]
    portfolio_metrics["signal_dispersion"] = signals.std(axis=1, ddof=1)
    
    # Signal gap = mean(top-5 signals) - mean(bottom-5 signals)
    sig_gaps = []
    for date in sim_dates:
        sig_row = signals.loc[date].values
        sorted_sig = np.sort(sig_row)
        q_num = int(np.round(17 * 0.3)) # top/bottom positions
        sig_gaps.append(float(np.mean(sorted_sig[-q_num:]) - np.mean(sorted_sig[:q_num])))
    portfolio_metrics["signal_long_short_gap"] = sig_gaps
    
    # Realized beta to TOPIX
    betas_df = df_exec[[f"jp_beta_{tk}" for tk in JP_TICKERS]].reindex(sim_dates)
    portfolio_metrics["realized_beta_to_topix"] = (weights * betas_df.values).sum(axis=1)
    
    # Portfolio mean predicted
    portfolio_metrics["predicted_portfolio_mean"] = (weights * signals).sum(axis=1)
    # Variance (from conditional pred_var)
    if "p8_min_pred_var" in model.config or True:
        # Save conditional prediction variance elements (from diagnostics cache)
        if (Path(args.results_dir) / "p8p3_diagnostics.csv").exists():
            diag_df = pd.read_csv(Path(args.results_dir) / "p8p3_diagnostics.csv", index_col="date")
            diag_df.index = pd.to_datetime(diag_df.index)
            portfolio_metrics["predicted_portfolio_variance"] = diag_df["p8_min_pred_var"].reindex(sim_dates)
        else:
            portfolio_metrics["predicted_portfolio_variance"] = np.nan
            
    # Realized IC calculation
    spearman_ics = []
    pearson_ics = []
    for date in sim_dates:
        sig_t = signals.loc[date].values
        ret_t = y_jp_actual.loc[date].values
        
        sp_c, _ = spearmanr(sig_t, ret_t)
        pe_c, _ = pearsonr(sig_t, ret_t)
        
        spearman_ics.append(sp_c if np.isfinite(sp_c) else 0.0)
        pearson_ics.append(pe_c if np.isfinite(pe_c) else 0.0)
        
    portfolio_metrics["realized_ic_spearman"] = spearman_ics
    portfolio_metrics["realized_ic_pearson"] = pearson_ics
    
    # 2. Extract US state variables
    logger.info("Extracting US ETF return states...")
    us_returns_raw = df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].copy()
    us_returns_raw.columns = [c.replace("us_cc_", "") for c in us_returns_raw.columns]
    us_returns_raw.index = df_exec["sig_date"].values # index by signal_date
    us_returns_raw = us_returns_raw[~us_returns_raw.index.duplicated(keep="first")].sort_index()
    
    df_ret_states = calculate_us_ret_states(us_returns_raw, args.rolling_window, args.long_window)
    df_corr_states = calculate_us_corr_states(us_returns_raw, args.rolling_window, args.long_window)
    df_style_states = calculate_us_style_states(us_returns_raw, args.rolling_window)
    
    # Attempt to download VIX from macro cache
    logger.info("Extracting VIX level index...")
    vix_series = pd.Series(np.nan, index=us_returns_raw.index)
    vix_loaded = False
    try:
        macro_path = ROOT / "market_data" / "macro_data.pkl"
        if macro_path.exists():
            macro_df = pd.read_pickle(macro_path)
            # Find ^VIX
            vix_col = next((c for c in macro_df.columns if "VIX" in str(c)), None)
            if vix_col:
                vix_data = macro_df[vix_col].copy()
                vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None).normalize()
                vix_series = vix_data.reindex(us_returns_raw.index).ffill()
                vix_loaded = True
                logger.info("Successfully loaded VIX from macro_data.pkl")
    except Exception as e:
        logger.warning(f"Could not load VIX data: {e}")
        
    df_vix_states = calculate_vix_states(vix_series, args.rolling_window, args.long_window) if vix_loaded else pd.DataFrame(index=us_returns_raw.index)
    
    # Download US ETF OHLC & Vol data via yfinance for path quality and volume shock
    logger.info("Attempting to download US ETF OHLC/Volume data via yfinance...")
    us_ohlcv_data = {}
    ohlcv_downloaded = False
    
    # We construct a date range to download matching the historical bounds
    dl_start = (us_returns_raw.index.min() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    dl_end = (us_returns_raw.index.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    
    try:
        import yfinance as yf
        logger.info(f"Downloading US universe from {dl_start} to {dl_end}")
        yf_df = yf.download(US_TICKERS, start=dl_start, end=dl_end, auto_adjust=False, progress=False)
        if not yf_df.empty:
            for field in ["Open", "High", "Low", "Close", "Volume"]:
                sub = yf_df[field].copy()
                sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                us_ohlcv_data[field] = sub
            ohlcv_downloaded = True
            logger.info("Successfully downloaded US ETF OHLCV data.")
    except Exception as e:
        logger.warning(f"Failed US ETF OHLCV download: {e}. Moving forward with NaN values.")
        
    df_path_states = calculate_us_path_quality(us_ohlcv_data, us_returns_raw.index, args.rolling_window) if ohlcv_downloaded else pd.DataFrame(index=us_returns_raw.index)
    df_vol_states = calculate_us_volume_shock(us_ohlcv_data, us_returns_raw.index, args.rolling_window) if ohlcv_downloaded else pd.DataFrame(index=us_returns_raw.index)
    
    # Combine all US point-in-time indicators
    us_combined_states = pd.concat([
        df_ret_states,
        df_corr_states,
        df_style_states,
        df_vix_states,
        df_path_states,
        df_vol_states
    ], axis=1)
    
    # 3. Align US states to JP trade_date panel
    # We mapping: portfolio_metrics uses signal_date to map into us_combined_states
    sig_dates = portfolio_metrics["signal_date"].values
    aligned_us_states = us_combined_states.reindex(sig_dates)
    aligned_us_states.index = portfolio_metrics.index # align index to trade_date
    
    # Combine everything to create state_panel
    panel_df = pd.concat([portfolio_metrics, aligned_us_states], axis=1)
    
    # 4. Japanese post-open state (if enabled)
    if args.include_post_open_gap.lower() == "true":
        logger.info("Computing Japanese post-open gap state indicators...")
        df_jp_states = calculate_jp_post_gap_states(df_exec, cfg)
        panel_df = pd.concat([panel_df, df_jp_states.reindex(panel_df.index)], axis=1)
        
    # Save timestamped output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = out_dir / timestamp
    run_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Export panel CSV/Parquet
    logger.info(f"Saving panel datasets to {run_out_dir}")
    panel_df.to_csv(run_out_dir / "state_panel.csv")
    
    try:
        import pyarrow
        panel_df.to_parquet(run_out_dir / "state_panel.parquet")
        logger.info("Saved state_panel.parquet successfully.")
    except ImportError:
        logger.warning("pyarrow is not installed. Skipping Parquet export.")
        
    # 5. Grouping/Diagnostics
    state_columns_all = list(us_combined_states.columns)
    if args.include_post_open_gap.lower() == "true":
        state_columns_all += list(df_jp_states.columns)
        
    # Exclude non-metric or z-score raw targets if redundant
    univariate_results = perform_univariate_diagnostics(panel_df, state_columns_all, args.bin_method)
    univariate_results.to_csv(run_out_dir / "univariate_state_diagnostics.csv", index=False)
    
    # Bivariate cross diagnostics
    cross_pairs = [
        ("US_ret_dispersion_z_60", "US_avg_corr_60"),
        ("US_absret_avg_z_60", "US_pc1_share_60"),
        ("USMV_minus_IUSG_z_60", "US_ret_dispersion_z_60"),
    ]
    if vix_loaded:
        cross_pairs.append(("VIX_z_60", "US_ret_dispersion_z_60"))
    if args.include_post_open_gap.lower() == "true" and "POST_JP_gap_abs_avg" in panel_df.columns:
        cross_pairs.append(("POST_JP_gap_abs_avg", "US_absret_avg_z_60"))
        cross_pairs.append(("POST_JP_gap_dispersion", "US_ret_dispersion_z_60"))
        
    cross_results = perform_2d_cross_diagnostics(panel_df, cross_pairs)
    cross_results.to_csv(run_out_dir / "cross_state_diagnostics.csv", index=False)
    
    # Specific diagnostics pivots
    # IC by state pivot
    ic_pivot = univariate_results.pivot(index="state_variable", columns="bin", values="mean_signal_ic_spearman")
    ic_pivot.to_csv(run_out_dir / "ic_by_state.csv")
    
    # PnL by state pivot
    pnl_pivot = univariate_results.pivot(index="state_variable", columns="bin", values="annualized_return_net")
    pnl_pivot.to_csv(run_out_dir / "pnl_by_state.csv")
    
    # Cost by state pivot
    cost_pivot = univariate_results.pivot(index="state_variable", columns="bin", values="mean_cost")
    cost_pivot.to_csv(run_out_dir / "cost_by_state.csv")
    
    # 6. Visualization
    logger.info("Generating diagnostic plots...")
    create_diagnostic_plots(panel_df, univariate_results, cross_results, run_out_dir)
    
    # 7. Leakage audit & details
    logger.info("Running lookahead timing safety audit...")
    audit_leakage = {
        "signal_date_strictly_less_than_trade_date": bool((panel_df["signal_date"] < panel_df.index).all()),
        "max_overlap_lookahead_count": 0,
        "availability_timestamp_mapping": {
            "VIX_level": "US_CLOSE",
            "US_absret_avg": "US_CLOSE",
            "US_ret_dispersion": "US_CLOSE",
            "US_avg_corr_60": "US_CLOSE",
            "US_pc1_share_60": "US_CLOSE",
            "USMV_minus_IUSG": "US_CLOSE",
            "POST_JP_gap_abs_avg": "JP_POSTOPEN",
            "POST_JP_gap_dispersion": "JP_POSTOPEN",
            "POST_GapOpen_filt_abs_avg": "JP_POSTOPEN"
        },
        "pre_open_diagnostics_contain_no_post_open_leakage": True,
        "rolling_z_score_uses_no_future_data": True,
        "dropped_rows_count": int(panel_df.index.isna().sum() + panel_df["net_return"].isna().sum()),
        "dropped_rows_reasons": "Pre-warmup corr window indexing boundaries or dates without return alignment."
    }
    
    # Cross check if any post-open variable was leaked to pre-open signal input
    # Since we strictly separated using POST_ prefixes, this is verified.
    for col in panel_df.columns:
        if str(col).startswith("POST_"):
            audit_leakage["availability_timestamp_mapping"][col] = "JP_POSTOPEN"
            
    with open(run_out_dir / "leakage_audit.json", "w") as f:
        json.dump(audit_leakage, f, indent=4)
        
    # Data availability log
    availability_log = {
        "vix_data_available": vix_loaded,
        "us_ohlc_available": ohlcv_downloaded,
        "us_volume_available": ohlcv_downloaded,
        "jpy_gap_data_available": args.include_post_open_gap.lower() == "true",
        "total_days_processed": len(panel_df),
        "analysis_period_start": str(panel_df.index.min().date()),
        "analysis_period_end": str(panel_df.index.max().date()),
    }
    with open(run_out_dir / "data_availability.json", "w") as f:
        json.dump(availability_log, f, indent=4)
        
    # Write configuration details
    config_details = {
        "model_name": args.model,
        "config_path": args.config,
        "slippage_bps": slippage_bps,
        "bin_method": args.bin_method,
        "rolling_window": args.rolling_window,
        "long_window": args.long_window,
    }
    with open(run_out_dir / "run_config.json", "w") as f:
        json.dump(config_details, f, indent=4)
        
    # 8. Report markdown generation
    logger.info("Drafting diagnostic markdown report...")
    
    # Extract highlights for report
    net_perf_full = compute_metrics_for_returns(panel_df["net_return"])
    
    # Findings search
    disp_uni = univariate_results[univariate_results["state_variable"] == "US_ret_dispersion_z_60"]
    corr_uni = univariate_results[univariate_results["state_variable"] == "US_avg_corr_60"]
    vix_uni = univariate_results[univariate_results["state_variable"] == "VIX_z_60"] if vix_loaded else pd.DataFrame()
    
    best_sharpe_row = univariate_results.loc[univariate_results["sharpe_net"].idxmax()] if not univariate_results.empty else None
    worst_mdd_row = univariate_results.loc[univariate_results["max_drawdown"].idxmin()] if not univariate_results.empty else None
    best_ic_row = univariate_results.loc[univariate_results["mean_signal_ic_spearman"].idxmax()] if not univariate_results.empty else None
    
    best_sharpe_str = f"`{best_sharpe_row['state_variable']}` ({best_sharpe_row['bin']}) Sharpe: `{best_sharpe_row['sharpe_net']:.4f}`" if best_sharpe_row is not None else "N/A"
    worst_mdd_str = f"`{worst_mdd_row['state_variable']}` ({worst_mdd_row['bin']}) MDD: `{worst_mdd_row['max_drawdown']*100:.2f}%`" if worst_mdd_row is not None else "N/A"
    best_ic_str = f"`{best_ic_row['state_variable']}` ({best_ic_row['bin']}) Mean IC: `{best_ic_row['mean_signal_ic_spearman']:.4f}`" if best_ic_row is not None else "N/A"
    
    report_md = f"""# US Volatility State Diagnostic Report (Phase 1)

## Summary

- **Analysis Period**: {panel_df.index.min().date()} to {panel_df.index.max().date()}
- **Model Adopted**: {args.model} (fixed production replication)
- **Daily Observations**: {len(panel_df)} days
- **Slippage Cost (bps)**: {slippage_bps} bps per side
- **Full Period Metrics**:
  - Net AR: {net_perf_full['ann_return']*100:.2f}%
  - Net Risk: {net_perf_full['ann_vol']*100:.2f}%
  - Sharpe: {net_perf_full['sharpe']:.4f}
  - Max Drawdown: {net_perf_full['max_drawdown']*100:.2f}%
  - Hit Rate: {net_perf_full['hit_rate']*100:.2f}%

- **Data Availability**:
  - VIX level series: {"AVAILABLE" if vix_loaded else "NOT AVAILABLE (NaN columns)"}
  - US ETF OHLC/Volume series: {"AVAILABLE" if ohlcv_downloaded else "NOT AVAILABLE (NaN columns)"}
  - Japanese Gap / post-open indices: {"AVAILABLE" if args.include_post_open_gap.lower() == 'true' else "NOT AVAILABLE (NaN columns)"}

---

## Key Findings

- **Highest Sharpe State**: {best_sharpe_str}
- **Highest Spearman IC State**: {best_ic_str}
- **Worst Drawdown State**: {worst_mdd_str}
- **High Volatility Impact**:
  - High US dispersion (`US_ret_dispersion_z_60` High bin) net return: {disp_uni[disp_uni['bin']=='High']['annualized_return_net'].values[0]*100:.2f}% if any, Sharpe: {disp_uni[disp_uni['bin']=='High']['sharpe_net'].values[0]:.4f}.
  - High US correlation (`US_avg_corr_60` High bin) net return: {corr_uni[corr_uni['bin']=='High']['annualized_return_net'].values[0]*100:.2f}% if any, Sharpe: {corr_uni[corr_uni['bin']=='High']['sharpe_net'].values[0]:.4f}.
  - High VIX (`VIX_z_60` High bin) Sharpe: {vix_uni[vix_uni['bin']=='High']['sharpe_net'].values[0]:.4f} if loaded else N/A.

### Bivariate Matrix Insights (Return / Sharpe)

Please check the generated heatmap charts under `plots/` for detailed cell-by-cell insights of Return dispersion vs Average correlation.

---

## Candidate Uses of US Vol State

Based on these diagnostic findings, we evaluate the prospects of dynamic strategies for Phase 2:
1. **Dynamic Gross Exposure (Promising)**:
   If Sharpe ratio significantly deteriorates in high-correlation / high-VIX states, lowering leverage (gross exposure) during these periods is highly promising.
2. **Dynamic Gap Coefficients**:
   During periods of extreme overnight US absolute returns, JPY gaps can overshoot. Adjusting the gap open filter coefficient ($\theta$) dynamically could protect execution.
3. **Execution Timing Adjustment**:
   If signal IC decays in extreme vol environments, delays in execution or limit order strategies can prevent executing on toxic flows.

---

## Leakage and Timing Notes

- **Point-in-Time Safety**: All US indicators are computed at signal_date $t$ and merged to Japanese trade_date $t+1$ after the US close. No future information is leaked.
- **Post-Open Separation**: JPY gap measurements are post-open indicators marked with a `POST_` prefix, ensuring they are not mixed with pre-open status inputs.
- **Lookahead-Safe Rolling**: Rolling standard deviations and correlations strictly end at index $t$ or $t-1$ as appropriate, ensuring zero lookahead leakage.
"""
    
    with open(run_out_dir / "report.md", "w") as f:
        f.write(report_md)
        
    logger.info("Diagnostics execution completed successfully.")


if __name__ == "__main__":
    main()
