#!/usr/bin/env python3
"""Historical Simulation of Hybrid Execution Strategy (Limit Order + Gap Residual)

This script simulates:
1. Pure Baseline (with 5bps/10bps round-trip slippage)
2. Pure Baseline (with 0bps slippage - theoretical upper bound)
3. Pure Gap Residual (with 5bps/10bps round-trip slippage)
4. Hybrid Strategy:
   - At 9:00: Place limit orders for 10 baseline-selected tickers.
   - At 9:10: Check fill status against P_open. Filled orders have 0 entry slippage (5bps exit). Unfilled orders are canceled.
   - At 9:10: Fallback the unfilled long/short budget using the gap_residual strategy (5bps entry + 5bps exit).

Author: Antigravity
Date: 2026-06-03
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Setup Paths
_TOOLS_DIR = Path(__file__).parent
_PROJECT_ROOT = _TOOLS_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
_DATA_DIR = _PROJECT_ROOT / "data"
_RESULTS_DIR = _PROJECT_ROOT / "results" / "hybrid_simulation"

sys.path.insert(0, str(_SRC_DIR))

from config import STRATEGY_DEFAULTS, N_US_ASSETS, N_JP_ASSETS
from domain.models.types import StrategyConfig
from domain.signals import lead_lag as signals
from data_loader import download_data, preprocess_data
from data.cache import is_decision_cache_valid, load_decision_cache, save_decision_cache
from performance import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid_sim")

def load_df_exec() -> pd.DataFrame:
    """Load execution DataFrame using decision cache or preprocess raw data."""
    if is_decision_cache_valid():
        logger.info("Loading df_exec from decision cache (fast path)...")
        return load_decision_cache()
    
    logger.info("Decision cache invalid/missing. Downloading and preprocessing raw data...")
    data = download_data(beta_window=STRATEGY_DEFAULTS["beta_window"])
    df_exec = preprocess_data(data, beta_window=STRATEGY_DEFAULTS["beta_window"])
    save_decision_cache(df_exec)
    return df_exec

def round_tse_tick(p: float) -> float:
    """Round price to the official Tokyo Stock Exchange (TSE) tick sizes."""
    if p <= 1000.0:
        return round(p, 1)
    elif p <= 10000.0:
        return float(round(p))
    elif p <= 100000.0:
        return float(round(p / 10.0) * 10)
    else:
        return float(round(p / 100.0) * 100)

def run_simulation(df_exec: pd.DataFrame, start_date: str, config: StrategyConfig) -> pd.DataFrame:
    """Run the historical simulation and return a daily returns DataFrame."""
    all_cc_cols = [c for c in df_exec.columns if c.startswith("us_cc_") or c.startswith("jp_cc_")]
    jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
    jp_close_sig_cols = [c for c in df_exec.columns if c.startswith("jp_close_sig_")]
    jp_open_trade_cols = [c for c in df_exec.columns if c.startswith("jp_open_trade_")]
    gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]
    topix_night_cols = "topix_night_return"

    all_returns = df_exec[all_cc_cols].values
    date_index = df_exec.index.values
    jp_oc = df_exec[jp_oc_cols].values
    jp_close_sig = df_exec[jp_close_sig_cols].values
    jp_open_trade = df_exec[jp_open_trade_cols].values
    jp_gap = df_exec[gap_cols].values
    jp_beta = df_exec[beta_cols].values
    topix_night = df_exec[topix_night_cols].values if topix_night_cols in df_exec.columns else None

    n_u = N_US_ASSETS
    n_j = N_JP_ASSETS
    q = config.q
    weight_mode = config.weight_mode
    dispersion_filter = config.dispersion_filter
    dispersion_metric = config.dispersion_metric
    gamma = config.gamma
    vol_adjusted_target = config.vol_adjusted_target

    # Pre-compute baseline correlation matrix
    c_full = signals.compute_baseline_correlation(all_returns, date_index, config.ewma_half_life)
    v0_static = signals.build_v3_static(n_u, n_j, config.include_v4_prior)
    base_vectors = signals.build_base_vectors(n_u, n_j)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    # Determine start index
    start_idx = max(
        df_exec.index.searchsorted(pd.to_datetime(start_date)),
        config.corr_window
    )

    # Initialize dispersion history
    disp_hist_base = []
    disp_hist_gap = []
    disp_hist_hybrid = []

    history_start = max(0, start_idx - 60)
    for hist_i in range(history_start, start_idx):
        # Base signal
        sig_res_b = signals.compute_signal(
            all_returns, hist_i, n_u, config.corr_window, c_full, v0_static, v1, v2,
            config.k, config.lambda_reg, config.lambda_lw, config.lw_target,
            config.ewma_half_life, v3_dynamic=(config.v3_mode == "dynamic"),
            gap_override=None, vol_adjusted_target=vol_adjusted_target
        )
        disp_b = signals.compute_dispersion_indicator(sig_res_b["signal"], q, n_j, dispersion_metric)
        disp_hist_base.append(disp_b)
        disp_hist_hybrid.append(disp_b)

        # Gap signal
        gap_hist = np.nan_to_num(jp_gap[hist_i], nan=0.0)
        betas_hist = np.asarray(jp_beta[hist_i], dtype=float)
        topix_hist = float(topix_night[hist_i]) if topix_night is not None else None
        sig_res_g = signals.compute_signal(
            all_returns, hist_i, n_u, config.corr_window, c_full, v0_static, v1, v2,
            config.k, config.lambda_reg, config.lambda_lw, config.lw_target,
            config.ewma_half_life, v3_dynamic=(config.v3_mode == "dynamic"),
            gap_override=gap_hist, gap_open_coef=config.gap_open_coef,
            topix_beta_coef=config.topix_beta_coef, betas_t=betas_hist,
            topix_night_t=topix_hist, vol_adjusted_target=vol_adjusted_target
        )
        disp_g = signals.compute_dispersion_indicator(sig_res_g["signal"], q, n_j, dispersion_metric)
        disp_hist_gap.append(disp_g)

    logger.info("Running simulation loop from %s (%d days)...", start_date, len(df_exec) - start_idx)

    daily_results = []
    
    for i in range(start_idx, len(df_exec)):
        t_trade = df_exec.index[i]
        
        # Current data
        r_oc_t1 = np.nan_to_num(jp_oc[i], nan=0.0)
        p_close_prev = jp_close_sig[i]
        p_open = jp_open_trade[i]
        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0)
        betas_t = np.asarray(jp_beta[i], dtype=float)
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        # ----------------------------------------------------
        # 1. Pure Baseline (with & without slippage)
        # ----------------------------------------------------
        sig_res_base = signals.compute_signal(
            all_returns, i, n_u, config.corr_window, c_full, v0_static, v1, v2,
            config.k, config.lambda_reg, config.lambda_lw, config.lw_target,
            config.ewma_half_life, v3_dynamic=(config.v3_mode == "dynamic"),
            gap_override=None, vol_adjusted_target=vol_adjusted_target
        )
        sig_base = np.asarray(sig_res_base["signal"], dtype=float)
        sigma_s = float(sig_res_base["sigma_s"])
        
        w_base = signals.build_weights(sig_base, q, n_j, weight_mode)
        disp_b = signals.compute_dispersion_indicator(sig_base, q, n_j, dispersion_metric)
        scale_b = signals.dispersion_scale(disp_b, disp_hist_base, dispersion_filter)
        disp_hist_base.append(disp_b)
        scaled_w_base = w_base * scale_b

        gross_exposure_base = float(np.sum(np.abs(scaled_w_base)))
        ret_base_gross = float(np.sum(scaled_w_base * r_oc_t1))
        # 10bps round-trip slippage
        slip_base = 2.0 * 0.0005 * gross_exposure_base
        ret_base_slip = ret_base_gross - slip_base
        ret_base_noslip = ret_base_gross

        # ----------------------------------------------------
        # 2. Pure Gap Residual (with slippage)
        # ----------------------------------------------------
        sig_res_gap = signals.compute_signal(
            all_returns, i, n_u, config.corr_window, c_full, v0_static, v1, v2,
            config.k, config.lambda_reg, config.lambda_lw, config.lw_target,
            config.ewma_half_life, v3_dynamic=(config.v3_mode == "dynamic"),
            gap_override=gap_t1, gap_open_coef=config.gap_open_coef,
            topix_beta_coef=config.topix_beta_coef, betas_t=betas_t,
            topix_night_t=topix_night_t, vol_adjusted_target=vol_adjusted_target
        )
        sig_gap = np.asarray(sig_res_gap["signal"], dtype=float)
        w_gap_pure = signals.build_weights(sig_gap, q, n_j, weight_mode)
        disp_g = signals.compute_dispersion_indicator(sig_gap, q, n_j, dispersion_metric)
        scale_g = signals.dispersion_scale(disp_g, disp_hist_gap, dispersion_filter)
        disp_hist_gap.append(disp_g)
        scaled_w_gap = w_gap_pure * scale_g

        gross_exposure_gap = float(np.sum(np.abs(scaled_w_gap)))
        ret_gap_gross = float(np.sum(scaled_w_gap * r_oc_t1))
        slip_gap_pure = 2.0 * 0.0005 * gross_exposure_gap
        ret_gap_slip = ret_gap_gross - slip_gap_pure

        # ----------------------------------------------------
        # 3. Hybrid Strategy Simulation
        # ----------------------------------------------------
        disp_h = signals.compute_dispersion_indicator(sig_base, q, n_j, dispersion_metric)
        scale_h = signals.dispersion_scale(disp_h, disp_hist_hybrid, dispersion_filter)
        disp_hist_hybrid.append(disp_h)

        # Baseline physical orders placed at 9:00 (scaled by dispersion filter)
        target_w_base = w_base * scale_h
        
        # Calculate pre-opening limit prices
        limit_prices = np.full(n_j, np.nan)
        for j in range(n_j):
            if abs(target_w_base[j]) > 1e-12:
                prev_close = p_close_prev[j]
                if not (np.isfinite(prev_close) and prev_close > 0):
                    continue
                if target_w_base[j] > 0: # BUY
                    lp = prev_close * (1.0 + gamma * sig_base[j] * sigma_s)
                else: # SELL
                    lp = prev_close * (1.0 - gamma * abs(sig_base[j]) * sigma_s)
                limit_prices[j] = round_tse_tick(lp)

        # 9:10 Open fill check
        filled_limit = np.zeros(n_j, dtype=bool)
        for j in range(n_j):
            if abs(target_w_base[j]) > 1e-12:
                po = p_open[j]
                lp = limit_prices[j]
                if not (np.isfinite(po) and po > 0 and np.isfinite(lp)):
                    continue
                if target_w_base[j] > 0: # BUY
                    if po <= lp:
                        filled_limit[j] = True
                else: # SELL
                    if po >= lp:
                        filled_limit[j] = True

        # Unfilled orders are canceled. Keep the filled ones.
        w_filled_base = np.zeros(n_j)
        w_filled_base[filled_limit] = target_w_base[filled_limit]

        # Calculate remaining budgets to allocate via gap_residual
        W_filled_long = float(np.sum(w_filled_base[w_filled_base > 0]))
        W_filled_short = float(np.sum(np.abs(w_filled_base[w_filled_base < 0])))
        
        B_long = max(0.0, scale_h - W_filled_long)
        B_short = max(0.0, scale_h - W_filled_short)

        # We allocate B_long and B_short among unfilled assets using gap_residual signals
        unfilled_mask = ~filled_limit
        unfilled_indices = np.where(unfilled_mask)[0]
        
        w_gap_hybrid = np.zeros(n_j)
        
        if len(unfilled_indices) > 0 and (B_long > 1e-12 or B_short > 1e-12):
            # Select long and short candidates from the unfilled pool
            # Center the gap signals
            s_centered = sig_gap - np.median(sig_gap)
            
            # Sort unfilled indices by gap signal
            sorted_unfilled = unfilled_indices[np.argsort(sig_gap[unfilled_indices])]
            
            # Target number of long/short positions in gap residual fallback
            num_long_target = max(0, int(np.floor(n_j * q)) - int(np.sum((w_filled_base > 0))))
            num_short_target = max(0, int(np.floor(n_j * q)) - int(np.sum((w_filled_base < 0))))
            
            # Long candidates (highest signals among unfilled)
            if num_long_target > 0 and len(sorted_unfilled) > 0:
                long_sel = sorted_unfilled[-num_long_target:]
            else:
                long_sel = np.array([], dtype=int)
                
            # Short candidates (lowest signals among unfilled)
            if num_short_target > 0 and len(sorted_unfilled) > 0:
                short_sel = sorted_unfilled[:num_short_target]
            else:
                short_sel = np.array([], dtype=int)

            if weight_mode == "signal":
                # Long allocation
                if len(long_sel) > 0 and B_long > 1e-12:
                    raw_w = np.maximum(s_centered[long_sel], 1e-8)
                    denom = np.sum(raw_w)
                    if denom > 0:
                        w_gap_hybrid[long_sel] = B_long * (raw_w / denom)
                # Short allocation
                if len(short_sel) > 0 and B_short > 1e-12:
                    raw_w = np.maximum(-s_centered[short_sel], 1e-8)
                    denom = np.sum(raw_w)
                    if denom > 0:
                        w_gap_hybrid[short_sel] = -B_short * (raw_w / denom)
            else: # equal weighting
                if len(long_sel) > 0 and B_long > 1e-12:
                    w_gap_hybrid[long_sel] = B_long / len(long_sel)
                if len(short_sel) > 0 and B_short > 1e-12:
                    w_gap_hybrid[short_sel] = -B_short / len(short_sel)

        # Final hybrid portfolio weights
        w_hybrid = w_filled_base + w_gap_hybrid

        # Verify dollar-neutrality and gross exposure matching scale_h
        gross_exposure_hybrid = float(np.sum(np.abs(w_hybrid)))
        net_exposure_hybrid = float(np.sum(w_hybrid))
        
        # Calculate returns
        ret_hybrid_gross = float(np.sum(w_hybrid * r_oc_t1))
        
        # Slippage cost:
        # - Filled limit orders: 0 entry slippage, 5bps exit slippage (one-way = 5bps)
        # - Fallback gap residual orders: 5bps entry slippage, 5bps exit slippage (round-trip = 10bps)
        slip_filled_base = float(np.sum(np.abs(w_filled_base))) * 0.0005
        slip_gap_fallback = float(np.sum(np.abs(w_gap_hybrid))) * 0.0010
        slip_hybrid = slip_filled_base + slip_gap_fallback
        ret_hybrid_slip = ret_hybrid_gross - slip_hybrid

        # Fill metrics
        n_active_base = int(np.sum(np.abs(target_w_base) > 1e-12))
        n_filled_base = int(np.sum(filled_limit))
        fill_rate = n_filled_base / n_active_base if n_active_base > 0 else np.nan

        daily_results.append({
            "trade_date": t_trade,
            # Returns
            "ret_base_slip": ret_base_slip,
            "ret_base_noslip": ret_base_noslip,
            "ret_gap_slip": ret_gap_slip,
            "ret_hybrid_slip": ret_hybrid_slip,
            "ret_hybrid_gross": ret_hybrid_gross,
            # Slippage costs
            "slip_base": slip_base,
            "slip_gap": slip_gap_pure,
            "slip_hybrid": slip_hybrid,
            # Exposures
            "gross_base": gross_exposure_base,
            "gross_gap": gross_exposure_gap,
            "gross_hybrid": gross_exposure_hybrid,
            "net_hybrid": net_exposure_hybrid,
            # Fill stats
            "n_active_base": n_active_base,
            "n_filled_base": n_filled_base,
            "fill_rate": fill_rate,
            "w_filled_base_gross": float(np.sum(np.abs(w_filled_base))),
            "w_gap_hybrid_gross": float(np.sum(np.abs(w_gap_hybrid))),
        })

    df_results = pd.DataFrame(daily_results).set_index("trade_date")
    return df_results

def print_performance_table(df_results: pd.DataFrame, start: str, end: str = None) -> pd.DataFrame:
    """Calculate and print performance summary table."""
    sub = df_results
    if start:
        sub = sub[sub.index >= pd.to_datetime(start)]
    if end:
        sub = sub[sub.index < pd.to_datetime(end)]
        
    strategies = {
        "Baseline (10bps Slip)": "ret_base_slip",
        "Baseline (0bps Slip)": "ret_base_noslip",
        "Gap Residual (10bps Slip)": "ret_gap_slip",
        "Hybrid Strategy (Slip)": "ret_hybrid_slip"
    }
    
    rows = []
    for name, col in strategies.items():
        metrics = calculate_metrics(sub[col])
        # Add fill rate details if hybrid
        if name == "Hybrid Strategy (Slip)":
            fill_mean = sub["fill_rate"].mean() * 100
        else:
            fill_mean = np.nan
            
        rows.append({
            "Strategy": name,
            "Annual Return (AR)": f"{metrics.get('AR', 0.0)*100:.2f}%",
            "Annual Risk (RISK)": f"{metrics.get('RISK', 0.0)*100:.2f}%",
            "R/R Ratio": f"{metrics.get('R/R', 0.0):.3f}",
            "Sharpe": f"{metrics.get('Sharpe', 0.0):.3f}",
            "Max Drawdown (MDD)": f"{metrics.get('MDD', 0.0)*100:.2f}%",
            "Avg Fill Rate": f"{fill_mean:.1f}%" if np.isfinite(fill_mean) else "N/A"
        })
        
    df_perf = pd.DataFrame(rows)
    return df_perf

def main():
    parser = argparse.ArgumentParser(description="Simulate Hybrid Execution Strategy")
    parser.add_argument("--start-date", default="2015-01-01", help="Simulation start date")
    parser.add_argument("--vol-adjusted", action="store_true", default=True, help="Use volatility adjusted target Z-score")
    parser.add_argument("--no-vol-adjusted", action="store_false", dest="vol_adjusted", help="Disable volatility adjusted target")
    args = parser.parse_args()

    logger.info("Initializing Hybrid Strategy Simulation...")
    df_exec = load_df_exec()
    
    # Configure strategy defaults
    config = StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
        dispersion_metric=STRATEGY_DEFAULTS.get("dispersion_metric", "long_short_mean_gap"),
        v3_mode=STRATEGY_DEFAULTS["v3_mode"],
        ewma_half_life=STRATEGY_DEFAULTS["ewma_half_life"],
        lambda_lw=STRATEGY_DEFAULTS["lambda_lw"],
        lw_target=STRATEGY_DEFAULTS["lw_target"],
        corr_window=STRATEGY_DEFAULTS["corr_window"],
        include_v4_prior=STRATEGY_DEFAULTS["include_v4_prior"],
        signal_mode="gap_residual",
        gap_open_coef=STRATEGY_DEFAULTS["gap_open_coef"],
        topix_beta_coef=STRATEGY_DEFAULTS["topix_beta_coef"],
        beta_window=STRATEGY_DEFAULTS["beta_window"],
        gamma=STRATEGY_DEFAULTS.get("gamma", 0.5),
        vol_adjusted_target=args.vol_adjusted
    )
    
    logger.info("Strategy Configuration:")
    logger.info("  K: %d", config.k)
    logger.info("  gamma: %.2f", config.gamma)
    logger.info("  vol_adjusted_target: %s", config.vol_adjusted_target)
    logger.info("  gap_open_coef: %.2f", config.gap_open_coef)
    logger.info("  topix_beta_coef: %.2f", config.topix_beta_coef)

    df_results = run_simulation(df_exec, args.start_date, config)

    # Output directory
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(_RESULTS_DIR / "daily_results.csv")
    logger.info("Daily results saved to: %s", _RESULTS_DIR / "daily_results.csv")

    # 1. Print Overall Performance
    print("\n" + "="*80)
    print(f"OVERALL PERFORMANCE COMPARISON ({args.start_date} to Present)")
    print("="*80)
    perf_overall = print_performance_table(df_results, args.start_date)
    print(perf_overall.to_markdown(index=False))
    print("="*80)

    # 2. Yearly Breakdown
    years = sorted(list(set(df_results.index.year)))
    yearly_rows = []
    print("\n" + "="*80)
    print("YEAR-BY-YEAR HYBRID STRATEGY METRICS")
    print("="*80)
    for y in years:
        sub_yr = df_results[df_results.index.year == y]
        metrics_hybrid = calculate_metrics(sub_yr["ret_hybrid_slip"])
        metrics_gap = calculate_metrics(sub_yr["ret_gap_slip"])
        metrics_base = calculate_metrics(sub_yr["ret_base_slip"])
        
        yearly_rows.append({
            "Year": y,
            "Baseline AR": f"{metrics_base.get('AR', 0.0)*100:.1f}%",
            "Gap Residual AR": f"{metrics_gap.get('AR', 0.0)*100:.1f}%",
            "Hybrid AR": f"{metrics_hybrid.get('AR', 0.0)*100:.1f}%",
            "Hybrid RISK": f"{metrics_hybrid.get('RISK', 0.0)*100:.1f}%",
            "Hybrid R/R": f"{metrics_hybrid.get('R/R', 0.0):.3f}",
            "Hybrid MDD": f"{metrics_hybrid.get('MDD', 0.0)*100:.1f}%",
            "Fill Rate": f"{sub_yr['fill_rate'].mean()*100:.1f}%",
            "Limit Exposure Ratio": f"{(sub_yr['w_filled_base_gross'] / sub_yr['gross_hybrid']).mean()*100:.1f}%"
        })
    df_yearly = pd.DataFrame(yearly_rows)
    print(df_yearly.to_markdown(index=False))
    print("="*80)
    df_yearly.to_csv(_RESULTS_DIR / "yearly_breakdown.csv", index=False)

    # 3. Cumulative Return Plot
    plt.figure(figsize=(12, 7))
    plt.plot((1.0 + df_results["ret_base_noslip"]).cumprod(), label="Baseline (0bps Slip - Theoretical)", color="#2ca02c", linewidth=1.5, alpha=0.7)
    plt.plot((1.0 + df_results["ret_base_slip"]).cumprod(), label="Baseline (10bps Slip)", color="#d62728", linewidth=1.5, alpha=0.7)
    plt.plot((1.0 + df_results["ret_gap_slip"]).cumprod(), label="Gap Residual (10bps Slip)", color="#ff7f0e", linewidth=1.5, alpha=0.7)
    plt.plot((1.0 + df_results["ret_hybrid_slip"]).cumprod(), label="Hybrid: Limit + Fallback (Slip)", color="#1f77b4", linewidth=2.5)
    
    plt.yscale("log")
    plt.title("Cumulative Return Comparison (Log Scale)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Cumulative Growth of 1.0", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    
    plot_path = _RESULTS_DIR / "cumulative_returns.png"
    plt.savefig(plot_path, dpi=150)
    logger.info("Saved cumulative returns plot to: %s", plot_path)
    print(f"\nPlot saved to: {plot_path}")

if __name__ == "__main__":
    main()
