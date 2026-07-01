"""src/leadlag/diagnostics/sprint1_experiments.py — Sprint 1 feasibility diagnostics.

Implements targets panel separation, liquidity constrained simulations,
TOPIX beta-neutral optimization, and rolling RuleD calibration.
"""

from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import yfinance as yf

from leadlag.data.cache import load_df_exec_from_local_cache, load_intraday_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS, TOPIX_TICKER
from experiments.diagnostics.sprint0 import find_latest_distribution_diagnostics, compute_rolling_beta
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Rolling Beta Series
# ---------------------------------------------------------------------------
def compute_rolling_beta_series(r_asset: pd.Series, r_market: pd.Series, window: int) -> pd.Series:
    """Compute lookahead-safe rolling beta of asset series vs market (shifted by 1 day)."""
    cov = r_asset.rolling(window).cov(r_market)
    var = r_market.rolling(window).var().clip(lower=1e-8)
    beta = cov.divide(var)
    return beta.shift(1)


# ---------------------------------------------------------------------------
# 1. Target Separation Panel
# ---------------------------------------------------------------------------
def generate_targets_panel(
    df_exec: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Separate Close-to-Close, Gap, Open-to-Close, and Entry-to-Close target returns."""
    df_exec = df_exec.copy()
    for tk in JP_TICKERS:
        for suffix in ["gap", "oc"]:
            col = f"jp_{suffix}_{tk}"
            if col in df_exec.columns:
                df_exec[col] = df_exec[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
    sim_dates = df_exec.index
    
    # Load 5m cache to identify days where 9:10 prices exist
    df_5m = load_intraday_cache("5m")
    has_5m_dates = set()
    r_open_910_dict = {}
    if df_5m is not None and not df_5m.empty:
        has_5m_dates = set(pd.Series(df_5m.index.date).unique())
        
        # Calculate ret_open_910 for true 9:10 days
        dates_5m = pd.Series(df_5m.index.date).unique()
        for dt in dates_5m:
            dt_ts = pd.Timestamp(dt)
            day_data = df_5m[df_5m.index.date == dt]
            idx_910 = pd.Timestamp(f"{dt} 09:10:00")
            row_910 = day_data.loc[idx_910] if idx_910 in day_data.index else None
            
            ticker_returns = {}
            for ticker in JP_TICKERS:
                p_910 = np.nan
                if row_910 is not None:
                    high = row_910.get(("High", ticker))
                    low = row_910.get(("Low", ticker))
                    close = row_910.get(("Close", ticker))
                    p_910 = (high + low) / 2 if (pd.notna(high) and pd.notna(low)) else close
                    
                p_open_5m = np.nan
                for time_str in ["09:00:00", "09:05:00", "09:10:00"]:
                    idx_time = pd.Timestamp(f"{dt} {time_str}")
                    if idx_time in day_data.index:
                        row_time = day_data.loc[idx_time]
                        op = row_time.get(("Open", ticker))
                        cl = row_time.get(("Close", ticker))
                        val = op if pd.notna(op) else cl
                        if pd.notna(val):
                            p_open_5m = val
                            break
                
                ret = 0.0
                if pd.notna(p_910) and pd.notna(p_open_5m) and p_open_5m > 0:
                    ret = float(p_910 / p_open_5m - 1.0)
                ticker_returns[ticker] = ret
            r_open_910_dict[dt_ts] = ticker_returns

    rows = []
    r_topix_cc = df_exec["topix_cc_trade"]
    r_topix_night = df_exec["topix_night_return"]
    r_topix_oc = df_exec["topix_oc_return"]

    # Precalculate beta for each ticker and window to ensure efficient lookahead-free estimation
    betas_60d = {}
    betas_120d = {}
    
    for tk in JP_TICKERS:
        jp_gap = df_exec[f"jp_gap_{tk}"]
        jp_oc = df_exec[f"jp_oc_{tk}"]
        
        # Calculate returns
        r_cc_tk = (1.0 + jp_gap) * (1.0 + jp_oc) - 1.0
        
        # Construct entry_to_close and gap returns dynamically based on is_true_0910
        r_entry_to_close_tk = pd.Series(0.0, index=sim_dates)
        r_gap_tk = pd.Series(0.0, index=sim_dates)
        
        for dt in sim_dates:
            date_ts = pd.Timestamp(dt)
            is_true = date_ts.date() in has_5m_dates
            ret_open_910 = r_open_910_dict.get(date_ts, {}).get(tk, 0.0) if is_true else 0.0
            
            if is_true:
                denom = 1.0 + ret_open_910
                if denom > 0.01:
                    r_entry_to_close_tk.loc[dt] = (1.0 + jp_oc.loc[dt]) / denom - 1.0
                else:
                    r_entry_to_close_tk.loc[dt] = jp_oc.loc[dt]
                r_gap_tk.loc[dt] = (1.0 + jp_gap.loc[dt]) * (1.0 + ret_open_910) - 1.0
            else:
                r_entry_to_close_tk.loc[dt] = jp_oc.loc[dt]
                r_gap_tk.loc[dt] = jp_gap.loc[dt]
                
        # Calculate rolling beta (using entry_to_close vs topix_oc as proxy since TOPIX 9:10 is not in cache)
        betas_60d[tk] = compute_rolling_beta_series(r_entry_to_close_tk, r_topix_oc, 60)
        betas_120d[tk] = compute_rolling_beta_series(r_entry_to_close_tk, r_topix_oc, 120)

    for dt in sim_dates:
        if start_date and dt < pd.to_datetime(start_date):
            continue
        if end_date and dt > pd.to_datetime(end_date):
            continue
            
        date_ts = pd.Timestamp(dt)
        is_true = date_ts.date() in has_5m_dates
        
        for tk in JP_TICKERS:
            jp_gap = df_exec.loc[dt, f"jp_gap_{tk}"]
            jp_oc = df_exec.loc[dt, f"jp_oc_{tk}"]
            ret_open_910 = r_open_910_dict.get(date_ts, {}).get(tk, 0.0) if is_true else 0.0
            
            close_to_close = (1.0 + jp_gap) * (1.0 + jp_oc) - 1.0
            
            if is_true:
                denom = 1.0 + ret_open_910
                if denom > 0.01:
                    entry_to_close = (1.0 + jp_oc) / denom - 1.0
                else:
                    entry_to_close = jp_oc
                gap_ret = (1.0 + jp_gap) * (1.0 + ret_open_910) - 1.0
                entry_price_type = "true_0910"
            else:
                entry_to_close = jp_oc
                gap_ret = jp_gap
                entry_price_type = "open"
                
            topix_cc = r_topix_cc.loc[dt]
            topix_night = r_topix_night.loc[dt]
            topix_oc = r_topix_oc.loc[dt]
            
            # topix same window proxy
            topix_same = topix_oc  # for entry-to-close, topix_oc is the fallback
            
            beta_60 = betas_60d[tk].loc[dt]
            beta_120 = betas_120d[tk].loc[dt]
            
            res_60 = entry_to_close - (beta_60 * topix_same) if pd.notna(beta_60) else np.nan
            res_120 = entry_to_close - (beta_120 * topix_same) if pd.notna(beta_120) else np.nan
            
            rows.append({
                "date": dt,
                "ticker": tk,
                "close_to_close_return": close_to_close,
                "gap_return": gap_ret,
                "open_to_close_return": jp_oc,
                "entry_to_close_return": entry_to_close,
                "entry_price_type": entry_price_type,
                "is_true_0910": is_true,
                "topix_return_same_window": topix_same,
                "beta_topix_60d": beta_60,
                "beta_topix_120d": beta_120,
                "residual_return_60d": res_60,
                "residual_return_120d": res_120,
            })
            
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Portfolio Solver for Pattern 2, 3, and 4
# ---------------------------------------------------------------------------
def solve_portfolio_weights(
    w_0: np.ndarray,
    beta_topix: np.ndarray,
    cap_bounds: np.ndarray,
    target_gross_limit: float,
    include_beta_neutral: bool = True,
) -> tuple[np.ndarray, bool]:
    """Solve optimization: minimize ||w_new - w_original||^2 with dollar/beta neutrality and box limits.

    Splits weight w_j = u_j - v_j (u_j >= 0, v_j >= 0) to make it smooth.
    """
    n = len(w_0)
    x0 = np.zeros(2 * n) # [u, v]
    
    # Objective: sum_j (u_j - v_j - w0_j)^2
    def obj_fun(x):
        diff = (x[:n] - x[n:]) - w_0
        return np.sum(diff**2)
        
    def obj_grad(x):
        diff = (x[:n] - x[n:]) - w_0
        grad = np.zeros(2 * n)
        grad[:n] = 2.0 * diff
        grad[n:] = -2.0 * diff
        return grad
        
    # Bounds: 0 <= u_j <= cap_j, 0 <= v_j <= cap_j
    bounds = []
    for cap in cap_bounds:
        bounds.append((0.0, float(cap)))
    for cap in cap_bounds:
        bounds.append((0.0, float(cap)))
        
    # Equality: sum(u_j - v_j) = 0 (dollar neutral)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:n] - x[n:])},
        # Inequality: target_gross_limit - sum(u_j + v_j) >= 0 (gross limit)
        {'type': 'ineq', 'fun': lambda x: target_gross_limit - np.sum(x)}
    ]
    
    if include_beta_neutral:
        # Equality: sum((u_j - v_j) * beta_j) = 0
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum((x[:n] - x[n:]) * beta_topix)})
        
    res = minimize(
        fun=obj_fun,
        x0=x0,
        jac=obj_grad,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 100, 'ftol': 1e-8}
    )
    
    if res.success:
        w_opt = res.x[:n] - res.x[n:]
        return w_opt, True
    else:
        return np.zeros(n), False


# ---------------------------------------------------------------------------
# 3. Liquidity and Beta Neutral Experiment Runner
# ---------------------------------------------------------------------------
def run_sprint1_backtests(
    df_exec: pd.DataFrame,
    w_ruled_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Run simulations for liquidity constraints, beta neutrality, and combinations."""
    logger.info("Starting Sprint 1 backtests and optimization experiments...")
    
    # Extract config parameters
    aum_scenarios = config.get("aum_scenarios_jpy", [100000000])
    adv_caps = config.get("adv_caps", [0.05])
    min_adv_scenarios = config.get("min_adv_jpy", [0])
    adv_windows = config.get("adv_windows", [20])
    beta_windows = config.get("beta_windows", [60])
    impact_etas = config.get("impact_eta", [0.05])
    static_costs = config.get("static_cost_bps", [15])
    target_gross = config.get("target_gross", 2.0)
    max_weight_per_name = config.get("max_abs_weight_per_name", 0.25)

    sim_dates = w_ruled_df.index
    
    # 1. Download price and volume for JP ETFs to compute ADV
    logger.info("Retrieving volume data from cache or yfinance...")
    yf_data = yf.download(
        JP_TICKERS,
        start=sim_dates.min().strftime("%Y-%m-%d"),
        end=sim_dates.max().strftime("%Y-%m-%d"),
        auto_adjust=False
    )
    volume_df = yf_data["Volume"].reindex(sim_dates).ffill()
    close_df = yf_data["Close"].reindex(sim_dates).ffill()
    
    # Daily ADV = close * volume
    adv_daily = close_df * volume_df
    
    # Intraday returns for PnL calculation
    r_intra = targets_df.pivot(index="date", columns="ticker", values="entry_to_close_return")
    r_topix_oc = df_exec["topix_oc_return"]
    
    # Load Quote width spreads if available, else static
    spread_path = "/Users/takahashimasatoshi/Library/Mobile Documents/com~apple~CloudDocs/個別株/日米ラグ_2.1/results/sector_relative_ensemble_execution_cost/quote_width_by_ticker.csv"
    if os.path.exists(spread_path):
        spread_df = pd.read_csv(spread_path)
        spread_df["trade_date"] = pd.to_datetime(spread_df["trade_date"]).dt.normalize()
        spread_df = spread_df.set_index("trade_date").reindex(sim_dates).ffill().fillna(0.0010)
    else:
        spread_df = pd.DataFrame(0.0010, index=sim_dates, columns=JP_TICKERS)

    # Precalculate rolling volatility of entry-to-close returns (shifted by 1 day)
    vol_20d = r_intra.rolling(20).std().shift(1).fillna(0.01)

    backtest_results_list = []
    
    # Loop over parameters
    for adv_w in adv_windows:
        # Precalculate rolling ADV (shifted by 1 day)
        adv_rolling = adv_daily.rolling(adv_w).mean().shift(1).replace(0.0, np.nan)
        
        for beta_w in beta_windows:
            # Extract beta series
            beta_cols = f"beta_topix_{beta_w}d"
            beta_topix_df = targets_df.pivot(index="date", columns="ticker", values=beta_cols).fillna(0.0)
            
            # Filter dates to those having valid non-NaN beta data
            valid_dates = w_ruled_df.index.intersection(beta_topix_df.dropna().index)
            if len(valid_dates) == 0:
                continue

            for aum in aum_scenarios:
                for phi in adv_caps:
                    for min_adv in min_adv_scenarios:
                        for strat in ["scale_down", "clip_by_name", "skip_illiquid"]:
                            # Optimization 1: Skip redundant runs
                            # min_adv is only used by skip_illiquid.
                            # For scale_down and clip_by_name, running for min_adv > 0 is identical to min_adv = 0.
                            if strat in ["scale_down", "clip_by_name"] and min_adv != 0:
                                continue
                                
                            config_name = f"AUM_{aum/1e6:.0f}M_phi_{phi:.2f}_strat_{strat}_advW_{adv_w}_betaW_{beta_w}_minADV_{min_adv/1e6:.0f}M"
                            logger.info("Running: %s", config_name)
                            
                            # Extract 2D matrices over all valid_dates
                            w_t_mat = w_ruled_df.loc[valid_dates].values.copy()
                            adv_t_mat = adv_rolling.loc[valid_dates].values
                            adv_t_mat = np.nan_to_num(adv_t_mat, nan=0.0)
                            
                            if strat == "skip_illiquid":
                                skipped_mask = (adv_t_mat < min_adv) | (adv_t_mat <= 0.0)
                            else:
                                skipped_mask = adv_t_mat <= 0.0
                                
                            w_t_mat[skipped_mask] = 0.0
                            num_skipped_mat = np.sum(skipped_mask, axis=1)
                            
                            # Restore neutrality if weights are changed
                            any_skipped = np.any(skipped_mask, axis=1)
                            any_w_nonzero = np.sum(np.abs(w_t_mat), axis=1) > 0.0
                            neutralize_rows = any_skipped & any_w_nonzero
                            
                            if np.any(neutralize_rows):
                                w_t_mat[neutralize_rows] = restore_dollar_neutrality_2d(w_t_mat[neutralize_rows])
                                
                            # ADV caps
                            caps = np.where(adv_t_mat > 0.0, phi * adv_t_mat / aum, 0.0)
                            caps = np.minimum(caps, max_weight_per_name)
                            
                            # Strategy checks
                            if strat in ["scale_down", "skip_illiquid"]:
                                ratios = np.where(caps > 0.0, np.abs(w_t_mat) / caps, np.where(w_t_mat != 0.0, np.inf, 0.0))
                                max_ratio = np.max(ratios, axis=1, keepdims=True)
                                w_opt_mat = np.where(max_ratio > 1.0, w_t_mat / max_ratio, w_t_mat)
                            elif strat == "clip_by_name":
                                w_clipped = np.sign(w_t_mat) * np.minimum(np.abs(w_t_mat), caps)
                                w_opt_mat = restore_dollar_neutrality_2d(w_clipped)
                            
                            # Zero out completely inactive rows
                            inactive_rows = np.sum(np.abs(w_t_mat), axis=1) == 0.0
                            w_opt_mat[inactive_rows] = 0.0
                            
                            # Calculate ADV trade statistics
                            trade_val = np.abs(w_opt_mat) * aum
                            trade_adv_ratios = np.where(adv_t_mat > 0.0, trade_val / adv_t_mat, 0.0)
                            max_trade_adv_mat = np.max(trade_adv_ratios, axis=1)
                            avg_trade_adv_mat = np.mean(trade_adv_ratios, axis=1)
                            
                            w_opt_df = pd.DataFrame(w_opt_mat, index=valid_dates, columns=JP_TICKERS)
                            strat_returns = w_opt_df.multiply(r_intra.loc[valid_dates]).sum(axis=1)
                            
                            w_diff = w_opt_df.diff().abs()
                            w_diff.iloc[0] = w_opt_df.iloc[0].abs()
                            turnover_daily = w_diff.sum(axis=1) / 2.0
                            
                            # Precalculate PnL stats vectorized
                            long_pnl_vec = np.sum(np.maximum(w_opt_mat, 0.0) * r_intra.loc[valid_dates].values, axis=1)
                            short_pnl_vec = np.sum(np.minimum(w_opt_mat, 0.0) * r_intra.loc[valid_dates].values, axis=1)
                            gross_exp_vec = np.sum(np.abs(w_opt_mat), axis=1)
                            net_exp_vec = np.sum(w_opt_mat, axis=1)
                            
                            # Optimization 2: Move cost loops out of the daily weight capping loop!
                            for eta in impact_etas:
                                w_vals = w_opt_df.abs().values
                                adv_vals = adv_rolling.loc[valid_dates].values
                                adv_vals = np.nan_to_num(adv_vals, nan=0.0)
                                vol_vals = vol_20d.loc[valid_dates].values
                                trade_vals = w_vals * aum
                                
                                mask = (adv_vals > 0.0) & (w_vals > 0.0)
                                impact_cost_matrix = np.zeros_like(w_vals)
                                impact_cost_matrix[mask] = eta * vol_vals[mask] * np.sqrt(trade_vals[mask] / adv_vals[mask]) * w_vals[mask]
                                            
                                daily_impact_cost = np.sum(impact_cost_matrix, axis=1)
                                daily_spread_cost = np.sum(0.5 * spread_df.loc[valid_dates].values * w_diff.values, axis=1)
                                combined_costs = daily_spread_cost + daily_impact_cost
                                net_ret_combined = strat_returns - combined_costs
                                
                                for c_bps in static_costs:
                                    cost_series = turnover_daily * c_bps / 10000.0
                                    net_ret = strat_returns - cost_series
                                    
                                    # Collect daily records
                                    for idx, dt in enumerate(valid_dates):
                                        backtest_results_list.append({
                                            "date": dt,
                                            "AUM": aum,
                                            "phi": phi,
                                            "strategy": strat,
                                            "adv_window": adv_w,
                                            "beta_window": beta_w,
                                            "min_adv": min_adv,
                                            "eta": eta,
                                            "static_cost_bps": c_bps,
                                            "strategy_return": strat_returns.iloc[idx],
                                            "net_return_after_cost": net_ret.iloc[idx],
                                            "net_return_after_combined_cost": net_ret_combined.iloc[idx],
                                            "realized_gross_exposure": gross_exp_vec[idx],
                                            "realized_net_exposure": net_exp_vec[idx],
                                            "avg_trade_adv": avg_trade_adv_mat[idx],
                                            "max_trade_adv": max_trade_adv_mat[idx],
                                            "is_constrained": (max_trade_adv_mat[idx] > phi),
                                            "num_skipped_tickers": num_skipped_mat[idx],
                                            "turnover": turnover_daily.iloc[idx],
                                            "long_leg_pnl": long_pnl_vec[idx],
                                            "short_leg_pnl": short_pnl_vec[idx],
                                            "optimization_failed": False
                                        })
                                        
            # 4. Beta neutral optimization comparison
            # Only run SLSQP for the reference window (adv_w=20, beta_w=60) to avoid ~14min of extra compute.
            if adv_w != 20 or beta_w != 60:
                continue
                
            # Optimization 3: Solve Pattern 2 (beta_neutral_only) once per beta_w, since it doesn't depend on AUM or phi
            w_opt_2_list = []
            opt_failed_2_list = []
            for dt in valid_dates:
                w_0 = w_ruled_df.loc[dt].values.copy()
                beta_topix = beta_topix_df.loc[dt].values
                cap_bounds_2 = np.full(17, max_weight_per_name)
                w_opt_2, success_2 = run_solver_with_fallbacks(w_0, beta_topix, cap_bounds_2, target_gross, include_beta_neutral=True)
                w_opt_2_list.append(w_opt_2)
                opt_failed_2_list.append(not success_2)
                
            # Run SLSQP solver for Pattern 3 and 4 across AUM and phi
            for aum in aum_scenarios:
                for phi in adv_caps:
                    # We only need the optimization patterns for the reference scenario in our charts and reports
                    # (AUM = 100M, phi = 0.05) to avoid thousands of slow SLSQP runs.
                    if aum != 100000000 or not np.isclose(phi, 0.05):
                        continue
                        
                    config_name_opt = f"Opt_AUM_{aum/1e6:.0f}M_phi_{phi:.2f}"
                    logger.info("Running Optimization Patterns for %s", config_name_opt)
                    
                    w_patterns = {
                        "current": [],
                        "beta_neutral_only": w_opt_2_list,
                        "liquidity_constrained_only": [],
                        "full_practical": []
                    }
                    opt_failed = {
                        "beta_neutral_only": opt_failed_2_list,
                        "liquidity_constrained_only": [],
                        "full_practical": []
                    }
                    
                    for dt in valid_dates:
                        w_0 = w_ruled_df.loc[dt].values.copy()
                        beta_topix = beta_topix_df.loc[dt].values
                        adv_t = adv_rolling.loc[dt].values
                        adv_t = np.nan_to_num(adv_t, nan=0.0)
                        
                        # Set missing/zero ADV to cap = 0
                        caps_full = np.zeros(17)
                        for j in range(17):
                            if adv_t[j] > 0:
                                caps_full[j] = phi * adv_t[j] / aum
                            else:
                                caps_full[j] = 0.0
                        caps_full = np.minimum(caps_full, max_weight_per_name)
                        
                        # Pattern 1: current
                        w_patterns["current"].append(w_0)
                        
                        # Pattern 3: liquidity_constrained_only (no beta neutrality)
                        w_opt_3, success_3 = run_solver_with_fallbacks(w_0, beta_topix, caps_full, target_gross, include_beta_neutral=False)
                        w_patterns["liquidity_constrained_only"].append(w_opt_3)
                        opt_failed["liquidity_constrained_only"].append(not success_3)
                        
                        # Pattern 4: full_practical (both)
                        w_opt_4, success_4 = run_solver_with_fallbacks(w_0, beta_topix, caps_full, target_gross, include_beta_neutral=True)
                        w_patterns["full_practical"].append(w_opt_4)
                        opt_failed["full_practical"].append(not success_4)
                        
                    # Save results of the patterns
                    for pattern, w_list in w_patterns.items():
                        w_df = pd.DataFrame(w_list, index=valid_dates, columns=JP_TICKERS)
                        strat_returns = w_df.multiply(r_intra.loc[valid_dates]).sum(axis=1)
                        w_diff = w_df.diff().abs()
                        w_diff.iloc[0] = w_df.iloc[0].abs()
                        turnover_daily = w_diff.sum(axis=1) / 2.0
                        
                        failed_list = opt_failed.get(pattern, [False] * len(valid_dates))
                        
                        # Save under AUM, phi, but label the strategy as the pattern name
                        for idx, dt in enumerate(valid_dates):
                            cost_bps_default = 15
                            cost_series = turnover_daily * cost_bps_default / 10000.0
                            net_ret = strat_returns - cost_series
                            
                            backtest_results_list.append({
                                "date": dt,
                                "AUM": aum,
                                "phi": phi,
                                "strategy": f"pattern_{pattern}",
                                "adv_window": adv_w,
                                "beta_window": beta_w,
                                "min_adv": 0.0,
                                "eta": 0.05,
                                "static_cost_bps": cost_bps_default,
                                "strategy_return": strat_returns.iloc[idx],
                                "net_return_after_cost": net_ret.iloc[idx],
                                "net_return_after_combined_cost": net_ret.iloc[idx],
                                "realized_gross_exposure": np.sum(np.abs(w_df.iloc[idx])),
                                "realized_net_exposure": np.sum(w_df.iloc[idx]),
                                "avg_trade_adv": 0.0,
                                "max_trade_adv": 0.0,
                                "is_constrained": False,
                                "num_skipped_tickers": 0,
                                "turnover": turnover_daily.iloc[idx],
                                "long_leg_pnl": np.sum(np.maximum(w_df.iloc[idx], 0.0) * r_intra.loc[dt]),
                                "short_leg_pnl": np.sum(np.minimum(w_df.iloc[idx], 0.0) * r_intra.loc[dt]),
                                "optimization_failed": failed_list[idx]
                            })

    return pd.DataFrame(backtest_results_list)


# ---------------------------------------------------------------------------
# Solver helper with iterative scaling fallbacks
# ---------------------------------------------------------------------------
def run_solver_with_fallbacks(
    w_0: np.ndarray,
    beta_topix: np.ndarray,
    cap_bounds: np.ndarray,
    target_gross: float,
    include_beta_neutral: bool
) -> tuple[np.ndarray, bool]:
    """Iteratively solve portfolio weights, scaling down target gross if SLSQP fails."""
    for scale in [1.0, 0.8, 0.6, 0.4, 0.2]:
        w_opt, success = solve_portfolio_weights(
            w_0=w_0,
            beta_topix=beta_topix,
            cap_bounds=cap_bounds,
            target_gross_limit=target_gross * scale,
            include_beta_neutral=include_beta_neutral
        )
        if success:
            return w_opt, True
            
    # Stop trading fallback
    return np.zeros(len(w_0)), False


# ---------------------------------------------------------------------------
# Helper: Restore Dollar Neutrality
# ---------------------------------------------------------------------------
def restore_dollar_neutrality(weights: np.ndarray) -> np.ndarray:
    """Restores dollar neutrality using a box-bound safe scaling of the larger leg."""
    w = weights.copy()
    long_mask = w > 0.0
    short_mask = w < 0.0
    
    L = np.sum(w[long_mask])
    S = np.sum(np.abs(w[short_mask]))
    
    if L == 0.0 or S == 0.0:
        return np.zeros(len(w))
        
    if L > S:
        w[long_mask] = w[long_mask] * (S / L)
    elif S > L:
        w[short_mask] = w[short_mask] * (L / S)
        
    return w


def restore_dollar_neutrality_2d(weights: np.ndarray) -> np.ndarray:
    """Restores dollar neutrality for a 2D array of weights (shape N x 17) using a box-bound safe scaling of the larger leg."""
    w = weights.copy()
    long_mask = w > 0.0
    short_mask = w < 0.0
    
    L = np.sum(np.where(long_mask, w, 0.0), axis=1, keepdims=True)
    S = np.sum(np.where(short_mask, np.abs(w), 0.0), axis=1, keepdims=True)
    
    L_safe = np.where(L == 0.0, 1e-8, L)
    S_safe = np.where(S == 0.0, 1e-8, S)
    
    scale_long = S / L_safe
    scale_short = L / S_safe
    
    # Apply scaling where L > S
    w = np.where((L > S) & long_mask, w * scale_long, w)
    # Apply scaling where S > L
    w = np.where((S > L) & short_mask, w * scale_short, w)
    
    # Zero out rows where either leg is zero
    zero_rows = (L == 0.0) | (S == 0.0)
    w[zero_rows.squeeze(axis=1)] = 0.0
    
    return w


# ---------------------------------------------------------------------------
# 4. RuleD Rolling Calibration Validation
# ---------------------------------------------------------------------------
def run_ruled_rolling_calibration(
    df_exec: pd.DataFrame,
    w_ruled_df: pd.DataFrame,
    valid_dates_beta: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Re-implements calibration splits using full-sample, rolling 252d, and expanding quantiles."""
    logger.info("Running RuleD rolling calibration check...")
    
    diag_file = find_latest_distribution_diagnostics()
    if not diag_file:
        logger.warning("Diagnostics csv not found. Cannot run calibration leak check.")
        return pd.DataFrame()
        
    diag_df = pd.read_csv(diag_file)
    diag_df["trade_date"] = pd.to_datetime(diag_df["trade_date"]).dt.normalize()
    diag_df = diag_df.set_index("trade_date")
    
    # Extract ex-ante IR values
    ex_ante_ir = pd.Series(np.nan, index=valid_dates_beta)
    for dt in valid_dates_beta:
        if dt in diag_df.index:
            ir_val = diag_df.loc[dt, "pred_ir_gap_exante_cost"]
            if isinstance(ir_val, pd.Series):
                ir_val = ir_val.iloc[0]
            ex_ante_ir[dt] = float(ir_val)
            
    strategy_returns = w_ruled_df.multiply(df_exec["jp_oc_1617.T"].reindex(valid_dates_beta).fillna(0.0), axis=0).sum(axis=1) # dummy fallback
    # Wait, we want the actual baseline strategy return (before RuleD scaling, i.e., w_baseline * r_intraday)
    # Let's re-run base calculation to get the true daily portfolio returns.
    # To keep this clean, let's load strategy returns from the diagnostics run or calculate it directly.
    # Let's use the actual strategy_return before RuleD scaling:
    # return is strategy_return = sum(w_baseline * r_intraday)
    # Wait, w_ruled_df is scaled by multipliers. We can un-scale it by dividing by multipliers if multipliers > 0,
    # or just load the baseline returns from df_exec or other components.
    # Let's check how the strategy return was calculated in sprint0.py:
    # strategy_returns = w_ruled_df.multiply(r_intraday.loc[valid_dates_beta]).sum(axis=1) / multipliers
    # To be extremely clean, let's calculate:
    # PnL = strategy_returns
    
    # Load 5m target returns to match intraday return profile
    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    r_oc = df_exec[jp_oc_cols].copy()
    r_oc.columns = JP_TICKERS
    r_intra = compute_jp_target_returns(df_exec, JP_TICKERS)
    r_intra_df = pd.DataFrame(r_intra, index=df_exec.index, columns=JP_TICKERS)
    
    # We un-scale w_ruled_df to get w_baseline_df
    # In find_latest_distribution_diagnostics, we have the multipliers
    multipliers = pd.Series(1.0, index=valid_dates_beta)
    for dt in valid_dates_beta:
        if dt in diag_df.index:
            gross_exp = diag_df.loc[dt, "gross_exposure"]
            if isinstance(gross_exp, pd.Series):
                gross_exp = gross_exp.iloc[0]
            multipliers[dt] = float(gross_exp) / 2.0
            
    # Daily baseline weight
    w_baseline_df = w_ruled_df.divide(multipliers, axis=0).fillna(0.0)
    realized_returns = w_baseline_df.multiply(r_intra_df.loc[valid_dates_beta]).sum(axis=1)
    
    calib_df = pd.DataFrame({
        "ex_ante_ir": ex_ante_ir,
        "realized_return": realized_returns,
        "multiplier": multipliers
    }, index=valid_dates_beta).dropna()
    
    n_days = len(calib_df)
    if n_days == 0:
        return pd.DataFrame()
        
    # 1. Full sample quantile split
    calib_df["full_sample_bin"] = pd.qcut(calib_df["ex_ante_ir"], 3, labels=["Low", "Medium", "High"])
    
    # 2. Rolling 252d quantile split
    rolling_bins = []
    # 3. Expanding quantile split
    expanding_bins = []
    
    for idx, dt in enumerate(calib_df.index):
        # Rolling 252d
        if idx < 252:
            rolling_bins.append("Medium")
        else:
            window_dates = calib_df.index[idx-252:idx]
            window_ir = calib_df.loc[window_dates, "ex_ante_ir"]
            q33 = window_ir.quantile(0.333)
            q66 = window_ir.quantile(0.667)
            val = calib_df.loc[dt, "ex_ante_ir"]
            if val <= q33:
                rolling_bins.append("Low")
            elif val <= q66:
                rolling_bins.append("Medium")
            else:
                rolling_bins.append("High")
                
        # Expanding
        if idx < 60:
            expanding_bins.append("Medium")
        else:
            window_dates = calib_df.index[:idx]
            window_ir = calib_df.loc[window_dates, "ex_ante_ir"]
            q33 = window_ir.quantile(0.333)
            q66 = window_ir.quantile(0.667)
            val = calib_df.loc[dt, "ex_ante_ir"]
            if val <= q33:
                expanding_bins.append("Low")
            elif val <= q66:
                expanding_bins.append("Medium")
            else:
                expanding_bins.append("High")
                
    calib_df["rolling_252_bin"] = rolling_bins
    calib_df["expanding_bin"] = expanding_bins
    
    # Calculate PnL with multiplier
    # Low: 0.5, Medium: 1.0, High: 1.5
    bin_mult = {"Low": 0.5, "Medium": 1.0, "High": 1.5}
    
    calib_df["pnl_multiplier_full"] = calib_df.apply(
        lambda r: r["realized_return"] * bin_mult[r["full_sample_bin"]], axis=1
    )
    calib_df["pnl_multiplier_rolling"] = calib_df.apply(
        lambda r: r["realized_return"] * bin_mult[r["rolling_252_bin"]], axis=1
    )
    calib_df["pnl_multiplier_expanding"] = calib_df.apply(
        lambda r: r["realized_return"] * bin_mult[r["expanding_bin"]], axis=1
    )
    
    return calib_df
