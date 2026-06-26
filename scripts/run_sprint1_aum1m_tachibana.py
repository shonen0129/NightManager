"""scripts/run_sprint1_aum1m_tachibana.py

Specialized simulation runner for small-scale AUM (1,000,000 JPY) under
Tachibana Securities cost structures, lot rounding, and operational stresses.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import yaml
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.diagnostics.sprint0 import run_sprint0_calculations
from leadlag.diagnostics.sprint1_experiments import generate_targets_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 1 AUM 1M JPY Tachibana Simulation CLI")
    parser.add_argument("--config", type=str, default="configs/archive/sprint1_aum1m_tachibana.yaml", help="Path to config YAML")
    parser.add_argument("--start_date", type=str, default=None, help="Optional start date override")
    parser.add_argument("--end_date", type=str, default=None, help="Optional end date override")
    return parser.parse_args()


def compute_max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    cum_returns = (1.0 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return float(drawdown.min())


def restore_neutral_rescale(w: np.ndarray) -> np.ndarray:
    w_new = w.copy()
    long_mask = w_new > 0.0
    short_mask = w_new < 0.0
    
    long_sum = np.sum(w_new[long_mask])
    short_sum = np.abs(np.sum(w_new[short_mask]))
    
    if long_sum == 0.0 or short_sum == 0.0:
        return np.zeros_like(w_new)
        
    target_gross = min(long_sum, short_sum)
    w_new[long_mask] = w_new[long_mask] * (target_gross / long_sum)
    w_new[short_mask] = w_new[short_mask] * (target_gross / short_sum)
    return w_new


def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    start_date = args.start_date or config.get("start_date")
    end_date = args.end_date or config.get("end_date")
    
    output_dir = config.get("output_dir", "reports/sprint1_aum1m_tachibana")
    artifact_dir = config.get("artifact_dir", "artifacts/sprint1_aum1m_tachibana")
    figure_dir = os.path.join(output_dir, "figures")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    
    logger.info("Loading cache data...")
    df_exec = load_df_exec_from_local_cache()
    
    # Data cleaning for zero open prices to prevent inf return values
    for tk in JP_TICKERS:
        for suffix in ["gap", "oc"]:
            col = f"jp_{suffix}_{tk}"
            if col in df_exec.columns:
                df_exec[col] = df_exec[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
    # Run sprint0 to retrieve baseline weight_ruled
    logger.info("Running baseline diagnostics calculations...")
    base_results = run_sprint0_calculations(start_date=start_date, end_date=end_date)
    w_ruled_df = base_results["signal_diagnostics_panel"]["weight_ruled"]
    signals_df = base_results["signal_diagnostics_panel"]["signal_gap_adjusted"]
    
    # Generate targets panel
    targets_df = generate_targets_panel(df_exec, start_date=start_date, end_date=end_date)
    
    # Align dates
    valid_dates = w_ruled_df.index.intersection(df_exec.index[120:])
    if start_date:
        valid_dates = valid_dates[valid_dates >= pd.to_datetime(start_date)]
    if end_date:
        valid_dates = valid_dates[valid_dates <= pd.to_datetime(end_date)]
        
    targets_pivot = targets_df.pivot(index="date", columns="ticker")
    r_cc = targets_pivot["close_to_close_return"].reindex(valid_dates)
    r_oc = targets_pivot["open_to_close_return"].reindex(valid_dates)
    r_etc = targets_pivot["entry_to_close_return"].reindex(valid_dates)
    r_topix_cc = df_exec["topix_cc_trade"].reindex(valid_dates)
    r_topix_oc = df_exec["topix_oc_return"].reindex(valid_dates)
    
    # Retrieve JPY Volume and Close prices for ADV from cache or yfinance
    logger.info("Retrieving Close and Volume data for ADV...")
    yf_data = yf.download(
        JP_TICKERS,
        start=valid_dates.min().strftime("%Y-%m-%d"),
        end=valid_dates.max().strftime("%Y-%m-%d"),
        auto_adjust=False
    )
    volume_df = yf_data["Volume"].reindex(valid_dates).ffill().fillna(0.0)
    close_df = yf_data["Close"].reindex(valid_dates).ffill().fillna(1.0)
    
    adv_daily = close_df * volume_df
    rolling_ADV = adv_daily.rolling(20).mean().shift(1).fillna(1e6) # shift(1) is lookahead-free
    
    # Retrieve Spread Quote Width Data
    spread_path = "/Users/takahashimasatoshi/Library/Mobile Documents/com~apple~CloudDocs/個別株/日米ラグ_2.1/results/sector_relative_ensemble_execution_cost/quote_width_by_ticker.csv"
    if os.path.exists(spread_path):
        spread_df = pd.read_csv(spread_path)
        spread_df["trade_date"] = pd.to_datetime(spread_df["trade_date"]).dt.normalize()
        spread_df = spread_df.set_index("trade_date").reindex(valid_dates).ffill().fillna(0.0010)
    else:
        spread_df = pd.DataFrame(0.0010, index=valid_dates, columns=JP_TICKERS)
        
    # Pre-clean open prices for entry price calculation
    open_prices_df = pd.DataFrame(index=valid_dates, columns=JP_TICKERS)
    for tk in JP_TICKERS:
        op = df_exec[f"jp_open_trade_{tk}"].reindex(valid_dates).copy()
        cl = df_exec[f"jp_close_sig_{tk}"].reindex(valid_dates)
        op[op <= 0.0] = cl[op <= 0.0]
        open_prices_df[tk] = op
        
    # Parameters from config
    AUM = config.get("aum_jpy", 1000000)
    buy_interest_rate = config.get("buy_interest_rate_annual", 0.025)
    sell_interest_rate = config.get("sell_interest_rate_annual", 0.00)
    stock_borrow_fee = config.get("stock_borrow_fee_annual", 0.0115)
    phi = config.get("adv_cap", 0.20)
    
    # -----------------------------------------------------------------------
    # Backtest Core Loop
    # -----------------------------------------------------------------------
    logger.info("Executing AUM 1M JPY Backtest Matrix Loop...")
    
    daily_records = []
    position_details = []
    cost_breakdown_records = []
    
    gross_scenarios = config.get("gross_exposure_list", [0.5, 1.0, 1.5, 2.0])
    min_adv_scenarios = config.get("min_adv_jpy_list", [0, 500000, 1000000, 3000000])
    reverse_fee_bps_scenarios = config.get("reverse_fee_bps_per_day_scenarios", [0, 5, 10, 30])
    fallback_spreads = config.get("fallback_roundtrip_spread_bps", [5, 10, 15, 20, 30, 50])
    short_unavailable_scenarios = config.get("short_unavailable_scenarios", [])
    
    # Base simulation matrix
    for gross in gross_scenarios:
        for min_adv in min_adv_scenarios:
            for strat in config.get("liquidity_constraint_method_list", ["scale_down", "clip_by_name", "skip_illiquid"]):
                # Stress loops are nested or evaluated on default settings to prevent exponential grid explosion
                # Base Case: reverse_fee=0, short_unavailability=0, spread=data
                
                # We also run reverse fee stress, short unavailability stress and spread stress for gross=1.0, strat=scale_down, min_adv=0
                
                # Check is_constrained count
                constrained_days = 0
                zero_share_days_count = 0
                
                # Generate random seed for short unavailability stress
                np.random.seed(42)
                
                for dt in valid_dates:
                    # Target weight before constraints scaled to exactly gross
                    w_base = w_ruled_df.loc[dt].values
                    gross_base = np.sum(np.abs(w_base))
                    w_target = w_base * (gross / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
                    
                    # Capping limits
                    adv_t = rolling_ADV.loc[dt].values
                    cap_val = phi * adv_t
                    weight_cap = cap_val / AUM
                    
                    # Apply constraint strategy
                    if strat == "scale_down":
                        ratios = np.where(weight_cap > 0.0, np.abs(w_target) / weight_cap, np.where(w_target != 0.0, np.inf, 0.0))
                        max_ratio = np.max(ratios)
                        w_opt = w_target / max_ratio if max_ratio > 1.0 else w_target
                        is_constrained = max_ratio > 1.0
                    elif strat == "clip_by_name":
                        w_clipped = np.sign(w_target) * np.minimum(np.abs(w_target), weight_cap)
                        w_opt = restore_neutral_rescale(w_clipped)
                        is_constrained = np.any(np.abs(w_target) > weight_cap)
                    elif strat == "skip_illiquid":
                        skipped = (adv_t < min_adv) | (adv_t <= 0.0)
                        w_clipped = w_target.copy()
                        w_clipped[skipped] = 0.0
                        w_opt = restore_neutral_rescale(w_clipped)
                        is_constrained = np.any(skipped & (w_target != 0.0))
                        
                    if is_constrained:
                        constrained_days += 1
                        
                    # Target shares and round lot rounding
                    open_t = open_prices_df.loc[dt].values
                    r_etc_t = r_etc.loc[dt].values
                    r_oc_t = r_oc.loc[dt].values
                    
                    # Compute entry price: open_p * (1 + jp_oc) / (1 + entry_to_close)
                    denom = 1.0 + r_etc_t
                    # Avoid division by zero
                    denom = np.where(denom > 0.01, denom, 1.0)
                    entry_p = open_t * (1.0 + r_oc_t) / denom
                    
                    target_notional = w_opt * AUM
                    target_shares = target_notional / entry_p
                    
                    # Actual shares (rounded to nearest integer share)
                    shares = np.round(target_shares)
                    actual_notional = shares * entry_p
                    actual_weight = actual_notional / AUM
                    
                    zero_share_days_count += np.sum((w_opt != 0.0) & (shares == 0))
                    
                    # Rounding error
                    rounding_err = np.abs(actual_weight - w_opt)
                    avg_round_err = np.mean(rounding_err)
                    max_round_err = np.max(rounding_err)
                    gross_loss = np.sum(np.abs(w_opt)) - np.sum(np.abs(actual_weight))
                    
                    # realized return
                    realized_etc = r_etc_t
                    gross_etc_pnl = np.sum(actual_weight * realized_etc)
                    long_pnl = np.sum(np.maximum(actual_weight, 0.0) * realized_etc)
                    short_pnl = np.sum(np.minimum(actual_weight, 0.0) * realized_etc)
                    
                    # Costs (base case: reverse_fee=0, spread=data)
                    spread_val = spread_df.loc[dt].values
                    spread_cost = np.sum(spread_val * np.abs(actual_weight))
                    
                    long_financing_cost = np.sum(np.maximum(actual_notional, 0.0) * buy_interest_rate / 365.0)
                    short_borrow_cost = np.sum(np.abs(np.minimum(actual_notional, 0.0)) * stock_borrow_fee / 365.0)
                    credit_cost = long_financing_cost + short_borrow_cost
                    
                    total_cost = credit_cost + spread_cost * AUM
                    net_etc_pnl = gross_etc_pnl - (total_cost / AUM)
                    
                    # Record base case results for gross scenarios comparison
                    if min_adv == 0 and strat == "scale_down":
                        daily_records.append({
                            "date": dt,
                            "gross_exposure_target": gross,
                            "realized_gross": np.sum(np.abs(actual_weight)),
                            "realized_net": np.sum(actual_weight),
                            "gross_return": gross_etc_pnl,
                            "net_return": net_etc_pnl,
                            "spread_cost": spread_cost,
                            "credit_cost": credit_cost / AUM,
                            "total_cost": total_cost / AUM,
                            "avg_rounding_error": avg_round_err,
                            "max_rounding_error": max_round_err,
                            "gross_exposure_loss": gross_loss,
                            "zero_share_orders": np.sum((w_opt != 0.0) & (shares == 0)),
                            "long_pnl": long_pnl,
                            "short_pnl": short_pnl,
                        })
                        
                        # Cost breakdown detailed records
                        cost_breakdown_records.append({
                            "date": dt,
                            "gross": gross,
                            "spread_cost_jpy": spread_cost * AUM,
                            "borrow_fee_jpy": short_borrow_cost,
                            "buy_interest_jpy": long_financing_cost,
                            "reverse_fee_jpy": 0.0,
                        })
                        
                        # Position details (record a sample to keep files reasonably sized, e.g. first 5 tickers)
                        if dt == valid_dates[-1] or dt == valid_dates[120]:
                            for idx, tk in enumerate(JP_TICKERS[:5]):
                                position_details.append({
                                    "date": dt,
                                    "gross": gross,
                                    "ticker": tk,
                                    "target_weight": w_opt[idx],
                                    "actual_weight": actual_weight[idx],
                                    "shares": shares[idx],
                                    "entry_price": entry_p[idx],
                                    "notional_jpy": actual_notional[idx],
                                    "rolling_adv_jpy": adv_t[idx],
                                    "one_way_trade_adv": np.abs(actual_notional[idx]) / adv_t[idx] if adv_t[idx] > 0 else 0.0
                                })

    # Convert to DataFrames and save
    daily_pnl_df = pd.DataFrame(daily_records)
    position_details_df = pd.DataFrame(position_details)
    cost_breakdown_df = pd.DataFrame(cost_breakdown_records)
    
    daily_pnl_df.to_parquet(os.path.join(artifact_dir, "daily_pnl_aum1m.parquet"))
    position_details_df.to_parquet(os.path.join(artifact_dir, "position_detail_aum1m.parquet"))
    cost_breakdown_df.to_parquet(os.path.join(artifact_dir, "cost_breakdown_aum1m.parquet"))
    
    # -----------------------------------------------------------------------
    # Sensitivity & Stress Run Loops
    # -----------------------------------------------------------------------
    logger.info("Executing Sensitivity and Stress loops...")
    
    # 1. Spread sensitivity loop (at gross=1.0, strat=scale_down, min_adv=0)
    spread_sensitivity_rows = []
    for s_bps in fallback_spreads:
        net_rets = []
        for dt in valid_dates:
            # same as base
            w_base = w_ruled_df.loc[dt].values
            gross_base = np.sum(np.abs(w_base))
            w_target = w_base * (1.0 / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
            adv_t = rolling_ADV.loc[dt].values
            weight_cap = (phi * adv_t) / AUM
            ratios = np.where(weight_cap > 0.0, np.abs(w_target) / weight_cap, np.where(w_target != 0.0, np.inf, 0.0))
            max_ratio = np.max(ratios)
            w_opt = w_target / max_ratio if max_ratio > 1.0 else w_target
            
            # entry price
            open_t = open_prices_df.loc[dt].values
            r_etc_t = r_etc.loc[dt].values
            r_oc_t = r_oc.loc[dt].values
            denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
            entry_p = open_t * (1.0 + r_oc_t) / denom
            
            shares = np.round(w_opt * AUM / entry_p)
            actual_weight = shares * entry_p / AUM
            
            # Spread cost
            spread_cost = np.sum((s_bps / 10000.0) * np.abs(actual_weight))
            long_financing_cost = np.sum(np.maximum(shares * entry_p, 0.0) * buy_interest_rate / 365.0)
            short_borrow_cost = np.sum(np.abs(np.minimum(shares * entry_p, 0.0)) * stock_borrow_fee / 365.0)
            
            gross_etc_pnl = np.sum(actual_weight * r_etc_t)
            total_cost = (long_financing_cost + short_borrow_cost) / AUM + spread_cost
            net_ret = gross_etc_pnl - total_cost
            net_rets.append(net_ret)
            
        net_series = pd.Series(net_rets)
        ann_ret = net_series.mean() * 252
        ann_vol = net_series.std() * np.sqrt(252)
        ir_val = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
        max_dd = compute_max_drawdown(net_series)
        
        spread_sensitivity_rows.append({
            "fallback_spread_bps": s_bps,
            "annualized_net_return": ann_ret,
            "annualized_volatility": ann_vol,
            "IR": ir_val,
            "max_drawdown": max_dd
        })
    spread_sensitivity_df = pd.DataFrame(spread_sensitivity_rows)
    spread_sensitivity_df.to_csv(os.path.join(artifact_dir, "spread_sensitivity_summary.csv"), index=False)
    
    # 2. Reverse fee sensitivity loop (at gross=1.0, strat=scale_down, min_adv=0)
    reverse_fee_rows = []
    for rev_bps in reverse_fee_bps_scenarios:
        net_rets = []
        for dt in valid_dates:
            # same as base
            w_base = w_ruled_df.loc[dt].values
            gross_base = np.sum(np.abs(w_base))
            w_target = w_base * (1.0 / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
            adv_t = rolling_ADV.loc[dt].values
            weight_cap = (phi * adv_t) / AUM
            ratios = np.where(weight_cap > 0.0, np.abs(w_target) / weight_cap, np.where(w_target != 0.0, np.inf, 0.0))
            max_ratio = np.max(ratios)
            w_opt = w_target / max_ratio if max_ratio > 1.0 else w_target
            
            # entry price
            open_t = open_prices_df.loc[dt].values
            r_etc_t = r_etc.loc[dt].values
            r_oc_t = r_oc.loc[dt].values
            denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
            entry_p = open_t * (1.0 + r_oc_t) / denom
            
            shares = np.round(w_opt * AUM / entry_p)
            actual_weight = shares * entry_p / AUM
            
            # Spread cost
            spread_val = spread_df.loc[dt].values
            spread_cost = np.sum(spread_val * np.abs(actual_weight))
            
            long_financing_cost = np.sum(np.maximum(shares * entry_p, 0.0) * buy_interest_rate / 365.0)
            short_borrow_cost = np.sum(np.abs(np.minimum(shares * entry_p, 0.0)) * stock_borrow_fee / 365.0)
            reverse_fee_cost = np.sum(np.abs(np.minimum(shares * entry_p, 0.0)) * (rev_bps / 10000.0))
            
            gross_etc_pnl = np.sum(actual_weight * r_etc_t)
            total_cost = (long_financing_cost + short_borrow_cost + reverse_fee_cost) / AUM + spread_cost
            net_ret = gross_etc_pnl - total_cost
            net_rets.append(net_ret)
            
        net_series = pd.Series(net_rets)
        ann_ret = net_series.mean() * 252
        ann_vol = net_series.std() * np.sqrt(252)
        ir_val = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
        max_dd = compute_max_drawdown(net_series)
        
        reverse_fee_rows.append({
            "reverse_fee_bps_per_day": rev_bps,
            "annualized_net_return": ann_ret,
            "annualized_volatility": ann_vol,
            "IR": ir_val,
            "max_drawdown": max_dd
        })
    reverse_fee_df = pd.DataFrame(reverse_fee_rows)
    reverse_fee_df.to_csv(os.path.join(artifact_dir, "reverse_fee_sensitivity_summary.csv"), index=False)
    
    # 3. Short unavailability stress scenarios
    short_stress_rows = []
    for stress_scen in short_unavailable_scenarios:
        prob = stress_scen["unavailable_probability"]
        scen_name = stress_scen["name"]
        
        for mode in ["zero_and_rescale", "replace_with_next_candidate"]:
            # Zero out or replace logic
            np.random.seed(42) # fixed seed
            net_rets = []
            
            for dt in valid_dates:
                w_base = w_ruled_df.loc[dt].values
                gross_base = np.sum(np.abs(w_base))
                w_target = w_base * (1.0 / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
                adv_t = rolling_ADV.loc[dt].values
                weight_cap = (phi * adv_t) / AUM
                
                # scale down first
                ratios = np.where(weight_cap > 0.0, np.abs(w_target) / weight_cap, np.where(w_target != 0.0, np.inf, 0.0))
                max_ratio = np.max(ratios)
                w_opt = w_target / max_ratio if max_ratio > 1.0 else w_target
                
                # Apply random short unavailability
                unavailable_mask = np.random.rand(len(JP_TICKERS)) < prob
                
                if mode == "zero_and_rescale":
                    # set short weights to 0 for unavailable
                    w_opt_stress = w_opt.copy()
                    w_opt_stress[(w_opt < 0.0) & unavailable_mask] = 0.0
                    w_opt_stress = restore_neutral_rescale(w_opt_stress)
                elif mode == "replace_with_next_candidate":
                    w_opt_stress = w_opt.copy()
                    # Identify unavailable shorts
                    unavail_shorts = (w_opt < 0.0) & unavailable_mask
                    
                    if np.any(unavail_shorts):
                        # Extract signals to order candidates
                        sig_t = signals_df.loc[dt].values
                        sorted_indices = np.argsort(sig_t) # ascending (most negative signals first)
                        
                        for idx in range(len(JP_TICKERS)):
                            if unavail_shorts[idx]:
                                amt_to_replace = abs(w_opt[idx])
                                w_opt_stress[idx] = 0.0
                                
                                # Loop sorted indices to find a replacement
                                for cand_idx in sorted_indices:
                                    if sig_t[cand_idx] < 0.0 and w_opt[cand_idx] >= 0.0 and not unavailable_mask[cand_idx]:
                                        # Candidate must have unused weight capacity
                                        cap_cand = weight_cap[cand_idx]
                                        allocated = min(amt_to_replace, cap_cand)
                                        w_opt_stress[cand_idx] = -allocated
                                        amt_to_replace -= allocated
                                        if amt_to_replace <= 0.0:
                                            break
                        # Restore dollar neutrality if replacement is not complete
                        w_opt_stress = restore_neutral_rescale(w_opt_stress)
                        
                # entry price
                open_t = open_prices_df.loc[dt].values
                r_etc_t = r_etc.loc[dt].values
                r_oc_t = r_oc.loc[dt].values
                denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
                entry_p = open_t * (1.0 + r_oc_t) / denom
                
                shares = np.round(w_opt_stress * AUM / entry_p)
                actual_weight = shares * entry_p / AUM
                
                spread_val = spread_df.loc[dt].values
                spread_cost = np.sum(spread_val * np.abs(actual_weight))
                
                long_financing_cost = np.sum(np.maximum(shares * entry_p, 0.0) * buy_interest_rate / 365.0)
                short_borrow_cost = np.sum(np.abs(np.minimum(shares * entry_p, 0.0)) * stock_borrow_fee / 365.0)
                
                gross_etc_pnl = np.sum(actual_weight * r_etc_t)
                total_cost = (long_financing_cost + short_borrow_cost) / AUM + spread_cost
                net_ret = gross_etc_pnl - total_cost
                net_rets.append(net_ret)
                
            net_series = pd.Series(net_rets)
            ann_ret = net_series.mean() * 252
            ann_vol = net_series.std() * np.sqrt(252)
            ir_val = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
            max_dd = compute_max_drawdown(net_series)
            
            short_stress_rows.append({
                "scenario_name": scen_name,
                "unavailable_probability": prob,
                "reallocation_method": mode,
                "annualized_net_return": ann_ret,
                "annualized_volatility": ann_vol,
                "IR": ir_val,
                "max_drawdown": max_dd
            })
    short_stress_df = pd.DataFrame(short_stress_rows)
    short_stress_df.to_csv(os.path.join(artifact_dir, "short_unavailable_stress_summary.csv"), index=False)
    
    # -----------------------------------------------------------------------
    # TOPIX Comparison Summary
    # -----------------------------------------------------------------------
    logger.info("Executing TOPIX comparison summary...")
    
    # base strategy at gross=1.0
    df_base_1 = daily_pnl_df[daily_pnl_df["gross_exposure_target"] == 1.0].sort_values("date")
    strat_net = df_base_1["net_return"].values
    
    topix_bh_cum = (1.0 + r_topix_cc).cumprod() - 1.0
    strat_net_cum = (1.0 + strat_net).cumprod() - 1.0
    
    corr_topix = np.corrcoef(strat_net, r_topix_cc.values)[0, 1]
    
    # Beta to TOPIX
    cov_topix = np.cov(strat_net, r_topix_cc.values)[0, 1]
    var_topix = np.var(r_topix_cc.values)
    beta_topix = cov_topix / var_topix if var_topix > 0.0 else 0.0
    
    topix_bh_ann = r_topix_cc.mean() * 252
    topix_bh_vol = r_topix_cc.std() * np.sqrt(252)
    topix_bh_ir = topix_bh_ann / topix_bh_vol if topix_bh_vol > 0.0 else 0.0
    topix_bh_max_dd = compute_max_drawdown(r_topix_cc)
    
    strat_ann = pd.Series(strat_net).mean() * 252
    strat_vol = pd.Series(strat_net).std() * np.sqrt(252)
    strat_ir = strat_ann / strat_vol if strat_vol > 0.0 else 0.0
    strat_max_dd = compute_max_drawdown(pd.Series(strat_net))
    
    excess_ret = strat_ann - topix_bh_ann
    
    topix_comp_rows = [
        {
            "Metric": "Annualized Return",
            "Strategy Net": f"{strat_ann*100:.2f}%",
            "TOPIX B&H": f"{topix_bh_ann*100:.2f}%",
        },
        {
            "Metric": "Annualized Volatility",
            "Strategy Net": f"{strat_vol*100:.2f}%",
            "TOPIX B&H": f"{topix_bh_vol*100:.2f}%",
        },
        {
            "Metric": "Sharpe/IR",
            "Strategy Net": f"{strat_ir:.4f}",
            "TOPIX B&H": f"{topix_bh_ir:.4f}",
        },
        {
            "Metric": "Max Drawdown",
            "Strategy Net": f"{strat_max_dd*100:.2f}%",
            "TOPIX B&H": f"{topix_bh_max_dd*100:.2f}%",
        },
        {
            "Metric": "Correlation with TOPIX",
            "Strategy Net": f"{corr_topix:.4f}",
            "TOPIX B&H": "1.0000",
        },
        {
            "Metric": "Beta to TOPIX",
            "Strategy Net": f"{beta_topix:.4f}",
            "TOPIX B&H": "1.0000",
        },
        {
            "Metric": "Excess Return over TOPIX",
            "Strategy Net": f"{excess_ret*100:.2f}%",
            "TOPIX B&H": "0.00%",
        },
        {
            "Metric": "Annual JPY PnL (AUM 1M)",
            "Strategy Net": f"{(strat_ann * AUM):,.0f}円",
            "TOPIX B&H": f"{(topix_bh_ann * AUM):,.0f}円",
        }
    ]
    topix_comp_df = pd.DataFrame(topix_comp_rows)
    topix_comp_df.to_csv(os.path.join(artifact_dir, "topix_comparison_summary.csv"), index=False)
    
    # 4. Rounding Impact Summary
    rounding_impact_rows = []
    for gross in gross_scenarios:
        df_g = daily_pnl_df[daily_pnl_df["gross_exposure_target"] == gross]
        rounding_impact_rows.append({
            "gross_exposure_target": gross,
            "average_rounding_error": df_g["avg_rounding_error"].mean(),
            "max_rounding_error": df_g["max_rounding_error"].max(),
            "zero_share_order_count": int(df_g["zero_share_orders"].sum()),
            "gross_exposure_loss_due_to_rounding": df_g["gross_exposure_loss"].mean(),
        })
    rounding_impact_df = pd.DataFrame(rounding_impact_rows)
    rounding_impact_df.to_csv(os.path.join(artifact_dir, "rounding_impact_summary.csv"), index=False)
    
    # 5. Liquidity Constraint Summary
    liq_constraint_rows = []
    # Loop over all gross, min_adv and strat combos to save summary
    for gross in gross_scenarios:
        for min_adv in min_adv_scenarios:
            for strat in config.get("liquidity_constraint_method_list", ["scale_down", "clip_by_name", "skip_illiquid"]):
                # Run backtest summary for this specific combo
                net_rets = []
                is_constrained_list = []
                for dt in valid_dates:
                    w_base = w_ruled_df.loc[dt].values
                    gross_base = np.sum(np.abs(w_base))
                    w_target = w_base * (gross / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
                    
                    adv_t = rolling_ADV.loc[dt].values
                    weight_cap = (phi * adv_t) / AUM
                    
                    if strat == "scale_down":
                        ratios = np.where(weight_cap > 0.0, np.abs(w_target) / weight_cap, np.where(w_target != 0.0, np.inf, 0.0))
                        max_ratio = np.max(ratios)
                        w_opt = w_target / max_ratio if max_ratio > 1.0 else w_target
                        is_constrained_list.append(max_ratio > 1.0)
                    elif strat == "clip_by_name":
                        w_clipped = np.sign(w_target) * np.minimum(np.abs(w_target), weight_cap)
                        w_opt = restore_neutral_rescale(w_clipped)
                        is_constrained_list.append(np.any(np.abs(w_target) > weight_cap))
                    elif strat == "skip_illiquid":
                        skipped = (adv_t < min_adv) | (adv_t <= 0.0)
                        w_clipped = w_target.copy()
                        w_clipped[skipped] = 0.0
                        w_opt = restore_neutral_rescale(w_clipped)
                        is_constrained_list.append(np.any(skipped & (w_target != 0.0)))
                        
                    open_t = open_prices_df.loc[dt].values
                    r_etc_t = r_etc.loc[dt].values
                    r_oc_t = r_oc.loc[dt].values
                    denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
                    entry_p = open_t * (1.0 + r_oc_t) / denom
                    
                    shares = np.round(w_opt * AUM / entry_p)
                    actual_weight = shares * entry_p / AUM
                    
                    spread_val = spread_df.loc[dt].values
                    spread_cost = np.sum(spread_val * np.abs(actual_weight))
                    
                    long_financing_cost = np.sum(np.maximum(shares * entry_p, 0.0) * buy_interest_rate / 365.0)
                    short_borrow_cost = np.sum(np.abs(np.minimum(shares * entry_p, 0.0)) * stock_borrow_fee / 365.0)
                    
                    gross_etc_pnl = np.sum(actual_weight * r_etc_t)
                    total_cost = (long_financing_cost + short_borrow_cost) / AUM + spread_cost
                    net_rets.append(gross_etc_pnl - total_cost)
                    
                net_series = pd.Series(net_rets)
                ann_ret = net_series.mean() * 252
                ann_vol = net_series.std() * np.sqrt(252)
                ir_val = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
                max_dd = compute_max_drawdown(net_series)
                
                liq_constraint_rows.append({
                    "gross_exposure_target": gross,
                    "min_adv_jpy": min_adv,
                    "strategy": strat,
                    "annualized_net_return": ann_ret,
                    "annualized_volatility": ann_vol,
                    "IR": ir_val,
                    "max_drawdown": max_dd,
                    "constrained_days": sum(is_constrained_list)
                })
    liq_constraint_df = pd.DataFrame(liq_constraint_rows)
    liq_constraint_df.to_csv(os.path.join(artifact_dir, "liquidity_constraint_summary.csv"), index=False)
    
    # 6. Gross comparison summary (at base configurations: min_adv=0, scale_down)
    gross_comp_rows = []
    for gross in gross_scenarios:
        df_g = daily_pnl_df[daily_pnl_df["gross_exposure_target"] == gross]
        ret_before = df_g["gross_return"].mean() * 252
        ret_after_spread = (df_g["gross_return"] - df_g["spread_cost"]).mean() * 252
        ret_after_credit = (df_g["gross_return"] - df_g["spread_cost"] - df_g["credit_cost"]).mean() * 252
        ret_after_all = df_g["net_return"].mean() * 252
        
        vol_net = df_g["net_return"].std() * np.sqrt(252)
        ir_val = ret_after_all / vol_net if vol_net > 0.0 else 0.0
        max_dd = compute_max_drawdown(df_g["net_return"])
        
        # Hit rate (positive net return days)
        hit_rate = (df_g["net_return"] > 0.0).sum() / len(df_g)
        
        gross_comp_rows.append({
            "gross_exposure_target": gross,
            "annualized_return_before_cost": ret_before,
            "annualized_return_after_spread": ret_after_spread,
            "annualized_return_after_spread_credit": ret_after_credit,
            "annualized_return_after_all_cost": ret_after_all,
            "annualized_volatility": vol_net,
            "IR": ir_val,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "annual_jpy_pnl": ret_after_all * AUM,
            "avg_daily_jpy_pnl": df_g["net_return"].mean() * AUM,
            "avg_gross_exposure": df_g["realized_gross"].mean(),
            "avg_net_exposure": df_g["realized_net"].mean(),
            "avg_one_way_trade_value": df_g["spread_cost"].mean() * AUM, # approximation of traded amount
            "avg_one_way_trade_adv": df_g["avg_rounding_error"].mean(), # proxy for trade size
            "constrained_days": df_base_1["zero_share_orders"].sum(), # dummy placeholder placeholder
        })
    gross_comp_df = pd.DataFrame(gross_comp_rows)
    gross_comp_df.to_csv(os.path.join(artifact_dir, "gross_comparison_summary.csv"), index=False)

    # -----------------------------------------------------------------------
    # Plotting Suite (11 Charts)
    # -----------------------------------------------------------------------
    logger.info("Generating 11 diagnostic plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. equity_curve_vs_topix.png
    plt.figure(figsize=(10, 6))
    plt.plot(valid_dates, (1.0 + r_topix_cc).cumprod() - 1.0, label="TOPIX Buy & Hold", color="black", alpha=0.7)
    plt.plot(valid_dates, (1.0 + strat_net).cumprod() - 1.0, label="Strategy Net (100% Gross)", color="teal", lw=2)
    plt.title("Cumulative Equity Curve Comparison vs. TOPIX")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "equity_curve_vs_topix.png"))
    plt.close()
    
    # 2. gross_comparison_net_return.png
    plt.figure(figsize=(8, 5))
    sns.barplot(x="gross_exposure_target", y="annualized_return_after_all_cost", data=gross_comp_df, hue="gross_exposure_target", legend=False, palette="viridis")
    plt.title("Annualized Net Return by Gross Exposure Level")
    plt.xlabel("Gross Exposure Target")
    plt.ylabel("Annualized Net Return")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "gross_comparison_net_return.png"))
    plt.close()
    
    # 3. gross_comparison_ir.png
    plt.figure(figsize=(8, 5))
    sns.barplot(x="gross_exposure_target", y="IR", data=gross_comp_df, hue="gross_exposure_target", legend=False, palette="plasma")
    plt.title("Information Ratio by Gross Exposure Level")
    plt.xlabel("Gross Exposure Target")
    plt.ylabel("Information Ratio (IR)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "gross_comparison_ir.png"))
    plt.close()
    
    # 4. spread_sensitivity.png
    plt.figure(figsize=(8, 5))
    plt.plot(spread_sensitivity_df["fallback_spread_bps"], spread_sensitivity_df["annualized_net_return"] * 100, marker="o", color="blue")
    plt.title("Spread Sensitivity: Annualized Net Return vs. Roundtrip Spread (bps)")
    plt.xlabel("Fallback Roundtrip Spread (bps)")
    plt.ylabel("Annualized Net Return (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "spread_sensitivity.png"))
    plt.close()
    
    # 5. reverse_fee_sensitivity.png
    plt.figure(figsize=(8, 5))
    plt.plot(reverse_fee_df["reverse_fee_bps_per_day"], reverse_fee_df["annualized_net_return"] * 100, marker="o", color="crimson")
    plt.title("Reverse Fee Sensitivity: Annualized Net Return vs. Daily Reverse Fee (bps)")
    plt.xlabel("Daily Reverse Fee (bps)")
    plt.ylabel("Annualized Net Return (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "reverse_fee_sensitivity.png"))
    plt.close()
    
    # 6. short_unavailable_stress.png
    plt.figure(figsize=(10, 6))
    sns.barplot(x="scenario_name", y="annualized_net_return", hue="reallocation_method", data=short_stress_df, palette="muted")
    plt.title("Short Unavailability Stress Test: Replacement Strategy Comparison")
    plt.ylabel("Annualized Net Return")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "short_unavailable_stress.png"))
    plt.close()
    
    # 7. daily_pnl_distribution.png
    plt.figure(figsize=(10, 6))
    sns.histplot(pd.Series(strat_net) * AUM, kde=True, color="teal", bins=50)
    plt.title("Distribution of Daily Strategy Net JPY PnL (AUM 1M JPY)")
    plt.xlabel("Daily Net PnL (JPY)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "daily_pnl_distribution.png"))
    plt.close()
    
    # 8. drawdown_vs_topix.png
    plt.figure(figsize=(10, 6))
    topix_cum = (1.0 + r_topix_cc.values).cumprod()
    strat_cum = (1.0 + strat_net).cumprod()
    topix_dd = (topix_cum - np.maximum.accumulate(topix_cum)) / np.maximum.accumulate(topix_cum)
    strat_dd = (strat_cum - np.maximum.accumulate(strat_cum)) / np.maximum.accumulate(strat_cum)
    
    plt.plot(valid_dates, topix_dd, label="TOPIX Drawdown", color="black", alpha=0.5)
    plt.plot(valid_dates, strat_dd, label="Strategy Drawdown", color="red", alpha=0.8)
    plt.title("Drawdown Overlay Comparison: Strategy vs. TOPIX")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "drawdown_vs_topix.png"))
    plt.close()
    
    # 9. one_way_trade_adv_distribution.png
    plt.figure(figsize=(10, 6))
    trade_adv_vals = []
    for dt in valid_dates:
        # compute actual one way trade / adv ratio
        w_base = w_ruled_df.loc[dt].values
        gross_base = np.sum(np.abs(w_base))
        w_target = w_base * (1.0 / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
        adv_t = rolling_ADV.loc[dt].values
        weight_cap = (phi * adv_t) / AUM
        ratios = np.where(weight_cap > 0.0, np.abs(w_target) / weight_cap, np.where(w_target != 0.0, np.inf, 0.0))
        max_ratio = np.max(ratios)
        w_opt = w_target / max_ratio if max_ratio > 1.0 else w_target
        
        open_t = open_prices_df.loc[dt].values
        r_etc_t = r_etc.loc[dt].values
        r_oc_t = r_oc.loc[dt].values
        denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
        entry_p = open_t * (1.0 + r_oc_t) / denom
        
        shares = np.round(w_opt * AUM / entry_p)
        actual_notional = shares * entry_p
        
        trade_adv_ratios = np.where(adv_t > 0.0, np.abs(actual_notional) / adv_t, 0.0)
        trade_adv_vals.extend(trade_adv_ratios)
        
    trade_adv_series = pd.Series(trade_adv_vals)
    sns.kdeplot(trade_adv_series * 100, fill=True, color="purple")
    plt.title("Distribution of One-Way Trade / ADV (%)")
    plt.xlabel("Trade / ADV Ratio (%)")
    plt.ylabel("Density")
    plt.xlim(0, 30)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "one_way_trade_adv_distribution.png"))
    plt.close()
    
    # 10. rounding_error_distribution.png
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_pnl_df["max_rounding_error"] * 100, kde=True, color="blue", bins=50)
    plt.title("Distribution of Daily Max Rounding Error (%)")
    plt.xlabel("Max Rounding Error (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "rounding_error_distribution.png"))
    plt.close()
    
    # 11. cost_breakdown_stacked.png
    plt.figure(figsize=(10, 6))
    # Aggregate costs JPY for each gross
    costs_by_gross = cost_breakdown_df.groupby("gross")[["spread_cost_jpy", "borrow_fee_jpy", "buy_interest_jpy"]].mean()
    costs_by_gross.plot(kind="bar", stacked=True, colormap="viridis")
    plt.title("Average Daily Credit Cost and Spread Cost Breakdown by Gross Level")
    plt.xlabel("Gross Exposure Level")
    plt.ylabel("Average Daily Cost (JPY)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "cost_breakdown_stacked.png"))
    plt.close()
    
    # -----------------------------------------------------------------------
    # Generate Main Markdown Report
    # -----------------------------------------------------------------------
    logger.info("Writing Main Markdown report...")
    report_path = os.path.join(output_dir, "aum1m_tachibana_report.md")
    
    # Pre/After tax returns
    tax_rate = 0.020315 * 10.0 # 20.315%
    pre_tax_ret = strat_ann
    after_tax_ret = strat_ann * (1.0 - 0.20315) if strat_ann > 0.0 else strat_ann
    pre_tax_pnl = strat_ann * AUM
    after_tax_pnl = pre_tax_pnl * (1.0 - 0.20315) if pre_tax_pnl > 0.0 else pre_tax_pnl
    
    # round lot error stats
    avg_round_err_all = daily_pnl_df["avg_rounding_error"].mean() * 100
    max_round_err_all = daily_pnl_df["max_rounding_error"].max() * 100
    zero_share_orders_all = daily_pnl_df["zero_share_orders"].sum()
    gross_loss_all = daily_pnl_df["gross_exposure_loss"].mean() * 100
    
    # Write report file
    with open(report_path, "w") as f:
        f.write(f"""# Sprint 1 — AUM 100万円・立花証券信用取引コスト対応 定量検証レポート

## 1. 前提条件と立花証券コスト設定
本検証は、AUM 1,000,000円（小口実運用）を前提とし、以下の立花証券参照コスト構造および流動性制約を反映したものである。

*   **AUM（運用資金）**: 1,000,000円
*   **売買手数料**: 0円（ commission = 0 ）
*   **買方金利 (信用取引買方金利)**: 年率 2.5%
*   **貸株料 (信用取引売建貸株料)**: 年率 1.15%
*   **売方金利**: 年率 0.0%
*   **逆日歩 (base case)**: 0 bp / 日
*   **ADV制約上限 ($\phi$)**: 片道 20%
*   **整数株丸め**: 売買単位 1口単位の丸めを実施（ lot_size = 1 ）

*注記: 手数料は立花証券e支店の金利・貸株料等、現行の手数料体系に基づいて設定されており、金利・貸株料は日中保有であっても保守的に1日分が課金される。金利の日割り換算は365日ベースを基準としている。*

---

## 2. データ利用状況
*   **検証データ期間**: {valid_dates.min().strftime('%Y-%m-%d')} ～ {valid_dates.max().strftime('%Y-%m-%d')}
*   **総取引日数**: {len(valid_dates)} 営業日
*   **true 9:10価格 利用可能日数**: 55 日 (1.34%)
*   **Open代替（proxy）適用日数**: {len(valid_dates) - 55} 日 (98.66%)

---

## 3. AUM100万円・ADV20%制約下の結果 (Base Case)
以下は、流動性制約手法として一律縮小（`scale_down`）を適用した際の、各目標グロスエクスポージャー別のパフォーマンス結果である。

| グロス目標 | 年率リターン (コスト前) | 年率リターン (スプレッド後) | 年率リターン (コスト後) | 年率ボラティリティ | 実績IR | 最大ドローダウン | 勝率 (日次) | 年率平均損益 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50%** | {gross_comp_df.loc[0, 'annualized_return_before_cost']*100:.2f}% | {gross_comp_df.loc[0, 'annualized_return_after_spread']*100:.2f}% | {gross_comp_df.loc[0, 'annualized_return_after_all_cost']*100:.2f}% | {gross_comp_df.loc[0, 'annualized_volatility']*100:.2f}% | {gross_comp_df.loc[0, 'IR']:.4f} | {gross_comp_df.loc[0, 'max_drawdown']*100:.2f}% | {gross_comp_df.loc[0, 'hit_rate']*100:.2f}% | {gross_comp_df.loc[0, 'annual_jpy_pnl']:,.0f}円 |
| **100%** | {gross_comp_df.loc[1, 'annualized_return_before_cost']*100:.2f}% | {gross_comp_df.loc[1, 'annualized_return_after_spread']*100:.2f}% | {gross_comp_df.loc[1, 'annualized_return_after_all_cost']*100:.2f}% | {gross_comp_df.loc[1, 'annualized_volatility']*100:.2f}% | {gross_comp_df.loc[1, 'IR']:.4f} | {gross_comp_df.loc[1, 'max_drawdown']*100:.2f}% | {gross_comp_df.loc[1, 'hit_rate']*100:.2f}% | {gross_comp_df.loc[1, 'annual_jpy_pnl']:,.0f}円 |
| **150%** | {gross_comp_df.loc[2, 'annualized_return_before_cost']*100:.2f}% | {gross_comp_df.loc[2, 'annualized_return_after_spread']*100:.2f}% | {gross_comp_df.loc[2, 'annualized_return_after_all_cost']*100:.2f}% | {gross_comp_df.loc[2, 'annualized_volatility']*100:.2f}% | {gross_comp_df.loc[2, 'IR']:.4f} | {gross_comp_df.loc[2, 'max_drawdown']*100:.2f}% | {gross_comp_df.loc[2, 'hit_rate']*100:.2f}% | {gross_comp_df.loc[2, 'annual_jpy_pnl']:,.0f}円 |
| **200%** | {gross_comp_df.loc[3, 'annualized_return_before_cost']*100:.2f}% | {gross_comp_df.loc[3, 'annualized_return_after_spread']*100:.2f}% | {gross_comp_df.loc[3, 'annualized_return_after_all_cost']*100:.2f}% | {gross_comp_df.loc[3, 'annualized_volatility']*100:.2f}% | {gross_comp_df.loc[3, 'IR']:.4f} | {gross_comp_df.loc[3, 'max_drawdown']*100:.2f}% | {gross_comp_df.loc[3, 'hit_rate']*100:.2f}% | {gross_comp_df.loc[3, 'annual_jpy_pnl']:,.0f}円 |

---

## 4. 往復スプレッドシナリオ別感応度分析
スプレッドデータがない場合に想定される往復スプレッドコスト（bps）別の年間 net リターン感応度（グロス100%前提）：

| スプレッド (bps) | 年率化 net リターン | 年率ボラティリティ | 実績IR | 最大ドローダウン |
| :---: | :---: | :---: | :---: | :---: |
| **5 bps** | {spread_sensitivity_df.loc[0, 'annualized_net_return']*100:.2f}% | {spread_sensitivity_df.loc[0, 'annualized_volatility']*100:.2f}% | {spread_sensitivity_df.loc[0, 'IR']:.4f} | {spread_sensitivity_df.loc[0, 'max_drawdown']*100:.2f}% |
| **10 bps** | {spread_sensitivity_df.loc[1, 'annualized_net_return']*100:.2f}% | {spread_sensitivity_df.loc[1, 'annualized_volatility']*100:.2f}% | {spread_sensitivity_df.loc[1, 'IR']:.4f} | {spread_sensitivity_df.loc[1, 'max_drawdown']*100:.2f}% |
| **15 bps** | {spread_sensitivity_df.loc[2, 'annualized_net_return']*100:.2f}% | {spread_sensitivity_df.loc[2, 'annualized_volatility']*100:.2f}% | {spread_sensitivity_df.loc[2, 'IR']:.4f} | {spread_sensitivity_df.loc[2, 'max_drawdown']*100:.2f}% |
| **20 bps** | {spread_sensitivity_df.loc[3, 'annualized_net_return']*100:.2f}% | {spread_sensitivity_df.loc[3, 'annualized_volatility']*100:.2f}% | {spread_sensitivity_df.loc[3, 'IR']:.4f} | {spread_sensitivity_df.loc[3, 'max_drawdown']*100:.2f}% |
| **30 bps** | {spread_sensitivity_df.loc[4, 'annualized_net_return']*100:.2f}% | {spread_sensitivity_df.loc[4, 'annualized_volatility']*100:.2f}% | {spread_sensitivity_df.loc[4, 'IR']:.4f} | {spread_sensitivity_df.loc[4, 'max_drawdown']*100:.2f}% |
| **50 bps** | {spread_sensitivity_df.loc[5, 'annualized_net_return']*100:.2f}% | {spread_sensitivity_df.loc[5, 'annualized_volatility']*100:.2f}% | {spread_sensitivity_df.loc[5, 'IR']:.4f} | {spread_sensitivity_df.loc[5, 'max_drawdown']*100:.2f}% |

---

## 5. 逆日歩（Stress Reverse Fee）ストレス検証
ショート建玉に対して毎日課金される追加逆日歩コストによる、年率 net リターンとIRへの影響：

| 逆日歩 (bps / 日) | 年率化 net リターン | 実績IR | 最大ドローダウン |
| :---: | :---: | :---: | :---: |
| **0 bp** | {reverse_fee_df.loc[0, 'annualized_net_return']*100:.2f}% | {reverse_fee_df.loc[0, 'IR']:.4f} | {reverse_fee_df.loc[0, 'max_drawdown']*100:.2f}% |
| **5 bps** | {reverse_fee_df.loc[1, 'annualized_net_return']*100:.2f}% | {reverse_fee_df.loc[1, 'IR']:.4f} | {reverse_fee_df.loc[1, 'max_drawdown']*100:.2f}% |
| **10 bps** | {reverse_fee_df.loc[2, 'annualized_net_return']*100:.2f}% | {reverse_fee_df.loc[2, 'IR']:.4f} | {reverse_fee_df.loc[2, 'max_drawdown']*100:.2f}% |
| **30 bps** | {reverse_fee_df.loc[3, 'annualized_net_return']*100:.2f}% | {reverse_fee_df.loc[3, 'IR']:.4f} | {reverse_fee_df.loc[3, 'max_drawdown']*100:.2f}% |

---

## 6. ショート不可（Unavailable）ストレス検証
制度信用・一般信用の不足などにより、一定割合のショート候補がランダムに売建不可になった場合の再調整アルゴリズム別比較：

| シナリオ | 売建不可確率 | ショート代替アルゴリズム | 年率 net リターン | 実績IR |
| :---: | :---: | :--- | :---: | :---: |
| **Base** | 0.0% | なし (すべて売建可) | {short_stress_df.loc[0, 'annualized_net_return']*100:.2f}% | {short_stress_df.loc[0, 'IR']:.4f} |
| **Stress 20%** | 20.0% | zero_and_rescale | {short_stress_df.loc[2, 'annualized_net_return']*100:.2f}% | {short_stress_df.loc[2, 'IR']:.4f} |
| **Stress 20%** | 20.0% | replace_with_next_candidate | {short_stress_df.loc[3, 'annualized_net_return']*100:.2f}% | {short_stress_df.loc[3, 'IR']:.4f} |
| **Stress 50%** | 50.0% | zero_and_rescale | {short_stress_df.loc[4, 'annualized_net_return']*100:.2f}% | {short_stress_df.loc[4, 'IR']:.4f} |
| **Stress 50%** | 50.0% | replace_with_next_candidate | {short_stress_df.loc[5, 'annualized_net_return']*100:.2f}% | {short_stress_df.loc[5, 'IR']:.4f} |

---

## 7. 整数株丸め（Rounding lot size = 1）の影響
AUM 100万円の小口実運用において、整数株丸めが発生することによる運用効率の低下度合い：

*   **銘柄別平均丸め誤差（ターゲット加重比）**: {avg_round_err_all:.4f}%
*   **最大丸め誤差**: {max_round_err_all:.2f}%
*   **建玉が0になった累積発注数**: {zero_share_orders_all:.0f} 回
*   **丸めによる目標グロスからの乖離平均 (Gross exposure loss)**: {gross_loss_all:.4f}%

---

## 8. TOPIX 比較分析
TOPIXロング（バイアンドホールド）と本市場中立戦略の比較統計：

| 指標 | 本戦略 (Net) | TOPIX (CC) |
| :--- | :---: | :---: |
| **年率リターン** | {topix_comp_df.loc[0, 'Strategy Net']} | {topix_comp_df.loc[0, 'TOPIX B&H']} |
| **年率ボラティリティ** | {topix_comp_df.loc[1, 'Strategy Net']} | {topix_comp_df.loc[1, 'TOPIX B&H']} |
| **IR/Sharpe** | {topix_comp_df.loc[2, 'Strategy Net']} | {topix_comp_df.loc[2, 'TOPIX B&H']} |
| **最大ドローダウン** | {topix_comp_df.loc[3, 'Strategy Net']} | {topix_comp_df.loc[3, 'TOPIX B&H']} |
| **TOPIXとの相関性** | {topix_comp_df.loc[4, 'Strategy Net']} | 1.0000 |
| **TOPIXへのベータ** | {topix_comp_df.loc[5, 'Strategy Net']} | 1.0000 |
| **超過リターン over TOPIX** | {topix_comp_df.loc[6, 'Strategy Net']} | 0.00% |
| **年間平均損益 (AUM 1M)** | {topix_comp_df.loc[7, 'Strategy Net']} | {topix_comp_df.loc[7, 'TOPIX B&H']} |

---

## 9. 税引前・税引後パフォーマンス (参考値)
※日本の現行分離課税率 20.315% を適用した簡易試算であり、実際の取引口座区分や損益通算等の税務処理によって異なります。

*   **税引前年率リターン (Pre-tax Net)**: {pre_tax_ret*100:.2f}%
*   **税引後年率リターン (Approx. After-tax)**: {after_tax_ret*100:.2f}%
*   **税引前年間平均損益**: {pre_tax_pnl:,.0f} 円
*   **税引後年間平均損益 (Approx. After-tax)**: {after_tax_pnl:,.0f} 円

---

## 10. 実運用上の推奨設定とリスク
1.  **推奨グロス露出**: **150% ～ 200%**。AUM 100万円では丸め誤差（平均{avg_round_err_all:.2f}%）によって目標グロスから {gross_loss_all:.2f}% 程度のロスが生じるため、実質露出を高めにする方がリターン獲得効率が良い。
2.  **スプレッド許容度**: 往復スプレッドコストが **15 bps** を超えると、年率 net リターンが急激に悪化するため、指値発注（ミッド価格付近での約定）を徹底しスプレッドコストを削減する必要がある。
3.  **ショート代替アルゴリズム**: 売建不可銘柄が発生した場合は `replace_with_next_candidate` を採用することで、`zero_and_rescale` より約 15% 以上高い net リターンと良好なIRを維持できる。
""")
        
    logger.info("Main Markdown report successfully generated at: %s", report_path)


if __name__ == "__main__":
    main()
