"""src/leadlag/diagnostics/sprint0_qa.py — Sprint 0-B Diagnostics QA audits.

Verifies target mismatch splits, date lags, signal signs, units, capacity inputs, costs,
and ex-ante IR calibration leaks.
"""

from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf

from leadlag.data.cache import load_df_exec_from_local_cache, load_intraday_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS, TOPIX_TICKER
from experiments.diagnostics.sprint0 import run_sprint0_calculations, find_latest_distribution_diagnostics

logger = logging.getLogger(__name__)


def run_sprint0_qa(
    start_date: str | None = None,
    end_date: str | None = None,
    config: dict | None = None,
) -> dict:
    """Run all 8 QA checks and audits for Sprint 0."""
    if config is None:
        config = {}

    # Run base calculations
    logger.info("Running base diagnostics calculations...")
    base_results = run_sprint0_calculations(start_date=start_date, end_date=end_date, config=config)

    # Load shared inputs
    df_exec = load_df_exec_from_local_cache()
    sim_dates = df_exec.index
    
    # 5m cache to find dates with real 9:10 prices
    df_5m = load_intraday_cache("5m")
    has_5m_dates = set()
    if df_5m is not None and not df_5m.empty:
        has_5m_dates = set(pd.Series(df_5m.index.date).unique())

    # Get data panels from base results
    returns_panel = base_results["returns_panel"]
    residual_returns_panel = base_results["residual_returns_panel"]
    signal_diagnostics_panel = base_results["signal_diagnostics_panel"]
    
    valid_dates_beta = base_results["ic_timeseries"].index
    
    # QA 1: 9:10 Prices vs Open Proxy Separation
    qa1_results = run_qa1_separation(
        returns_panel, residual_returns_panel, signal_diagnostics_panel, has_5m_dates, valid_dates_beta
    )

    # QA 2: Date Alignment Check
    qa2_results = run_qa2_alignment(
        returns_panel, residual_returns_panel, signal_diagnostics_panel, valid_dates_beta
    )

    # QA 3: Signal Sign Validation
    qa3_results = run_qa3_signs(
        returns_panel, residual_returns_panel, signal_diagnostics_panel, valid_dates_beta
    )

    # QA 4: Bps, Annualization, and PnL Unit Check
    qa4_results = run_qa4_units(
        returns_panel, residual_returns_panel, signal_diagnostics_panel, valid_dates_beta, base_results
    )

    # QA 5: Long-Short Sign Clarification
    qa5_results = run_qa5_long_short_signs(
        returns_panel, signal_diagnostics_panel, valid_dates_beta
    )

    # QA 6: Capacity Calculation Unit Audit
    qa6_results = run_qa6_capacity_units(
        signal_diagnostics_panel, valid_dates_beta
    )

    # QA 7: Cost vs Capacity Consistency
    qa7_results = run_qa7_cost_consistency(
        base_results, signal_diagnostics_panel, residual_returns_panel, valid_dates_beta
    )

    # QA 8: Ex-Ante IR Calibration Leak Check
    qa8_results = run_qa8_calibration_leak(
        base_results, valid_dates_beta
    )

    return {
        "qa1": qa1_results,
        "qa2": qa2_results,
        "qa3": qa3_results,
        "qa4": qa4_results,
        "qa5": qa5_results,
        "qa6": qa6_results,
        "qa7": qa7_results,
        "qa8": qa8_results,
    }


def run_qa1_separation(
    returns_panel: pd.DataFrame,
    residual_returns_panel: pd.DataFrame,
    signal_diagnostics_panel: pd.DataFrame,
    has_5m_dates: set,
    valid_dates_beta: pd.DatetimeIndex,
) -> dict:
    """QA 1: Separate real 9:10-to-close returns from open-to-close proxy returns."""
    # Define groups
    dates_full = valid_dates_beta
    dates_proxy = pd.DatetimeIndex([d for d in valid_dates_beta if d.date() not in has_5m_dates])
    dates_real = pd.DatetimeIndex([d for d in valid_dates_beta if d.date() in has_5m_dates])

    groups = {
        "mostly Open->Close proxy (Full Period)": dates_full,
        "Open->Close proxy only (No 9:10)": dates_proxy,
        "true 9:10-to-Close only": dates_real,
    }

    comparison_rows = []
    for g_name, dates in groups.items():
        if len(dates) == 0:
            comparison_rows.append({
                "Group": g_name,
                "Days": 0,
                "Rank IC Mean": np.nan,
                "Rank IC Std": np.nan,
                "Rank ICIR": np.nan,
                "Rank IC Hit Rate": np.nan,
                "Long-Short Spread (bps)": np.nan,
                "PnL Hit Rate": np.nan,
            })
            continue

        ic_list = []
        p_ic_list = []
        ls_spreads = []
        pnl_pos = 0

        # Extract values
        sig = signal_diagnostics_panel.loc[dates, "signal_gap_adjusted"]
        y_res = residual_returns_panel.loc[dates, "y_res_intraday"]
        w_ruled = signal_diagnostics_panel.loc[dates, "weight_ruled"]
        r_intra = returns_panel.loc[dates, "r_intraday"]

        for dt in dates:
            s_t = sig.loc[dt].values
            y_t = y_res.loc[dt].values
            w_t = w_ruled.loc[dt].values
            r_t = r_intra.loc[dt].values

            # Daily Rank IC
            rank_ic, _ = stats.spearmanr(s_t, y_t)
            ic_list.append(rank_ic)

            # Daily Pearson IC
            pear_ic, _ = stats.pearsonr(s_t, y_t)
            p_ic_list.append(pear_ic)

            # Quantiles
            ranks = sig.loc[dt].rank(method="first")
            q1_mask = ranks <= np.floor(len(s_t) * 0.3)
            q3_mask = ranks > np.ceil(len(s_t) * 0.7)
            ls_spreads.append(y_t[q3_mask].mean() - y_t[q1_mask].mean())

            # PnL
            pnl_t = np.sum(w_t * r_t)
            if pnl_t > 0:
                pnl_pos += 1

        ic_series = pd.Series(ic_list).dropna()
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic > 0 else 0.0
        hit_rate = (ic_series > 0).mean()

        mean_spread = pd.Series(ls_spreads).mean() * 10000 # to bps
        pnl_hit_rate = pnl_pos / len(dates)

        comparison_rows.append({
            "Group": g_name,
            "Days": len(dates),
            "Rank IC Mean": mean_ic,
            "Rank IC Std": std_ic,
            "Rank ICIR": icir,
            "Rank IC Hit Rate": hit_rate,
            "Long-Short Spread (bps)": mean_spread,
            "PnL Hit Rate": pnl_hit_rate,
        })

    comparison_df = pd.DataFrame(comparison_rows).set_index("Group")
    return {"comparison_table": comparison_df}


def run_qa2_alignment(
    returns_panel: pd.DataFrame,
    residual_returns_panel: pd.DataFrame,
    signal_diagnostics_panel: pd.DataFrame,
    valid_dates_beta: pd.DatetimeIndex,
) -> dict:
    """QA 2: Date alignment validation (evaluate lag/lead effects)."""
    # Align signals and returns with lags -2 to +2
    targets = {
        "y_res_cc": residual_returns_panel["y_res_cc"],
        "y_res_intraday": residual_returns_panel["y_res_intraday"],
        "r_intraday": returns_panel["r_intraday"],
    }

    sig = signal_diagnostics_panel["signal_gap_adjusted"]

    # We shift target return relative to signal.
    # return[t+k] aligned with signal[t]
    # In pandas, to get return[t+k] aligned with signal[t], we shift return series by -k.
    lags = [-2, -1, 0, 1, 2]
    alignment_results = {}

    for t_name, ret_df in targets.items():
        lag_means = {}
        for k in lags:
            ic_list = []
            # We align signal[t] with return[t+k]
            # Slicing dates to avoid boundary NaNs
            dates_slice = valid_dates_beta[2:-2]
            for dt in dates_slice:
                dt_idx = valid_dates_beta.get_loc(dt)
                target_dt = valid_dates_beta[dt_idx + k]
                
                s_t = sig.loc[dt].values
                r_tk = ret_df.loc[target_dt].values
                
                if np.isnan(s_t).any() or np.isnan(r_tk).any():
                    continue
                rank_ic, _ = stats.spearmanr(s_t, r_tk)
                ic_list.append(rank_ic)

            lag_means[f"Lag {k}"] = pd.Series(ic_list).mean()
        alignment_results[t_name] = pd.Series(lag_means)

    alignment_df = pd.DataFrame(alignment_results)
    return {"alignment_table": alignment_df}


def run_qa3_signs(
    returns_panel: pd.DataFrame,
    residual_returns_panel: pd.DataFrame,
    signal_diagnostics_panel: pd.DataFrame,
    valid_dates_beta: pd.DatetimeIndex,
) -> dict:
    """QA 3: Validate signal sign (+signal vs -signal)."""
    sig = signal_diagnostics_panel["signal_gap_adjusted"]
    y_res_cc = residual_returns_panel["y_res_cc"]
    y_res_intra = residual_returns_panel["y_res_intraday"]

    targets = {
        "y_res_cc_60": y_res_cc,
        "y_res_intraday_60": y_res_intra,
    }

    results = {}
    for s_mode, factor in [("Positive Signal (Normal)", 1.0), ("Negative Signal (-Signal)", -1.0)]:
        mode_results = {}
        for t_name, ret_df in targets.items():
            ic_list = []
            p_ic_list = []
            ls_spreads = []
            q1_returns = []
            q2_returns = []
            q3_returns = []

            for dt in valid_dates_beta:
                s_t = sig.loc[dt].values * factor
                r_t = ret_df.loc[dt].values

                if np.isnan(s_t).any() or np.isnan(r_t).any():
                    continue

                rank_ic, _ = stats.spearmanr(s_t, r_t)
                pear_ic, _ = stats.pearsonr(s_t, r_t)
                ic_list.append(rank_ic)
                p_ic_list.append(pear_ic)

                ranks = pd.Series(s_t).rank(method="first")
                q1_mask = ranks <= np.floor(len(s_t) * 0.3)
                q3_mask = ranks > np.ceil(len(s_t) * 0.7)
                q2_mask = ~(q1_mask | q3_mask)

                q1_ret = r_t[q1_mask].mean()
                q2_ret = r_t[q2_mask].mean()
                q3_ret = r_t[q3_mask].mean()

                q1_returns.append(q1_ret)
                q2_returns.append(q2_ret)
                q3_returns.append(q3_ret)
                ls_spreads.append(q3_ret - q1_ret)

            mode_results[f"{t_name} Rank IC"] = pd.Series(ic_list).mean()
            mode_results[f"{t_name} Pearson IC"] = pd.Series(p_ic_list).mean()
            mode_results[f"{t_name} Long-Short Spread (bps)"] = pd.Series(ls_spreads).mean() * 10000
            mode_results[f"{t_name} Q1 (Bottom 30%) bps"] = pd.Series(q1_returns).mean() * 10000
            mode_results[f"{t_name} Q2 (Middle 40%) bps"] = pd.Series(q2_returns).mean() * 10000
            mode_results[f"{t_name} Q3 (Top 30%) bps"] = pd.Series(q3_returns).mean() * 10000

        results[s_mode] = pd.Series(mode_results)

    sign_df = pd.DataFrame(results)
    return {"sign_comparison_table": sign_df}


def run_qa4_units(
    returns_panel: pd.DataFrame,
    residual_returns_panel: pd.DataFrame,
    signal_diagnostics_panel: pd.DataFrame,
    valid_dates_beta: pd.DatetimeIndex,
    base_results: dict,
) -> dict:
    """QA 4: Bps, Annualization, and PnL unit check."""
    # Representative 5 days (last 5 days)
    rep_dates = valid_dates_beta[-5:]
    
    rep_rows = []
    for dt in rep_dates:
        s_t = signal_diagnostics_panel.loc[dt, "signal_gap_adjusted"]
        w_t = signal_diagnostics_panel.loc[dt, "weight_ruled"]
        r_t = returns_panel.loc[dt, "r_intraday"]

        # Sort indices
        ranks = s_t.rank(method="first")
        long_tickers = list(s_t.index[ranks > np.ceil(len(s_t) * 0.7)])
        short_tickers = list(s_t.index[ranks <= np.floor(len(s_t) * 0.3)])

        long_pnl = np.sum([w_t[tk] * r_t[tk] for tk in long_tickers])
        short_pnl = np.sum([w_t[tk] * r_t[tk] for tk in short_tickers])
        total_pnl = np.sum(w_t * r_t)

        rep_rows.append({
            "Trade Date": dt.strftime("%Y-%m-%d"),
            "Long Tickers": ", ".join([t.replace(".T", "") for t in long_tickers]),
            "Short Tickers": ", ".join([t.replace(".T", "") for t in short_tickers]),
            "Long PnL (bps)": long_pnl * 10000,
            "Short PnL (bps)": short_pnl * 10000,
            "Total PnL (bps)": total_pnl * 10000,
            "Gross Exposure": np.sum(np.abs(w_t)),
        })

    rep_df = pd.DataFrame(rep_rows).set_index("Trade Date")

    # Audit formulas
    c_stats = base_results["cost_impact_summary"]
    # Static cost Sharpe check
    daily_mean_pnl = base_results["beta_exposure_timeseries"]["strategy_return"].mean()
    daily_std_pnl = base_results["beta_exposure_timeseries"]["strategy_return"].dropna().std()

    formula_check = {
        "Annualization Factor (Days)": 252,
        "Annual Return Formula": "mean * 252",
        "Annual Vol Formula": "std * sqrt(252)",
        "Calculated Sharpe Formula": "ann_return / ann_vol",
        "Daily PnL Formula": "sum(weight * return)",
        "Weight scale is decimal": np.all(np.abs(signal_diagnostics_panel["weight_ruled"].dropna().sum(axis=1)) < 3.0),
    }

    return {
        "representative_days_table": rep_df,
        "formula_audit": formula_check,
    }


def run_qa5_long_short_signs(
    returns_panel: pd.DataFrame,
    signal_diagnostics_panel: pd.DataFrame,
    valid_dates_beta: pd.DatetimeIndex,
) -> dict:
    """QA 5: Audit long leg and short leg returns and PnL signs separately."""
    sig = signal_diagnostics_panel["signal_gap_adjusted"]
    w_ruled = signal_diagnostics_panel["weight_ruled"]
    r_intra = returns_panel["r_intraday"]

    long_basket_returns = []
    short_basket_returns = []
    long_leg_pnls = []
    short_leg_pnls = []
    total_pnls = []

    for dt in valid_dates_beta:
        s_t = sig.loc[dt].values
        w_t = w_ruled.loc[dt].values
        r_t = r_intra.loc[dt].values

        ranks = pd.Series(s_t).rank(method="first")
        long_mask = ranks > np.ceil(len(s_t) * 0.7)
        short_mask = ranks <= np.floor(len(s_t) * 0.3)

        # Average raw returns of the assets selected
        long_basket_returns.append(r_t[long_mask].mean())
        short_basket_returns.append(r_t[short_mask].mean())

        # PnLs
        w_long = np.maximum(w_t, 0.0)
        w_short = np.minimum(w_t, 0.0)

        long_leg_pnls.append(np.sum(w_long * r_t))
        short_leg_pnls.append(np.sum(w_short * r_t))
        total_pnls.append(np.sum(w_t * r_t))

    ls_df = pd.DataFrame({
        "long_basket_raw_return_bps": pd.Series(long_basket_returns, index=valid_dates_beta) * 10000,
        "short_basket_raw_return_bps": pd.Series(short_basket_returns, index=valid_dates_beta) * 10000,
        "long_leg_pnl_bps": pd.Series(long_leg_pnls, index=valid_dates_beta) * 10000,
        "short_leg_pnl_bps": pd.Series(short_leg_pnls, index=valid_dates_beta) * 10000,
        "total_pnl_bps": pd.Series(total_pnls, index=valid_dates_beta) * 10000,
    })

    summary = {
        "Long Basket Raw Return Mean (bps)": ls_df["long_basket_raw_return_bps"].mean(),
        "Short Basket Raw Return Mean (bps)": ls_df["short_basket_raw_return_bps"].mean(),
        "Long Leg PnL Mean (bps)": ls_df["long_leg_pnl_bps"].mean(),
        "Short Leg PnL Mean (bps)": ls_df["short_leg_pnl_bps"].mean(),
        "Total Strategy PnL Mean (bps)": ls_df["total_pnl_bps"].mean(),
        "Long Basket Return Hit Rate": (ls_df["long_basket_raw_return_bps"] > 0).mean(),
        "Short Basket Return Hit Rate": (ls_df["short_basket_raw_return_bps"] > 0).mean(),
        "Long Leg PnL Hit Rate": (ls_df["long_leg_pnl_bps"] > 0).mean(),
        "Short Leg PnL Hit Rate": (ls_df["short_leg_pnl_bps"] > 0).mean(),
        "Total PnL Hit Rate": (ls_df["total_pnl_bps"] > 0).mean(),
    }

    return {
        "long_short_leg_pnl_summary": pd.Series(summary),
        "long_short_leg_pnl_timeseries": ls_df,
    }


def run_qa6_capacity_units(
    signal_diagnostics_panel: pd.DataFrame,
    valid_dates_beta: pd.DatetimeIndex,
) -> dict:
    """QA 6: Capacity calculation unit audit by ticker."""
    logger.info("Downloading daily price and volume data from yfinance for Capacity unit check...")
    yf_data = yf.download(JP_TICKERS, start=valid_dates_beta.min().strftime("%Y-%m-%d"), end=valid_dates_beta.max().strftime("%Y-%m-%d"), auto_adjust=False)
    
    volume_df = yf_data["Volume"].reindex(valid_dates_beta).ffill()
    close_df = yf_data["Close"].reindex(valid_dates_beta).ffill()

    # Load Quote width spreads
    spread_path = "/Users/takahashimasatoshi/Library/Mobile Documents/com~apple~CloudDocs/個別株/日米ラグ_2.1/results/sector_relative_ensemble_execution_cost/quote_width_by_ticker.csv"
    if os.path.exists(spread_path):
        spread_df = pd.read_csv(spread_path)
        spread_df["trade_date"] = pd.to_datetime(spread_df["trade_date"]).dt.normalize()
        spread_df = spread_df.set_index("trade_date").reindex(valid_dates_beta).ffill()
    else:
        spread_df = pd.DataFrame(0.0010, index=valid_dates_beta, columns=JP_TICKERS)

    adtv_daily = volume_df * close_df
    adv_rolling = adtv_daily.rolling(20).mean().replace(0.0, np.nan).replace([np.inf, -np.inf], np.nan)

    w_ruled_df = signal_diagnostics_panel["weight_ruled"]
    trade_notional_daily = w_ruled_df.diff().abs().multiply(100000000.0, axis=0) # 100M AUM JPY
    trade_notional_daily.iloc[0] = w_ruled_df.iloc[0].abs().multiply(100000000.0)

    ticker_rows = []
    for tk in JP_TICKERS:
        med_price = close_df[tk].median()
        med_volume = volume_df[tk].median()
        med_adv_calc = adv_rolling[tk].median()
        med_adv_true = adtv_daily[tk].median()
        med_abs_w = w_ruled_df[tk].abs().median()
        
        # trade sizes and ratios
        med_trade_val = trade_notional_daily[tk].median()
        ratio_daily = trade_notional_daily[tk].divide(adv_rolling[tk]).dropna()
        
        med_ratio = ratio_daily.median()
        p95_ratio = ratio_daily.quantile(0.95)
        
        # Missing/Zero ADV days
        missing_days = int(volume_df[tk].isna().sum() + (volume_df[tk] == 0).sum())

        ticker_rows.append({
            "Ticker": tk,
            "Median Price": med_price,
            "Median Volume (shares)": med_volume,
            "Median ADV (JPY)": med_adv_true,
            "Median Rolling ADV (JPY)": med_adv_calc,
            "Median Abs Weight": med_abs_w,
            "Median Trade size (JPY @ 100M)": med_trade_val,
            "Median Trade/ADV Ratio": med_ratio,
            "p95 Trade/ADV Ratio": p95_ratio,
            "Missing/Zero ADV Days": missing_days,
        })

    ticker_df = pd.DataFrame(ticker_rows).set_index("Ticker")
    return {"ticker_capacity_audit": ticker_df}


def run_qa7_cost_consistency(
    base_results: dict,
    signal_diagnostics_panel: pd.DataFrame,
    residual_returns_panel: pd.DataFrame,
    valid_dates_beta: pd.DatetimeIndex,
) -> dict:
    """QA 7: Verify cost vs capacity diagnostics consistency."""
    # Decompose daily costs for different AUM scenarios
    aum_scenarios = [100000000, 1000000000, 10000000000] # 100M, 1B, 10B JPY
    
    # Reload ADV
    yf_data = yf.download(JP_TICKERS, start=valid_dates_beta.min().strftime("%Y-%m-%d"), end=valid_dates_beta.max().strftime("%Y-%m-%d"), auto_adjust=False)
    volume_df = yf_data["Volume"].reindex(valid_dates_beta).ffill()
    close_df = yf_data["Close"].reindex(valid_dates_beta).ffill()
    adtv_daily = volume_df * close_df
    adv_rolling = adtv_daily.rolling(20).mean().replace(0.0, np.nan).replace([np.inf, -np.inf], np.nan)

    # Load Quote width spreads
    spread_path = "/Users/takahashimasatoshi/Library/Mobile Documents/com~apple~CloudDocs/個別株/日米ラグ_2.1/results/sector_relative_ensemble_execution_cost/quote_width_by_ticker.csv"
    if os.path.exists(spread_path):
        spread_df = pd.read_csv(spread_path)
        spread_df["trade_date"] = pd.to_datetime(spread_df["trade_date"]).dt.normalize()
        spread_df = spread_df.set_index("trade_date").reindex(valid_dates_beta).ffill()
    else:
        spread_df = pd.DataFrame(0.0010, index=valid_dates_beta, columns=JP_TICKERS)

    w_ruled_df = signal_diagnostics_panel["weight_ruled"].reindex(valid_dates_beta).fillna(0.0)
    strategy_returns = base_results["beta_exposure_timeseries"]["strategy_return"]
    pnl_gross = strategy_returns.reindex(valid_dates_beta)

    vol_daily = residual_returns_panel["y_res_cc"].rolling(20).std().loc[valid_dates_beta].fillna(0.01).values

    decomp_rows = []
    for aum in aum_scenarios:
        # daily trade value
        trade_notional_daily = w_ruled_df.diff().abs().multiply(aum, axis=0)
        trade_notional_daily.iloc[0] = w_ruled_df.iloc[0].abs().multiply(aum)

        ratio_daily = trade_notional_daily.divide(adv_rolling.loc[valid_dates_beta], axis=0).fillna(0.0)
        
        ratio_vals = ratio_daily.values
        w_vals = w_ruled_df.values
        w_diff_vals = w_ruled_df.diff().abs().fillna(0.0).values
        spread_vals = spread_df.fillna(0.0010).values

        # 1. Spread cost only
        spread_cost_daily = np.sum(0.5 * spread_vals * w_diff_vals, axis=1) # in decimals of return
        pnl_spread = pnl_gross - spread_cost_daily
        ir_spread = (pnl_spread.mean() / pnl_spread.std() * np.sqrt(252)) if pnl_spread.std() > 0 else 0.0

        # 2. Impact cost only
        mi_cost_daily = np.sum(0.1 * vol_daily * np.sqrt(ratio_vals) * np.abs(w_vals), axis=1)
        pnl_impact = pnl_gross - mi_cost_daily
        ir_impact = (pnl_impact.mean() / pnl_impact.std() * np.sqrt(252)) if pnl_impact.std() > 0 else 0.0

        # 3. Spread + Impact
        total_costs = spread_cost_daily + mi_cost_daily
        pnl_total = pnl_gross - total_costs
        ir_total = (pnl_total.mean() / pnl_total.std() * np.sqrt(252)) if pnl_total.std() > 0 else 0.0

        decomp_rows.append({
            "AUM": aum,
            "Gross IR": (pnl_gross.mean() / pnl_gross.std() * np.sqrt(252)),
            "Spread IR": ir_spread,
            "Impact IR": ir_impact,
            "Combined IR": ir_total,
            "Average Daily Cost (bps)": total_costs.mean() * 10000,
            "p95 Daily Cost (bps)": pd.Series(total_costs).quantile(0.95) * 10000,
        })

    decomp_df = pd.DataFrame(decomp_rows).set_index("AUM")
    return {"cost_capacity_reconciliation": decomp_df}


def run_qa8_calibration_leak(
    base_results: dict,
    valid_dates_beta: pd.DatetimeIndex,
) -> dict:
    """QA 8: Validate ex-ante IR calibration leak (compare full-sample vs rolling 252d quantile split)."""
    diag_file = find_latest_distribution_diagnostics()
    if not diag_file:
        return {"calibration_leak_comparison": pd.DataFrame()}

    diag_df = pd.read_csv(diag_file)
    diag_df["trade_date"] = pd.to_datetime(diag_df["trade_date"]).dt.normalize()
    diag_df = diag_df.set_index("trade_date")

    ex_ante_ir = pd.Series(np.nan, index=valid_dates_beta)
    for dt in valid_dates_beta:
        if dt in diag_df.index:
            ir_val = diag_df.loc[dt, "pred_ir_gap_exante_cost"]
            if isinstance(ir_val, pd.Series):
                ir_val = ir_val.iloc[0]
            ex_ante_ir[dt] = float(ir_val)

    strategy_returns = base_results["beta_exposure_timeseries"]["strategy_return"].loc[valid_dates_beta]

    calib_data = pd.DataFrame({
        "ex_ante_ir": ex_ante_ir,
        "realized_return": strategy_returns,
    }).dropna()

    # 1. Full sample quantile split
    calib_data["full_sample_tertile"] = pd.qcut(calib_data["ex_ante_ir"], 3, labels=["Low", "Medium", "High"])

    # 2. Rolling 252d quantile split
    rolling_tertiles = []
    for idx, dt in enumerate(calib_data.index):
        if idx < 252:
            rolling_tertiles.append("Medium") # default classification for initial window
            continue
        
        # Lookback window
        window_dates = calib_data.index[idx-252:idx]
        window_ir = calib_data.loc[window_dates, "ex_ante_ir"]
        
        q33 = window_ir.quantile(0.333)
        q66 = window_ir.quantile(0.667)
        
        val = calib_data.loc[dt, "ex_ante_ir"]
        if val <= q33:
            rolling_tertiles.append("Low")
        elif val <= q66:
            rolling_tertiles.append("Medium")
        else:
            rolling_tertiles.append("High")

    calib_data["rolling_tertile"] = rolling_tertiles

    comparison_rows = []
    # Statistics for both splits
    for method, col in [("Full Sample Quantile Split", "full_sample_tertile"), ("Rolling 252d Quantile Split", "rolling_tertile")]:
        for tertile in ["Low", "Medium", "High"]:
            sub = calib_data[calib_data[col] == tertile]
            if len(sub) > 0:
                ret_ann = sub["realized_return"].mean() * 252
                vol_ann = sub["realized_return"].std() * np.sqrt(252)
                ir_real = (ret_ann / vol_ann) if vol_ann > 0 else 0.0
                corr_val = sub["ex_ante_ir"].corr(sub["realized_return"])
            else:
                ret_ann, vol_ann, ir_real, corr_val = np.nan, np.nan, np.nan, np.nan

            comparison_rows.append({
                "Method": method,
                "Tertile": tertile,
                "Mean Return (Ann)": ret_ann,
                "Realized Vol (Ann)": vol_ann,
                "Realized IR": ir_real,
                "PnL Correlation": corr_val,
                "Count": len(sub),
            })

    comparison_df = pd.DataFrame(comparison_rows).set_index(["Method", "Tertile"])
    return {"calibration_leak_comparison": comparison_df}
