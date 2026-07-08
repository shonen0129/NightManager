"""src/diagnostics/sprint0.py — Sprint 0 core diagnostics math and data processing.

Calculates target return mismatch, cross-sectional ICs, TOPIX beta exposure,
long-short contribution, liquidity ADV, cost scenarios, and capacity limits.
"""

from __future__ import annotations

import logging
import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf

from leadlag.data.cache import load_df_exec_from_local_cache, load_intraday_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS, TOPIX_TICKER
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)


def find_latest_distribution_diagnostics() -> str | None:
    """Find the path of the latest portfolio_gap_distribution_diagnostics.csv in results."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "gap_adjusted_distribution")
    if not os.path.exists(base_dir):
        return None
    subdirs = glob.glob(os.path.join(base_dir, "202*"))
    if not subdirs:
        return None
    latest_subdir = max(subdirs)
    diag_file = os.path.join(latest_subdir, "portfolio_gap_distribution_diagnostics.csv")
    if os.path.exists(diag_file):
        return diag_file
    return None


def compute_rolling_beta(r_asset: pd.DataFrame, r_market: pd.Series, window: int) -> pd.DataFrame:
    """Compute lookahead-safe rolling betas of assets vs market (shifted by 1 day)."""
    cov = r_asset.rolling(window).cov(r_market)
    var = r_market.rolling(window).var().clip(lower=1e-8)
    beta = cov.divide(var, axis=0)
    return beta.shift(1)


def run_sprint0_calculations(
    start_date: str | None = None,
    end_date: str | None = None,
    config: dict | None = None,
) -> dict:
    """Run all calculations for Sprint 0 diagnostics pipeline.

    Args:
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        config: Configuration dictionary.

    Returns:
        Dict of results and dataframes.
    """
    if config is None:
        config = {}

    # 1. Load aligned execution data
    df_exec_full = load_df_exec_from_local_cache()
    df_exec = df_exec_full.copy()
    sim_dates = df_exec.index

    # Define target analysis dates
    analysis_dates = sim_dates
    if start_date:
        analysis_dates = analysis_dates[analysis_dates >= pd.to_datetime(start_date)]
    if end_date:
        analysis_dates = analysis_dates[analysis_dates <= pd.to_datetime(end_date)]

    logger.info("Running diagnostics: full history has %d trading days, analysis window has %d trading days (%s to %s)",
                len(df_exec), len(analysis_dates), analysis_dates.min().date(), analysis_dates.max().date())

    # 2. Return definitions & pricing identity
    r_cc = pd.DataFrame(index=sim_dates, columns=JP_TICKERS, dtype=float)
    r_intraday = pd.DataFrame(index=sim_dates, columns=JP_TICKERS, dtype=float)
    gap = pd.DataFrame(index=sim_dates, columns=JP_TICKERS, dtype=float)

    # Load 5m cache to identify days where 9:10 prices exist
    df_5m = load_intraday_cache("5m")
    has_5m_dates = set()
    if df_5m is not None and not df_5m.empty:
        has_5m_dates = set(pd.Series(df_5m.index.date).unique())

    # Compute actual 9:10 target return series (y_jp_target)
    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    y_jp_target_df = pd.DataFrame(y_jp_target, index=sim_dates, columns=JP_TICKERS)

    fallback_count = 0
    actual_count = 0

    for tk in JP_TICKERS:
        # Close-to-Close: (1 + jp_gap) * (1 + jp_oc) - 1
        r_cc[tk] = (1.0 + df_exec[f"jp_gap_{tk}"]) * (1.0 + df_exec[f"jp_oc_{tk}"]) - 1.0
        # 9:10 to Close
        r_intraday[tk] = y_jp_target_df[tk]
        # Gap Open: (1 + r_cc) / (1 + r_intraday) - 1
        gap[tk] = (1.0 + r_cc[tk]) / (1.0 + r_intraday[tk]) - 1.0

    # Sanitize infinities and NaNs in returns
    r_cc = r_cc.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    r_intraday = r_intraday.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    gap = gap.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    # Count actual vs fallback dates over analysis window
    for dt in analysis_dates:
        if dt.date() in has_5m_dates:
            actual_count += 1
        else:
            fallback_count += 1

    # pricing identity deviation
    pricing_dev = (1.0 + r_cc.loc[analysis_dates]) - (1.0 + gap.loc[analysis_dates]) * (1.0 + r_intraday.loc[analysis_dates])
    max_dev = float(pricing_dev.abs().max().max())
    mean_dev = float(pricing_dev.abs().mean().mean())
    logger.info("Pricing identity absolute deviation check (analysis window): Max=%.2e, Mean=%.2e", max_dev, mean_dev)
    if max_dev > 1e-10:
        logger.warning("Pricing identity deviation detected: maximum absolute deviation = %.2e", max_dev)

    # 3. Residual returns & OLS Betas
    r_topix_cc = df_exec["topix_cc_trade"]
    # Fallback to topix_oc_return as proxy for TOPIX 9:10→Close return
    r_topix_intraday = df_exec["topix_oc_return"] 

    betas_60_cc = compute_rolling_beta(r_cc, r_topix_cc, 60)
    betas_120_cc = compute_rolling_beta(r_cc, r_topix_cc, 120)

    betas_60_intraday = compute_rolling_beta(r_intraday, r_topix_intraday, 60)
    betas_120_intraday = compute_rolling_beta(r_intraday, r_topix_intraday, 120)

    y_res_cc_60 = (r_cc - betas_60_cc.multiply(r_topix_cc, axis=0)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_res_cc_120 = (r_cc - betas_120_cc.multiply(r_topix_cc, axis=0)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    y_res_intraday_60 = (r_intraday - betas_60_intraday.multiply(r_topix_intraday, axis=0)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_res_intraday_120 = (r_intraday - betas_120_intraday.multiply(r_topix_intraday, axis=0)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    # Filter out early dates that lack sufficient history for rolling betas (first 120 days)
    # and restrict to the analysis dates window
    valid_dates_beta = analysis_dates.intersection(sim_dates[120:])

    # 4. Target Mismatch Diagnostics
    # Global correlations (analysis window)
    corr_cc_intraday_by_ticker = pd.Series({tk: r_cc.loc[analysis_dates, tk].corr(r_intraday.loc[analysis_dates, tk]) for tk in JP_TICKERS})
    corr_res_by_ticker = pd.Series({
        tk: y_res_cc_60.loc[valid_dates_beta, tk].corr(y_res_intraday_60.loc[valid_dates_beta, tk]) 
        for tk in JP_TICKERS
    })

    # Rolling correlations (cross-sectional daily correlations or average rolling correlation)
    # We compute the average of rolling 60-day correlation across all tickers
    rolling_corr_df = pd.DataFrame(index=sim_dates, columns=JP_TICKERS)
    for tk in JP_TICKERS:
        rolling_corr_df[tk] = r_cc[tk].rolling(60).corr(r_intraday[tk])
    avg_rolling_corr = rolling_corr_df.loc[analysis_dates].mean(axis=1)

    # Year-by-year correlation
    years = analysis_dates.year.unique()
    yearly_corr = {}
    for yr in years:
        mask_yr = analysis_dates.year == yr
        dates_yr = analysis_dates[mask_yr]
        r_cc_yr = r_cc.loc[dates_yr]
        r_intra_yr = r_intraday.loc[dates_yr]
        corr_val = np.mean([r_cc_yr[tk].corr(r_intra_yr[tk]) for tk in JP_TICKERS if len(r_cc_yr) > 1])
        yearly_corr[yr] = corr_val
    yearly_corr_df = pd.Series(yearly_corr, name="average_ticker_correlation")

    # Regime-wise correlation (VIX & USDJPY)
    macro_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "market_data", "macro_data.pkl")
    macro_df = None
    regime_results = {}
    if os.path.exists(macro_path):
        macro_df = pd.read_pickle(macro_path)
        macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
        macro_df = macro_df.reindex(sim_dates).ffill()

        vix = macro_df.loc[analysis_dates, "^VIX"]
        usdjpy = macro_df.loc[analysis_dates, "USDJPY=X"]
        
        vix_med = vix.median()
        usdjpy_med = usdjpy.median()

        regime_masks = {
            "vix_high": vix > vix_med,
            "vix_low": vix <= vix_med,
            "usdjpy_weak_yen": usdjpy > usdjpy_med,
            "usdjpy_strong_yen": usdjpy <= usdjpy_med
        }

        for r_name, mask in regime_masks.items():
            r_cc_r = r_cc.loc[analysis_dates][mask]
            r_intra_r = r_intraday.loc[analysis_dates][mask]
            if len(r_cc_r) > 5:
                corr_val = np.mean([r_cc_r[tk].corr(r_intra_r[tk]) for tk in JP_TICKERS])
                beta_regime_dates = r_cc_r.index.intersection(valid_dates_beta)
                if len(beta_regime_dates) > 5:
                    res_corr = np.mean([
                        y_res_cc_60.loc[beta_regime_dates, tk].corr(
                            y_res_intraday_60.loc[beta_regime_dates, tk]
                        )
                        for tk in JP_TICKERS
                    ])
                else:
                    res_corr = np.nan
                regime_results[r_name] = {"raw_corr": corr_val, "residual_corr": res_corr}
            else:
                regime_results[r_name] = {"raw_corr": np.nan, "residual_corr": np.nan}

    # Variance decomposition
    # Log variance decomposition: r_cc_log = gap_log + intraday_log
    # Since returns are small, we can approximate: var(r_cc) = var(gap) + var(r_intraday) + 2*cov(gap, r_intraday)
    var_decomp = []
    for tk in JP_TICKERS:
        v_cc = r_cc.loc[analysis_dates, tk].var()
        v_gap = gap.loc[analysis_dates, tk].var()
        v_intra = r_intraday.loc[analysis_dates, tk].var()
        cov_val = r_cc.loc[analysis_dates, tk].cov(gap.loc[analysis_dates, tk])
        cov_intra = r_cc.loc[analysis_dates, tk].cov(r_intraday.loc[analysis_dates, tk])
        var_decomp.append({
            "ticker": tk,
            "var_cc": v_cc,
            "var_gap": v_gap,
            "var_intraday": v_intra,
            "prop_explained_by_gap": cov_val / v_cc if v_cc > 0 else 0,
            "prop_explained_by_intraday": cov_intra / v_cc if v_cc > 0 else 0,
        })
    var_decomp_df = pd.DataFrame(var_decomp).set_index("ticker")

    # 5. Generate Signals (with and without gap adjustment)
    from pathlib import Path
    prod_config_path = Path(__file__).resolve().parents[3] / "configs" / "production" / "production.yaml"
    import yaml
    with open(prod_config_path) as f:
        prod_config = yaml.safe_load(f)

    # Standard model (with gap adjustment)
    model_gap = SectorRelativeEnsembleBLPEnhancedModel(prod_config)
    pred_gap = model_gap.predict_signals(df_exec)
    signals_gap = pred_gap["signals"]  # post-gap adjustment combined signal
    residual_blpx_signals = pred_gap["residual_blpx_signals"]

    # Model without gap adjustment
    prod_config_no_gap = prod_config.copy()
    prod_config_no_gap["blpx"]["gap_open_coef"] = 0.0
    model_no_gap = SectorRelativeEnsembleBLPEnhancedModel(prod_config_no_gap)
    pred_no_gap = model_no_gap.predict_signals(df_exec)
    signals_no_gap = pred_no_gap["signals"]
    residual_blpx_no_gap_signals = pred_no_gap["residual_blpx_signals"]

    # 6. Signal IC Analysis
    # We evaluate IC over valid_dates_beta
    ic_results = []
    
    # We calculate daily Rank IC (Spearman)
    for dt in valid_dates_beta:
        y_cc_t = y_res_cc_60.loc[dt].values
        y_intra_t = y_res_intraday_60.loc[dt].values
        
        # Signals
        sig_gap_t = residual_blpx_signals.loc[dt].values
        sig_nogap_t = residual_blpx_no_gap_signals.loc[dt].values

        if np.isnan(y_cc_t).any() or np.isnan(y_intra_t).any():
            continue

        # Rank IC (Spearman)
        spearman_gap_cc, _ = stats.spearmanr(sig_gap_t, y_cc_t)
        spearman_gap_intra, _ = stats.spearmanr(sig_gap_t, y_intra_t)
        spearman_nogap_cc, _ = stats.spearmanr(sig_nogap_t, y_cc_t)
        spearman_nogap_intra, _ = stats.spearmanr(sig_nogap_t, y_intra_t)

        # Pearson IC
        pearson_gap_cc, _ = stats.pearsonr(sig_gap_t, y_cc_t)
        pearson_gap_intra, _ = stats.pearsonr(sig_gap_t, y_intra_t)
        pearson_nogap_cc, _ = stats.pearsonr(sig_nogap_t, y_cc_t)
        pearson_nogap_intra, _ = stats.pearsonr(sig_nogap_t, y_intra_t)

        ic_results.append({
            "trade_date": dt,
            "ic_gap_cc": pearson_gap_cc,
            "ic_gap_intra": pearson_gap_intra,
            "ic_nogap_cc": pearson_nogap_cc,
            "ic_nogap_intra": pearson_nogap_intra,
            "rank_ic_gap_cc": spearman_gap_cc,
            "rank_ic_gap_intra": spearman_gap_intra,
            "rank_ic_nogap_cc": spearman_nogap_cc,
            "rank_ic_nogap_intra": spearman_nogap_intra,
        })
    ic_df = pd.DataFrame(ic_results).set_index("trade_date")

    # IC summary
    ic_cols = ["ic_gap_cc", "ic_gap_intra", "ic_nogap_cc", "ic_nogap_intra", "rank_ic_gap_cc", "rank_ic_gap_intra", "rank_ic_nogap_cc", "rank_ic_nogap_intra"]
    ic_summary = {}
    for col in ic_cols:
        series = ic_df[col].dropna()
        mean_val = series.mean()
        std_val = series.std()
        icir = mean_val / std_val * np.sqrt(252) if std_val > 0 else 0
        hit_rate = (series > 0).mean()
        ic_summary[col] = {
            "mean": mean_val,
            "std": std_val,
            "icir": icir,
            "hit_rate": hit_rate
        }
    ic_summary_df = pd.DataFrame(ic_summary).T

    # 7. Score Quantile Returns
    # We rank tickers daily based on signal and look at average realized returns (y_res_cc, y_res_intraday, raw r_intraday)
    # We check 3 quantiles: Q1 (bottom 30%), Q2 (middle 40%), Q3 (top 30%)
    quantile_returns = []
    
    # We compute the quantile returns for standard gap-adjusted signals
    for dt in valid_dates_beta:
        sig_t = residual_blpx_signals.loc[dt]
        y_cc_t = y_res_cc_60.loc[dt]
        y_intra_t = y_res_intraday_60.loc[dt]
        y_raw_t = r_intraday.loc[dt]

        # Ranks
        ranks = sig_t.rank(method="first")
        n_assets = len(sig_t)
        
        # Lower 30% (indices 1 to floor(n_assets*0.3))
        q1_mask = ranks <= np.floor(n_assets * 0.3)
        # Upper 30% (indices ceiling(n_assets*0.7) to n_assets)
        q3_mask = ranks > np.ceil(n_assets * 0.7)
        # Middle
        q2_mask = ~(q1_mask | q3_mask)

        quantile_returns.append({
            "trade_date": dt,
            "q1_cc": y_cc_t[q1_mask].mean(),
            "q2_cc": y_cc_t[q2_mask].mean(),
            "q3_cc": y_cc_t[q3_mask].mean(),
            "q1_intra": y_intra_t[q1_mask].mean(),
            "q2_intra": y_intra_t[q2_mask].mean(),
            "q3_intra": y_intra_t[q3_mask].mean(),
            "q1_raw": y_raw_t[q1_mask].mean(),
            "q2_raw": y_raw_t[q2_mask].mean(),
            "q3_raw": y_raw_t[q3_mask].mean(),
        })
    q_df = pd.DataFrame(quantile_returns).set_index("trade_date")
    q_summary = q_df.mean()

    # 8. Portfolio Weights & TOPIX Beta Exposure & Long-Short PnL
    # Run backtester for standard model
    # Wait, SRE config has blpx settings. We can build weights using model_gap.build_weights
    w_baseline = np.zeros((len(valid_dates_beta), model_gap.n_j))
    for idx, dt in enumerate(valid_dates_beta):
        sig_t = residual_blpx_signals.loc[dt].values
        w_baseline[idx] = model_gap.build_weights(sig_t)
    w_baseline_df = pd.DataFrame(w_baseline, index=valid_dates_beta, columns=JP_TICKERS)

    # Match RuleD multipliers if portfolio_gap_distribution_diagnostics.csv exists
    diag_file = find_latest_distribution_diagnostics()
    multipliers = pd.Series(1.0, index=valid_dates_beta)
    ex_ante_ir = pd.Series(np.nan, index=valid_dates_beta)
    pit_bins = pd.Series("Medium", index=valid_dates_beta)
    if diag_file:
        logger.info("Found gap distribution diagnostics file: %s", diag_file)
        diag_df = pd.read_csv(diag_file)
        diag_df["trade_date"] = pd.to_datetime(diag_df["trade_date"]).dt.normalize()
        diag_df = diag_df.set_index("trade_date")
        
        # Join multipliers
        for dt in valid_dates_beta:
            if dt in diag_df.index:
                # gross_exposure is 2.0 * multiplier
                gross_exp = diag_df.loc[dt, "gross_exposure"]
                if isinstance(gross_exp, pd.Series):
                    gross_exp = gross_exp.iloc[0]
                multipliers[dt] = float(gross_exp) / 2.0
                
                # Ex-ante IR
                ir_val = diag_df.loc[dt, "pred_ir_gap_exante_cost"]
                if isinstance(ir_val, pd.Series):
                    ir_val = ir_val.iloc[0]
                ex_ante_ir[dt] = float(ir_val)

                # PIT binning
                bin_val = diag_df.loc[dt, "pit_bin"] if "pit_bin" in diag_df.columns else "Medium"
                if isinstance(bin_val, pd.Series):
                    bin_val = bin_val.iloc[0]
                pit_bins[dt] = bin_val
    else:
        logger.warning("portfolio_gap_distribution_diagnostics.csv not found. Defaulting RuleD gross multiplier to 1.0 (no scaling).")

    # RuleD weights
    w_ruled_df = w_baseline_df.multiply(multipliers, axis=0)

    # Beta exposures
    # We use betas_60_intraday[j, t-1] since it is lookahead-free and matches trade target return residualization
    beta_exp_series = pd.Series(index=valid_dates_beta, dtype=float)
    gross_series = pd.Series(index=valid_dates_beta, dtype=float)
    net_series = pd.Series(index=valid_dates_beta, dtype=float)

    for dt in valid_dates_beta:
        w_t = w_ruled_df.loc[dt].values
        b_t = betas_60_intraday.loc[dt].values
        beta_exp_series[dt] = np.sum(w_t * b_t)
        gross_series[dt] = np.sum(np.abs(w_t))
        net_series[dt] = np.sum(w_t)

    # Realized target returns
    r_target_panel = r_intraday.loc[valid_dates_beta].values
    strategy_returns = w_ruled_df.multiply(r_intraday.loc[valid_dates_beta]).sum(axis=1)

    # TOPIX beta exposure statistics
    beta_exp_mean = beta_exp_series.mean()
    beta_exp_std = beta_exp_series.std()
    beta_exp_max = beta_exp_series.max()
    beta_exp_min = beta_exp_series.min()
    
    # Correlation with strategy PnL
    corr_beta_pnl = beta_exp_series.corr(strategy_returns)
    # Relation with TOPIX return (correlation between beta_exposure and TOPIX return)
    corr_beta_topix = beta_exp_series.corr(r_topix_intraday.loc[valid_dates_beta])

    # 9. Long/Short contribution decomposition
    long_pnl_decomp = pd.DataFrame(index=valid_dates_beta, columns=["long_pnl", "short_pnl", "long_gross", "short_gross"])
    for dt in valid_dates_beta:
        w_t = w_ruled_df.loc[dt].values
        r_t = r_target_panel[valid_dates_beta.get_loc(dt)]
        
        w_long = np.maximum(w_t, 0)
        w_short = np.minimum(w_t, 0)
        
        long_pnl = np.sum(w_long * r_t)
        short_pnl = np.sum(w_short * r_t)
        
        long_pnl_decomp.loc[dt] = {
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_gross": np.sum(w_long),
            "short_gross": np.sum(np.abs(w_short))
        }

    long_mean_pnl = long_pnl_decomp["long_pnl"].mean()
    short_mean_pnl = long_pnl_decomp["short_pnl"].mean()
    long_hit_rate = (long_pnl_decomp["long_pnl"] > 0).mean()
    short_hit_rate = (long_pnl_decomp["short_pnl"] > 0).mean()

    # Cumulative contribution by ticker
    ticker_contribution = w_ruled_df.multiply(r_intraday.loc[valid_dates_beta]).cumsum()

    # 10. Liquidity & ADV Diagnostics
    # Download Volume from yfinance
    logger.info("Downloading daily trading Volume and Close prices from yfinance for liquidity diagnostics...")
    start_yf = sim_dates.min().strftime("%Y-%m-%d")
    yf_data = yf.download(JP_TICKERS, start=start_yf, end=sim_dates.max().strftime("%Y-%m-%d"), auto_adjust=False)
    
    volume_df = yf_data["Volume"].reindex(sim_dates).ffill()
    close_df = yf_data["Close"].reindex(sim_dates).ffill()
    
    # ADV (rolling 20-day mean of traded value in JPY)
    adtv_daily = volume_df * close_df
    adv_rolling = adtv_daily.rolling(20).mean()
    # Replace zero and infinities with NaN to prevent division by zero in capacity ratios
    adv_rolling = adv_rolling.replace(0.0, np.nan).replace([np.inf, -np.inf], np.nan)

    # Load Quote width spreads
    spread_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "sector_relative_ensemble_execution_cost", "quote_width_by_ticker.csv")
    if os.path.exists(spread_path):
        spread_df = pd.read_csv(spread_path)
        spread_df["trade_date"] = pd.to_datetime(spread_df["trade_date"]).dt.normalize()
        spread_df = spread_df.set_index("trade_date").reindex(sim_dates).ffill()
    else:
        # If file missing, fallback to 10bps spread
        spread_df = pd.DataFrame(0.0010, index=sim_dates, columns=JP_TICKERS)
        logger.warning("quote_width_by_ticker.csv not found. Defaulting daily spreads to 10bps.")

    # Liquidity stats by ticker
    liquidity_stats = []
    for tk in JP_TICKERS:
        mean_vol = volume_df[tk].mean()
        mean_value = adtv_daily[tk].mean()
        median_value = adtv_daily[tk].median()
        mean_spread = spread_df[tk].mean()
        
        liquidity_stats.append({
            "ticker": tk,
            "mean_volume": mean_vol,
            "mean_adv_jpy": mean_value,
            "median_adv_jpy": median_value,
            "mean_spread_bps": mean_spread * 10000,
        })
    liquidity_summary_df = pd.DataFrame(liquidity_stats).set_index("ticker")

    # 11. Transaction Cost Scenarios
    # Scenarios: low (5bp round-trip), base (15bp round-trip), high (30bp round-trip)
    # Round-trip cost is applied to daily turnover
    turnover = w_ruled_df.diff().abs().sum(axis=1) / 2.0
    # First day has turnover equal to half the gross exposure
    turnover.iloc[0] = w_ruled_df.iloc[0].abs().sum() / 2.0

    pnl_gross = strategy_returns
    costs_low = turnover * (5.0 / 10000.0)
    costs_base = turnover * (15.0 / 10000.0)
    costs_high = turnover * (30.0 / 10000.0)

    pnl_net_low = pnl_gross - costs_low
    pnl_net_base = pnl_gross - costs_base
    pnl_net_high = pnl_gross - costs_high

    cost_comparison = {
        "gross": {"pnl_mean": pnl_gross.mean() * 252, "sharpe": pnl_gross.mean() / pnl_gross.std() * np.sqrt(252)},
        "low": {"pnl_mean": pnl_net_low.mean() * 252, "sharpe": pnl_net_low.mean() / pnl_net_low.std() * np.sqrt(252)},
        "base": {"pnl_mean": pnl_net_base.mean() * 252, "sharpe": pnl_net_base.mean() / pnl_net_base.std() * np.sqrt(252)},
        "high": {"pnl_mean": pnl_net_high.mean() * 252, "sharpe": pnl_net_high.mean() / pnl_net_high.std() * np.sqrt(252)}
    }
    cost_comparison_df = pd.DataFrame(cost_comparison).T

    # 12. Capacity Diagnostics by AUM
    aum_scenarios = [100000000, 500000000, 1000000000, 3000000000, 5000000000, 10000000000]
    capacity_summary = []
    
    # We calculate capacity metrics for each AUM
    for aum in aum_scenarios:
        # daily trade value in JPY for each ticker
        trade_notional_daily = w_ruled_df.diff().abs().multiply(aum, axis=0)
        # First day initial trade value
        trade_notional_daily.iloc[0] = w_ruled_df.iloc[0].abs().multiply(aum)
        
        # ADV ratio: trade_value / ADV
        ratio_daily = trade_notional_daily.divide(adv_rolling.loc[valid_dates_beta], axis=0)
        # Drop days where all columns are NaN
        ratio_daily = ratio_daily.dropna(how="all")

        max_ratio_by_ticker = ratio_daily.max()
        mean_ratio_by_ticker = ratio_daily.mean()
        p95_ratio_by_ticker = ratio_daily.quantile(0.95)

        # Count warning and critical warning days (any ticker ratio > threshold)
        warning_days = (ratio_daily > 0.05).any(axis=1).sum()
        critical_days = (ratio_daily > 0.10).any(axis=1).sum()

        # Net returns with market impact and spread cost
        # Market impact: eta * sigma * sqrt(trade_value / ADV) * abs(w)
        # For simplicity, we use eta = 0.1 and rolling JP volatility.
        # If ADV is missing, ratio_daily has NaNs. Let's fill NaNs in ratio with 0.0
        ratio_vals = ratio_daily.reindex(valid_dates_beta).fillna(0.0).values
        w_vals = w_ruled_df.reindex(valid_dates_beta).values
        
        # Calculate daily market impact cost
        # We use a standard rolling std of returns as proxy for volatility (annual std / sqrt(252))
        vol_daily = y_res_cc_60.rolling(20).std().reindex(valid_dates_beta).fillna(0.01).values
        # market impact = sum_j (0.1 * vol_j_t * sqrt(trade_value_j_t / ADV_j_t) * abs(w_j_t))
        mi_cost_daily = np.sum(0.1 * vol_daily * np.sqrt(ratio_vals) * np.abs(w_vals), axis=1)
        # spread cost = sum_j (0.5 * spread_j_t * abs_trade_w_j_t)
        # Traded weight is abs(w_t - w_t-1)
        w_diff_vals = w_ruled_df.diff().abs().reindex(valid_dates_beta).fillna(0.0).values
        spread_vals = spread_df.reindex(valid_dates_beta).fillna(0.0010).values
        spread_cost_daily = np.sum(0.5 * spread_vals * w_diff_vals, axis=1)

        total_costs_jpy = mi_cost_daily + spread_cost_daily
        pnl_net_aum = pnl_gross.reindex(valid_dates_beta) - total_costs_jpy
        
        net_mean = pnl_net_aum.mean()
        net_std = pnl_net_aum.std()
        net_ir = (net_mean / net_std * np.sqrt(252)) if net_std > 0 else 0.0

        capacity_summary.append({
            "AUM": aum,
            "max_ratio": max_ratio_by_ticker.max(),
            "mean_ratio": mean_ratio_by_ticker.mean(),
            "p95_ratio": p95_ratio_by_ticker.max(),
            "warning_days": int(warning_days),
            "critical_days": int(critical_days),
            "cost_adjusted_ir": net_ir,
        })
    capacity_summary_df = pd.DataFrame(capacity_summary).set_index("AUM")

    # 13. Ex-Ante vs Realized IR Calibration
    calibration_df = pd.DataFrame()
    if diag_file:
        calib_data = pd.DataFrame({
            "ex_ante_ir": ex_ante_ir,
            "realized_return": strategy_returns,
            "pit_bin": pit_bins,
        }).dropna()
        
        # Sort into tertiles by ex_ante_ir
        calib_data["ex_ante_tertile"] = pd.qcut(calib_data["ex_ante_ir"], 3, labels=["Low", "Medium", "High"])
        
        calib_summary = []
        for tertile in ["Low", "Medium", "High"]:
            sub = calib_data[calib_data["ex_ante_tertile"] == tertile]
            calib_summary.append({
                "ex_ante_bin": tertile,
                "mean_ex_ante_ir": sub["ex_ante_ir"].mean(),
                "realized_mean_return_ann": sub["realized_return"].mean() * 252,
                "realized_vol_ann": sub["realized_return"].std() * np.sqrt(252),
                "realized_ir": (sub["realized_return"].mean() / sub["realized_return"].std() * np.sqrt(252)) if sub["realized_return"].std() > 0 else 0.0,
                "count": len(sub)
            })
        calibration_df = pd.DataFrame(calib_summary).set_index("ex_ante_bin")
        
        # Also compute correlation of ex-ante IR with next-day PnL and max drawdown
        corr_ir_pnl = calib_data["ex_ante_ir"].corr(calib_data["realized_return"])
        # Next-day drawdown relation: we can find the maximum drawdown over next 5 days
        rolling_5d_mdd = calib_data["realized_return"].rolling(5).min()
        corr_ir_drawdown = calib_data["ex_ante_ir"].corr(rolling_5d_mdd)

        calibration_metrics = {
            "corr_ir_pnl": corr_ir_pnl,
            "corr_ir_drawdown": corr_ir_drawdown
        }
    else:
        calibration_metrics = {
            "corr_ir_pnl": np.nan,
            "corr_ir_drawdown": np.nan
        }

    # Gather data panels for Parquet files
    # Returns panel: daily r_cc, gap, r_intraday for each JP ETF
    returns_panel = pd.concat({
        "r_cc": r_cc,
        "gap": gap,
        "r_intraday": r_intraday
    }, axis=1).loc[analysis_dates]

    # Residual returns panel (60d)
    residual_returns_panel = pd.concat({
        "y_res_cc": y_res_cc_60,
        "y_res_intraday": y_res_intraday_60
    }, axis=1).loc[analysis_dates]

    # Signal diagnostics panel: daily signals, normalized signals, weights
    signal_diagnostics_panel = pd.concat({
        "signal_gap_adjusted": residual_blpx_signals,
        "signal_raw": residual_blpx_no_gap_signals,
        "weight_ruled": w_ruled_df,
    }, axis=1).loc[analysis_dates]

    return {
        # Panels
        "returns_panel": returns_panel,
        "residual_returns_panel": residual_returns_panel,
        "signal_diagnostics_panel": signal_diagnostics_panel,
        # DataFrames
        "ic_timeseries": ic_df,
        "ic_summary": ic_summary_df,
        "quantile_return_summary": q_df,
        "quantile_return_mean": q_summary,
        "beta_exposure_timeseries": pd.DataFrame({
            "beta_exposure": beta_exp_series,
            "gross_exposure": gross_series,
            "net_exposure": net_series,
            "strategy_return": strategy_returns,
        }),
        "beta_exposure_stats": {
            "mean": beta_exp_mean,
            "std": beta_exp_std,
            "max": beta_exp_max,
            "min": beta_exp_min,
            "corr_pnl": corr_beta_pnl,
            "corr_topix": corr_beta_topix,
        },
        "long_short_pnl_decomposition": long_pnl_decomp,
        "long_short_stats": {
            "long_mean": long_mean_pnl,
            "short_mean": short_mean_pnl,
            "long_hit_rate": long_hit_rate,
            "short_hit_rate": short_hit_rate,
        },
        "ticker_contribution": ticker_contribution,
        "liquidity_summary": liquidity_summary_df,
        "cost_impact_summary": cost_comparison_df,
        "capacity_summary": capacity_summary_df,
        "predicted_ir_calibration": calibration_df,
        "calibration_metrics": calibration_metrics,
        "data_availability": {
            "total_days": len(analysis_dates),
            "9_10_actual_days": actual_count,
            "9_10_fallback_days": fallback_count,
            "pct_9_10_available": actual_count / len(analysis_dates) if len(analysis_dates) > 0 else 0.0,
        },
        "regime_correlations": regime_results,
        "var_decomposition": var_decomp_df,
    }
