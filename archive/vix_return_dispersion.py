#!/usr/bin/env python
"""VIX-regime stratified return dispersion and realized PnL analysis.

IC is scale-invariant (Spearman rank). Actual PnL depends on:
  PnL ≈ IC × cross-sectional dispersion of returns

This script measures:
1. JP intraday residual return dispersion by VIX regime
2. Realized long-short spread by VIX regime
3. Annualized return / Sharpe by VIX regime
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)


def compute_rolling_beta(r_asset: pd.DataFrame, r_market: pd.Series, window: int) -> pd.DataFrame:
    cov = r_asset.rolling(window).cov(r_market)
    var = r_market.rolling(window).var().clip(lower=1e-8)
    beta = cov.divide(var, axis=0)
    return beta.shift(1)


def main():
    print("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    sim_dates = df_exec.index

    # Load config and run model
    config_path = os.path.join(ROOT, "configs", "production.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Running BLPX model...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    pred = model.predict_signals(df_exec)
    signals_df = pred["signals"]
    residual_blpx_signals = pred.get("residual_blpx_signals", signals_df)

    # JP intraday returns (open-to-close)
    jp_oc = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]]
    jp_oc.columns = JP_TICKERS

    # TOPIX for residualization
    topix_cc = df_exec["topix_night_return"].values + df_exec["topix_oc_return"].values
    topix_cc_series = pd.Series(topix_cc, index=sim_dates)

    beta_df = compute_rolling_beta(jp_oc, topix_cc_series, 60)
    y_res_intraday = jp_oc - beta_df.multiply(topix_cc_series, axis=0)
    y_res_intraday = y_res_intraday.dropna(how="all")

    # Also raw intraday (non-residualized) for reference
    jp_oc_clean = jp_oc.dropna(how="all")

    # Load VIX
    macro_path = os.path.join(ROOT, "market_data", "macro_data.pkl")
    macro_df = pd.read_pickle(macro_path)
    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
    vix = macro_df["^VIX"].reindex(sim_dates).ffill()

    # Valid dates
    corr_window = 60
    valid_start = sim_dates[corr_window]
    common_dates = signals_df.index.intersection(y_res_intraday.index)
    common_dates = common_dates[common_dates >= valid_start]
    common_dates = common_dates.intersection(vix.dropna().index)

    print(f"Analysis period: {common_dates.min()} to {common_dates.max()} ({len(common_dates)} days)")

    # Compute daily metrics
    daily_metrics = []
    for dt in common_dates:
        sig_t = residual_blpx_signals.loc[dt].values
        y_intra_t = y_res_intraday.loc[dt].values
        y_raw_t = jp_oc_clean.loc[dt].values if dt in jp_oc_clean.index else None

        if len(sig_t) < 3 or np.isnan(y_intra_t).any() or np.isnan(sig_t).any():
            continue

        sp_ic, _ = spearmanr(sig_t, y_intra_t)
        if not np.isfinite(sp_ic):
            continue

        # Cross-sectional dispersion (std of returns across tickers)
        cs_disp_res = np.std(y_intra_t)
        cs_disp_raw = np.std(y_raw_t) if y_raw_t is not None and not np.isnan(y_raw_t).any() else np.nan

        # Long-short spread: top 5 - bottom 5 by signal
        n_long = 5
        n_short = 5
        sorted_idx = np.argsort(sig_t)
        short_idx = sorted_idx[:n_short]
        long_idx = sorted_idx[-n_long:]
        ls_spread_res = np.mean(y_intra_t[long_idx]) - np.mean(y_intra_t[short_idx])
        ls_spread_raw = np.nan
        if y_raw_t is not None and not np.isnan(y_raw_t).any():
            ls_spread_raw = np.mean(y_raw_t[long_idx]) - np.mean(y_raw_t[short_idx])

        daily_metrics.append({
            "date": dt,
            "rank_ic": sp_ic,
            "cs_disp_res": cs_disp_res,
            "cs_disp_raw": cs_disp_raw,
            "ls_spread_res": ls_spread_res,
            "ls_spread_raw": ls_spread_raw,
            "long_ret_res": np.mean(y_intra_t[long_idx]),
            "short_ret_res": np.mean(y_intra_t[short_idx]),
            "vix": vix.loc[dt],
        })

    df = pd.DataFrame(daily_metrics).set_index("date")
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"Computed metrics for {len(df)} days")
    print(f"  NaN/inf in ls_spread_res: {df['ls_spread_res'].isna().sum()}")

    # Stratify by VIX tertile
    df["vix_tertile"] = pd.qcut(df["vix"], 3, labels=["Low", "Medium", "High"])
    df["vix_quintile"] = pd.qcut(df["vix"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    # Convert to bps for readability
    bps = 10000

    print("\n" + "=" * 90)
    print("VIX-Stratified Return Dispersion & Realized L/S Spread")
    print("=" * 90)

    # Tertile summary
    print(f"\n{'Regime':<10} {'N':>6} {'VIX':>8} {'Mean IC':>9} {'CS Disp(res)':>13} {'L/S Spread':>12} {'L/S(bps)':>10} {'Ann.Ret':>10} {'Sharpe':>8}")
    print(f"{'':<10} {'':>6} {'avg':>8} {'':>9} {'(bps)':>13} {'(res,bps)':>12} {'':>10} {'(bps/day)':>10} {'':>8}")
    print("-" * 90)
    for label in ["Low", "Medium", "High"]:
        sub = df[df["vix_tertile"] == label]
        n = len(sub)
        mean_vix = sub["vix"].mean()
        mean_ic = sub["rank_ic"].mean()
        mean_disp = sub["cs_disp_res"].mean() * bps
        mean_ls = sub["ls_spread_res"].mean() * bps
        ls_std = sub["ls_spread_res"].std() * bps
        ann_ret = mean_ls * 252
        sharpe = mean_ls / ls_std * np.sqrt(252) if ls_std > 0 else 0
        print(f"{label:<10} {n:>6} {mean_vix:>8.1f} {mean_ic:>9.4f} {mean_disp:>13.2f} {mean_ls:>12.2f} {mean_ls:>10.2f} {ann_ret:>10.1f} {sharpe:>8.2f}")

    # Quintile summary
    print(f"\n{'Quintile':<10} {'N':>6} {'VIX':>8} {'Mean IC':>9} {'CS Disp(res)':>13} {'L/S Spread':>12} {'Ann.Ret':>10} {'Sharpe':>8}")
    print(f"{'':<10} {'':>6} {'range':>8} {'':>9} {'(bps)':>13} {'(res,bps)':>12} {'(bps/yr)':>10} {'':>8}")
    print("-" * 90)
    for label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        sub = df[df["vix_quintile"] == label]
        n = len(sub)
        vix_lo = sub["vix"].min()
        vix_hi = sub["vix"].max()
        mean_ic = sub["rank_ic"].mean()
        mean_disp = sub["cs_disp_res"].mean() * bps
        mean_ls = sub["ls_spread_res"].mean() * bps
        ls_std = sub["ls_spread_res"].std() * bps
        ann_ret = mean_ls * 252
        sharpe = mean_ls / ls_std * np.sqrt(252) if ls_std > 0 else 0
        vix_str = f"{vix_lo:.0f}-{vix_hi:.0f}"
        print(f"{label:<10} {n:>6} {vix_str:>8} {mean_ic:>9.4f} {mean_disp:>13.2f} {mean_ls:>12.2f} {ann_ret:>10.1f} {sharpe:>8.2f}")

    # Correlation between VIX and dispersion / L/S spread
    print(f"\n--- Correlations ---")
    corr_vix_disp = df["vix"].corr(df["cs_disp_res"])
    corr_vix_ls = df["vix"].corr(df["ls_spread_res"])
    corr_vix_ic = df["vix"].corr(df["rank_ic"])
    corr_disp_ls = df["cs_disp_res"].corr(df["ls_spread_res"])
    corr_ic_ls = df["rank_ic"].corr(df["ls_spread_res"])
    print(f"  Corr(VIX, CS dispersion)     = {corr_vix_disp:.4f}")
    print(f"  Corr(VIX, L/S spread)        = {corr_vix_ls:.4f}")
    print(f"  Corr(VIX, IC)                = {corr_vix_ic:.4f}")
    print(f"  Corr(CS disp, L/S spread)    = {corr_disp_ls:.4f}")
    print(f"  Corr(IC, L/S spread)         = {corr_ic_ls:.4f}")

    # Decomposition: L/S spread ≈ IC × dispersion
    # Approximate: daily L/S spread should scale with IC * dispersion
    print(f"\n--- PnL Decomposition: L/S spread vs IC × dispersion ---")
    df["ic_x_disp"] = df["rank_ic"] * df["cs_disp_res"]
    for label in ["Low", "Medium", "High"]:
        sub = df[df["vix_tertile"] == label]
        mean_ls = sub["ls_spread_res"].mean() * bps
        mean_ic_x_disp = sub["ic_x_disp"].mean() * bps
        ratio = mean_ls / mean_ic_x_disp if mean_ic_x_disp != 0 else np.nan
        print(f"  {label:<8}  L/S={mean_ls:>8.2f}bps  IC×disp={mean_ic_x_disp:>8.2f}bps  ratio={ratio:.2f}")

    # Extreme low VIX periods
    print(f"\n--- Extreme Low VIX (bottom 10%) ---")
    vix_p10 = df["vix"].quantile(0.10)
    low_extreme = df[df["vix"] <= vix_p10]
    print(f"  VIX <= {vix_p10:.1f} ({len(low_extreme)} days)")
    print(f"  Mean IC:       {low_extreme['rank_ic'].mean():.4f}")
    print(f"  Mean CS disp:  {low_extreme['cs_disp_res'].mean()*bps:.2f} bps")
    print(f"  Mean L/S:      {low_extreme['ls_spread_res'].mean()*bps:.2f} bps")
    print(f"  Ann. return:   {low_extreme['ls_spread_res'].mean()*bps*252:.1f} bps/yr")
    ls_std = low_extreme["ls_spread_res"].std() * bps
    print(f"  Sharpe:        {low_extreme['ls_spread_res'].mean()*bps/ls_std*np.sqrt(252):.2f}" if ls_std > 0 else "  Sharpe:        N/A")

    print(f"\n--- Extreme High VIX (top 10%) ---")
    vix_p90 = df["vix"].quantile(0.90)
    high_extreme = df[df["vix"] >= vix_p90]
    print(f"  VIX >= {vix_p90:.1f} ({len(high_extreme)} days)")
    print(f"  Mean IC:       {high_extreme['rank_ic'].mean():.4f}")
    print(f"  Mean CS disp:  {high_extreme['cs_disp_res'].mean()*bps:.2f} bps")
    print(f"  Mean L/S:      {high_extreme['ls_spread_res'].mean()*bps:.2f} bps")
    print(f"  Ann. return:   {high_extreme['ls_spread_res'].mean()*bps*252:.1f} bps/yr")
    ls_std = high_extreme["ls_spread_res"].std() * bps
    print(f"  Sharpe:        {high_extreme['ls_spread_res'].mean()*bps/ls_std*np.sqrt(252):.2f}" if ls_std > 0 else "  Sharpe:        N/A")

    print("\nDone.")


if __name__ == "__main__":
    main()
