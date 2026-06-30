#!/usr/bin/env python
"""VIX-regime stratified US→JP signal IC analysis.

Computes daily cross-sectional Rank IC (Spearman) between production model
signals and realized JP intraday residual returns, then stratifies by VIX level.

Output: table of mean IC, ICIR, hit rate by VIX tertile (Low/Medium/High)
and median split (vix_low / vix_high).
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
    # 1. Load data
    print("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    sim_dates = df_exec.index
    T = len(df_exec)
    print(f"  shape={df_exec.shape}, range={sim_dates.min()} to {sim_dates.max()}")

    # 2. Load production config and run model
    config_path = os.path.join(ROOT, "configs", "production.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Running BLPX model (production_residual_blpx)...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    pred = model.predict_signals(df_exec)
    signals_df = pred["signals"]  # T x N_J
    residual_blpx_signals = pred.get("residual_blpx_signals", signals_df)
    print(f"  signals shape: {signals_df.shape}")

    # 3. Compute realized target: JP intraday residual returns
    # Use jp_oc (open-to-close) as proxy for 9:10-to-close
    jp_oc = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]]
    jp_oc.columns = JP_TICKERS

    # TOPIX close-to-close for residualization
    topix_cc = df_exec["topix_night_return"].values + df_exec["topix_oc_return"].values
    topix_cc_series = pd.Series(topix_cc, index=sim_dates)

    # Rolling beta (60d, shifted) for residualization
    beta_df = compute_rolling_beta(jp_oc, topix_cc_series, 60)
    y_res_intraday = jp_oc - beta_df.multiply(topix_cc_series, axis=0)
    y_res_intraday = y_res_intraday.dropna(how="all")

    # Also compute CC residual for reference
    jp_cc = df_exec[[f"jp_cc_{tk}" for tk in JP_TICKERS]] if any(
        c.startswith("jp_cc_") for c in df_exec.columns
    ) else None
    if jp_cc is not None:
        jp_cc.columns = JP_TICKERS
        beta_cc = compute_rolling_beta(jp_cc, topix_cc_series, 60)
        y_res_cc = jp_cc - beta_cc.multiply(topix_cc_series, axis=0)
        y_res_cc = y_res_cc.dropna(how="all")
    else:
        y_res_cc = None

    # 4. Compute daily Rank IC
    common_dates = signals_df.index.intersection(y_res_intraday.index)
    # Also need to skip the first corr_window (60) rows for signal validity
    corr_window = 60
    valid_start = sim_dates[corr_window]
    common_dates = common_dates[common_dates >= valid_start]

    ic_records = []
    for dt in common_dates:
        sig_t = residual_blpx_signals.loc[dt].values
        y_intra_t = y_res_intraday.loc[dt].values

        if len(sig_t) < 3 or np.isnan(y_intra_t).any() or np.isnan(sig_t).any():
            continue

        sp_ic, _ = spearmanr(sig_t, y_intra_t)
        if not np.isfinite(sp_ic):
            continue

        rec = {"date": dt, "rank_ic_intraday": sp_ic}

        if y_res_cc is not None and dt in y_res_cc.index:
            y_cc_t = y_res_cc.loc[dt].values
            if not np.isnan(y_cc_t).any():
                sp_ic_cc, _ = spearmanr(sig_t, y_cc_t)
                if np.isfinite(sp_ic_cc):
                    rec["rank_ic_cc"] = sp_ic_cc

        ic_records.append(rec)

    ic_df = pd.DataFrame(ic_records).set_index("date")
    print(f"\nIC computed for {len(ic_df)} days")
    print(f"  Intraday Rank IC: mean={ic_df['rank_ic_intraday'].mean():.4f}, "
          f"std={ic_df['rank_ic_intraday'].std():.4f}, "
          f"ICIR={ic_df['rank_ic_intraday'].mean()/ic_df['rank_ic_intraday'].std()*np.sqrt(252):.2f}, "
          f"hit_rate={(ic_df['rank_ic_intraday']>0).mean():.2%}")

    # 5. Load VIX
    macro_path = os.path.join(ROOT, "market_data", "macro_data.pkl")
    macro_df = pd.read_pickle(macro_path)
    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
    vix = macro_df["^VIX"].reindex(ic_df.index).ffill()

    # Drop NaN VIX
    valid_mask = vix.notna()
    ic_df = ic_df[valid_mask]
    vix = vix[valid_mask]
    print(f"  After VIX merge: {len(ic_df)} days")

    # 6. Stratify by VIX
    # Tertile bins
    vix_tertile = pd.qcut(vix, 3, labels=["Low", "Medium", "High"])
    # Median split
    vix_median = vix.median()
    vix_binary = pd.Series(np.where(vix > vix_median, "vix_high", "vix_low"), index=vix.index)

    print(f"\nVIX range: {vix.min():.2f} to {vix.max():.2f}, median={vix_median:.2f}")
    print(f"Tertile boundaries: {pd.qcut(vix, 3).cat.categories.tolist()}")

    # 7. Results table
    print("\n" + "=" * 80)
    print("VIX-Stratified US→JP Signal IC (Residual-BLPX, production config)")
    print("=" * 80)

    # Tertile analysis
    print("\n--- Tertile Bins ---")
    print(f"{'Regime':<12} {'N_days':>7} {'Mean IC':>10} {'Std IC':>10} {'ICIR':>10} {'Hit Rate':>10} {'Mean VIX':>10}")
    print("-" * 79)
    for label in ["Low", "Medium", "High"]:
        mask = vix_tertile == label
        n = mask.sum()
        if n == 0:
            continue
        ic_vals = ic_df.loc[mask, "rank_ic_intraday"]
        mean_ic = ic_vals.mean()
        std_ic = ic_vals.std()
        icir = mean_ic / std_ic * np.sqrt(252) if std_ic > 0 else 0
        hit = (ic_vals > 0).mean()
        mean_vix = vix[mask].mean()
        print(f"{label:<12} {n:>7} {mean_ic:>10.4f} {std_ic:>10.4f} {icir:>10.2f} {hit:>10.2%} {mean_vix:>10.2f}")

    # Median split
    print("\n--- Median Split ---")
    print(f"{'Regime':<12} {'N_days':>7} {'Mean IC':>10} {'Std IC':>10} {'ICIR':>10} {'Hit Rate':>10} {'Mean VIX':>10}")
    print("-" * 79)
    for label in ["vix_low", "vix_high"]:
        mask = vix_binary == label
        n = mask.sum()
        if n == 0:
            continue
        ic_vals = ic_df.loc[mask, "rank_ic_intraday"]
        mean_ic = ic_vals.mean()
        std_ic = ic_vals.std()
        icir = mean_ic / std_ic * np.sqrt(252) if std_ic > 0 else 0
        hit = (ic_vals > 0).mean()
        mean_vix = vix[mask].mean()
        print(f"{label:<12} {n:>7} {mean_ic:>10.4f} {std_ic:>10.4f} {icir:>10.2f} {hit:>10.2%} {mean_vix:>10.2f}")

    # Also show CC IC if available
    if "rank_ic_cc" in ic_df.columns:
        print("\n--- CC Target IC (for reference) ---")
        print(f"{'Regime':<12} {'N_days':>7} {'Mean IC_CC':>12} {'ICIR_CC':>10} {'Hit Rate':>10}")
        print("-" * 59)
        for label in ["Low", "Medium", "High"]:
            mask = vix_tertile == label
            n = mask.sum()
            if n == 0:
                continue
            ic_vals = ic_df.loc[mask, "rank_ic_cc"].dropna()
            if len(ic_vals) == 0:
                continue
            mean_ic = ic_vals.mean()
            std_ic = ic_vals.std()
            icir = mean_ic / std_ic * np.sqrt(252) if std_ic > 0 else 0
            hit = (ic_vals > 0).mean()
            print(f"{label:<12} {n:>7} {mean_ic:>12.4f} {icir:>10.2f} {hit:>10.2%}")

    # VIX quintile for finer detail
    print("\n--- Quintile Bins (finer) ---")
    vix_quintile = pd.qcut(vix, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    print(f"{'Quintile':<10} {'N_days':>7} {'Mean IC':>10} {'ICIR':>10} {'Hit Rate':>10} {'VIX range':>20}")
    print("-" * 69)
    for label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        mask = vix_quintile == label
        n = mask.sum()
        if n == 0:
            continue
        ic_vals = ic_df.loc[mask, "rank_ic_intraday"]
        mean_ic = ic_vals.mean()
        std_ic = ic_vals.std()
        icir = mean_ic / std_ic * np.sqrt(252) if std_ic > 0 else 0
        hit = (ic_vals > 0).mean()
        vix_range = f"{vix[mask].min():.1f}-{vix[mask].max():.1f}"
        print(f"{label:<10} {n:>7} {mean_ic:>10.4f} {icir:>10.2f} {hit:>10.2%} {vix_range:>20}")

    # Correlation between VIX level and daily IC
    corr_vix_ic = vix.corr(ic_df["rank_ic_intraday"])
    print(f"\nCorrelation(VIX level, daily IC) = {corr_vix_ic:.4f}")

    # Rolling 60d IC vs VIX
    rolling_ic = ic_df["rank_ic_intraday"].rolling(60, min_periods=30).mean()
    rolling_vix = vix.rolling(60, min_periods=30).mean()
    corr_rolling = rolling_vix.corr(rolling_ic)
    print(f"Correlation(60d rolling VIX, 60d rolling IC) = {corr_rolling:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
