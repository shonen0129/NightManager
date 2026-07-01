"""Phase 2D: Cross-Sectional Feature Integration.

Tests blending of cross-sectional momentum, mean-reversion, and dispersion
signals with the BLPX baseline signal.

Features:
  1. CS Momentum (5d): 5-day cumulative return, cross-sectionally ranked
  2. CS Momentum (20d): 20-day cumulative return, cross-sectionally ranked
  3. CS Mean Reversion (20d): Negative 20-day return (mean-reversion signal)
  4. CS Dispersion: Rolling cross-sectional std (high dispersion = stronger signal)
  5. CS Rank Reversal: 1-day rank change (reversal signal)

Blend: z(blpx) + w * z(feature) for various weights w.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
    _BLP_CORR_CACHE,
    _RAW_PCA_RESIDUAL_PCA_CACHE,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_PARAMS = {
    "alpha_xx": 0.20, "alpha_yy": 0.50, "alpha_yx": 0.15,
    "lambda_pca": 0.10, "lambda_sector": 0.60, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 504,
    "blp_ewma_halflife": 120, "sector_eta": 0.5, "sector_gamma": 4.0,
}

SIGNAL_WEIGHTS = {
    "raw_pca": {"enabled": True, "weight": 0.2},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": True, "weight": 0.8},
    "residual_blpx": {"enabled": False, "weight": 0.0},
}

BLEND_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        cfg.setdefault("blpx", {}).update(blpx_overrides)
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


def compute_cs_momentum(df_exec, horizon, col_prefix="jp_oc"):
    """Cross-sectional momentum: rolling cumulative return per ticker."""
    cols = [f"{col_prefix}_{tk}" for tk in JP_TICKERS]
    df = df_exec[cols].copy()
    df.columns = JP_TICKERS
    mom = df.shift(1).rolling(horizon).sum()
    return mom


def compute_cs_mean_reversion(df_exec, horizon, col_prefix="jp_oc"):
    """Cross-sectional mean reversion: negative rolling return."""
    mom = compute_cs_momentum(df_exec, horizon, col_prefix)
    return -mom


def compute_cs_dispersion(df_exec, window=20, col_prefix="jp_oc"):
    """Cross-sectional dispersion: rolling std across tickers."""
    cols = [f"{col_prefix}_{tk}" for tk in JP_TICKERS]
    df = df_exec[cols].copy()
    df.columns = JP_TICKERS
    cs_std = df.shift(1).rolling(window).std(ddof=1).mean(axis=1)
    return cs_std


def compute_cs_rank_reversal(df_exec, col_prefix="jp_oc"):
    """1-day rank reversal: change in cross-sectional rank."""
    cols = [f"{col_prefix}_{tk}" for tk in JP_TICKERS]
    df = df_exec[cols].copy()
    df.columns = JP_TICKERS
    ranks = df.shift(1).rank(axis=1)
    rank_change = ranks.diff()
    return -rank_change


def cross_sectional_zscore(df):
    """Z-score each row (cross-sectional normalization)."""
    centered = df.sub(df.median(axis=1), axis=0)
    std = centered.std(axis=1)
    std_safe = std.where(std > 1e-8, 1.0)
    return centered.div(std_safe, axis=0)


def compute_rank_ic(signals_df, y_target, sim_dates, start_idx):
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)
    ic_list = []
    for i in range(start_idx, len(sim_dates)):
        date = sim_dates[i]
        if date not in signals_df.index:
            continue
        sig_t = signals_df.loc[date].values
        y_t = y_df.loc[date].values
        valid = ~(np.isnan(sig_t) | np.isnan(y_t))
        if valid.sum() >= 3:
            rho, _ = stats.spearmanr(sig_t[valid], y_t[valid])
            if np.isfinite(rho):
                ic_list.append(float(rho))
    if not ic_list:
        return np.nan, np.nan
    ic_arr = np.array(ic_list)
    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr, ddof=1))
    icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic > 1e-8 else np.nan
    return mean_ic, icir


def run_backtest(name, model, df_exec, y_target, slippage_bps=5.0):
    t0 = time.perf_counter()
    results = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-01",
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0, slippage_bps=slippage_bps,
    )
    elapsed = time.perf_counter() - t0
    dr = results["daily_returns"]
    ar = float(dr.mean() * 245)
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(results["daily_turnover"].mean())
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
    return {
        "name": name, "AR_net": ar, "Vol_net": vol, "Sharpe_net": sharpe,
        "MDD": mdd, "Turnover": turnover, "Mean_Rank_IC": mean_ic,
        "ICIR": icir, "elapsed_s": elapsed,
    }


class CSFeatureBlendModel:
    """Blends BLPX baseline signal with cross-sectional feature signals."""

    def __init__(self, blpx_signals, feature_signals, weights, n_j):
        self.blpx_signals = blpx_signals
        self.feature_signals = feature_signals
        self.weights = weights  # dict: {feature_name: weight}
        self.n_j = n_j
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"

    def predict_signals(self, df_exec):
        T = len(df_exec)
        sim_dates = df_exec.index

        blpx = self.blpx_signals.reindex(sim_dates).fillna(0.0)
        z_blpx = cross_sectional_zscore(blpx)

        combined = z_blpx.copy()
        for fname, w in self.weights.items():
            feat = self.feature_signals[fname].reindex(sim_dates).fillna(0.0)
            z_feat = cross_sectional_zscore(feat)
            combined = combined + w * z_feat

        combined = combined.fillna(0.0)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "raw_pca_signals": empty,
            "residual_pca_signals": empty,
            "p4_signals": empty,
            "signals": combined,
            "normalized_signals": combined,
            "y_jp_oc_df": y_jp_oc_df,
        }

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        q_val = q if q is not None else self.q
        return build_weights(signal=signal, q=q_val, n_j=self.n_j,
                             weight_mode=self.weight_mode, enforce_sign=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 2D: Cross-Sectional Feature Integration")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/phase2d_cs_features")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    cfg_base = build_config(yaml_path, blpx_overrides=BASE_PARAMS,
                            signal_components=SIGNAL_WEIGHTS)

    logger.info("Computing baseline BLPX signals...")
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    pred_base = model_base.predict_signals(df_exec)
    blpx_signals = pred_base["signals"]

    logger.info("Computing cross-sectional features...")
    features = {
        "mom_5d": compute_cs_momentum(df_exec, 5),
        "mom_20d": compute_cs_momentum(df_exec, 20),
        "mr_20d": compute_cs_mean_reversion(df_exec, 20),
        "rank_reversal": compute_cs_rank_reversal(df_exec),
    }

    # Also compute gap-based momentum
    gap_mom_5d = compute_cs_momentum(df_exec, 5, col_prefix="jp_gap")
    features["gap_mom_5d"] = gap_mom_5d

    all_results = []

    # Baseline
    logger.info("=== Baseline (BLPX only) ===")
    m_base = run_backtest("baseline_blpx", model_base, df_exec, y_target, slippage_bps=args.slippage_bps)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                m_base["Sharpe_net"], m_base["Mean_Rank_IC"], m_base["ICIR"],
                m_base["MDD"] * 100, m_base["elapsed_s"])
    all_results.append({**m_base, "scheme": "baseline", "feature": "none", "weight": 0.0})

    # Single feature blends
    for fname, feat_df in features.items():
        for w in BLEND_WEIGHTS:
            if w == 0.0:
                continue
            name = f"{fname}_w{w:.2f}"
            logger.info("=== %s ===", name)
            blend_model = CSFeatureBlendModel(blpx_signals, features, {fname: w}, len(JP_TICKERS))
            m = run_backtest(name, blend_model, df_exec, y_target, slippage_bps=args.slippage_bps)
            m["scheme"] = "single_feature"
            m["feature"] = fname
            m["weight"] = w
            all_results.append(m)
            logger.info("  %s: Sharpe=%.4f IC=%.4f (delta=%+.4f, %.1fs)",
                        name, m["Sharpe_net"], m["Mean_Rank_IC"],
                        m["Sharpe_net"] - m_base["Sharpe_net"], m["elapsed_s"])

    # Combined features
    combos = [
        ("mom5_mom20", {"mom_5d": 0.10, "mom_20d": 0.10}),
        ("mom5_mr20", {"mom_5d": 0.10, "mr_20d": 0.10}),
        ("all_mom", {"mom_5d": 0.05, "mom_20d": 0.05, "gap_mom_5d": 0.05}),
        ("all_features", {"mom_5d": 0.05, "mom_20d": 0.05, "mr_20d": 0.05, "rank_reversal": 0.05}),
        ("mom5_gap5", {"mom_5d": 0.10, "gap_mom_5d": 0.10}),
    ]
    for combo_name, combo_weights in combos:
        name = f"combo_{combo_name}"
        logger.info("=== %s ===", name)
        blend_model = CSFeatureBlendModel(blpx_signals, features, combo_weights, len(JP_TICKERS))
        m = run_backtest(name, blend_model, df_exec, y_target, slippage_bps=args.slippage_bps)
        m["scheme"] = "combo"
        m["feature"] = combo_name
        m["weight"] = sum(combo_weights.values())
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f IC=%.4f (delta=%+.4f, %.1fs)",
                    name, m["Sharpe_net"], m["Mean_Rank_IC"],
                    m["Sharpe_net"] - m_base["Sharpe_net"], m["elapsed_s"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    print("\n" + "=" * 100)
    print("PHASE 2D — CROSS-SECTIONAL FEATURE INTEGRATION — RESULTS")
    print("=" * 100)
    print(f"\nBaseline (BLPX only): Sharpe={m_base['Sharpe_net']:.4f} IC={m_base['Mean_Rank_IC']:.4f} ICIR={m_base['ICIR']:.2f} MDD={m_base['MDD']*100:.2f}%")

    print(f"\n{'Name':<30} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Delta':<8}")
    print("-" * 74)
    for r in all_results:
        delta = r["Sharpe_net"] - m_base["Sharpe_net"] if np.isfinite(r["Sharpe_net"]) else np.nan
        print(f"{r['name']:<30} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {delta:+.4f}")

    valid = [r for r in all_results if r.get("scheme") != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} (delta={best['Sharpe_net']-m_base['Sharpe_net']:+.4f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
