"""Phase 2A: Multi-Horizon Signal Blending.

Combines 1-day, 3-day, and 5-day horizon BLPX signals.
For each horizon, creates cumulative multi-day returns as alternative targets
and runs the model, then blends the resulting signals.

Approach:
  - h1: Standard 1-day signal (baseline)
  - h3: 3-day cumulative JP target with 3-day cumulative US input
  - h5: 5-day cumulative JP target with 5-day cumulative US input
  - Blend: w1 * z(h1) + w3 * z(h3) + w5 * z(h5)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
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

PHASE1A_BEST = {
    "alpha_xx": 0.20, "alpha_yy": 0.50, "alpha_yx": 0.15,
    "lambda_pca": 0.10, "lambda_sector": 0.60, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 504,
    "blp_ewma_halflife": 120, "sector_eta": 0.5, "sector_gamma": 4.0,
}

PHASE1B_WEIGHTS = {
    "raw_pca": {"enabled": True, "weight": 0.2},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": True, "weight": 0.8},
    "residual_blpx": {"enabled": False, "weight": 0.0},
}

BLEND_CONFIGS = [
    {"w1": 1.0, "w3": 0.0, "w5": 0.0, "name": "h1_only"},
    {"w1": 0.0, "w3": 1.0, "w5": 0.0, "name": "h3_only"},
    {"w1": 0.0, "w3": 0.0, "w5": 1.0, "name": "h5_only"},
    {"w1": 0.6, "w3": 0.3, "w5": 0.1, "name": "blend_60_30_10"},
    {"w1": 0.5, "w3": 0.3, "w5": 0.2, "name": "blend_50_30_20"},
    {"w1": 0.4, "w3": 0.4, "w5": 0.2, "name": "blend_40_40_20"},
    {"w1": 0.7, "w3": 0.2, "w5": 0.1, "name": "blend_70_20_10"},
    {"w1": 0.8, "w3": 0.1, "w5": 0.1, "name": "blend_80_10_10"},
    {"w1": 0.5, "w3": 0.5, "w5": 0.0, "name": "blend_50_50_0"},
    {"w1": 0.34, "w3": 0.33, "w5": 0.33, "name": "blend_equal"},
]


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        cfg.setdefault("blpx", {}).update(blpx_overrides)
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


def compute_cumulative_returns(df_exec, horizon):
    """Create a modified df_exec with cumulative h-day returns for US and JP."""
    df_mod = df_exec.copy()

    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    jp_gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]

    # Cumulative US returns (shift to avoid lookahead: use t-h+1 to t)
    for col in us_cols:
        df_mod[col] = df_exec[col].rolling(horizon).sum()

    # Cumulative JP open-close returns
    for col in jp_oc_cols:
        df_mod[col] = df_exec[col].rolling(horizon).sum()

    # Cumulative JP gap returns
    for col in jp_gap_cols:
        df_mod[col] = df_exec[col].rolling(horizon).sum()

    # Cumulative TOPIX returns
    for col in ["topix_night_return", "topix_oc_return", "topix_cc_trade"]:
        if col in df_exec.columns:
            df_mod[col] = df_exec[col].rolling(horizon).sum()

    return df_mod


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


def run_variant(name, model, df_exec, y_target, slippage_bps=5.0):
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


class MultiHorizonBlendModel:
    """Wraps multiple BLPX models at different horizons and blends signals."""

    def __init__(self, configs, horizons, weights, df_exec_raw):
        self.configs = configs
        self.horizons = horizons
        self.weights = weights
        self.df_exec_raw = df_exec_raw
        self.n_j = len(JP_TICKERS)
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"

        self._models = {}
        self._df_mods = {}
        self._signals = {}

    def _prepare(self):
        from leadlag.core import signal as sig_module
        for h, cfg in zip(self.horizons, self.configs):
            df_mod = compute_cumulative_returns(self.df_exec_raw, h)
            self._df_mods[h] = df_mod
            _BLP_CORR_CACHE.clear()
            _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            pred = model.predict_signals(df_mod)
            self._signals[h] = pred["signals"]
            self._models[h] = model

    def predict_signals(self, df_exec):
        if not self._signals:
            self._prepare()

        T = len(df_exec)
        sim_dates = df_exec.index
        combined = np.zeros((T, self.n_j))

        for h, w in zip(self.horizons, self.weights):
            sig_h = self._signals[h].reindex(sim_dates).fillna(0.0).values
            z_h = np.zeros_like(sig_h)
            for i in range(T):
                z_h[i] = self._normalize(sig_h[i])
            combined += w * z_h

        from leadlag.models.sre import compute_jp_target_returns
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )

        empty_df = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        return {
            "raw_pca_signals": empty_df,
            "residual_pca_signals": empty_df,
            "p4_signals": empty_df,
            "signals": pd.DataFrame(combined, index=sim_dates, columns=JP_TICKERS),
            "normalized_signals": pd.DataFrame(combined, index=sim_dates, columns=JP_TICKERS),
            "y_jp_oc_df": y_jp_oc_df,
        }

    def _normalize(self, sig):
        centered = sig - np.median(sig)
        std = np.std(centered)
        std_safe = std if std > 1e-8 else 1.0
        return centered / std_safe

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        q_val = q if q is not None else self.q
        return build_weights(signal=signal, q=q_val, n_j=self.n_j,
                             weight_mode=self.weight_mode, enforce_sign=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 2A: Multi-Horizon Signal Blending")
    parser.add_argument("--stage", choices=["single", "blend", "all"], default="all")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/phase2a_multi_horizon")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    all_results = []

    # Baseline (h1 only, current production)
    logger.info("=== Baseline (h1 only) ===")
    cfg_base = build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                            signal_components=PHASE1B_WEIGHTS)
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    m_base = run_variant("baseline_h1", model_base, df_exec, y_target, slippage_bps=args.slippage_bps)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                m_base["Sharpe_net"], m_base["Mean_Rank_IC"], m_base["ICIR"],
                m_base["MDD"] * 100, m_base["elapsed_s"])
    all_results.append({**m_base, "scheme": "baseline", "w1": 1.0, "w3": 0.0, "w5": 0.0})

    # Single horizon tests
    if args.stage in ("single", "all"):
        for h in [3, 5]:
            logger.info("=== Single horizon h%d ===", h)
            df_mod = compute_cumulative_returns(df_exec, h)
            _BLP_CORR_CACHE.clear()
            _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
            m = run_variant(f"h{h}_only", model, df_mod, y_target, slippage_bps=args.slippage_bps)
            m["scheme"] = "single"
            m["w1"] = 0.0
            m["w3"] = 1.0 if h == 3 else 0.0
            m["w5"] = 1.0 if h == 5 else 0.0
            all_results.append(m)
            logger.info("h%d: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                        h, m["Sharpe_net"], m["Mean_Rank_IC"], m["ICIR"],
                        m["MDD"] * 100, m["elapsed_s"])

    # Blend tests
    if args.stage in ("blend", "all"):
        logger.info("=== Multi-Horizon Blends ===")
        for bc in BLEND_CONFIGS:
            if bc["w1"] == 1.0 and bc["w3"] == 0.0 and bc["w5"] == 0.0:
                continue
            logger.info("Blend: %s (w1=%.1f w3=%.1f w5=%.1f)", bc["name"], bc["w1"], bc["w3"], bc["w5"])
            configs = [cfg_base, cfg_base, cfg_base]
            horizons = [1, 3, 5]
            weights = [bc["w1"], bc["w3"], bc["w5"]]
            blend_model = MultiHorizonBlendModel(configs, horizons, weights, df_exec)
            m = run_variant(bc["name"], blend_model, df_exec, y_target, slippage_bps=args.slippage_bps)
            m["scheme"] = "blend"
            m["w1"] = bc["w1"]
            m["w3"] = bc["w3"]
            m["w5"] = bc["w5"]
            all_results.append(m)
            logger.info("  %s: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                        bc["name"], m["Sharpe_net"], m["Mean_Rank_IC"], m["ICIR"],
                        m["MDD"] * 100, m["elapsed_s"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    print("\n" + "=" * 100)
    print("PHASE 2A — MULTI-HORIZON SIGNAL BLENDING — RESULTS")
    print("=" * 100)
    print(f"\nBaseline (h1 only): Sharpe={m_base['Sharpe_net']:.4f} IC={m_base['Mean_Rank_IC']:.4f} ICIR={m_base['ICIR']:.2f} MDD={m_base['MDD']*100:.2f}%")

    print(f"\n{'Name':<25} {'w1':<5} {'w3':<5} {'w5':<5} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Delta':<8}")
    print("-" * 84)
    for r in all_results:
        delta = r["Sharpe_net"] - m_base["Sharpe_net"] if np.isfinite(r["Sharpe_net"]) else np.nan
        print(f"{r['name']:<25} {r.get('w1',0):<5.1f} {r.get('w3',0):<5.1f} {r.get('w5',0):<5.1f} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {delta:+.4f}")

    valid = [r for r in all_results if r.get("scheme") != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} (delta={best['Sharpe_net']-m_base['Sharpe_net']:+.4f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
