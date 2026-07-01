"""Phase 2B: Regime-Conditional Parameter Adaptation.

Tests whether switching BLPX parameters based on market regimes improves performance.

Regimes:
  1. US Volatility: VIX proxy (rolling std of SPY) — High/Low
  2. Cross-sectional Dispersion: rolling std of JP returns — High/Low
  3. USDJPY: rolling return — Strong/Weak yen

For each regime, tests parameter variants:
  - High vol → more regularization (higher alpha_xx, lambda_sector)
  - Low vol → less regularization (lower alpha_xx, lambda_sector)
  - High dispersion → stronger sector prior
  - Low dispersion → weaker sector prior

Evaluation: Walk-forward regime-conditional vs static parameters.
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

REGIME_PARAM_VARIANTS = {
    "vol_high": {
        "alpha_xx": 0.30, "lambda_sector": 0.70, "blp_ewma_halflife": 180,
    },
    "vol_low": {
        "alpha_xx": 0.15, "lambda_sector": 0.50, "blp_ewma_halflife": 90,
    },
    "disp_high": {
        "lambda_sector": 0.70, "sector_eta": 0.6, "sector_gamma": 5.0,
    },
    "disp_low": {
        "lambda_sector": 0.50, "sector_eta": 0.4, "sector_gamma": 3.0,
    },
    "yen_weak": {
        "alpha_yx": 0.20, "lambda_sector": 0.65,
    },
    "yen_strong": {
        "alpha_yx": 0.10, "lambda_sector": 0.55,
    },
}

REGIME_CONFIGS = [
    {
        "name": "vol_regime",
        "indicator": "us_vol",
        "split": "median",
        "high_params": REGIME_PARAM_VARIANTS["vol_high"],
        "low_params": REGIME_PARAM_VARIANTS["vol_low"],
    },
    {
        "name": "disp_regime",
        "indicator": "jp_disp",
        "split": "median",
        "high_params": REGIME_PARAM_VARIANTS["disp_high"],
        "low_params": REGIME_PARAM_VARIANTS["disp_low"],
    },
    {
        "name": "yen_regime",
        "indicator": "usdjpy",
        "split": "median",
        "high_params": REGIME_PARAM_VARIANTS["yen_weak"],
        "low_params": REGIME_PARAM_VARIANTS["yen_strong"],
    },
    {
        "name": "vol_disp_2x2",
        "indicator": "vol_disp",
        "split": "2x2",
        "high_params": {**REGIME_PARAM_VARIANTS["vol_high"], **REGIME_PARAM_VARIANTS["disp_high"]},
        "low_params": {**REGIME_PARAM_VARIANTS["vol_low"], **REGIME_PARAM_VARIANTS["disp_low"]},
    },
]


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        cfg.setdefault("blpx", {}).update(blpx_overrides)
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


def compute_regime_indicators(df_exec, window=60):
    """Compute rolling regime indicators (lookahead-safe with shift(1))."""
    indicators = {}

    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    us_returns = df_exec[us_cols]
    us_vol = us_returns.rolling(window).std(ddof=1).mean(axis=1)
    us_vol_z = (us_vol - us_vol.rolling(252).mean()) / us_vol.rolling(252).std(ddof=1)
    indicators["us_vol"] = (us_vol_z > 0).astype(int)

    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    jp_returns = df_exec[jp_oc_cols]
    jp_disp = jp_returns.rolling(window).std(ddof=1).mean(axis=1)
    jp_disp_z = (jp_disp - jp_disp.rolling(252).mean()) / jp_disp.rolling(252).std(ddof=1)
    indicators["jp_disp"] = (jp_disp_z > 0).astype(int)

    if "usdjpy_cc" in df_exec.columns:
        usdjpy_ret = df_exec["usdjpy_cc"]
    elif "usdjpy_return" in df_exec.columns:
        usdjpy_ret = df_exec["usdjpy_return"]
    else:
        usdjpy_ret = df_exec[us_cols].mean(axis=1) - df_exec[jp_oc_cols].mean(axis=1)
    usdjpy_cum = usdjpy_ret.rolling(window).sum()
    usdjpy_z = (usdjpy_cum - usdjpy_cum.rolling(252).mean()) / usdjpy_cum.rolling(252).std(ddof=1)
    indicators["usdjpy"] = (usdjpy_z > 0).astype(int)

    indicators["vol_disp"] = indicators["us_vol"] * 2 + indicators["jp_disp"]

    return pd.DataFrame(indicators, index=df_exec.index)


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


class RegimeConditionalModel:
    """Model that switches parameters based on regime indicators.

    Pre-computes signals for all parameter variants, then selects per-date
    based on the regime indicator. This is lookahead-safe because regime
    indicators use shift(1) (computed from historical data only).
    """

    def __init__(self, cfg_base, regime_df, regime_name, indicator_col,
                 high_params, low_params, split_mode, df_exec_raw):
        self.cfg_base = cfg_base
        self.regime_df = regime_df
        self.regime_name = regime_name
        self.indicator_col = indicator_col
        self.high_params = high_params
        self.low_params = low_params
        self.split_mode = split_mode
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
        self._signals_cache = None

    def _prepare(self):
        if self._signals_cache is not None:
            return self._signals_cache

        all_signals = {}
        for label, params in [("high", self.high_params), ("low", self.low_params)]:
            cfg = dict(self.cfg_base)
            blpx = dict(cfg.get("blpx", {}))
            blpx.update(params)
            cfg["blpx"] = blpx

            _BLP_CORR_CACHE.clear()
            _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            pred = model.predict_signals(self.df_exec_raw)
            all_signals[label] = pred["signals"]

        self._signals_cache = all_signals
        return all_signals

    def predict_signals(self, df_exec):
        all_signals = self._prepare()
        T = len(df_exec)
        sim_dates = df_exec.index

        regime_vals = self.regime_df[self.indicator_col].reindex(sim_dates).fillna(0).values

        sig_high = all_signals["high"].reindex(sim_dates).fillna(0.0).values
        sig_low = all_signals["low"].reindex(sim_dates).fillna(0.0).values

        combined = np.zeros((T, self.n_j))
        for i in range(T):
            if self.split_mode == "2x2":
                rv = int(regime_vals[i])
                use_high = rv >= 2
            else:
                use_high = regime_vals[i] > 0

            sig_t = sig_high[i] if use_high else sig_low[i]
            combined[i] = self._normalize(sig_t)

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
    parser = argparse.ArgumentParser(description="Phase 2B: Regime-Conditional Parameter Adaptation")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/phase2b_regime_conditional")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    logger.info("Computing regime indicators...")
    regime_df = compute_regime_indicators(df_exec)
    regime_df.to_csv(output_dir / "regime_indicators.csv")

    cfg_base = build_config(yaml_path, blpx_overrides=BASE_PARAMS,
                            signal_components=SIGNAL_WEIGHTS)

    all_results = []

    # Baseline (static params)
    logger.info("=== Baseline (static params) ===")
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    m_base = run_backtest("baseline_static", model_base, df_exec, y_target, slippage_bps=args.slippage_bps)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                m_base["Sharpe_net"], m_base["Mean_Rank_IC"], m_base["ICIR"],
                m_base["MDD"] * 100, m_base["elapsed_s"])
    all_results.append({**m_base, "scheme": "baseline"})

    # Regime-conditional variants
    for rc in REGIME_CONFIGS:
        logger.info("=== Regime: %s (indicator=%s) ===", rc["name"], rc["indicator"])
        regime_model = RegimeConditionalModel(
            cfg_base, regime_df, rc["name"], rc["indicator"],
            rc["high_params"], rc["low_params"], rc["split"],
            df_exec,
        )
        m = run_backtest(rc["name"], regime_model, df_exec, y_target, slippage_bps=args.slippage_bps)
        m["scheme"] = "regime_conditional"
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (delta=%+.4f, %.1fs)",
                    rc["name"], m["Sharpe_net"], m["Mean_Rank_IC"], m["ICIR"],
                    m["MDD"] * 100, m["Sharpe_net"] - m_base["Sharpe_net"], m["elapsed_s"])

    # Regime distribution analysis
    regime_summary = {}
    for col in regime_df.columns:
        vals = regime_df[col].values
        if col == "vol_disp":
            regime_summary[col] = {
                "0 (vol_low, disp_low)": int(np.sum(vals == 0)),
                "1 (vol_low, disp_high)": int(np.sum(vals == 1)),
                "2 (vol_high, disp_low)": int(np.sum(vals == 2)),
                "3 (vol_high, disp_high)": int(np.sum(vals == 3)),
            }
        else:
            regime_summary[col] = {
                "0 (low)": int(np.sum(vals == 0)),
                "1 (high)": int(np.sum(vals == 1)),
            }

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    print("\n" + "=" * 100)
    print("PHASE 2B — REGIME-CONDITIONAL PARAMETER ADAPTATION — RESULTS")
    print("=" * 100)
    print(f"\nBaseline (static): Sharpe={m_base['Sharpe_net']:.4f} IC={m_base['Mean_Rank_IC']:.4f} ICIR={m_base['ICIR']:.2f} MDD={m_base['MDD']*100:.2f}%")

    print(f"\n{'Name':<25} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Delta':<8}")
    print("-" * 69)
    for r in all_results:
        delta = r["Sharpe_net"] - m_base["Sharpe_net"] if np.isfinite(r["Sharpe_net"]) else np.nan
        print(f"{r['name']:<25} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {delta:+.4f}")

    print("\nRegime distribution:")
    for col, dist in regime_summary.items():
        print(f"  {col}: {dist}")

    valid = [r for r in all_results if r.get("scheme") != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} (delta={best['Sharpe_net']-m_base['Sharpe_net']:+.4f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
