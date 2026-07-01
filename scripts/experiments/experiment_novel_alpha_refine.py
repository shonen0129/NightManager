"""Refine novel alpha: select only top-N |IC| pairs and re-test.

Tests N = 10, 20, 30, 50, 100 and blend_weight = 0.20, 0.30, 0.40, 0.50.
Uses cached novel data (no re-download).
"""

from __future__ import annotations

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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


def build_combined_signal(blpx_signals, alt_data, sig_pairs, sim_dates, jp_tickers, blend_weight):
    combined = blpx_signals.reindex(sim_dates).fillna(0.0).copy()
    ticker_signals = {}
    for signal_name, ticker, ic in sig_pairs:
        ticker_signals.setdefault(ticker, []).append((signal_name, ic))
    for j, tk in enumerate(jp_tickers):
        if tk not in ticker_signals:
            continue
        for signal_name, ic in ticker_signals[tk]:
            if signal_name not in alt_data:
                continue
            alt_series = alt_data[signal_name].reindex(sim_dates).shift(1)
            rmean = alt_series.rolling(252, min_periods=60).mean()
            rstd = alt_series.rolling(252, min_periods=60).std()
            z_alt = (alt_series - rmean) / rstd.replace(0, np.nan)
            combined[tk] = combined[tk] + (blend_weight * ic * z_alt).reindex(sim_dates).fillna(0.0).values
    return combined


class CombinedModel:
    def __init__(self, combined_signals, df_exec):
        self.combined_signals = combined_signals
        self.df_exec = df_exec
        self.n_j = len(JP_TICKERS)
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"
        self._wc = 0

    def predict_signals(self, df_exec):
        si = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), self.corr_window)
        self._wc = si
        T = len(df_exec)
        sd = df_exec.index
        blpx = self.combined_signals.reindex(sd).fillna(0.0)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sd, columns=JP_TICKERS)
        y_oc = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(columns=lambda c: c.replace("jp_oc_", ""))
        return {"raw_pca_signals": empty, "residual_pca_signals": empty, "p4_signals": empty,
                "signals": blpx, "normalized_signals": blpx, "y_jp_oc_df": y_oc}

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        if self._wc < len(self.df_exec):
            w = build_weights(signal, q=self.q, n_j=self.n_j, weight_mode="signal", enforce_sign=False)
            self._wc += 1
            return w
        return np.zeros(self.n_j)


def run_bt(name, model, df_exec, y_target):
    results = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-01",
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0, slippage_bps=5.0)
    dr = results["daily_returns"]
    ar = float(dr.mean() * 245)
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(results["daily_turnover"].mean())
    sd = df_exec.index
    si = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    y_df = pd.DataFrame(y_target, index=sd, columns=JP_TICKERS)
    ic_list = []
    for i in range(si, len(sd)):
        d = sd[i]
        if d not in results["signals"].index:
            continue
        s = results["signals"].loc[d].values
        y = y_df.loc[d].values
        v = ~(np.isnan(s) | np.isnan(y))
        if v.sum() >= 3:
            rho, _ = stats.spearmanr(s[v], y[v])
            if np.isfinite(rho):
                ic_list.append(float(rho))
    mean_ic = float(np.mean(ic_list)) if ic_list else np.nan
    return {"name": name, "Sharpe": sharpe, "AR": ar, "Vol": vol, "MDD": mdd,
            "Turnover": turnover, "IC": mean_ic}


def main():
    output_dir = ROOT / "artifacts" / "novel_alpha"
    yaml_path = str(ROOT / "configs" / "production.yaml")

    logger.info("Loading data...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)

    alt_data = pd.read_pickle(output_dir / "novel_data.pkl")
    ic_df = pd.read_csv(output_dir / "ic_diagnostic.csv")
    sig_all = ic_df[ic_df["significant"]].sort_values("rank_ic", ascending=False)

    logger.info("Computing baseline BLPX...")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("blpx", {}).update(BASE_PARAMS)
    cfg["signal_components"] = SIGNAL_WEIGHTS
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    blpx_signals = model_base.predict_signals(df_exec)["signals"]

    top_n_values = [10, 20, 30, 50, 100, 268]
    blend_weights = [0.20, 0.30, 0.40, 0.50]
    all_results = []

    # Baseline
    logger.info("=== baseline ===")
    m = CombinedModel(blpx_signals, df_exec)
    r = run_bt("baseline", m, df_exec, y_target)
    all_results.append(r)
    logger.info("  Sharpe=%.4f", r["Sharpe"])

    for top_n in top_n_values:
        sig_pairs = [(row["signal"], row["ticker"], row["rank_ic"])
                     for _, row in sig_all.head(top_n).iterrows()]
        for bw in blend_weights:
            name = f"top{top_n}_bw{bw:.2f}"
            logger.info("=== %s ===", name)
            combined = build_combined_signal(blpx_signals, alt_data, sig_pairs,
                                              sim_dates, JP_TICKERS, bw)
            m = CombinedModel(combined, df_exec)
            r = run_bt(name, m, df_exec, y_target)
            r["top_n"] = top_n
            r["bw"] = bw
            all_results.append(r)
            logger.info("  Sharpe=%.4f MDD=%.2f%% IC=%.4f", r["Sharpe"], r["MDD"]*100, r["IC"])

    # Print results
    base_s = all_results[0]["Sharpe"]
    print("\n" + "=" * 110)
    print("TOP-N REFINEMENT RESULTS")
    print("=" * 110)
    print(f"\n{'Name':<25} {'Sharpe':<10} {'dSharpe':<10} {'MDD%':<8} {'Turnover':<10} {'IC':<10}")
    print("-" * 80)
    for r in all_results:
        ds = r["Sharpe"] - base_s if np.isfinite(r["Sharpe"]) else np.nan
        print(f"{r['name']:<25} {r['Sharpe']:<10.4f} {ds:<+10.4f} {r['MDD']*100:<8.2f} {r['Turnover']:<10.2f} {r['IC']:<10.4f}")

    valid = [r for r in all_results if r["name"] != "baseline" and np.isfinite(r["Sharpe"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe']:.4f} (dSharpe={best['Sharpe']-base_s:+.4f})")

    pd.DataFrame(all_results).to_csv(output_dir / "refinement_results.csv", index=False)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
