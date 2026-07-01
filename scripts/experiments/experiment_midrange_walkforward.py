"""Mid-range refinement: top50/top100 walk-forward OOS test.

Tests the sweet spot between quality filtering and diversification.
"""

from __future__ import annotations

import logging
import sys
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
from leadlag.core.signal import build_weights

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

TRAIN_WINDOW = 504
TEST_WINDOW = 63
STEP = 63
PURGE = 1
MIN_IC = 0.02
P_THRESHOLD = 0.05


def compute_train_ic_topn(alt_data, y_target, sim_dates, jp_tickers,
                           train_start, train_end, top_n):
    pairs = []
    for signal_name, signal_series in alt_data.items():
        sig_aligned = signal_series.reindex(sim_dates)
        for j, tk in enumerate(jp_tickers):
            y_j = y_target[:, j]
            vals = []
            for i in range(train_start, train_end):
                sv = sig_aligned.iloc[i]
                yv = y_j[i]
                if np.isfinite(sv) and np.isfinite(yv):
                    vals.append((sv, yv))
            if len(vals) < 50:
                continue
            arr = np.array(vals)
            rho, pval = stats.spearmanr(arr[:, 0], arr[:, 1])
            if pval < P_THRESHOLD and abs(rho) >= MIN_IC:
                pairs.append((signal_name, tk, float(rho), abs(float(rho))))
    pairs.sort(key=lambda x: x[3], reverse=True)
    return [(p[0], p[1], p[2]) for p in pairs[:top_n]]


def build_combined(blpx_signals, alt_data, sig_pairs, sim_dates, jp_tickers, bw):
    combined = blpx_signals.reindex(sim_dates).fillna(0.0).copy()
    ts = {}
    for sn, tk, ic in sig_pairs:
        ts.setdefault(tk, []).append((sn, ic))
    for j, tk in enumerate(jp_tickers):
        if tk not in ts:
            continue
        for sn, ic in ts[tk]:
            if sn not in alt_data:
                continue
            s = alt_data[sn].reindex(sim_dates).shift(1)
            rm = s.rolling(252, min_periods=60).mean()
            rs = s.rolling(252, min_periods=60).std()
            z = (s - rm) / rs.replace(0, np.nan)
            combined[tk] = combined[tk] + (bw * ic * z).reindex(sim_dates).fillna(0.0).values
    return combined


class WFModel:
    def __init__(self, cs, df_exec, ws, we):
        self.cs = cs; self.df_exec = df_exec; self.ws = ws; self.we = we
        self.n_j = len(JP_TICKERS); self.n_u = len(US_TICKERS)
        self.corr_window = 60; self.slippage_bps = 5.0; self.q = 0.3
        self.overnight_alpha_long = 0.0; self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"; self._wc = 0; self._w = None

    def _cw(self):
        T = len(self.df_exec)
        w = np.zeros((T, self.n_j))
        for i in range(self.ws, self.we):
            if i >= T: break
            s = self.cs.iloc[i].values
            if np.isfinite(s).any():
                w[i] = build_weights(s, q=self.q, n_j=self.n_j, weight_mode="signal", enforce_sign=False)
        self._w = w

    def predict_signals(self, df_exec):
        if self._w is None: self._cw()
        si = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), self.corr_window)
        self._wc = si
        T = len(df_exec); sd = df_exec.index
        blpx = self.cs.reindex(sd).fillna(0.0)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sd, columns=JP_TICKERS)
        y_oc = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(columns=lambda c: c.replace("jp_oc_", ""))
        return {"raw_pca_signals": empty, "residual_pca_signals": empty, "p4_signals": empty,
                "signals": blpx, "normalized_signals": blpx, "y_jp_oc_df": y_oc}

    def build_weights(self, signal, q=None):
        if self._w is not None and self._wc < len(self._w):
            w = self._w[self._wc]; self._wc += 1; return w
        return np.zeros(self.n_j)


def run_wf_window(combined, df_exec, y_target, te_s, te_e):
    m = WFModel(combined, df_exec, te_s, te_e)
    return BacktestEngine.run_backtest(
        m, df_exec=df_exec, start_date="2015-01-01",
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0, slippage_bps=5.0)


def main():
    output_dir = ROOT / "artifacts" / "novel_alpha"
    yaml_path = str(ROOT / "configs" / "production.yaml")

    logger.info("Loading data...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    T = len(sim_dates)
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    alt_data = pd.read_pickle(output_dir / "novel_data.pkl")

    logger.info("Computing BLPX baseline...")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("blpx", {}).update(BASE_PARAMS)
    cfg["signal_components"] = SIGNAL_WEIGHTS
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    blpx_signals = model_base.predict_signals(df_exec)["signals"]

    windows = []
    ws = start_idx + TRAIN_WINDOW + PURGE
    while ws + TEST_WINDOW <= T:
        tr_s = ws - TRAIN_WINDOW - PURGE; tr_e = ws - PURGE
        te_s = ws; te_e = min(ws + TEST_WINDOW, T)
        windows.append((tr_s, tr_e, te_s, te_e))
        ws += STEP
    logger.info("Windows: %d", len(windows))

    # Test configs: mid-range top_n × moderate bw
    configs = [
        (30, 0.20), (30, 0.30),
        (50, 0.20), (50, 0.30),
        (75, 0.20), (75, 0.30),
        (100, 0.20), (100, 0.30),
        (150, 0.20), (150, 0.30),
        (268, 0.20),  # reference: all pairs
    ]
    all_results = []

    # Baseline
    logger.info("=== baseline ===")
    all_dr = []
    for tr_s, tr_e, te_s, te_e in windows:
        r = run_wf_window(blpx_signals, df_exec, y_target, te_s, te_e)
        td = sim_dates[te_s:te_e]
        all_dr.append(r["daily_returns"].reindex(td).dropna())
    dr_base = pd.concat(all_dr)
    ar = float(dr_base.mean() * 245); vol = float(dr_base.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr_base).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    all_results.append({"name": "baseline", "top_n": 0, "bw": 0.0,
                        "Sharpe": sharpe, "AR": ar, "MDD": mdd, "avg_pairs": 0})
    logger.info("  baseline: Sharpe=%.4f", sharpe)

    for top_n, bw in configs:
        name = f"top{top_n}_bw{bw:.2f}"
        logger.info("=== %s ===", name)
        all_dr = []; n_pairs_list = []
        for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            sig_pairs = compute_train_ic_topn(alt_data, y_target, sim_dates, JP_TICKERS, tr_s, tr_e, top_n)
            n_pairs_list.append(len(sig_pairs))
            if not sig_pairs:
                combined = blpx_signals
            else:
                combined = build_combined(blpx_signals, alt_data, sig_pairs, sim_dates, JP_TICKERS, bw)
            r = run_wf_window(combined, df_exec, y_target, te_s, te_e)
            td = sim_dates[te_s:te_e]
            dr_w = r["daily_returns"].reindex(td).dropna()
            if len(dr_w) > 0:
                all_dr.append(dr_w)
            if wi % 10 == 0:
                logger.info("  W%d/%d pairs=%d", wi+1, len(windows), len(sig_pairs))

        dr_all = pd.concat(all_dr) if all_dr else pd.Series(dtype=float)
        ar = float(dr_all.mean() * 245); vol = float(dr_all.std(ddof=1) * np.sqrt(245))
        sharpe = ar / vol if vol > 0 else np.nan
        wealth = (1.0 + dr_all).cumprod()
        mdd = float(((wealth / wealth.cummax()) - 1.0).min())
        avg_pairs = float(np.mean(n_pairs_list)) if n_pairs_list else 0
        all_results.append({"name": name, "top_n": top_n, "bw": bw,
                            "Sharpe": sharpe, "AR": ar, "MDD": mdd, "avg_pairs": avg_pairs})
        logger.info("  %s: Sharpe=%.4f MDD=%.2f%% avg_pairs=%.0f", name, sharpe, mdd*100, avg_pairs)

    base_s = all_results[0]["Sharpe"]
    print("\n" + "=" * 100)
    print("MID-RANGE REFINEMENT WALK-FORWARD OOS")
    print("=" * 100)
    print(f"\n{'Name':<25} {'Sharpe':<10} {'dSharpe':<10} {'AR':<10} {'MDD%':<8} {'avgPairs':<10}")
    print("-" * 80)
    for r in all_results:
        ds = r["Sharpe"] - base_s if np.isfinite(r["Sharpe"]) else np.nan
        print(f"{r['name']:<25} {r['Sharpe']:<10.4f} {ds:<+10.4f} {r['AR']:<10.4f} "
              f"{r['MDD']*100:<8.2f} {r['avg_pairs']:<10.0f}")

    valid = [r for r in all_results if r["name"] != "baseline" and np.isfinite(r["Sharpe"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe']:.4f} (dSharpe={best['Sharpe']-base_s:+.4f})")
        best_mdd = min(valid, key=lambda x: x["MDD"])
        print(f"Best MDD: {best_mdd['name']} MDD={best_mdd['MDD']*100:.2f}%")

    pd.DataFrame(all_results).to_csv(output_dir / "midrange_walkforward.csv", index=False)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
