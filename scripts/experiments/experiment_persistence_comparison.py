"""Persistence comparison: top5/10/20/50 per-window OOS Sharpe."""

from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel, _BLP_CORR_CACHE, _RAW_PCA_RESIDUAL_PCA_CACHE,
)
from leadlag.core.signal import build_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_PARAMS = {"alpha_xx":0.20,"alpha_yy":0.50,"alpha_yx":0.15,"lambda_pca":0.10,
    "lambda_sector":0.60,"beta_conf":0.25,"rho":0.01,"winsor_sigma":3.0,
    "blp_window":504,"blp_ewma_halflife":120,"sector_eta":0.5,"sector_gamma":4.0}
SIGNAL_WEIGHTS = {"raw_pca":{"enabled":True,"weight":0.2},"residual_pca":{"enabled":False,"weight":0.0},
    "raw_blpx":{"enabled":True,"weight":0.8},"residual_blpx":{"enabled":False,"weight":0.0}}
TRAIN_WINDOW=504; TEST_WINDOW=63; STEP=63; PURGE=1
MIN_IC=0.02; P_THRESHOLD=0.05; BW=0.20
TOP_NS = [5, 10, 20, 50]


def compute_train_ic_topn(alt_data, y_target, sim_dates, jp_tickers, tr_s, tr_e, top_n):
    pairs = []
    for sn, ss in alt_data.items():
        sa = ss.reindex(sim_dates)
        for j, tk in enumerate(jp_tickers):
            yj = y_target[:, j]
            vals = [(sa.iloc[i], yj[i]) for i in range(tr_s, tr_e) if np.isfinite(sa.iloc[i]) and np.isfinite(yj[i])]
            if len(vals) < 50: continue
            arr = np.array(vals)
            rho, pval = stats.spearmanr(arr[:,0], arr[:,1])
            if pval < P_THRESHOLD and abs(rho) >= MIN_IC:
                pairs.append((sn, tk, float(rho), abs(float(rho))))
    pairs.sort(key=lambda x: x[3], reverse=True)
    return [(p[0], p[1], p[2]) for p in pairs[:top_n]]


def build_combined(blpx, alt_data, sig_pairs, sim_dates, jp_tickers, bw):
    combined = blpx.reindex(sim_dates).fillna(0.0).copy()
    ts = {}
    for sn, tk, ic in sig_pairs: ts.setdefault(tk, []).append((sn, ic))
    for j, tk in enumerate(jp_tickers):
        if tk not in ts: continue
        for sn, ic in ts[tk]:
            if sn not in alt_data: continue
            s = alt_data[sn].reindex(sim_dates).shift(1)
            rm = s.rolling(252, min_periods=60).mean()
            rs = s.rolling(252, min_periods=60).std()
            z = (s - rm) / rs.replace(0, np.nan)
            combined[tk] = combined[tk] + (bw * ic * z).reindex(sim_dates).fillna(0.0).values
    return combined


class WFModel:
    def __init__(self, cs, df_exec, ws, we):
        self.cs=cs; self.df_exec=df_exec; self.ws=ws; self.we=we
        self.n_j=len(JP_TICKERS); self.n_u=15; self.corr_window=60
        self.slippage_bps=5.0; self.q=0.3; self.overnight_alpha_long=0.0
        self.overnight_alpha_short=0.0; self.normalization_method="zscore"
        self._wc=0; self._w=None
    def _cw(self):
        T=len(self.df_exec); w=np.zeros((T, self.n_j))
        for i in range(self.ws, self.we):
            if i>=T: break
            s=self.cs.iloc[i].values
            if np.isfinite(s).any():
                w[i]=build_weights(s, q=self.q, n_j=self.n_j, weight_mode="signal", enforce_sign=False)
        self._w=w
    def predict_signals(self, df_exec):
        if self._w is None: self._cw()
        si=max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), self.corr_window)
        self._wc=si; T=len(df_exec); sd=df_exec.index
        blpx=self.cs.reindex(sd).fillna(0.0)
        empty=pd.DataFrame(np.zeros((T, self.n_j)), index=sd, columns=JP_TICKERS)
        y_oc=df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(columns=lambda c:c.replace("jp_oc_",""))
        return {"raw_pca_signals":empty,"residual_pca_signals":empty,"p4_signals":empty,
                "signals":blpx,"normalized_signals":blpx,"y_jp_oc_df":y_oc}
    def build_weights(self, signal, q=None):
        if self._w is not None and self._wc < len(self._w):
            w=self._w[self._wc]; self._wc+=1; return w
        return np.zeros(self.n_j)


def run_window(combined, df_exec, y_target, te_s, te_e):
    m = WFModel(combined, df_exec, te_s, te_e)
    r = BacktestEngine.run_backtest(m, df_exec=df_exec, start_date="2015-01-01",
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0, slippage_bps=5.0)
    return r


def main():
    output_dir = ROOT / "artifacts" / "novel_alpha"
    yaml_path = str(ROOT / "configs" / "production.yaml")
    logger.info("Loading...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index; T = len(sim_dates)
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    alt_data = pd.read_pickle(output_dir / "novel_data.pkl")
    with open(yaml_path) as f: cfg = yaml.safe_load(f)
    cfg.setdefault("blpx", {}).update(BASE_PARAMS)
    cfg["signal_components"] = SIGNAL_WEIGHTS
    _BLP_CORR_CACHE.clear(); _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    blpx = SectorRelativeEnsembleBLPEnhancedModel(cfg).predict_signals(df_exec)["signals"]

    windows = []
    ws = start_idx + TRAIN_WINDOW + PURGE
    while ws + TEST_WINDOW <= T:
        tr_s = ws - TRAIN_WINDOW - PURGE; tr_e = ws - PURGE
        te_s = ws; te_e = min(ws + TEST_WINDOW, T)
        windows.append((tr_s, tr_e, te_s, te_e))
        ws += STEP

    # Precompute IC pairs for each top_n per window
    logger.info("Precomputing IC pairs...")
    ic_cache = {}
    for top_n in TOP_NS:
        ic_cache[top_n] = []
        for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            pairs = compute_train_ic_topn(alt_data, y_target, sim_dates, JP_TICKERS, tr_s, tr_e, top_n)
            ic_cache[top_n].append(pairs)

    # Run baseline once per window
    logger.info("Running baseline + %d top_n variants per window", len(TOP_NS))
    rows = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        td = sim_dates[te_s:te_e]
        # Baseline
        r_base = run_window(blpx, df_exec, y_target, te_s, te_e)
        dr_base = r_base["daily_returns"].reindex(td).dropna()
        def metrics(dr):
            if len(dr) < 5: return np.nan, np.nan
            ar = dr.mean() * 245; vol = dr.std(ddof=1) * np.sqrt(245)
            sh = ar/vol if vol > 0 else np.nan
            wealth = (1+dr).cumprod()
            mdd = ((wealth/wealth.cummax())-1).min()
            return sh, mdd
        sh_b, mdd_b = metrics(dr_base)
        row = {"window": wi+1, "start": str(sim_dates[te_s].date()), "end": str(sim_dates[te_e-1].date()),
               "base_sharpe": sh_b, "base_mdd": mdd_b}
        # Each top_n
        for top_n in TOP_NS:
            pairs = ic_cache[top_n][wi]
            if pairs:
                combined = build_combined(blpx, alt_data, pairs, sim_dates, JP_TICKERS, BW)
            else:
                combined = blpx
            r = run_window(combined, df_exec, y_target, te_s, te_e)
            dr = r["daily_returns"].reindex(td).dropna()
            sh, mdd = metrics(dr)
            row[f"top{top_n}_sharpe"] = sh
            row[f"top{top_n}_delta"] = sh - sh_b if np.isfinite(sh) and np.isfinite(sh_b) else np.nan
            row[f"top{top_n}_mdd"] = mdd
            row[f"top{top_n}_pairs"] = len(pairs)
        rows.append(row)
        logger.info("W%d/%d done", wi+1, len(windows))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "persistence_comparison.csv", index=False)

    # Print per-window table
    print("\n" + "=" * 130)
    print("PER-WINDOW OOS SHARPE: baseline vs top5/10/20/50 (bw=0.20)")
    print("=" * 130)
    hdr = f"{'W':<4} {'Start':<12} {'Base':<8}"
    for tn in TOP_NS:
        hdr += f" {'T'+str(tn):<8} {'d'+str(tn):<8}"
    print(hdr)
    print("-" * 80)
    for _, r in df.iterrows():
        line = f"{r['window']:<4} {r['start']:<12} {r['base_sharpe']:<8.2f}"
        for tn in TOP_NS:
            line += f" {r[f'top{tn}_sharpe']:<8.2f} {r[f'top{tn}_delta']:<+8.2f}"
        print(line)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Delta Sharpe Statistics")
    print(f"{'='*80}")
    print(f"{'Config':<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'WinRate':<10} {'Max':<10} {'Min':<10}")
    print("-" * 75)
    for tn in TOP_NS:
        d = df[f'top{tn}_delta'].dropna()
        wr = (d > 0).sum() / len(d) * 100
        print(f"top{tn}_bw0.20    {d.mean():<+10.4f} {d.median():<+10.4f} {d.std():<10.4f} {wr:<10.0f} {d.max():<+10.2f} {d.min():<+10.2f}")

    # Yearly
    df['year'] = df['start'].str[:4]
    print(f"\n{'='*80}")
    print("YEARLY WIN RATE (%)")
    print(f"{'='*80}")
    hdr = f"{'Year':<6}"
    for tn in TOP_NS:
        hdr += f" {'top'+str(tn):<12}"
    print(hdr)
    print("-" * 60)
    for yr, g in df.groupby('year'):
        line = f"{yr:<6}"
        for tn in TOP_NS:
            d = g[f'top{tn}_delta'].dropna()
            wr = (d > 0).sum() / len(d) * 100 if len(d) > 0 else 0
            line += f" {wr:<12.0f}"
        print(line)

    print(f"\n{'='*80}")
    print("YEARLY MEAN DELTA SHARPE")
    print(f"{'='*80}")
    hdr = f"{'Year':<6}"
    for tn in TOP_NS:
        hdr += f" {'top'+str(tn):<12}"
    print(hdr)
    print("-" * 60)
    for yr, g in df.groupby('year'):
        line = f"{yr:<6}"
        for tn in TOP_NS:
            d = g[f'top{tn}_delta'].dropna()
            line += f" {d.mean():<+12.4f}"
        print(line)

    # Consistency: how many consecutive winning windows?
    print(f"\n{'='*80}")
    print("CONSISTENCY: Consecutive winning windows (max streak)")
    print(f"{'='*80}")
    for tn in TOP_NS:
        d = df[f'top{tn}_delta'].dropna().values
        max_streak = 0; cur = 0
        for v in d:
            if v > 0: cur += 1; max_streak = max(max_streak, cur)
            else: cur = 0
        print(f"top{tn}: max winning streak = {max_streak} windows ({max_streak*63} trading days)")
        max_lose = 0; cur = 0
        for v in d:
            if v <= 0: cur += 1; max_lose = max(max_lose, cur)
            else: cur = 0
        print(f"top{tn}: max losing streak  = {max_lose} windows ({max_lose*63} trading days)")

    print(f"\nSaved to {output_dir / 'persistence_comparison.csv'}")


if __name__ == "__main__":
    main()
