"""Time-series persistence check: per-window OOS Sharpe for top50 vs baseline."""

from __future__ import annotations
import logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
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
MIN_IC=0.02; P_THRESHOLD=0.05; TOP_N=50; BW=0.20


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

    logger.info("Running per-window comparison: baseline vs top50_bw0.20")
    rows = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        # Baseline
        r_base = run_window(blpx, df_exec, y_target, te_s, te_e)
        td = sim_dates[te_s:te_e]
        dr_base = r_base["daily_returns"].reindex(td).dropna()

        # Combined
        sig_pairs = compute_train_ic_topn(alt_data, y_target, sim_dates, JP_TICKERS, tr_s, tr_e, TOP_N)
        if sig_pairs:
            combined = build_combined(blpx, alt_data, sig_pairs, sim_dates, JP_TICKERS, BW)
        else:
            combined = blpx
        r_comb = run_window(combined, df_exec, y_target, te_s, te_e)
        dr_comb = r_comb["daily_returns"].reindex(td).dropna()

        # Per-window metrics
        def metrics(dr):
            if len(dr) < 5: return np.nan, np.nan, np.nan
            ar = dr.mean() * 245; vol = dr.std(ddof=1) * np.sqrt(245)
            sh = ar/vol if vol > 0 else np.nan
            wealth = (1+dr).cumprod()
            mdd = ((wealth/wealth.cummax())-1).min()
            return sh, ar, mdd

        sh_b, ar_b, mdd_b = metrics(dr_base)
        sh_c, ar_c, mdd_c = metrics(dr_comb)

        rows.append({
            "window": wi+1,
            "start": str(sim_dates[te_s].date()),
            "end": str(sim_dates[te_e-1].date()),
            "n_pairs": len(sig_pairs),
            "base_sharpe": sh_b, "comb_sharpe": sh_c,
            "delta_sharpe": sh_c - sh_b if np.isfinite(sh_b) and np.isfinite(sh_c) else np.nan,
            "base_ar": ar_b, "comb_ar": ar_c,
            "base_mdd": mdd_b, "comb_mdd": mdd_c,
            "base_days": len(dr_base), "comb_days": len(dr_comb),
        })
        logger.info("W%d: %s~%s pairs=%d base_sh=%.2f comb_sh=%.2f delta=%+.2f",
                    wi+1, sim_dates[te_s].date(), sim_dates[te_e-1].date(),
                    len(sig_pairs), sh_b, sh_c, sh_c-sh_b)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "per_window_persistence.csv", index=False)

    # Print
    print("\n" + "=" * 120)
    print("PER-WINDOW OOS SHARPE: baseline vs top50_bw0.20")
    print("=" * 120)
    print(f"\n{'W':<4} {'Start':<12} {'End':<12} {'Pairs':<7} {'Base_SH':<10} {'Comb_SH':<10} {'Delta':<10} {'Base_MDD%':<10} {'Comb_MDD%':<10}")
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"{r['window']:<4} {r['start']:<12} {r['end']:<12} {r['n_pairs']:<7} "
              f"{r['base_sharpe']:<10.2f} {r['comb_sharpe']:<10.2f} {r['delta_sharpe']:<+10.2f} "
              f"{r['base_mdd']*100:<10.2f} {r['comb_mdd']*100:<10.2f}")

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Windows where combined > baseline: {(df['delta_sharpe']>0).sum()} / {len(df)}")
    print(f"Windows where combined < baseline: {(df['delta_sharpe']<0).sum()} / {len(df)}")
    print(f"Mean delta Sharpe: {df['delta_sharpe'].mean():+.4f}")
    print(f"Median delta Sharpe: {df['delta_sharpe'].median():+.4f}")
    print(f"Std delta Sharpe: {df['delta_sharpe'].std():.4f}")
    print(f"Max delta: {df['delta_sharpe'].max():+.4f} (W{df.loc[df['delta_sharpe'].idxmax(),'window']})")
    print(f"Min delta: {df['delta_sharpe'].min():+.4f} (W{df.loc[df['delta_sharpe'].idxmin(),'window']})")

    # Yearly breakdown
    df['year'] = df['start'].str[:4]
    print(f"\n--- By Year ---")
    print(f"{'Year':<6} {'N':<4} {'MeanDelta':<12} {'WinRate':<10} {'MeanPairs':<10}")
    for yr, g in df.groupby('year'):
        print(f"{yr:<6} {len(g):<4} {g['delta_sharpe'].mean():<+12.4f} {(g['delta_sharpe']>0).mean()*100:<10.0f} {g['n_pairs'].mean():<10.0f}")


if __name__ == "__main__":
    main()
