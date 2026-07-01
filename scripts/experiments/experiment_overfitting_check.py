"""Overfitting check: (1) Holdout 2026, (2) Deflated Sharpe Ratio.

1. Holdout: Re-run walk-forward on 2017-2025 only, then test top50_bw0.20 on 2026 holdout.
2. DSR: Bailey & López de Prado (2014) multiple-testing correction.
"""

from __future__ import annotations
import logging, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.stats import norm

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
MIN_IC=0.02; P_THRESHOLD=0.05; TOP_N=50; BW=0.20

# Number of trials for DSR (total configurations tested across all experiments)
N_TRIALS = 42  # 25 (high-precision) + 11 (mid-range) + 6 (reference methods)


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


def deflated_sharpe_ratio(returns, sr_observed, n_trials, freq=245):
    """Compute Deflated Sharpe Ratio (Bailey & López de Prado 2014).
    
    Returns DSR = P(SR_true > E[max(SR)|N] | observed data)
    DSR > 0.95 means the observed SR is statistically significant even after
    accounting for multiple testing.
    """
    T = len(returns)
    r = returns.dropna().values
    
    # Skewness and kurtosis of returns
    skew = float(stats.skew(r))
    kurt_excess = float(stats.kurtosis(r))  # excess kurtosis (kurt - 3)
    
    # Sharpe ratio (annualized -> per-period for formula)
    sr_per = sr_observed / math.sqrt(freq)
    
    # Variance of SR estimator (Lo 2002, corrected for higher moments)
    # Var(SR) = (1/(T-1)) * (1 - SR^2/4 * (skew^2 + (kurt_excess^2)/4) + SR*skew - SR^2)
    # Using the formula from Bailey & LdP:
    sr_var = (1.0 / (T - 1)) * (
        1.0 
        - 0.25 * sr_per * skew 
        + 0.25 * sr_per**2 * (kurt_excess / 4.0)
        + 0.5 * sr_per**2
    )
    sr_var = max(sr_var, 1e-10)
    sr_std = math.sqrt(sr_var)
    
    # Expected max SR under null (N independent trials)
    # E[max(Z_1...Z_N)] ≈ sqrt(2*ln(N)) - (gamma - ln(2*ln(N))) / (2*sqrt(2*ln(N)))
    # where gamma = Euler-Mascheroni constant ≈ 0.5772
    gamma = 0.5772156649
    if n_trials > 1:
        e_max = math.sqrt(2 * math.log(n_trials))
        correction = (gamma - math.log(2 * math.log(n_trials))) / (2 * math.sqrt(2 * math.log(n_trials)))
        e_max_sr = (e_max - correction) * sr_std * math.sqrt(freq)  # annualized
    else:
        e_max_sr = 0.0
    
    # DSR = Phi((SR_observed - E[max]) / SE(SR))
    # Per-period
    e_max_per = e_max_sr / math.sqrt(freq)
    dsr = float(norm.cdf((sr_per - e_max_per) / sr_std))
    
    return {
        "DSR": dsr,
        "E_max_SR": e_max_sr,
        "SR_std": sr_std * math.sqrt(freq),  # annualized
        "T": T,
        "skew": skew,
        "kurt_excess": kurt_excess,
        "n_trials": n_trials,
    }


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

    # Split: train-validation (pre-2026) vs holdout (2026+)
    HOLDOUT_DATE = pd.to_datetime("2026-01-01")
    val_windows = [(tr_s, tr_e, te_s, te_e) for tr_s, tr_e, te_s, te_e in windows 
                   if sim_dates[te_s] < HOLDOUT_DATE]
    holdout_windows = [(tr_s, tr_e, te_s, te_e) for tr_s, tr_e, te_s, te_e in windows 
                       if sim_dates[te_s] >= HOLDOUT_DATE]
    
    logger.info("Total windows: %d, Validation: %d, Holdout: %d", 
                len(windows), len(val_windows), len(holdout_windows))

    # === PART 1: Holdout validation ===
    logger.info("=" * 60)
    logger.info("PART 1: HOLDOUT VALIDATION (2026+)")
    logger.info("=" * 60)
    
    # Run baseline and top50 on validation period
    val_base_dr = []; val_comb_dr = []; val_deltas = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(val_windows):
        td = sim_dates[te_s:te_e]
        r_base = run_window(blpx, df_exec, y_target, te_s, te_e)
        dr_base = r_base["daily_returns"].reindex(td).dropna()
        val_base_dr.append(dr_base)
        
        pairs = compute_train_ic_topn(alt_data, y_target, sim_dates, JP_TICKERS, tr_s, tr_e, TOP_N)
        if pairs:
            combined = build_combined(blpx, alt_data, pairs, sim_dates, JP_TICKERS, BW)
        else:
            combined = blpx
        r_comb = run_window(combined, df_exec, y_target, te_s, te_e)
        dr_comb = r_comb["daily_returns"].reindex(td).dropna()
        val_comb_dr.append(dr_comb)
        
        def sh(dr):
            if len(dr) < 5: return np.nan
            ar = dr.mean() * 245; vol = dr.std(ddof=1) * np.sqrt(245)
            return ar/vol if vol > 0 else np.nan
        sb, sc = sh(dr_base), sh(dr_comb)
        val_deltas.append(sc - sb if np.isfinite(sb) and np.isfinite(sc) else np.nan)
        logger.info("Val W%d: base=%.2f comb=%.2f delta=%+.2f", wi+1, sb, sc, sc-sb)

    val_base_all = pd.concat(val_base_dr)
    val_comb_all = pd.concat(val_comb_dr)
    
    def full_metrics(dr):
        ar = dr.mean() * 245; vol = dr.std(ddof=1) * np.sqrt(245)
        sh = ar/vol if vol > 0 else np.nan
        wealth = (1+dr).cumprod()
        mdd = ((wealth/wealth.cummax())-1).min()
        return sh, ar, vol, mdd
    
    sh_b, ar_b, vol_b, mdd_b = full_metrics(val_base_all)
    sh_c, ar_c, vol_c, mdd_c = full_metrics(val_comb_all)
    
    print("\n" + "=" * 80)
    print("VALIDATION PERIOD (pre-2026)")
    print("=" * 80)
    print(f"  Windows: {len(val_windows)}")
    print(f"  Baseline:  Sharpe={sh_b:.4f}  AR={ar_b:.4f}  Vol={vol_b:.4f}  MDD={mdd_b*100:.2f}%")
    print(f"  Top50:     Sharpe={sh_c:.4f}  AR={ar_c:.4f}  Vol={vol_c:.4f}  MDD={mdd_c*100:.2f}%")
    print(f"  Delta:     Sharpe={sh_c-sh_b:+.4f}  AR={ar_c-ar_b:+.4f}  MDD={(mdd_c-mdd_b)*100:+.2f}%")
    print(f"  Win rate:  {sum(1 for d in val_deltas if d > 0)}/{len(val_deltas)} ({sum(1 for d in val_deltas if d > 0)/len(val_deltas)*100:.0f}%)")

    # Run on holdout
    logger.info("Running holdout (2026+)...")
    hold_base_dr = []; hold_comb_dr = []; hold_deltas = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(holdout_windows):
        td = sim_dates[te_s:te_e]
        r_base = run_window(blpx, df_exec, y_target, te_s, te_e)
        dr_base = r_base["daily_returns"].reindex(td).dropna()
        hold_base_dr.append(dr_base)
        
        pairs = compute_train_ic_topn(alt_data, y_target, sim_dates, JP_TICKERS, tr_s, tr_e, TOP_N)
        if pairs:
            combined = build_combined(blpx, alt_data, pairs, sim_dates, JP_TICKERS, BW)
        else:
            combined = blpx
        r_comb = run_window(combined, df_exec, y_target, te_s, te_e)
        dr_comb = r_comb["daily_returns"].reindex(td).dropna()
        hold_comb_dr.append(dr_comb)
        
        def sh(dr):
            if len(dr) < 5: return np.nan
            ar = dr.mean() * 245; vol = dr.std(ddof=1) * np.sqrt(245)
            return ar/vol if vol > 0 else np.nan
        sb, sc = sh(dr_base), sh(dr_comb)
        hold_deltas.append(sc - sb if np.isfinite(sb) and np.isfinite(sc) else np.nan)
        logger.info("Holdout W%d: base=%.2f comb=%.2f delta=%+.2f", wi+1, sb, sc, sc-sb)

    hold_base_all = pd.concat(hold_base_dr)
    hold_comb_all = pd.concat(hold_comb_dr)
    
    sh_hb, ar_hb, vol_hb, mdd_hb = full_metrics(hold_base_all)
    sh_hc, ar_hc, vol_hc, mdd_hc = full_metrics(hold_comb_all)
    
    print("\n" + "=" * 80)
    print("HOLDOUT PERIOD (2026+)")
    print("=" * 80)
    print(f"  Windows: {len(holdout_windows)}")
    print(f"  Baseline:  Sharpe={sh_hb:.4f}  AR={ar_hb:.4f}  Vol={vol_hb:.4f}  MDD={mdd_hb*100:.2f}%")
    print(f"  Top50:     Sharpe={sh_hc:.4f}  AR={ar_hc:.4f}  Vol={vol_hc:.4f}  MDD={mdd_hc*100:.2f}%")
    print(f"  Delta:     Sharpe={sh_hc-sh_hb:+.4f}  AR={ar_hc-ar_hb:+.4f}  MDD={(mdd_hc-mdd_hb)*100:+.2f}%")
    print(f"  Win rate:  {sum(1 for d in hold_deltas if d > 0)}/{len(hold_deltas)} ({sum(1 for d in hold_deltas if d > 0)/len(hold_deltas)*100:.0f}%)")

    # === PART 2: Deflated Sharpe Ratio ===
    logger.info("=" * 60)
    logger.info("PART 2: DEFLATED SHARPE RATIO")
    logger.info("=" * 60)
    
    # DSR for the combined strategy's OOS returns (full period)
    full_comb_dr = pd.concat([val_comb_all, hold_comb_all])
    full_base_dr = pd.concat([val_base_all, hold_base_all])
    
    sh_full_base, _, _, _ = full_metrics(full_base_dr)
    sh_full_comb, _, _, _ = full_metrics(full_comb_dr)
    delta_sharpe = sh_full_comb - sh_full_base
    
    # DSR on the combined strategy
    dsr_comb = deflated_sharpe_ratio(full_comb_dr, sh_full_comb, N_TRIALS)
    # DSR on baseline (1 trial, no multiple testing concern)
    dsr_base = deflated_sharpe_ratio(full_base_dr, sh_full_base, 1)
    # DSR on the delta (is the improvement real?)
    # For delta, we use the difference in daily returns
    delta_returns = full_comb_dr.reindex(full_base_dr.index) - full_base_dr
    delta_returns = delta_returns.dropna()
    sh_delta = delta_returns.mean() * 245 / (delta_returns.std(ddof=1) * np.sqrt(245))
    dsr_delta = deflated_sharpe_ratio(delta_returns, sh_delta, N_TRIALS)
    
    print("\n" + "=" * 80)
    print("DEFLATED SHARPE RATIO (Bailey & López de Prado 2014)")
    print("=" * 80)
    print(f"  Number of trials (total configurations tested): {N_TRIALS}")
    print(f"  OOS period: {full_comb_dr.index[0].date()} ~ {full_comb_dr.index[-1].date()}")
    print(f"  Trading days: {len(full_comb_dr)}")
    print()
    print(f"  {'Metric':<25} {'Value':<15} {'Details'}")
    print(f"  {'-'*70}")
    print(f"  {'Baseline Sharpe':<25} {sh_full_base:<15.4f}")
    print(f"  {'Combined Sharpe':<25} {sh_full_comb:<15.4f}")
    print(f"  {'Delta Sharpe':<25} {delta_sharpe:<15.4f}")
    print()
    print(f"  --- DSR for Combined Strategy ---")
    print(f"  {'DSR':<25} {dsr_comb['DSR']:<15.4f}  {'(>0.95 = significant after multiple testing)'}")
    print(f"  {'E[max SR|N]':<25} {dsr_comb['E_max_SR']:<15.4f}  {'(expected best SR under null)'}")
    print(f"  {'SE(SR)':<25} {dsr_comb['SR_std']:<15.4f}")
    print(f"  {'Skewness':<25} {dsr_comb['skew']:<15.4f}")
    print(f"  {'Excess Kurtosis':<25} {dsr_comb['kurt_excess']:<15.4f}")
    print()
    print(f"  --- DSR for Delta (improvement over baseline) ---")
    print(f"  {'Delta SR (annualized)':<25} {sh_delta:<15.4f}")
    print(f"  {'DSR(delta)':<25} {dsr_delta['DSR']:<15.4f}  {'(>0.95 = improvement is real)'}")
    print(f"  {'E[max SR|N] for delta':<25} {dsr_delta['E_max_SR']:<15.4f}")
    print(f"  {'SE(delta SR)':<25} {dsr_delta['SR_std']:<15.4f}")
    
    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    holdout_positive = sh_hc > sh_hb if np.isfinite(sh_hc) and np.isfinite(sh_hb) else False
    dsr_significant = dsr_comb['DSR'] > 0.95
    delta_significant = dsr_delta['DSR'] > 0.95
    
    print(f"  1. Holdout positive:    {'YES' if holdout_positive else 'NO'} (delta={sh_hc-sh_hb:+.4f})")
    print(f"  2. DSR(combined) > 0.95: {'YES' if dsr_significant else 'NO'} (DSR={dsr_comb['DSR']:.4f})")
    print(f"  3. DSR(delta) > 0.95:   {'YES' if delta_significant else 'NO'} (DSR={dsr_delta['DSR']:.4f})")
    print()
    if holdout_positive and dsr_significant and delta_significant:
        print("  => PASS: Improvement is statistically significant and holds in holdout.")
    elif holdout_positive and (dsr_significant or delta_significant):
        print("  => PARTIAL: Holdout positive but DSR marginal. Cautious adoption.")
    elif holdout_positive:
        print("  => WEAK: Holdout positive but DSR not significant. High overfitting risk.")
    else:
        print("  => FAIL: Holdout negative. Improvement likely overfit.")
    
    # Save
    results = {
        "validation": {"sharpe_base": sh_b, "sharpe_comb": sh_c, "delta": sh_c-sh_b,
                       "win_rate": sum(1 for d in val_deltas if d > 0)/len(val_deltas)},
        "holdout": {"sharpe_base": sh_hb, "sharpe_comb": sh_hc, "delta": sh_hc-sh_hb,
                    "win_rate": sum(1 for d in hold_deltas if d > 0)/len(hold_deltas) if hold_deltas else 0},
        "dsr": {"n_trials": N_TRIALS, "dsr_combined": dsr_comb['DSR'], 
                "dsr_delta": dsr_delta['DSR'], "e_max_sr": dsr_comb['E_max_SR']},
    }
    pd.DataFrame([results]).to_json(output_dir / "overfitting_check.json", indent=2)
    print(f"\nSaved to {output_dir / 'overfitting_check.json'}")


if __name__ == "__main__":
    main()
