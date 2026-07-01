"""Fundamental Architecture Improvements — Comprehensive Experiment.

Tests 5 fundamental portfolio construction improvements vs baseline:

  A. Baseline: Current production (signal-weighted, fixed 5/5)
  B. MVO: Mean-variance optimal weights using rolling covariance
  C. MVO + Turnover: MVO with turnover penalty
  D. Adaptive Positions: Dynamic long/short count by signal dispersion
  E. Continuous Scaling: Sigmoid gross multiplier by predicted IR
  F. Signal Shrinkage: Bayesian shrinkage by prediction uncertainty
  G. All Combined: B+C+D+E+F

Key insight: We compute Omega_gap (17x17 covariance) but ignore it in
portfolio construction. MVO uses this covariance for risk-aware allocation.
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


# ---------------------------------------------------------------------------
# Portfolio construction methods
# ---------------------------------------------------------------------------

def solve_mvo(mu: np.ndarray, Omega: np.ndarray, gross: float = 2.0,
              lambda_reg: float = 0.01) -> np.ndarray:
    """Mean-variance optimal weights (analytical approximation).

    maximize w'mu - (lambda/2) w'Omega w
    subject to sum(w) = 0, sum(|w|) = gross

    Uses ridge-regularized inverse then rescales to gross target.
    """
    n = len(mu)
    Omega_reg = Omega + lambda_reg * np.eye(n)
    try:
        w_raw = np.linalg.solve(Omega_reg, mu)
    except np.linalg.LinAlgError:
        w_raw = mu / (np.sqrt(np.maximum(np.diag(Omega), 1e-8)) + 1e-6)

    w_raw -= np.mean(w_raw)
    abs_sum = np.sum(np.abs(w_raw))
    if abs_sum > 1e-10:
        w_raw *= gross / abs_sum
    else:
        w_raw = np.zeros(n)
    return w_raw


def solve_mvo_turnover(mu: np.ndarray, Omega: np.ndarray, w_prev: np.ndarray,
                       gross: float = 2.0, lambda_reg: float = 0.01,
                       turnover_penalty: float = 0.003) -> np.ndarray:
    """MVO with turnover penalty.

    Adjusts mu by penalizing deviation from previous weights.
    """
    mu_adj = mu - turnover_penalty * np.sign(np.zeros_like(mu) - w_prev)
    return solve_mvo(mu_adj, Omega, gross=gross, lambda_reg=lambda_reg)


def compute_rolling_cov(df_exec: pd.DataFrame, i: int, window: int = 60) -> np.ndarray:
    """Rolling covariance of JP open-close returns (lookahead-safe).

    Uses returns up to t-1 (shifted by 1) to avoid lookahead.
    """
    n_j = len(JP_TICKERS)
    jp_oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    df_oc = df_exec[jp_oc_cols].copy()
    df_oc.columns = JP_TICKERS

    if i < window + 1:
        return np.eye(n_j) * 0.001

    returns_window = df_oc.iloc[i - window:i].values
    # Drop rows with NaN
    valid_mask = np.isfinite(returns_window).all(axis=1)
    valid_returns = returns_window[valid_mask]
    if len(valid_returns) < 20:
        return np.eye(n_j) * 0.001

    cov = np.cov(valid_returns, rowvar=False)
    if not np.isfinite(cov).all():
        return np.eye(n_j) * 0.001

    # Ensure PSD
    try:
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 0:
            cov += (abs(min_eig) + 1e-8) * np.eye(n_j)
    except np.linalg.LinAlgError:
        cov = np.eye(n_j) * 0.001

    return cov


def adaptive_position_count(scores: np.ndarray, n_j: int,
                            base_q: float = 0.3) -> int:
    """Dynamic long/short count based on signal dispersion."""
    centered = scores - np.median(scores)
    disp = np.std(centered)

    # Rolling dispersion would be ideal, but for simplicity use absolute scale
    # If dispersion is high, use more positions; if low, fewer
    if disp < 0.3:
        return max(3, int(n_j * 0.18))  # ~3 positions
    elif disp > 0.8:
        return min(8, int(n_j * 0.47))  # ~8 positions
    else:
        return int(n_j * base_q)  # ~5 positions (default)


def continuous_gross_multiplier(predicted_ir: float,
                                mult_min: float = 0.5,
                                mult_max: float = 1.5,
                                ir_mid: float = 0.5,
                                k: float = 3.0) -> float:
    """Sigmoid mapping of predicted IR to gross multiplier."""
    sigmoid = 1.0 / (1.0 + np.exp(-k * (predicted_ir - ir_mid)))
    return mult_min + (mult_max - mult_min) * sigmoid


def signal_shrinkage(signal: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """Bayesian shrinkage based on prediction uncertainty."""
    sigma = np.sqrt(np.maximum(np.diag(Omega), 1e-8))
    sigma_ref = np.median(sigma)

    # Confidence: high when sigma is low, low when sigma is high
    confidence = sigma_ref / (sigma + sigma_ref)

    # Shrink toward median
    med = np.median(signal)
    return signal * confidence + med * (1.0 - confidence)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class FundamentalImprovementModel:
    """Wraps BLPX signals with configurable portfolio construction.

    Pre-computes all weights in predict_signals, then returns them
    sequentially via build_weights.
    """

    def __init__(self, blpx_signals, df_exec, method: str, params: dict | None = None):
        self.blpx_signals = blpx_signals
        self.df_exec = df_exec
        self.method = method
        self.params = params or {}
        self.n_j = len(JP_TICKERS)
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"
        self._precomputed_weights = None
        self._weight_counter = 0

    def _compute_weights(self) -> pd.DataFrame:
        """Pre-compute weights for all dates based on method."""
        T = len(self.df_exec)
        sim_dates = self.df_exec.index
        signals = self.blpx_signals.reindex(sim_dates).fillna(0.0)
        weights = np.zeros((T, self.n_j))
        w_prev = np.zeros(self.n_j)

        cov_window = self.params.get("cov_window", 60)
        gross = self.params.get("gross", 2.0)
        lambda_reg = self.params.get("lambda_reg", 0.01)
        turnover_penalty = self.params.get("turnover_penalty", 0.003)

        for i in range(T):
            sig_i = signals.iloc[i].values

            if self.method == "baseline":
                from leadlag.core.signal import build_weights
                w_i = build_weights(sig_i, q=self.q, n_j=self.n_j,
                                    weight_mode="signal", enforce_sign=False)

            elif self.method == "mvo":
                Omega = compute_rolling_cov(self.df_exec, i, cov_window)
                w_i = solve_mvo(sig_i, Omega, gross=gross, lambda_reg=lambda_reg)

            elif self.method == "mvo_turnover":
                Omega = compute_rolling_cov(self.df_exec, i, cov_window)
                w_i = solve_mvo_turnover(sig_i, Omega, w_prev, gross=gross,
                                         lambda_reg=lambda_reg,
                                         turnover_penalty=turnover_penalty)

            elif self.method == "adaptive":
                n_pos = adaptive_position_count(sig_i, self.n_j)
                from leadlag.core.signal import build_weights
                q_eff = n_pos / self.n_j
                w_i = build_weights(sig_i, q=q_eff, n_j=self.n_j,
                                    weight_mode="signal", enforce_sign=False)

            elif self.method == "continuous_scale":
                from leadlag.core.signal import build_weights
                w_base = build_weights(sig_i, q=self.q, n_j=self.n_j,
                                       weight_mode="signal", enforce_sign=False)
                Omega = compute_rolling_cov(self.df_exec, i, cov_window)
                p_mean = float(np.dot(w_base, sig_i))
                p_vol = float(np.sqrt(max(0.0, np.dot(w_base, np.dot(Omega, w_base)))))
                pred_ir = p_mean / p_vol if p_vol > 1e-6 else 0.0
                mult = continuous_gross_multiplier(pred_ir)
                w_i = w_base * mult

            elif self.method == "shrinkage":
                Omega = compute_rolling_cov(self.df_exec, i, cov_window)
                sig_shrunk = signal_shrinkage(sig_i, Omega)
                from leadlag.core.signal import build_weights
                w_i = build_weights(sig_shrunk, q=self.q, n_j=self.n_j,
                                    weight_mode="signal", enforce_sign=False)

            elif self.method == "all_combined":
                Omega = compute_rolling_cov(self.df_exec, i, cov_window)
                sig_shrunk = signal_shrinkage(sig_i, Omega)
                n_pos = adaptive_position_count(sig_shrunk, self.n_j)
                w_i = solve_mvo_turnover(sig_shrunk, Omega, w_prev, gross=gross,
                                         lambda_reg=lambda_reg,
                                         turnover_penalty=turnover_penalty)
                # Continuous scaling
                p_mean = float(np.dot(w_i, sig_shrunk))
                p_vol = float(np.sqrt(max(0.0, np.dot(w_i, np.dot(Omega, w_i)))))
                pred_ir = p_mean / p_vol if p_vol > 1e-6 else 0.0
                mult = continuous_gross_multiplier(pred_ir)
                w_i = w_i * mult

            else:
                raise ValueError(f"Unknown method: {self.method}")

            weights[i] = w_i
            w_prev = w_i

        return pd.DataFrame(weights, index=sim_dates, columns=JP_TICKERS)

    def predict_signals(self, df_exec):
        if self._precomputed_weights is None:
            self._precomputed_weights = self._compute_weights()

        # Align counter with BacktestEngine's start_idx
        start_dt = pd.to_datetime("2015-01-01")
        start_idx = max(df_exec.index.searchsorted(start_dt), self.corr_window)
        self._weight_counter = start_idx

        T = len(df_exec)
        sim_dates = df_exec.index
        blpx = self.blpx_signals.reindex(sim_dates).fillna(0.0)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "raw_pca_signals": empty,
            "residual_pca_signals": empty,
            "p4_signals": empty,
            "signals": blpx,
            "normalized_signals": blpx,
            "y_jp_oc_df": y_jp_oc_df,
        }

    def build_weights(self, signal, q=None):
        if self._precomputed_weights is not None and self._weight_counter < len(self._precomputed_weights):
            w = self._precomputed_weights.iloc[self._weight_counter].values
            self._weight_counter += 1
            return w
        return np.zeros(self.n_j)


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

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
    gross_exp = float(results["daily_gross_exps"].mean())
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
    return {
        "name": name, "AR_net": ar, "Vol_net": vol, "Sharpe_net": sharpe,
        "MDD": mdd, "Turnover": turnover, "Gross_exp": gross_exp,
        "Mean_Rank_IC": mean_ic, "ICIR": icir, "elapsed_s": elapsed,
    }


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        cfg.setdefault("blpx", {}).update(blpx_overrides)
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fundamental Architecture Improvements Experiment")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/fundamental_improvements")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    # Compute baseline BLPX signals
    logger.info("Computing baseline BLPX signals...")
    cfg_base = build_config(yaml_path, blpx_overrides=BASE_PARAMS,
                            signal_components=SIGNAL_WEIGHTS)
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    pred_base = model_base.predict_signals(df_exec)
    blpx_signals = pred_base["signals"]

    # Define variants
    variants = [
        ("baseline", "baseline", {}),
        ("mvo", "mvo", {"gross": 2.0, "lambda_reg": 0.01, "cov_window": 60}),
        ("mvo_turnover", "mvo_turnover", {"gross": 2.0, "lambda_reg": 0.01,
                                          "cov_window": 60, "turnover_penalty": 0.003}),
        ("adaptive_pos", "adaptive", {}),
        ("continuous_scale", "continuous_scale", {"cov_window": 60}),
        ("signal_shrinkage", "shrinkage", {"cov_window": 60}),
        ("all_combined", "all_combined", {"gross": 2.0, "lambda_reg": 0.01,
                                          "cov_window": 60, "turnover_penalty": 0.003}),
    ]

    all_results = []

    for name, method, params in variants:
        logger.info("=== %s ===", name)
        model = FundamentalImprovementModel(blpx_signals, df_exec, method, params)
        m = run_backtest(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
        m["method"] = method
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f AR=%.4f Vol=%.4f MDD=%.2f%% Turnover=%.2f Gross=%.2f (%.1fs)",
                    name, m["Sharpe_net"], m["AR_net"], m["Vol_net"],
                    m["MDD"] * 100, m["Turnover"], m["Gross_exp"], m["elapsed_s"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Print comparison table
    print("\n" + "=" * 120)
    print("FUNDAMENTAL ARCHITECTURE IMPROVEMENTS — RESULTS")
    print("=" * 120)

    baseline_sharpe = all_results[0]["Sharpe_net"] if all_results else 0.0
    baseline_turnover = all_results[0]["Turnover"] if all_results else 0.0

    print(f"\n{'Name':<20} {'Sharpe':<10} {'AR':<10} {'Vol':<10} {'MDD%':<8} "
          f"{'Turnover':<10} {'Gross':<8} {'IC':<10} {'ICIR':<8} {'Delta':<8}")
    print("-" * 120)
    for r in all_results:
        delta = r["Sharpe_net"] - baseline_sharpe if np.isfinite(r["Sharpe_net"]) else np.nan
        print(f"{r['name']:<20} {r['Sharpe_net']:<10.4f} {r['AR_net']:<10.4f} {r['Vol_net']:<10.4f} "
              f"{r['MDD']*100:<8.2f} {r['Turnover']:<10.2f} {r['Gross_exp']:<8.2f} "
              f"{r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {delta:+.4f}")

    # Find best
    valid = [r for r in all_results if r["name"] != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} "
              f"(delta={best['Sharpe_net']-baseline_sharpe:+.4f}, "
              f"turnover delta={best['Turnover']-baseline_turnover:+.2f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
