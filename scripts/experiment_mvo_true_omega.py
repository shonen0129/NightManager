"""MVO with True BLPX Omega_gap — Fundamental Improvement Experiment.

Previous experiment used naive rolling covariance which degraded Sharpe.
This experiment extracts the TRUE Omega_gap from BLPX model internals:

  Omega_raw = diag(sigma_Y_denorm) @ C_YY @ diag(sigma_Y_denorm)
  D_gap = diag(1 / max(1 + gap_filt, 0.1))
  Omega_gap = D_gap @ Omega_raw @ D_gap

Then uses Omega_gap for mean-variance optimal portfolio construction.

Variants:
  A. baseline: signal-weighted (current production)
  B. mvo_true: MVO with true BLPX Omega_gap
  C. mvo_true_turnover: MVO + turnover penalty
  D. mvo_true_shrinkage: MVO + signal shrinkage
  E. mvo_true_combined: MVO + turnover + shrinkage
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


# ---------------------------------------------------------------------------
# MVO solvers
# ---------------------------------------------------------------------------

def solve_mvo(mu: np.ndarray, Omega: np.ndarray, gross: float = 2.0,
              lambda_reg: float = 0.01) -> np.ndarray:
    """Mean-variance optimal weights (ridge-regularized analytical solution)."""
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
    """MVO with turnover penalty via adjusted mu."""
    mu_adj = mu - turnover_penalty * np.sign(np.zeros_like(mu) - w_prev)
    return solve_mvo(mu_adj, Omega, gross=gross, lambda_reg=lambda_reg)


def signal_shrinkage(signal: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """Bayesian shrinkage based on prediction uncertainty."""
    sigma = np.sqrt(np.maximum(np.diag(Omega), 1e-8))
    sigma_ref = np.median(sigma)
    confidence = sigma_ref / (sigma + sigma_ref)
    med = np.median(signal)
    return signal * confidence + med * (1.0 - confidence)


# ---------------------------------------------------------------------------
# True Omega_gap extraction
# ---------------------------------------------------------------------------

def precompute_signals_and_omega(model, df_exec, cfg):
    """Loop through all dates, extract BLPX signals + true Omega_gap.

    Returns:
        signals_df: DataFrame of BLPX signals (T, n_j)
        mu_gap_arr: ndarray of mu_gap values (T, n_j)
        omega_gap_arr: ndarray of Omega_gap matrices (T, n_j, n_j)
    """
    inputs = model._prepare_common_inputs(df_exec)
    jp_gap = inputs["jp_gap"]
    jp_beta = inputs["jp_beta"]
    topix_night = inputs["topix_night"]
    jp_res_returns_p3 = inputs["jp_res_returns_p3"]
    c_full_p3 = inputs["c_full_p3"]
    v0_static = inputs["v0_static"]
    all_returns_raw = inputs["all_returns_raw"]

    T = len(df_exec)
    n_j = model.n_j
    signals = np.zeros((T, n_j))
    mu_gap_arr = np.zeros((T, n_j))
    omega_gap_arr = np.zeros((T, n_j, n_j))

    c = model.gap_open_coef
    b = model.topix_beta_coef

    start_idx = model.corr_window

    for i in range(T):
        if i < start_idx:
            continue

        gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else np.zeros(n_j)
        topix_night_t = float(topix_night[i]) if topix_night is not None else 0.0

        try:
            res = model.compute_blp_signal(
                all_returns_raw, i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
                rolling_std=None,
                v0_static=v0_static,
                c_full=c_full_p3,
                is_residual=True,
                return_matrices=True,
            )
        except Exception as e:
            logger.warning(f"compute_blp_signal failed at i={i}: {e}")
            continue

        signal = res["signal"]
        signals[i] = signal
        mu_gap_arr[i] = signal

        # Construct true Omega_gap
        sigma_Y_denorm = res.get("sigma_Y_denorm")
        C_YY = res.get("Sigma_YY")

        if sigma_Y_denorm is not None and C_YY is not None:
            sigma_Y_denorm = np.nan_to_num(sigma_Y_denorm, nan=0.01, posinf=0.01, neginf=0.01)
            sigma_Y_denorm = np.maximum(sigma_Y_denorm, 1e-6)
            C_YY = np.nan_to_num(C_YY, nan=0.0, posinf=0.0, neginf=0.0)

            Omega_raw = np.diag(sigma_Y_denorm) @ C_YY @ np.diag(sigma_Y_denorm)

            # Gap adjustment (same as compute_gap_adjusted_distribution.py)
            gap_syst = betas_t * topix_night_t
            gap_idio = gap_override - gap_syst
            gap_filt = c * gap_idio + (c - b) * gap_syst
            denom = np.maximum(1.0 + gap_filt, 0.1)
            D_gap = np.diag(1.0 / denom)

            Omega_gap = D_gap @ Omega_raw @ D_gap
            Omega_gap = 0.5 * (Omega_gap + Omega_gap.T)

            # Ensure PSD
            try:
                min_eig = np.min(np.linalg.eigvalsh(Omega_gap))
                if min_eig < 0:
                    Omega_gap += (abs(min_eig) + 1e-8) * np.eye(n_j)
            except np.linalg.LinAlgError:
                Omega_gap = np.eye(n_j) * 0.001

            omega_gap_arr[i] = Omega_gap
        else:
            omega_gap_arr[i] = np.eye(n_j) * 0.001

    signals_df = pd.DataFrame(signals, index=df_exec.index, columns=JP_TICKERS)
    return signals_df, mu_gap_arr, omega_gap_arr


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class MvoTrueOmegaModel:
    """Uses pre-computed BLPX signals + true Omega_gap for MVO weights."""

    def __init__(self, signals_df, mu_gap_arr, omega_gap_arr, df_exec,
                 method: str, params: dict | None = None):
        self.signals_df = signals_df
        self.mu_gap_arr = mu_gap_arr
        self.omega_gap_arr = omega_gap_arr
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
        T = len(self.df_exec)
        sim_dates = self.df_exec.index
        weights = np.zeros((T, self.n_j))
        w_prev = np.zeros(self.n_j)

        gross = self.params.get("gross", 2.0)
        lambda_reg = self.params.get("lambda_reg", 0.01)
        turnover_penalty = self.params.get("turnover_penalty", 0.003)

        for i in range(T):
            mu_i = self.mu_gap_arr[i]
            Omega_i = self.omega_gap_arr[i]

            if i < self.corr_window or not np.isfinite(mu_i).any():
                weights[i] = np.zeros(self.n_j)
                continue

            if self.method == "baseline":
                from leadlag.core.signal import build_weights
                w_i = build_weights(mu_i, q=self.q, n_j=self.n_j,
                                    weight_mode="signal", enforce_sign=False)

            elif self.method == "mvo_true":
                w_i = solve_mvo(mu_i, Omega_i, gross=gross, lambda_reg=lambda_reg)

            elif self.method == "mvo_true_turnover":
                w_i = solve_mvo_turnover(mu_i, Omega_i, w_prev, gross=gross,
                                         lambda_reg=lambda_reg,
                                         turnover_penalty=turnover_penalty)

            elif self.method == "mvo_true_shrinkage":
                mu_shrunk = signal_shrinkage(mu_i, Omega_i)
                w_i = solve_mvo(mu_shrunk, Omega_i, gross=gross, lambda_reg=lambda_reg)

            elif self.method == "mvo_true_combined":
                mu_shrunk = signal_shrinkage(mu_i, Omega_i)
                w_i = solve_mvo_turnover(mu_shrunk, Omega_i, w_prev, gross=gross,
                                         lambda_reg=lambda_reg,
                                         turnover_penalty=turnover_penalty)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            weights[i] = w_i
            w_prev = w_i

        return pd.DataFrame(weights, index=sim_dates, columns=JP_TICKERS)

    def predict_signals(self, df_exec):
        if self._precomputed_weights is None:
            self._precomputed_weights = self._compute_weights()

        start_dt = pd.to_datetime("2015-01-01")
        start_idx = max(df_exec.index.searchsorted(start_dt), self.corr_window)
        self._weight_counter = start_idx

        T = len(df_exec)
        sim_dates = df_exec.index
        blpx = self.signals_df.reindex(sim_dates).fillna(0.0)
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
# Helpers
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
    parser = argparse.ArgumentParser(description="MVO with True BLPX Omega_gap")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/mvo_true_omega")
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--turnover-penalty", type=float, default=0.003)
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    # Setup BLPX model
    logger.info("Setting up BLPX model...")
    cfg_base = build_config(yaml_path, blpx_overrides=BASE_PARAMS,
                            signal_components=SIGNAL_WEIGHTS)
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)

    # Pre-compute signals and true Omega_gap for all dates
    logger.info("Pre-computing BLPX signals + true Omega_gap (this may take a while)...")
    t0 = time.perf_counter()
    signals_df, mu_gap_arr, omega_gap_arr = precompute_signals_and_omega(model, df_exec, cfg_base)
    elapsed_pre = time.perf_counter() - t0
    logger.info(f"Pre-computation done in {elapsed_pre:.1f}s")

    # Define variants
    variants = [
        ("baseline", "baseline", {}),
        ("mvo_true", "mvo_true", {"gross": 2.0, "lambda_reg": args.lambda_reg}),
        ("mvo_true_turnover", "mvo_true_turnover",
         {"gross": 2.0, "lambda_reg": args.lambda_reg, "turnover_penalty": args.turnover_penalty}),
        ("mvo_true_shrinkage", "mvo_true_shrinkage",
         {"gross": 2.0, "lambda_reg": args.lambda_reg}),
        ("mvo_true_combined", "mvo_true_combined",
         {"gross": 2.0, "lambda_reg": args.lambda_reg, "turnover_penalty": args.turnover_penalty}),
    ]

    all_results = []

    for name, method, params in variants:
        logger.info("=== %s ===", name)
        mvo_model = MvoTrueOmegaModel(signals_df, mu_gap_arr, omega_gap_arr,
                                      df_exec, method, params)
        m = run_backtest(name, mvo_model, df_exec, y_target, slippage_bps=args.slippage_bps)
        m["method"] = method
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f AR=%.4f Vol=%.4f MDD=%.2f%% Turnover=%.2f Gross=%.2f (%.1fs)",
                    name, m["Sharpe_net"], m["AR_net"], m["Vol_net"],
                    m["MDD"] * 100, m["Turnover"], m["Gross_exp"], m["elapsed_s"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Print comparison table
    print("\n" + "=" * 120)
    print("MVO WITH TRUE BLPX Omega_gap — RESULTS")
    print("=" * 120)

    baseline_sharpe = all_results[0]["Sharpe_net"] if all_results else 0.0
    baseline_turnover = all_results[0]["Turnover"] if all_results else 0.0

    print(f"\n{'Name':<25} {'Sharpe':<10} {'AR':<10} {'Vol':<10} {'MDD%':<8} "
          f"{'Turnover':<10} {'Gross':<8} {'IC':<10} {'ICIR':<8} {'Delta':<8}")
    print("-" * 120)
    for r in all_results:
        delta = r["Sharpe_net"] - baseline_sharpe if np.isfinite(r["Sharpe_net"]) else np.nan
        print(f"{r['name']:<25} {r['Sharpe_net']:<10.4f} {r['AR_net']:<10.4f} {r['Vol_net']:<10.4f} "
              f"{r['MDD']*100:<8.2f} {r['Turnover']:<10.2f} {r['Gross_exp']:<8.2f} "
              f"{r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {delta:+.4f}")

    valid = [r for r in all_results if r["name"] != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} "
              f"(delta={best['Sharpe_net']-baseline_sharpe:+.4f}, "
              f"turnover delta={best['Turnover']-baseline_turnover:+.2f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
