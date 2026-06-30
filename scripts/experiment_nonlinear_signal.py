"""Non-Linear Signal Enhancement — Fundamental Improvement.

Sprint 3-A tested hinge features as overlay → FDR rejected all (0% selection).
This experiment takes a different approach: non-linear transformation of the
BLPX signal ITSELF, not adding external features.

Key insight: BLPX signal is z_hat = B_struct @ z_U (purely linear).
The lead-lag relationship may be NON-LINEAR in US return magnitude:
  - Large US moves → strong, reliable signal
  - Flat US days → signal is noise

Variants:
  A. baseline: Raw BLPX signal
  B. US magnitude amplification: signal * (1 + lambda * |z_U_mean|)
  C. US threshold gate: signal * max(|z_U_mean| - k, 0) / scale
  D. Per-sector conditioning: signal[j] *= (1 + lambda * |z_U[sector(j)]|)
  E. Quadratic amplification: signal + lambda * signal^2 * sign(signal)
  F. Rank-based non-linear: Replace signal with rank-based score
  G. Signal × US vol interaction: signal * f(realized_US_vol)
  H. Softmax-weighted: Non-linear cross-sectional weighting
  I. Signal confidence by prediction variance: signal / (1 + lambda * pred_var)
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
# Non-linear signal transforms
# ---------------------------------------------------------------------------

def transform_us_magnitude_amp(signal, z_U, lambda_amp=0.5):
    """Amplify signal by US return magnitude.

    signal_enhanced = signal * (1 + lambda * |mean(z_U)|)

    When US moves a lot, signal is more reliable → amplify.
    """
    us_mag = float(np.mean(np.abs(z_U)))
    return signal * (1.0 + lambda_amp * us_mag)


def transform_us_threshold_gate(signal, z_U, threshold=0.5, scale=1.0):
    """Gate signal by US return magnitude.

    signal_enhanced = signal * max(|mean(z_U)| - threshold, 0) / scale

    When US is flat, signal is noise → zero it out.
    """
    us_mag = float(np.mean(np.abs(z_U)))
    gate = max(us_mag - threshold, 0.0) / max(scale, 1e-6)
    return signal * gate


def transform_per_sector_conditioning(signal, z_U, n_u, n_j, lambda_amp=0.3):
    """Per-sector conditioning: scale each JP signal by corresponding US sector magnitude.

    For each JP ticker j, find the most correlated US ticker and scale signal by
    that US ticker's return magnitude.

    Uses a simple mapping: US sector i → JP sector i (mod n_u).
    """
    z_U_abs = np.abs(z_U)
    # Map each JP ticker to a US ticker (cyclic mapping as approximation)
    for j in range(n_j):
        us_idx = j % n_u
        signal[j] *= (1.0 + lambda_amp * z_U_abs[us_idx])
    return signal


def transform_quadratic_amp(signal, lambda_quad=0.1):
    """Quadratic amplification of strong signals.

    signal_enhanced = signal + lambda * signal^2 * sign(signal)

    Amplifies strong signals, leaves weak ones unchanged.
    """
    return signal + lambda_quad * signal**2 * np.sign(signal)


def transform_rank_nonlinear(signal, n_j, q=0.3):
    """Rank-based non-linear mapping.

    Replace signal values with a non-linear function of their rank.
    Uses tanh mapping to compress extreme ranks.
    """
    ranks = pd.Series(signal).rank().values  # 1 to n_j
    normalized_ranks = (ranks - (n_j + 1) / 2.0) / ((n_j - 1) / 2.0)  # -1 to 1
    # Tanh mapping: compresses extremes
    return np.tanh(2.0 * normalized_ranks) * np.std(signal)


def transform_us_vol_interaction(signal, z_U, window=20, lambda_vol=0.5):
    """Signal × US volatility interaction.

    Scale signal by recent US volatility regime.
    High US vol → stronger lead-lag → amplify signal.
    """
    us_vol = float(np.std(z_U))
    return signal * (1.0 + lambda_vol * us_vol)


def transform_softmax_weighted(signal, temperature=1.0):
    """Softmax-based non-linear cross-sectional weighting.

    Instead of linear signal-weighted allocation, use softmax to create
    a more peaked distribution.
    """
    # Center signal
    sig_centered = signal - np.median(signal)
    # Softmax
    exp_sig = np.exp(sig_centered / max(temperature, 1e-6))
    softmax_w = exp_sig / np.sum(exp_sig)

    # Map to long/short: top half long, bottom half short
    n_j = len(signal)
    long_mask = sig_centered > 0
    short_mask = ~long_mask

    result = np.zeros(n_j)
    if long_mask.any():
        result[long_mask] = softmax_w[long_mask] / softmax_w[long_mask].sum()
    if short_mask.any():
        result[short_mask] = -softmax_w[short_mask] / softmax_w[short_mask].sum()

    # Scale to gross=2.0
    abs_sum = np.sum(np.abs(result))
    if abs_sum > 1e-10:
        result *= 2.0 / abs_sum
    return result


def transform_confidence_by_var(signal, pred_var, lambda_conf=1.0):
    """Signal confidence by prediction variance.

    signal_enhanced = signal / (1 + lambda * pred_var)

    Shrink signal when prediction variance is high.
    """
    pred_var_safe = np.maximum(pred_var, 1e-8)
    return signal / (1.0 + lambda_conf * pred_var)


def transform_combined(signal, z_U, pred_var, n_u, n_j,
                        lambda_amp=0.3, lambda_conf=0.5, lambda_quad=0.05):
    """Combined: US magnitude amplification + confidence shrinkage + quadratic."""
    s = transform_us_magnitude_amp(signal, z_U, lambda_amp)
    s = transform_confidence_by_var(s, pred_var, lambda_conf)
    s = transform_quadratic_amp(s, lambda_quad)
    return s


# ---------------------------------------------------------------------------
# Model wrapper with pre-computed non-linear signals
# ---------------------------------------------------------------------------

class NonLinearSignalModel:
    """Applies non-linear transform to BLPX signals, then standard weight construction."""

    def __init__(self, blpx_signals, z_U_arr, pred_var_arr, df_exec,
                 method: str, params: dict | None = None):
        self.blpx_signals = blpx_signals
        self.z_U_arr = z_U_arr
        self.pred_var_arr = pred_var_arr
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
        self._enhanced_signals = None
        self._precomputed_weights = None
        self._weight_counter = 0

    def _compute_enhanced_signals(self):
        """Apply non-linear transform to each day's signal."""
        from leadlag.core.signal import build_weights

        T = len(self.df_exec)
        sim_dates = self.df_exec.index
        enhanced = np.zeros((T, self.n_j))
        weights = np.zeros((T, self.n_j))

        for i in range(T):
            sig_i = self.blpx_signals.iloc[i].values
            z_U_i = self.z_U_arr[i] if self.z_U_arr is not None else np.zeros(self.n_u)
            pred_var_i = self.pred_var_arr[i] if self.pred_var_arr is not None else np.zeros(self.n_j)

            if i < self.corr_window or not np.isfinite(sig_i).any():
                enhanced[i] = sig_i
                continue

            if self.method == "baseline":
                sig_enhanced = sig_i

            elif self.method == "us_mag_amp":
                lam = self.params.get("lambda_amp", 0.5)
                sig_enhanced = transform_us_magnitude_amp(sig_i, z_U_i, lam)

            elif self.method == "us_threshold_gate":
                thr = self.params.get("threshold", 0.5)
                scale = self.params.get("scale", 1.0)
                sig_enhanced = transform_us_threshold_gate(sig_i, z_U_i, thr, scale)

            elif self.method == "per_sector_cond":
                lam = self.params.get("lambda_amp", 0.3)
                sig_enhanced = transform_per_sector_conditioning(
                    sig_i.copy(), z_U_i, self.n_u, self.n_j, lam)

            elif self.method == "quadratic_amp":
                lam = self.params.get("lambda_quad", 0.1)
                sig_enhanced = transform_quadratic_amp(sig_i, lam)

            elif self.method == "rank_nonlinear":
                sig_enhanced = transform_rank_nonlinear(sig_i, self.n_j)

            elif self.method == "us_vol_interaction":
                lam = self.params.get("lambda_vol", 0.5)
                sig_enhanced = transform_us_vol_interaction(sig_i, z_U_i, lambda_vol=lam)

            elif self.method == "softmax_weighted":
                temp = self.params.get("temperature", 1.0)
                # This method directly produces weights
                weights[i] = transform_softmax_weighted(sig_i, temp)
                enhanced[i] = sig_i
                continue

            elif self.method == "confidence_by_var":
                lam = self.params.get("lambda_conf", 1.0)
                sig_enhanced = transform_confidence_by_var(sig_i, pred_var_i, lam)

            elif self.method == "combined":
                la = self.params.get("lambda_amp", 0.3)
                lc = self.params.get("lambda_conf", 0.5)
                lq = self.params.get("lambda_quad", 0.05)
                sig_enhanced = transform_combined(sig_i, z_U_i, pred_var_i,
                                                   self.n_u, self.n_j, la, lc, lq)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            enhanced[i] = sig_enhanced
            # Build weights from enhanced signal (except softmax which is direct)
            weights[i] = build_weights(sig_enhanced, q=self.q, n_j=self.n_j,
                                       weight_mode="signal", enforce_sign=False)

        self._enhanced_signals = pd.DataFrame(enhanced, index=sim_dates, columns=JP_TICKERS)
        return weights

    def predict_signals(self, df_exec):
        if self._precomputed_weights is None:
            self._precomputed_weights = self._compute_enhanced_signals()

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
            w = self._precomputed_weights[self._weight_counter]
            self._weight_counter += 1
            return w
        return np.zeros(self.n_j)


# ---------------------------------------------------------------------------
# Extract z_U and pred_var from BLPX model
# ---------------------------------------------------------------------------

def extract_zU_and_predvar(model, df_exec):
    """Extract z_U (US standardized returns) and pred_var for all dates."""
    inputs = model._prepare_common_inputs(df_exec)
    all_returns_raw = inputs["all_returns_raw"]
    jp_gap = inputs["jp_gap"]
    jp_beta = inputs["jp_beta"]
    topix_night = inputs["topix_night"]
    v0_static = inputs["v0_static"]
    c_full = inputs["c_full"]

    T = len(df_exec)
    n_u = model.n_u
    n_j = model.n_j
    z_U_arr = np.zeros((T, n_u))
    pred_var_arr = np.zeros((T, n_j))

    start_idx = model.corr_window

    for i in range(T):
        if i < start_idx:
            continue

        gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        try:
            res = model.compute_blp_signal(
                all_returns_raw, i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
                rolling_std=None,
                v0_static=v0_static,
                c_full=c_full,
                is_residual=True,
                return_matrices=True,
            )
            z_U_arr[i] = res.get("z_U", np.zeros(n_u))
            pred_var_arr[i] = res.get("pred_var_vec", np.zeros(n_j))
        except Exception:
            z_U_arr[i] = np.zeros(n_u)
            pred_var_arr[i] = np.zeros(n_j)

    return z_U_arr, pred_var_arr


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
    parser = argparse.ArgumentParser(description="Non-Linear Signal Enhancement Experiment")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/nonlinear_signal")
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

    # Extract z_U and pred_var
    logger.info("Extracting z_U and pred_var from BLPX model...")
    z_U_arr, pred_var_arr = extract_zU_and_predvar(model_base, df_exec)

    # Define variants
    variants = [
        ("baseline", "baseline", {}),
        ("us_mag_amp_03", "us_mag_amp", {"lambda_amp": 0.3}),
        ("us_mag_amp_05", "us_mag_amp", {"lambda_amp": 0.5}),
        ("us_mag_amp_10", "us_mag_amp", {"lambda_amp": 1.0}),
        ("us_gate_03", "us_threshold_gate", {"threshold": 0.3, "scale": 1.0}),
        ("us_gate_05", "us_threshold_gate", {"threshold": 0.5, "scale": 1.0}),
        ("per_sector_03", "per_sector_cond", {"lambda_amp": 0.3}),
        ("per_sector_05", "per_sector_cond", {"lambda_amp": 0.5}),
        ("quadratic_005", "quadratic_amp", {"lambda_quad": 0.05}),
        ("quadratic_010", "quadratic_amp", {"lambda_quad": 0.10}),
        ("rank_nonlinear", "rank_nonlinear", {}),
        ("us_vol_05", "us_vol_interaction", {"lambda_vol": 0.5}),
        ("softmax_05", "softmax_weighted", {"temperature": 0.5}),
        ("softmax_10", "softmax_weighted", {"temperature": 1.0}),
        ("confidence_var_05", "confidence_by_var", {"lambda_conf": 0.5}),
        ("confidence_var_10", "confidence_by_var", {"lambda_conf": 1.0}),
        ("combined", "combined", {"lambda_amp": 0.3, "lambda_conf": 0.5, "lambda_quad": 0.05}),
    ]

    all_results = []

    for name, method, params in variants:
        logger.info("=== %s ===", name)
        model = NonLinearSignalModel(blpx_signals, z_U_arr, pred_var_arr,
                                      df_exec, method, params)
        m = run_backtest(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
        m["method"] = method
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f AR=%.4f Vol=%.4f MDD=%.2f%% Turnover=%.2f IC=%.4f (%.1fs)",
                    name, m["Sharpe_net"], m["AR_net"], m["Vol_net"],
                    m["MDD"] * 100, m["Turnover"], m["Mean_Rank_IC"], m["elapsed_s"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Print comparison table
    print("\n" + "=" * 140)
    print("NON-LINEAR SIGNAL ENHANCEMENT — RESULTS")
    print("=" * 140)

    baseline_sharpe = all_results[0]["Sharpe_net"] if all_results else 0.0
    baseline_ic = all_results[0]["Mean_Rank_IC"] if all_results else 0.0

    print(f"\n{'Name':<25} {'Sharpe':<10} {'AR':<10} {'Vol':<10} {'MDD%':<8} "
          f"{'Turnover':<10} {'IC':<10} {'ICIR':<8} {'ΔSharpe':<8} {'ΔIC':<8}")
    print("-" * 140)
    for r in all_results:
        delta_s = r["Sharpe_net"] - baseline_sharpe if np.isfinite(r["Sharpe_net"]) else np.nan
        delta_ic = r["Mean_Rank_IC"] - baseline_ic if np.isfinite(r["Mean_Rank_IC"]) else np.nan
        print(f"{r['name']:<25} {r['Sharpe_net']:<10.4f} {r['AR_net']:<10.4f} {r['Vol_net']:<10.4f} "
              f"{r['MDD']*100:<8.2f} {r['Turnover']:<10.2f} {r['Mean_Rank_IC']:<10.4f} "
              f"{r['ICIR']:<8.2f} {delta_s:+.4f} {delta_ic:+.4f}")

    valid = [r for r in all_results if r["name"] != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} "
              f"(ΔSharpe={best['Sharpe_net']-baseline_sharpe:+.4f}, "
              f"ΔIC={best['Mean_Rank_IC']-baseline_ic:+.4f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
