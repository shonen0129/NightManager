"""Execution Efficiency & Signal Quality — Fundamental Improvements.

MVO didn't help because BLPX signal already incorporates covariance.
This experiment targets the real bottleneck: TURNOVER (1.58/day).

Tests 6 fundamental improvements:

  A. baseline: Current production (daily full rebalance)
  B. Signal decay profile: How long does alpha persist? (diagnostic)
  C. N-day holding: Rebalance every N days (2,3,5)
  D. Cost-aware threshold: Only rebalance positions where |Δw| > threshold
  E. Portfolio vol targeting: Scale gross to target constant portfolio vol
  F. Signal EMA smoothing: Exponential smoothing of signals to reduce noise
  G. Signal persistence weighting: Weight signal by agreement with recent history
  H. All combined: Best of C/D/E/F/G

Key insight: Reducing turnover from 1.58 to 0.8 saves ~8bps/day = ~20% AR improvement.
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
# Signal decay analysis (diagnostic)
# ---------------------------------------------------------------------------

def analyze_signal_decay(signals_df, y_target, sim_dates, start_idx, max_lag=10):
    """Measure how signal IC decays over 1-10 day lags.

    For each lag h, compute Rank IC between signal(t) and y(t+h).
    """
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)
    results = []

    for h in range(1, max_lag + 1):
        ic_list = []
        for i in range(start_idx, len(sim_dates) - h):
            date = sim_dates[i]
            future_date = sim_dates[i + h]
            if date not in signals_df.index or future_date not in y_df.index:
                continue
            sig_t = signals_df.loc[date].values
            y_future = y_df.loc[future_date].values
            valid = ~(np.isnan(sig_t) | np.isnan(y_future))
            if valid.sum() >= 3:
                rho, _ = stats.spearmanr(sig_t[valid], y_future[valid])
                if np.isfinite(rho):
                    ic_list.append(float(rho))

        if ic_list:
            ic_arr = np.array(ic_list)
            mean_ic = float(np.mean(ic_arr))
            std_ic = float(np.std(ic_arr, ddof=1))
            icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic > 1e-8 else np.nan
            t_stat = mean_ic / (std_ic / np.sqrt(len(ic_arr))) if std_ic > 1e-8 else np.nan
        else:
            mean_ic = std_ic = icir = t_stat = np.nan

        results.append({
            "lag_days": h, "mean_ic": mean_ic, "std_ic": std_ic,
            "icir": icir, "t_stat": t_stat, "n_obs": len(ic_list),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Portfolio construction methods
# ---------------------------------------------------------------------------

def build_n_day_holding_weights(signals_df, sim_dates, start_idx, n_j, q, hold_days):
    """Rebalance every N days, holding positions in between."""
    from leadlag.core.signal import build_weights
    T = len(sim_dates)
    weights = np.zeros((T, n_j))

    last_rebalance = -1
    for i in range(T):
        if i < start_idx:
            continue
        if i == start_idx or (i - last_rebalance) >= hold_days:
            sig_i = signals_df.iloc[i].values
            weights[i] = build_weights(sig_i, q=q, n_j=n_j,
                                       weight_mode="signal", enforce_sign=False)
            last_rebalance = i
        else:
            weights[i] = weights[i - 1] if i > 0 else np.zeros(n_j)

    return weights


def build_cost_aware_weights(signals_df, sim_dates, start_idx, n_j, q,
                              rebalance_threshold=0.15):
    """Only rebalance positions where |Δw| > threshold.

    For each position, if the target weight change is below threshold,
    keep the existing weight. This reduces unnecessary turnover from
    small signal fluctuations.
    """
    from leadlag.core.signal import build_weights
    T = len(sim_dates)
    weights = np.zeros((T, n_j))
    w_prev = np.zeros(n_j)

    for i in range(T):
        if i < start_idx:
            continue

        sig_i = signals_df.iloc[i].values
        w_target = build_weights(sig_i, q=q, n_j=n_j,
                                 weight_mode="signal", enforce_sign=False)

        # Only apply changes where |delta| > threshold
        delta = w_target - w_prev
        significant = np.abs(delta) > rebalance_threshold
        w_new = w_prev.copy()
        w_new[significant] = w_target[significant]

        # Renormalize to gross=2.0
        abs_sum = np.sum(np.abs(w_new))
        if abs_sum > 1e-10:
            w_new *= 2.0 / abs_sum

        weights[i] = w_new
        w_prev = w_new

    return weights


def build_vol_target_weights(signals_df, sim_dates, start_idx, n_j, q,
                              target_vol=0.15, vol_window=60, max_gross=4.0):
    """Scale gross exposure to target constant portfolio volatility.

    Uses rolling realized portfolio volatility to scale weights.
    """
    from leadlag.core.signal import build_weights
    T = len(sim_dates)
    weights = np.zeros((T, n_j))

    # Need returns for vol estimation
    # Use signal-weighted portfolio returns as proxy
    for i in range(T):
        if i < start_idx:
            continue

        sig_i = signals_df.iloc[i].values
        w_base = build_weights(sig_i, q=q, n_j=n_j,
                               weight_mode="signal", enforce_sign=False)

        # Estimate portfolio vol from recent signal-weighted returns
        if i >= vol_window + start_idx:
            # Use recent portfolio returns as vol proxy
            recent_weights = weights[i - vol_window:i]
            # Simple vol estimate: std of recent daily returns * sqrt(252)
            # Approximate with signal dispersion as vol proxy
            sig_centered = sig_i - np.median(sig_i)
            sig_disp = np.std(sig_centered)
            # Higher dispersion → higher expected vol
            est_vol = 0.10 + sig_disp * 0.5  # rough mapping
        else:
            est_vol = 0.15  # default

        # Scale to target vol
        scale = min(target_vol / max(est_vol, 0.01), max_gross / 2.0)
        weights[i] = w_base * scale

    return weights


def build_ema_smoothed_weights(signals_df, sim_dates, start_idx, n_j, q,
                                ema_halflife=3):
    """Exponential smoothing of signals before weight construction.

    Reduces signal noise by blending today's signal with recent history.
    """
    from leadlag.core.signal import build_weights
    T = len(sim_dates)
    weights = np.zeros((T, n_j))

    alpha = 1.0 - np.exp(-np.log(2) / ema_halflife)
    sig_ema = signals_df.iloc[0].values.copy()

    for i in range(T):
        if i < start_idx:
            continue

        sig_i = signals_df.iloc[i].values
        if i == start_idx:
            sig_ema = sig_i.copy()
        else:
            sig_ema = alpha * sig_i + (1.0 - alpha) * sig_ema

        weights[i] = build_weights(sig_ema, q=q, n_j=n_j,
                                   weight_mode="signal", enforce_sign=False)

    return weights


def build_persistence_weighted_weights(signals_df, sim_dates, start_idx, n_j, q,
                                        lookback=5, persistence_weight=0.3):
    """Weight signal by agreement with recent signal history.

    If today's signal agrees with recent signals (same direction),
    increase confidence. If signal flips, reduce confidence.
    """
    from leadlag.core.signal import build_weights
    T = len(sim_dates)
    weights = np.zeros((T, n_j))

    for i in range(T):
        if i < start_idx:
            continue

        sig_i = signals_df.iloc[i].values

        if i >= start_idx + lookback:
            # Compute sign agreement with recent lookback days
            recent_sigs = signals_df.iloc[i - lookback:i].values
            sig_sign = np.sign(sig_i)
            recent_signs = np.sign(recent_sigs)
            agreement = np.mean(recent_signs == sig_sign[np.newaxis, :], axis=0)

            # Weight signal by persistence: high agreement → keep, low → shrink
            persistence_factor = 0.5 + persistence_weight * (agreement - 0.5)
            sig_adjusted = sig_i * persistence_factor
        else:
            sig_adjusted = sig_i

        weights[i] = build_weights(sig_adjusted, q=q, n_j=n_j,
                                   weight_mode="signal", enforce_sign=False)

    return weights


def build_combined_weights(signals_df, sim_dates, start_idx, n_j, q,
                            hold_days=3, rebalance_threshold=0.12,
                            ema_halflife=3, target_vol=0.15):
    """Combined: EMA smoothing + cost-aware threshold + N-day holding + vol target."""
    from leadlag.core.signal import build_weights
    T = len(sim_dates)
    weights = np.zeros((T, n_j))
    w_prev = np.zeros(n_j)

    alpha_ema = 1.0 - np.exp(-np.log(2) / ema_halflife)
    sig_ema = signals_df.iloc[0].values.copy()
    last_rebalance = -1

    for i in range(T):
        if i < start_idx:
            continue

        sig_i = signals_df.iloc[i].values
        if i == start_idx:
            sig_ema = sig_i.copy()
        else:
            sig_ema = alpha_ema * sig_i + (1.0 - alpha_ema) * sig_ema

        # N-day holding check
        should_rebalance = (i == start_idx) or ((i - last_rebalance) >= hold_days)

        if should_rebalance:
            w_target = build_weights(sig_ema, q=q, n_j=n_j,
                                     weight_mode="signal", enforce_sign=False)

            # Cost-aware threshold
            delta = w_target - w_prev
            significant = np.abs(delta) > rebalance_threshold
            w_new = w_prev.copy()
            w_new[significant] = w_target[significant]

            # Renormalize
            abs_sum = np.sum(np.abs(w_new))
            if abs_sum > 1e-10:
                w_new *= 2.0 / abs_sum

            # Vol target scaling
            if i >= start_idx + 60:
                sig_centered = sig_ema - np.median(sig_ema)
                sig_disp = np.std(sig_centered)
                est_vol = 0.10 + sig_disp * 0.5
                scale = min(target_vol / max(est_vol, 0.01), 2.0)
                w_new *= scale

            weights[i] = w_new
            w_prev = w_new
            last_rebalance = i
        else:
            weights[i] = w_prev

    return weights


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ExecutionEfficiencyModel:
    """Pre-computed weights model for BacktestEngine compatibility."""

    def __init__(self, precomputed_weights, signals_df, df_exec):
        self._precomputed_weights = precomputed_weights
        self.signals_df = signals_df
        self.df_exec = df_exec
        self.n_j = len(JP_TICKERS)
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"
        self._weight_counter = 0

    def predict_signals(self, df_exec):
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
        if self._weight_counter < len(self._precomputed_weights):
            w = self._precomputed_weights[self._weight_counter]
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
    cost_mean = float(results["daily_costs"].mean() * 245)
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
    return {
        "name": name, "AR_net": ar, "Vol_net": vol, "Sharpe_net": sharpe,
        "MDD": mdd, "Turnover": turnover, "Gross_exp": gross_exp,
        "Annual_cost": cost_mean,
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
    parser = argparse.ArgumentParser(description="Execution Efficiency & Signal Quality Experiment")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/execution_efficiency")
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

    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    n_j = len(JP_TICKERS)
    q = 0.3

    # --- Signal decay analysis (diagnostic) ---
    logger.info("=== Signal Decay Analysis ===")
    decay_df = analyze_signal_decay(blpx_signals, y_target, sim_dates, start_idx, max_lag=10)
    print("\n=== SIGNAL DECAY PROFILE ===")
    print(f"{'Lag':<6} {'Mean IC':<10} {'Std IC':<10} {'ICIR':<10} {'t-stat':<10} {'N':<8}")
    print("-" * 54)
    for _, row in decay_df.iterrows():
        print(f"{int(row['lag_days']):<6} {row['mean_ic']:<10.4f} {row['std_ic']:<10.4f} "
              f"{row['icir']:<10.2f} {row['t_stat']:<10.2f} {int(row['n_obs']):<8}")
    decay_df.to_csv(output_dir / "signal_decay.csv", index=False)

    # --- Build weights for all variants ---
    logger.info("Building weights for all variants...")

    variants = {}

    # A. Baseline
    from leadlag.core.signal import build_weights
    w_baseline = np.zeros((len(sim_dates), n_j))
    for i in range(len(sim_dates)):
        if i >= start_idx:
            w_baseline[i] = build_weights(blpx_signals.iloc[i].values, q=q, n_j=n_j,
                                          weight_mode="signal", enforce_sign=False)
    variants["baseline"] = w_baseline

    # B. 2-day holding
    variants["hold_2d"] = build_n_day_holding_weights(blpx_signals, sim_dates, start_idx, n_j, q, 2)

    # C. 3-day holding
    variants["hold_3d"] = build_n_day_holding_weights(blpx_signals, sim_dates, start_idx, n_j, q, 3)

    # D. 5-day holding
    variants["hold_5d"] = build_n_day_holding_weights(blpx_signals, sim_dates, start_idx, n_j, q, 5)

    # E. Cost-aware threshold (0.10)
    variants["cost_aware_010"] = build_cost_aware_weights(blpx_signals, sim_dates, start_idx, n_j, q, 0.10)

    # F. Cost-aware threshold (0.15)
    variants["cost_aware_015"] = build_cost_aware_weights(blpx_signals, sim_dates, start_idx, n_j, q, 0.15)

    # G. Vol targeting (15%)
    variants["vol_target_15"] = build_vol_target_weights(blpx_signals, sim_dates, start_idx, n_j, q, 0.15)

    # H. EMA smoothing (halflife=3)
    variants["ema_3d"] = build_ema_smoothed_weights(blpx_signals, sim_dates, start_idx, n_j, q, 3)

    # I. EMA smoothing (halflife=5)
    variants["ema_5d"] = build_ema_smoothed_weights(blpx_signals, sim_dates, start_idx, n_j, q, 5)

    # J. Persistence weighting (lookback=5)
    variants["persistence_5"] = build_persistence_weighted_weights(blpx_signals, sim_dates, start_idx, n_j, q, 5)

    # K. Combined: EMA + cost-aware + 3-day hold + vol target
    variants["combined"] = build_combined_weights(blpx_signals, sim_dates, start_idx, n_j, q,
                                                   hold_days=3, rebalance_threshold=0.12,
                                                   ema_halflife=3, target_vol=0.15)

    # --- Run backtests ---
    all_results = []

    for name, weights in variants.items():
        logger.info("=== %s ===", name)
        model = ExecutionEfficiencyModel(weights, blpx_signals, df_exec)
        m = run_backtest(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f AR=%.4f Vol=%.4f MDD=%.2f%% Turnover=%.2f Cost=%.4f (%.1fs)",
                    name, m["Sharpe_net"], m["AR_net"], m["Vol_net"],
                    m["MDD"] * 100, m["Turnover"], m["Annual_cost"], m["elapsed_s"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    # Print comparison table
    print("\n" + "=" * 140)
    print("EXECUTION EFFICIENCY & SIGNAL QUALITY — RESULTS")
    print("=" * 140)

    baseline_sharpe = all_results[0]["Sharpe_net"] if all_results else 0.0
    baseline_turnover = all_results[0]["Turnover"] if all_results else 0.0
    baseline_cost = all_results[0]["Annual_cost"] if all_results else 0.0

    print(f"\n{'Name':<20} {'Sharpe':<10} {'AR':<10} {'Vol':<10} {'MDD%':<8} "
          f"{'Turnover':<10} {'Gross':<8} {'Ann.Cost':<10} {'IC':<10} {'ICIR':<8} {'ΔSharpe':<8} {'ΔTurn':<8}")
    print("-" * 140)
    for r in all_results:
        delta_s = r["Sharpe_net"] - baseline_sharpe if np.isfinite(r["Sharpe_net"]) else np.nan
        delta_t = r["Turnover"] - baseline_turnover
        print(f"{r['name']:<20} {r['Sharpe_net']:<10.4f} {r['AR_net']:<10.4f} {r['Vol_net']:<10.4f} "
              f"{r['MDD']*100:<8.2f} {r['Turnover']:<10.2f} {r['Gross_exp']:<8.2f} "
              f"{r['Annual_cost']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} "
              f"{delta_s:+.4f} {delta_t:+.2f}")

    valid = [r for r in all_results if r["name"] != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} "
              f"(ΔSharpe={best['Sharpe_net']-baseline_sharpe:+.4f}, "
              f"ΔTurnover={best['Turnover']-baseline_turnover:+.2f}, "
              f"ΔCost={best['Annual_cost']-baseline_cost:+.4f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
