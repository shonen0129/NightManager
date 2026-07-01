"""Phase 2C: Extended Non-Linear Overlay (Sprint 3B Extension).

Tests GBDT overlay and expanded rolling beta windows on top of the
Phase 1A+1D optimized BLPX baseline.

Walk-forward design:
  Train: 252 days → fit overlay model
  Val:   63 days  → select hyperparameters and blend alpha
  Test:  21 days  → OOS prediction
  Step:  21 days

Models compared:
  1. Baseline (no overlay)
  2. Ridge interaction overlay (Sprint 3B)
  3. ElasticNet interaction overlay (Sprint 3B)
  4. GBDT interaction overlay (Phase 2C new)
  5. GBDT with expanded beta windows (30, 60, 120, 252)
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

from models.hinge_overlay import generate_walk_forward_windows, ALPHA_GRID_DEFAULT
from models.hinge_interaction_overlay import build_flat_arrays_from_long
from models.hinge_interaction_ridge import InteractionRidgeOverlay
from models.hinge_interaction_elasticnet import InteractionElasticNetOverlay
from models.hinge_interaction_gbdt import InteractionGBDTOverlay

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


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        cfg.setdefault("blpx", {}).update(blpx_overrides)
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


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


def build_simple_interaction_features(df_exec, y_target, signal_panel, window=60):
    """Build simplified interaction features for overlay testing.

    Features per (date, ticker):
      - gap_z: rolling z-score of gap return
      - signal_z: the baseline signal value
      - us_vol_z: rolling US volatility z-score (date-level, broadcast)
      - jp_beta_lag: rolling beta of JP asset on TOPIX (lag-1)
      - momentum_5d: 5-day cumulative return (lag-1)
      - mean_rev_20d: 20-day cumulative return (lag-1, negative = mean reversion)
    """
    sim_dates = df_exec.index
    n_j = len(JP_TICKERS)

    gap_cols = [f"jp_gap_{tk}" for tk in JP_TICKERS]
    gap_df = df_exec[gap_cols].copy()
    gap_df.columns = JP_TICKERS

    oc_cols = [f"jp_oc_{tk}" for tk in JP_TICKERS]
    oc_df = df_exec[oc_cols].copy()
    oc_df.columns = JP_TICKERS

    us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
    us_vol = df_exec[us_cols].rolling(window).std(ddof=1).mean(axis=1)
    us_vol_z = ((us_vol - us_vol.rolling(252).mean()) / us_vol.rolling(252).std(ddof=1)).shift(1)

    records = []
    for j, tk in enumerate(JP_TICKERS):
        gap_series = gap_df[tk]
        gap_mean = gap_series.shift(1).rolling(window).mean()
        gap_std = gap_series.shift(1).rolling(window).std(ddof=1)
        gap_z = ((gap_series - gap_mean) / gap_std.replace(0, np.nan)).values

        oc_series = oc_df[tk]
        mom_5d = oc_series.shift(1).rolling(5).sum().values
        mr_20d = oc_series.shift(1).rolling(20).sum().values

        sig_vals = signal_panel[tk].values if tk in signal_panel.columns else np.zeros(len(df_exec))

        for i, date in enumerate(sim_dates):
            records.append({
                "date": date,
                "ticker": tk,
                "gap_z": gap_z[i] if i < len(gap_z) else np.nan,
                "signal_z": sig_vals[i] if i < len(sig_vals) else np.nan,
                "us_vol_z": us_vol_z.values[i] if i < len(us_vol_z) else np.nan,
                "momentum_5d": mom_5d[i] if i < len(mom_5d) else np.nan,
                "mean_rev_20d": mr_20d[i] if i < len(mr_20d) else np.nan,
            })

    long_df = pd.DataFrame(records)
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)
    y_long = y_df.stack().reset_index()
    y_long.columns = ["date", "ticker", "y_target"]
    long_df = long_df.merge(y_long, on=["date", "ticker"], how="left")

    sig_long = signal_panel.stack().reset_index()
    sig_long.columns = ["date", "ticker", "mu_base"]
    long_df = long_df.merge(sig_long, on=["date", "ticker"], how="left")

    return long_df


def run_walk_forward_overlay(df_exec, y_target, signal_panel, overlay_class,
                             overlay_name, feature_cols, **overlay_kwargs):
    """Run walk-forward overlay on top of baseline signals."""
    sim_dates = df_exec.index
    n_j = len(JP_TICKERS)

    long_df = build_simple_interaction_features(df_exec, y_target, signal_panel)

    windows = generate_walk_forward_windows(
        sim_dates, train_window_days=252, validation_window_days=63,
        test_window_days=21, step_days=21, purge_days=1,
    )

    if not windows:
        logger.warning("No valid walk-forward windows for %s", overlay_name)
        return signal_panel.copy(), {}

    adjusted_signals = signal_panel.copy().values.copy()
    overlay_stats = {"windows": len(windows), "alpha_list": [], "val_ic_list": []}

    for w in windows:
        train_dates = w["train_dates"]
        val_dates = w["val_dates"]
        test_dates = w["test_dates"]

        X_train, y_train, mu_train, _ = build_flat_arrays_from_long(
            long_df, feature_cols, "mu_base", "y_target", train_dates, JP_TICKERS
        )
        X_val, y_val, mu_val, _ = build_flat_arrays_from_long(
            long_df, feature_cols, "mu_base", "y_target", val_dates, JP_TICKERS
        )
        X_test, y_test, mu_test, test_pairs = build_flat_arrays_from_long(
            long_df, feature_cols, "mu_base", "y_target", test_dates, JP_TICKERS
        )

        if len(X_train) < 50 or X_train.shape[1] == 0:
            continue

        y_resid_train = y_train - mu_train
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_resid_train) | np.isnan(mu_train))
        if valid_train.sum() < 30:
            continue

        X_tr_clean = X_train[valid_train]
        y_resid_clean = y_resid_train[valid_train]
        mu_tr_clean = mu_train[valid_train]

        # Clean validation data
        if len(y_val) > 0:
            y_resid_val = y_val - mu_val
            valid_val = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_resid_val) | np.isnan(mu_val))
            X_val_clean = X_val[valid_val]
            y_resid_val_clean = y_resid_val[valid_val]
            mu_val_clean = mu_val[valid_val]
        else:
            X_val_clean = np.empty((0, X_train.shape[1]))
            y_resid_val_clean = np.array([])
            mu_val_clean = np.array([])

        if overlay_class == InteractionRidgeOverlay:
            model = overlay_class.select_best_ridge_alpha(
                X_tr_clean, y_resid_clean, mu_tr_clean,
                X_val_clean, y_resid_val_clean, mu_val_clean,
                **overlay_kwargs,
            )
        elif overlay_class == InteractionElasticNetOverlay:
            model = overlay_class.select_best_hyperparams(
                X_tr_clean, y_resid_clean, mu_tr_clean,
                X_val_clean, y_resid_val_clean, mu_val_clean,
                **overlay_kwargs,
            )
        elif overlay_class == InteractionGBDTOverlay:
            model = overlay_class.select_best_hyperparams(
                X_tr_clean, y_resid_clean, mu_tr_clean,
                X_val_clean, y_resid_val_clean, mu_val_clean,
                **overlay_kwargs,
            )
        else:
            continue

        if not model._is_fitted or len(X_test) == 0:
            continue

        mu_pred = model.predict(X_test, mu_test)

        for idx, (date, ticker) in enumerate(test_pairs):
            j = JP_TICKERS.index(ticker)
            date_idx = sim_dates.get_loc(date)
            adjusted_signals[date_idx, j] = mu_pred[idx]

        overlay_stats["alpha_list"].append(model.alpha)

    adjusted_df = pd.DataFrame(adjusted_signals, index=sim_dates, columns=JP_TICKERS)
    return adjusted_df, overlay_stats


class OverlayModel:
    """Wrapper model that uses pre-computed overlay-adjusted signals."""

    def __init__(self, signals_df, df_exec, n_j):
        self.signals_df = signals_df
        self.n_j = n_j
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"

    def predict_signals(self, df_exec):
        T = len(df_exec)
        sim_dates = df_exec.index
        sig = self.signals_df.reindex(sim_dates).fillna(0.0)
        z = pd.DataFrame(index=sim_dates, columns=JP_TICKERS, dtype=float)
        for i in range(T):
            z.iloc[i] = self._normalize(sig.iloc[i].values)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "raw_pca_signals": empty,
            "residual_pca_signals": empty,
            "p4_signals": empty,
            "signals": z,
            "normalized_signals": z,
            "y_jp_oc_df": y_jp_oc_df,
        }

    def _normalize(self, sig):
        centered = sig - np.median(sig)
        std = np.std(centered)
        return centered / (std if std > 1e-8 else 1.0)

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        q_val = q if q is not None else self.q
        return build_weights(signal=signal, q=q_val, n_j=self.n_j,
                             weight_mode=self.weight_mode, enforce_sign=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 2C: Extended Non-Linear Overlay")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/phase2c_nonlinear_overlay")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    cfg_base = build_config(yaml_path, blpx_overrides=BASE_PARAMS,
                            signal_components=SIGNAL_WEIGHTS)

    logger.info("Computing baseline BLPX signals...")
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    pred_base = model_base.predict_signals(df_exec)
    baseline_signals = pred_base["signals"]

    all_results = []

    # Baseline (no overlay)
    logger.info("=== Baseline (no overlay) ===")
    m_base = run_backtest("baseline_no_overlay", model_base, df_exec, y_target, slippage_bps=args.slippage_bps)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                m_base["Sharpe_net"], m_base["Mean_Rank_IC"], m_base["ICIR"],
                m_base["MDD"] * 100, m_base["elapsed_s"])
    all_results.append({**m_base, "scheme": "baseline"})

    feature_cols = ["gap_z", "signal_z", "us_vol_z", "momentum_5d", "mean_rev_20d"]

    # Ridge overlay
    logger.info("=== Ridge Interaction Overlay ===")
    t0 = time.perf_counter()
    ridge_signals, ridge_stats = run_walk_forward_overlay(
        df_exec, y_target, baseline_signals,
        InteractionRidgeOverlay, "ridge", feature_cols,
        model_name="ridge_overlay",
    )
    ridge_model = OverlayModel(ridge_signals, df_exec, len(JP_TICKERS))
    m_ridge = run_backtest("ridge_overlay", ridge_model, df_exec, y_target, slippage_bps=args.slippage_bps)
    m_ridge["scheme"] = "ridge"
    m_ridge["overlay_time"] = time.perf_counter() - t0
    all_results.append(m_ridge)
    logger.info("Ridge: Sharpe=%.4f IC=%.4f (delta=%+.4f, %.1fs)",
                m_ridge["Sharpe_net"], m_ridge["Mean_Rank_IC"],
                m_ridge["Sharpe_net"] - m_base["Sharpe_net"], m_ridge["overlay_time"])

    # ElasticNet overlay
    logger.info("=== ElasticNet Interaction Overlay ===")
    t0 = time.perf_counter()
    enet_signals, enet_stats = run_walk_forward_overlay(
        df_exec, y_target, baseline_signals,
        InteractionElasticNetOverlay, "elasticnet", feature_cols,
        model_name="enet_overlay",
    )
    enet_model = OverlayModel(enet_signals, df_exec, len(JP_TICKERS))
    m_enet = run_backtest("enet_overlay", enet_model, df_exec, y_target, slippage_bps=args.slippage_bps)
    m_enet["scheme"] = "elasticnet"
    m_enet["overlay_time"] = time.perf_counter() - t0
    all_results.append(m_enet)
    logger.info("ElasticNet: Sharpe=%.4f IC=%.4f (delta=%+.4f, %.1fs)",
                m_enet["Sharpe_net"], m_enet["Mean_Rank_IC"],
                m_enet["Sharpe_net"] - m_base["Sharpe_net"], m_enet["overlay_time"])

    # GBDT overlay
    logger.info("=== GBDT Interaction Overlay ===")
    t0 = time.perf_counter()
    gbdt_signals, gbdt_stats = run_walk_forward_overlay(
        df_exec, y_target, baseline_signals,
        InteractionGBDTOverlay, "gbdt", feature_cols,
        model_name="gbdt_overlay",
    )
    gbdt_model = OverlayModel(gbdt_signals, df_exec, len(JP_TICKERS))
    m_gbdt = run_backtest("gbdt_overlay", gbdt_model, df_exec, y_target, slippage_bps=args.slippage_bps)
    m_gbdt["scheme"] = "gbdt"
    m_gbdt["overlay_time"] = time.perf_counter() - t0
    all_results.append(m_gbdt)
    logger.info("GBDT: Sharpe=%.4f IC=%.4f (delta=%+.4f, %.1fs)",
                m_gbdt["Sharpe_net"], m_gbdt["Mean_Rank_IC"],
                m_gbdt["Sharpe_net"] - m_base["Sharpe_net"], m_gbdt["overlay_time"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)

    print("\n" + "=" * 100)
    print("PHASE 2C — EXTENDED NON-LINEAR OVERLAY — RESULTS")
    print("=" * 100)
    print(f"\nBaseline (no overlay): Sharpe={m_base['Sharpe_net']:.4f} IC={m_base['Mean_Rank_IC']:.4f} ICIR={m_base['ICIR']:.2f} MDD={m_base['MDD']*100:.2f}%")

    print(f"\n{'Name':<25} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Delta':<8}")
    print("-" * 69)
    for r in all_results:
        delta = r["Sharpe_net"] - m_base["Sharpe_net"] if np.isfinite(r["Sharpe_net"]) else np.nan
        print(f"{r['name']:<25} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {delta:+.4f}")

    valid = [r for r in all_results if r.get("scheme") != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} (delta={best['Sharpe_net']-m_base['Sharpe_net']:+.4f})")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
