"""Phase 1C: Correlation Estimation Improvements.

Three approaches:
  1. OAT sensitivity: lambda_lw, lambda_reg, ewma_half_life (PCA), corr_window
  2. Multi-horizon EWMA blend: blend short + long half-life correlations
  3. Dynamic lambda_lw: adjust LW shrinkage based on condition number

Uses monkey-patching for multi-horizon blend and dynamic lambda_lw.
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
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
    _BLP_CORR_CACHE,
)
from leadlag.core import correlation as corr_module

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Phase 1A best params as baseline
PHASE1A_BEST = {
    "alpha_xx": 0.20, "alpha_yy": 0.50, "alpha_yx": 0.15,
    "lambda_pca": 0.10, "lambda_sector": 0.60, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 504,
    "blp_ewma_halflife": 120,
}

# Phase 1B best ensemble weights
PHASE1B_WEIGHTS = {
    "raw_pca": {"enabled": True, "weight": 0.2},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": True, "weight": 0.8},
    "residual_blpx": {"enabled": False, "weight": 0.0},
}

# PCA-path params to sweep
PCA_PARAMS = {
    "lambda_lw": 0.5,
    "lambda_reg": 0.75,
    "ewma_half_life": 45,
    "corr_window": 60,
}

OAT_LEVELS = {
    "lambda_lw":      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda_reg":     [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    "ewma_half_life": [10, 21, 30, 45, 63, 90, 126, 189, 252],
    "corr_window":    [30, 40, 50, 60, 75, 90, 126],
}

# Multi-horizon EWMA blend configs
BLEND_CONFIGS = [
    {"short_hl": 21, "long_hl": 90, "blend_weight": 0.3},
    {"short_hl": 21, "long_hl": 90, "blend_weight": 0.5},
    {"short_hl": 21, "long_hl": 90, "blend_weight": 0.7},
    {"short_hl": 30, "long_hl": 120, "blend_weight": 0.3},
    {"short_hl": 30, "long_hl": 120, "blend_weight": 0.5},
    {"short_hl": 30, "long_hl": 120, "blend_weight": 0.7},
    {"short_hl": 21, "long_hl": 63, "blend_weight": 0.5},
    {"short_hl": 45, "long_hl": 189, "blend_weight": 0.5},
    {"short_hl": 30, "long_hl": 252, "blend_weight": 0.5},
]

# Dynamic lambda_lw configs
DYNAMIC_LW_CONFIGS = [
    {"base_lw": 0.3, "max_lw": 0.8, "cond_threshold": 50},
    {"base_lw": 0.3, "max_lw": 0.9, "cond_threshold": 30},
    {"base_lw": 0.5, "max_lw": 0.9, "cond_threshold": 50},
    {"base_lw": 0.2, "max_lw": 0.7, "cond_threshold": 100},
    {"base_lw": 0.4, "max_lw": 0.85, "cond_threshold": 40},
]


def build_config(yaml_path: str, blpx_overrides: dict | None = None,
                 pca_overrides: dict | None = None,
                 signal_components: dict | None = None) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        for key, val in blpx_overrides.items():
            cfg.setdefault("blpx", {})[key] = val
    if pca_overrides:
        for key, val in pca_overrides.items():
            if key in ("lambda_lw", "lambda_reg", "ewma_half_life", "corr_window"):
                cfg[key] = val
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


# ---------------------------------------------------------------------------
# Multi-horizon EWMA blend (monkey-patch)
# ---------------------------------------------------------------------------

_blend_state = {"short_hl": None, "long_hl": None, "blend_weight": 0.5}

_original_compute_correlation = corr_module.compute_correlation

# Import modules that hold references to compute_correlation / regularize_correlation
import leadlag.models.sector_relative_ensemble_blp_enhanced as _enhanced_mod
import leadlag.core.signal as _signal_mod
import leadlag.models.blp_base as _blp_base_mod

_orig_cc_enhanced = _enhanced_mod.compute_correlation
_orig_cc_signal = _signal_mod.compute_correlation
_orig_regcorr_enhanced = _enhanced_mod.regularize_correlation
_orig_regcorr_signal = _signal_mod.regularize_correlation


def _blended_compute_correlation(window_returns, ewma_half_life=None):
    """Blend two EWMA correlations: w * corr_short + (1-w) * corr_long."""
    short_hl = _blend_state["short_hl"]
    long_hl = _blend_state["long_hl"]
    w = _blend_state["blend_weight"]

    if short_hl is None or long_hl is None:
        return _original_compute_correlation(window_returns, ewma_half_life)

    mu_s, sigma_s, corr_s = _original_compute_correlation(window_returns, short_hl)
    mu_l, sigma_l, corr_l = _original_compute_correlation(window_returns, long_hl)

    mu = w * mu_s + (1 - w) * mu_l
    sigma = w * sigma_s + (1 - w) * sigma_l
    sigma = np.maximum(sigma, 1e-8)
    corr = w * corr_s + (1 - w) * corr_l
    np.fill_diagonal(corr, 1.0)

    return mu, sigma, corr


def enable_blend(short_hl, long_hl, blend_weight):
    _blend_state["short_hl"] = short_hl
    _blend_state["long_hl"] = long_hl
    _blend_state["blend_weight"] = blend_weight
    # Patch all modules that hold a reference to compute_correlation
    corr_module.compute_correlation = _blended_compute_correlation
    _enhanced_mod.compute_correlation = _blended_compute_correlation
    _signal_mod.compute_correlation = _blended_compute_correlation
    _BLP_CORR_CACHE.clear()


def disable_blend():
    _blend_state["short_hl"] = None
    _blend_state["long_hl"] = None
    corr_module.compute_correlation = _original_compute_correlation
    _enhanced_mod.compute_correlation = _orig_cc_enhanced
    _signal_mod.compute_correlation = _orig_cc_signal
    _BLP_CORR_CACHE.clear()


# ---------------------------------------------------------------------------
# Dynamic lambda_lw (monkey-patch regularize_correlation)
# ---------------------------------------------------------------------------

_dynamic_lw_state = {"base_lw": 0.5, "max_lw": 0.9, "cond_threshold": 50, "enabled": False}

_original_regularize_correlation = corr_module.regularize_correlation


def _dynamic_regularize_correlation(c_t, c_0_t, lambda_reg, lambda_lw, lw_target):
    """Adjust lambda_lw based on condition number of c_t."""
    if not _dynamic_lw_state["enabled"]:
        return _original_regularize_correlation(c_t, c_0_t, lambda_reg, lambda_lw, lw_target)

    try:
        eigvals = np.linalg.eigvalsh(c_t)
        eigvals_pos = eigvals[eigvals > 1e-10]
        if len(eigvals_pos) > 0:
            cond = float(eigvals_pos.max() / eigvals_pos.min())
        else:
            cond = 1000.0
    except Exception:
        cond = 1000.0

    base = _dynamic_lw_state["base_lw"]
    max_lw = _dynamic_lw_state["max_lw"]
    threshold = _dynamic_lw_state["cond_threshold"]

    if cond > threshold:
        ratio = min((cond - threshold) / (threshold * 2), 1.0)
        adj_lw = base + (max_lw - base) * ratio
    else:
        adj_lw = base

    return _original_regularize_correlation(c_t, c_0_t, lambda_reg, adj_lw, lw_target)


def enable_dynamic_lw(base_lw, max_lw, cond_threshold):
    _dynamic_lw_state["base_lw"] = base_lw
    _dynamic_lw_state["max_lw"] = max_lw
    _dynamic_lw_state["cond_threshold"] = cond_threshold
    _dynamic_lw_state["enabled"] = True
    corr_module.regularize_correlation = _dynamic_regularize_correlation
    _enhanced_mod.regularize_correlation = _dynamic_regularize_correlation
    _signal_mod.regularize_correlation = _dynamic_regularize_correlation
    _BLP_CORR_CACHE.clear()


def disable_dynamic_lw():
    _dynamic_lw_state["enabled"] = False
    corr_module.regularize_correlation = _original_regularize_correlation
    _enhanced_mod.regularize_correlation = _orig_regcorr_enhanced
    _signal_mod.regularize_correlation = _orig_regcorr_signal
    _BLP_CORR_CACHE.clear()


# ---------------------------------------------------------------------------
# Metric computation (reused from Phase 1A)
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


def run_variant(name, model, df_exec, y_target, start_date="2015-01-01", slippage_bps=5.0):
    t0 = time.perf_counter()
    results = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date=start_date,
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
    start_dt = pd.to_datetime(start_date)
    start_idx = max(df_exec.index.searchsorted(start_dt), getattr(model, "corr_window", 60))
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)

    return {
        "name": name, "AR_net": ar, "Vol_net": vol, "Sharpe_net": sharpe,
        "MDD": mdd, "Turnover": turnover, "Mean_Rank_IC": mean_ic,
        "ICIR": icir, "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1C: Correlation Estimation Improvements")
    parser.add_argument("--stage", choices=["oat", "blend", "dynamic", "wfo", "all"], default="all")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/phase1c_correlation_estimation")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec shape: %s", df_exec.shape)
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    all_results = []

    # --- Baseline (Phase 1A best + Phase 1B best ensemble) ---
    logger.info("=== Baseline (Phase 1A best + Phase 1B best ensemble) ===")
    cfg_base = build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                            signal_components=PHASE1B_WEIGHTS)
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    m_base = run_variant("baseline", model_base, df_exec, y_target, slippage_bps=args.slippage_bps)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                m_base["Sharpe_net"], m_base["Mean_Rank_IC"], m_base["ICIR"],
                m_base["MDD"] * 100, m_base["elapsed_s"])
    all_results.append({**m_base, "scheme": "baseline", "param": "baseline", "value": ""})

    # --- Stage 1: OAT sensitivity ---
    if args.stage in ("oat", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: OAT SENSITIVITY (lambda_lw, lambda_reg, ewma_half_life, corr_window)")
        logger.info("=" * 80)

        for param_name, levels in OAT_LEVELS.items():
            logger.info("=== OAT: %s ===", param_name)
            best_val = PCA_PARAMS[param_name]
            best_score = m_base["Sharpe_net"]

            for val in levels:
                if val == PCA_PARAMS[param_name]:
                    continue

                pca_overrides = PCA_PARAMS.copy()
                pca_overrides[param_name] = val
                cfg = build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                                   pca_overrides=pca_overrides,
                                   signal_components=PHASE1B_WEIGHTS)
                _BLP_CORR_CACHE.clear()
                model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
                name = f"oat_{param_name}_{val}"
                m = run_variant(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
                m["scheme"] = "oat"
                m["param"] = param_name
                m["value"] = val
                all_results.append(m)
                logger.info("  %s=%s: Sharpe=%.4f IC=%.4f ICIR=%.2f (%.1fs)",
                            param_name, val, m["Sharpe_net"], m["Mean_Rank_IC"],
                            m["ICIR"], m["elapsed_s"])

                if np.isfinite(m["Sharpe_net"]) and m["Sharpe_net"] > best_score:
                    best_score = m["Sharpe_net"]
                    best_val = val

            logger.info("  -> Best %s = %s (Sharpe=%.4f, delta=%+.4f)",
                        param_name, best_val, best_score, best_score - m_base["Sharpe_net"])

    # --- Stage 2: Multi-horizon EWMA blend ---
    if args.stage in ("blend", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: MULTI-HORIZON EWMA BLEND")
        logger.info("=" * 80)

        for cfg_blend in BLEND_CONFIGS:
            enable_blend(cfg_blend["short_hl"], cfg_blend["long_hl"], cfg_blend["blend_weight"])
            cfg = build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                               signal_components=PHASE1B_WEIGHTS)
            _BLP_CORR_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            name = f"blend_s{cfg_blend['short_hl']}_l{cfg_blend['long_hl']}_w{cfg_blend['blend_weight']}"
            m = run_variant(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
            m["scheme"] = "blend"
            m.update(cfg_blend)
            all_results.append(m)
            logger.info("  %s: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                        name, m["Sharpe_net"], m["Mean_Rank_IC"], m["ICIR"],
                        m["MDD"] * 100, m["elapsed_s"])

        disable_blend()
        _BLP_CORR_CACHE.clear()

    # --- Stage 3: Dynamic lambda_lw ---
    if args.stage in ("dynamic", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: DYNAMIC LAMBDA_LW (condition-number adaptive)")
        logger.info("=" * 80)

        for cfg_dyn in DYNAMIC_LW_CONFIGS:
            enable_dynamic_lw(cfg_dyn["base_lw"], cfg_dyn["max_lw"], cfg_dyn["cond_threshold"])
            cfg = build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                               signal_components=PHASE1B_WEIGHTS)
            _BLP_CORR_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            name = f"dyn_lw_b{cfg_dyn['base_lw']}_m{cfg_dyn['max_lw']}_c{cfg_dyn['cond_threshold']}"
            m = run_variant(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
            m["scheme"] = "dynamic_lw"
            m.update(cfg_dyn)
            all_results.append(m)
            logger.info("  %s: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                        name, m["Sharpe_net"], m["Mean_Rank_IC"], m["ICIR"],
                        m["MDD"] * 100, m["elapsed_s"])

        disable_dynamic_lw()
        _BLP_CORR_CACHE.clear()

    # --- Save all results ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    logger.info("Results saved to %s", output_dir / "all_results.csv")

    # --- Print summaries ---
    print("\n" + "=" * 100)
    print("PHASE 1C — CORRELATION ESTIMATION IMPROVEMENTS — RESULTS")
    print("=" * 100)

    print(f"\nBaseline (Phase 1A+1B best):")
    print(f"  Sharpe: {m_base['Sharpe_net']:.4f}")
    print(f"  IC:     {m_base['Mean_Rank_IC']:.4f}")
    print(f"  ICIR:   {m_base['ICIR']:.2f}")
    print(f"  MDD:    {m_base['MDD']*100:.2f}%")

    # OAT summary
    oat_results = [r for r in all_results if r.get("scheme") == "oat"]
    if oat_results:
        print("\n--- OAT Sensitivity Summary ---")
        print(f"{'Parameter':<20} {'Best Value':<12} {'Best Sharpe':<12} {'Delta':<10}")
        print("-" * 54)
        for param_name in OAT_LEVELS:
            param_rows = [r for r in oat_results if r["param"] == param_name]
            if param_rows:
                best = max(param_rows, key=lambda x: x["Sharpe_net"] if np.isfinite(x["Sharpe_net"]) else -999)
                print(f"{param_name:<20} {str(best['value']):<12} {best['Sharpe_net']:<12.4f} {best['Sharpe_net']-m_base['Sharpe_net']:+.4f}")

    # Blend summary
    blend_results = [r for r in all_results if r.get("scheme") == "blend"]
    if blend_results:
        print("\n--- Multi-Horizon EWMA Blend ---")
        print(f"{'Config':<45} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Delta':<8}")
        print("-" * 89)
        for r in sorted(blend_results, key=lambda x: -x["Sharpe_net"]):
            print(f"{r['name']:<45} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {r['Sharpe_net']-m_base['Sharpe_net']:+.4f}")

    # Dynamic LW summary
    dyn_results = [r for r in all_results if r.get("scheme") == "dynamic_lw"]
    if dyn_results:
        print("\n--- Dynamic lambda_lw ---")
        print(f"{'Config':<50} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Delta':<8}")
        print("-" * 94)
        for r in sorted(dyn_results, key=lambda x: -x["Sharpe_net"]):
            print(f"{r['name']:<50} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {r['Sharpe_net']-m_base['Sharpe_net']:+.4f}")

    # Best overall
    valid = [r for r in all_results if r.get("scheme") != "baseline" and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest Overall: {best['name']}")
        print(f"  Sharpe: {best['Sharpe_net']:.4f} (delta={best['Sharpe_net']-m_base['Sharpe_net']:+.4f})")
        print(f"  IC:     {best['Mean_Rank_IC']:.4f} (delta={best['Mean_Rank_IC']-m_base['Mean_Rank_IC']:+.4f})")
        print(f"  ICIR:   {best['ICIR']:.2f} (delta={best['ICIR']-m_base['ICIR']:+.2f})")
        print(f"  MDD:    {best['MDD']*100:.2f}%")

    # --- Walk-Forward OOS for top candidates ---
    if args.stage in ("wfo", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD OOS VALIDATION")
        logger.info("=" * 80)

        candidates = [("baseline", None, None)]

        # Top 3 overall
        if valid:
            top3 = sorted(valid, key=lambda x: -x["Sharpe_net"])[:3]
            for r in top3:
                candidates.append((r["name"], r.get("scheme"), r))

        eval_dates = df_exec.index[max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60):]
        n_eval = len(eval_dates)
        fold_size = n_eval // args.n_folds

        wfo_all = []
        for name, scheme, params in candidates:
            logger.info("WFO for %s", name)

            if scheme == "blend":
                enable_blend(params["short_hl"], params["long_hl"], params["blend_weight"])
            elif scheme == "dynamic_lw":
                enable_dynamic_lw(params["base_lw"], params["max_lw"], params["cond_threshold"])

            cfg = build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                               signal_components=PHASE1B_WEIGHTS)
            _BLP_CORR_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            results = BacktestEngine.run_backtest(
                model, df_exec=df_exec, start_date="2015-01-01",
                overnight_alpha_long=0.75, overnight_alpha_short=0.5,
                buy_interest_annual=0.025, borrow_fee_annual=0.0115,
                reverse_fee_bps=2.0, slippage_bps=args.slippage_bps,
            )

            if scheme == "blend":
                disable_blend()
            elif scheme == "dynamic_lw":
                disable_dynamic_lw()
            _BLP_CORR_CACHE.clear()

            dr = results["daily_returns"]
            for fold in range(args.n_folds):
                f_start = fold * fold_size
                f_end = min((fold + 1) * fold_size, n_eval)
                fold_dates = eval_dates[f_start:f_end]
                fold_dr = dr.reindex(fold_dates).dropna()
                if len(fold_dr) < 5:
                    continue
                fold_ar = float(fold_dr.mean() * 245)
                fold_vol = float(fold_dr.std(ddof=1) * np.sqrt(245))
                fold_sharpe = fold_ar / fold_vol if fold_vol > 0 else np.nan
                fold_wealth = (1.0 + fold_dr).cumprod()
                fold_mdd = float(((fold_wealth / fold_wealth.cummax()) - 1.0).min())

                wfo_all.append({
                    "candidate": name, "fold": fold + 1,
                    "start_date": fold_dates[0].strftime("%Y-%m-%d"),
                    "end_date": fold_dates[-1].strftime("%Y-%m-%d"),
                    "n_days": len(fold_dr),
                    "AR_net": fold_ar, "Sharpe_net": fold_sharpe,
                    "MDD": fold_mdd,
                })

        wfo_df = pd.DataFrame(wfo_all)
        wfo_df.to_csv(output_dir / "walk_forward_results.csv", index=False)

        print("\n--- Walk-Forward OOS Summary ---")
        for name, _, _ in candidates:
            folds = wfo_df[wfo_df["candidate"] == name]
            if len(folds) == 0:
                continue
            sharpes = folds["Sharpe_net"].dropna()
            print(f"  {name}:")
            print(f"    Mean Sharpe: {sharpes.mean():.4f} (std={sharpes.std():.4f})")
            print(f"    Min Sharpe:  {sharpes.min():.4f}")
            print(f"    Max MDD:     {folds['MDD'].min():.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
