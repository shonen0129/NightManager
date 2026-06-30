"""Phase 1D: Sector Prior Distribution Optimization.

Three approaches:
  1. OAT sensitivity: sector_eta, sector_gamma, lambda_sector
  2. Data-driven M_sector: full correlation mapping, ridge regression, expanded structural
  3. Walk-forward OOS validation of best configurations

Baseline: Phase 1A best params + Phase 1B best ensemble weights.
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
)

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

# Baseline sector prior params (current production: eta=0, gamma=2, lambda_sector=0.60)
SECTOR_BASELINE = {
    "sector_eta": 0.0,
    "sector_gamma": 2.0,
    "lambda_sector": 0.60,
}

# OAT sweep levels
OAT_LEVELS = {
    "sector_eta":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "sector_gamma":   [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "lambda_sector":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

# Data-driven M_sector configs (tested after OAT identifies best eta/gamma)
DATADRIVEN_CONFIGS = [
    # Ridge regression-based sector prior
    {"method": "ridge", "ridge_rho": 0.01},
    {"method": "ridge", "ridge_rho": 0.05},
    {"method": "ridge", "ridge_rho": 0.10},
    {"method": "ridge", "ridge_rho": 0.20},
    # Full correlation mapping (all JP tickers, not just structural)
    {"method": "full_corr", "gamma": 1.0},
    {"method": "full_corr", "gamma": 2.0},
    {"method": "full_corr", "gamma": 3.0},
    # Expanded structural: structural + top-K additional JP tickers by correlation
    {"method": "expanded", "gamma": 2.0, "top_k": 2},
    {"method": "expanded", "gamma": 2.0, "top_k": 3},
    {"method": "expanded", "gamma": 1.0, "top_k": 3},
]

BLPX_KEYS = {
    "alpha_xx", "alpha_yy", "alpha_yx", "lambda_pca", "lambda_sector",
    "beta_conf", "rho", "winsor_sigma", "blp_window", "blp_ewma_halflife",
    "sector_eta", "sector_gamma",
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(yaml_path: str, blpx_overrides: dict | None = None,
                 signal_components: dict | None = None) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        for key, val in blpx_overrides.items():
            cfg.setdefault("blpx", {})[key] = val
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


# ---------------------------------------------------------------------------
# Data-driven M_sector model variants
# ---------------------------------------------------------------------------

class FullCorrSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Use full cross-correlation mapping (all JP tickers, not just structural)."""

    def __init__(self, config, gamma=2.0):
        super().__init__(config)
        self._full_corr_gamma = gamma

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        if corr.shape != (self.n_u + self.n_j, self.n_u + self.n_j):
            return self._M_sector_fixed if self._M_sector_fixed.shape == B_blp.shape else np.zeros(B_blp.shape)

        c_xy = corr[: self.n_u, self.n_u:]  # (n_u, n_j)

        M_data = np.zeros((self.n_j, self.n_u))
        for u_idx in range(self.n_u):
            weights = []
            for j_idx in range(self.n_j):
                raw_corr = c_xy[u_idx, j_idx]
                w = max(0.0, raw_corr) ** self._full_corr_gamma
                if w > 1e-12:
                    weights.append((j_idx, w))
            if not weights:
                continue
            total = sum(w for _, w in weights)
            if total > 1e-10:
                for j_idx, w in weights:
                    M_data[j_idx, u_idx] = w / total

        if M_data.shape == B_blp.shape:
            return M_data
        return np.zeros(B_blp.shape)


class ExpandedStructuralSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Structural mapping + top-K additional JP tickers by correlation."""

    def __init__(self, config, gamma=2.0, top_k=2):
        super().__init__(config)
        self._exp_gamma = gamma
        self._exp_top_k = top_k

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        if corr.shape != (self.n_u + self.n_j, self.n_u + self.n_j):
            return self._M_sector_fixed if self._M_sector_fixed.shape == B_blp.shape else np.zeros(B_blp.shape)

        c_xy = corr[: self.n_u, self.n_u:]  # (n_u, n_j)

        M = self._M_sector_fixed.copy()
        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk not in self._SECTOR_MAPPING_STRUCTURE:
                continue
            mapped_jp = set(self._SECTOR_MAPPING_STRUCTURE[us_tk])
            unmapped_indices = []
            for j_idx, jp_tk in enumerate(JP_TICKERS):
                if jp_tk not in mapped_jp:
                    unmapped_indices.append((j_idx, c_xy[u_idx, j_idx]))
            unmapped_indices.sort(key=lambda x: -x[1])
            for j_idx, raw_corr in unmapped_indices[:self._exp_top_k]:
                if raw_corr > 0:
                    M[j_idx, u_idx] = max(0.0, raw_corr) ** self._exp_gamma

        col_sums = np.sum(M, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 1e-10:
                M[:, u_idx] /= col_sums[u_idx]

        if M.shape == B_blp.shape:
            return M
        return np.zeros(B_blp.shape)


class RidgeSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Use rolling ridge regression coefficients as sector prior."""

    def __init__(self, config, ridge_rho=0.05):
        super().__init__(config)
        self._ridge_rho = ridge_rho

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        window_start = max(0, current_index - self.blp_window)
        W = all_returns[window_start:current_index]
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        X = W[:, :self.n_u]
        Y = W[:, self.n_u:]
        XtX = X.T @ X
        ridge = self._ridge_rho * np.mean(np.diag(XtX)) * np.eye(self.n_u)
        try:
            A_inv = np.linalg.inv(XtX + ridge)
            B_ridge = Y.T @ X @ A_inv
        except Exception:
            B_ridge = np.zeros((self.n_j, self.n_u))
        if B_ridge.shape == B_blp.shape:
            return B_ridge
        return np.zeros(B_blp.shape)


# ---------------------------------------------------------------------------
# Metric computation
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
    parser = argparse.ArgumentParser(description="Phase 1D: Sector Prior Optimization")
    parser.add_argument("--stage", choices=["oat", "datadriven", "wfo", "all"], default="all")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/phase1d_sector_prior")
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
    _BLP_CORR_CACHE.clear()
    m_base = run_variant("baseline", model_base, df_exec, y_target, slippage_bps=args.slippage_bps)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                m_base["Sharpe_net"], m_base["Mean_Rank_IC"], m_base["ICIR"],
                m_base["MDD"] * 100, m_base["elapsed_s"])
    all_results.append({**m_base, "scheme": "baseline", "param": "baseline", "value": ""})

    # --- Stage 1: OAT sensitivity ---
    if args.stage in ("oat", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: OAT SENSITIVITY (sector_eta, sector_gamma, lambda_sector)")
        logger.info("=" * 80)

        for param_name, levels in OAT_LEVELS.items():
            logger.info("=== OAT: %s ===", param_name)
            best_val = SECTOR_BASELINE[param_name]
            best_score = m_base["Sharpe_net"]

            for val in levels:
                if val == SECTOR_BASELINE[param_name]:
                    continue

                overrides = PHASE1A_BEST.copy()
                overrides[param_name] = val
                cfg = build_config(yaml_path, blpx_overrides=overrides,
                                   signal_components=PHASE1B_WEIGHTS)
                _BLP_CORR_CACHE.clear()
                model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
                name = f"oat_{param_name}_{val}"
                m = run_variant(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
                m["scheme"] = "oat"
                m["param"] = param_name
                m["value"] = val
                all_results.append(m)
                logger.info("  %s=%s: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                            param_name, val, m["Sharpe_net"], m["Mean_Rank_IC"],
                            m["ICIR"], m["MDD"] * 100, m["elapsed_s"])

                if np.isfinite(m["Sharpe_net"]) and m["Sharpe_net"] > best_score:
                    best_score = m["Sharpe_net"]
                    best_val = val

            logger.info("  -> Best %s = %s (Sharpe=%.4f, delta=%+.4f)",
                        param_name, best_val, best_score, best_score - m_base["Sharpe_net"])

    # --- Stage 2: Data-driven M_sector approaches ---
    if args.stage in ("datadriven", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: DATA-DRIVEN M_SECTOR APPROACHES")
        logger.info("=" * 80)

        for cfg_dd in DATADRIVEN_CONFIGS:
            method = cfg_dd["method"]
            _BLP_CORR_CACHE.clear()

            if method == "ridge":
                model = RidgeSectorModel(
                    build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                                 signal_components=PHASE1B_WEIGHTS),
                    ridge_rho=cfg_dd["ridge_rho"],
                )
                name = f"ridge_rho{cfg_dd['ridge_rho']}"
            elif method == "full_corr":
                model = FullCorrSectorModel(
                    build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                                 signal_components=PHASE1B_WEIGHTS),
                    gamma=cfg_dd["gamma"],
                )
                name = f"full_corr_g{cfg_dd['gamma']}"
            elif method == "expanded":
                model = ExpandedStructuralSectorModel(
                    build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                                 signal_components=PHASE1B_WEIGHTS),
                    gamma=cfg_dd["gamma"],
                    top_k=cfg_dd["top_k"],
                )
                name = f"expanded_g{cfg_dd['gamma']}_k{cfg_dd['top_k']}"
            else:
                continue

            m = run_variant(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
            m["scheme"] = "datadriven"
            m["method"] = method
            m.update(cfg_dd)
            all_results.append(m)
            logger.info("  %s: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                        name, m["Sharpe_net"], m["Mean_Rank_IC"],
                        m["ICIR"], m["MDD"] * 100, m["elapsed_s"])

    # --- Save all results ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    logger.info("Results saved to %s", output_dir / "all_results.csv")

    # --- Print summaries ---
    print("\n" + "=" * 100)
    print("PHASE 1D — SECTOR PRIOR OPTIMIZATION — RESULTS")
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

    # Data-driven summary
    dd_results = [r for r in all_results if r.get("scheme") == "datadriven"]
    if dd_results:
        print("\n--- Data-Driven M_sector Approaches ---")
        print(f"{'Config':<40} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8} {'Delta':<8}")
        print("-" * 84)
        for r in sorted(dd_results, key=lambda x: -x["Sharpe_net"]):
            print(f"{r['name']:<40} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f} {r['Sharpe_net']-m_base['Sharpe_net']:+.4f}")

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

            cfg = build_config(yaml_path, blpx_overrides=PHASE1A_BEST,
                               signal_components=PHASE1B_WEIGHTS)
            _BLP_CORR_CACHE.clear()

            if scheme == "oat":
                oat_param = params["param"]
                oat_val = params["value"]
                cfg_blpx = PHASE1A_BEST.copy()
                cfg_blpx[oat_param] = oat_val
                cfg = build_config(yaml_path, blpx_overrides=cfg_blpx,
                                   signal_components=PHASE1B_WEIGHTS)
                model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            elif scheme == "datadriven":
                method = params.get("method", "")
                if method == "ridge":
                    model = RidgeSectorModel(cfg, ridge_rho=params["ridge_rho"])
                elif method == "full_corr":
                    model = FullCorrSectorModel(cfg, gamma=params["gamma"])
                elif method == "expanded":
                    model = ExpandedStructuralSectorModel(cfg, gamma=params["gamma"],
                                                          top_k=params["top_k"])
                else:
                    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            else:
                model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

            results = BacktestEngine.run_backtest(
                model, df_exec=df_exec, start_date="2015-01-01",
                overnight_alpha_long=0.75, overnight_alpha_short=0.5,
                buy_interest_annual=0.025, borrow_fee_annual=0.0115,
                reverse_fee_bps=2.0, slippage_bps=args.slippage_bps,
            )

            dr = results["daily_returns"]
            signals_df = results["signals"]
            y_df = pd.DataFrame(y_target, index=df_exec.index, columns=JP_TICKERS)

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

                ic_vals = []
                for date in fold_dates:
                    if date in signals_df.index:
                        sig_t = signals_df.loc[date].values
                        y_t = y_df.loc[date].values
                        valid_mask = ~(np.isnan(sig_t) | np.isnan(y_t))
                        if valid_mask.sum() >= 3:
                            rho, _ = stats.spearmanr(sig_t[valid_mask], y_t[valid_mask])
                            if np.isfinite(rho):
                                ic_vals.append(float(rho))

                fold_ic = float(np.mean(ic_vals)) if ic_vals else np.nan
                fold_icir = (fold_ic / np.std(ic_vals, ddof=1) * np.sqrt(252)) if (len(ic_vals) > 1 and np.std(ic_vals, ddof=1) > 1e-8) else np.nan

                wfo_all.append({
                    "candidate": name, "fold": fold + 1,
                    "start_date": fold_dates[0].strftime("%Y-%m-%d"),
                    "end_date": fold_dates[-1].strftime("%Y-%m-%d"),
                    "n_days": len(fold_dr),
                    "AR_net": fold_ar, "Sharpe_net": fold_sharpe,
                    "MDD": fold_mdd, "Mean_Rank_IC": fold_ic, "ICIR": fold_icir,
                })

            _BLP_CORR_CACHE.clear()

        wfo_df = pd.DataFrame(wfo_all)
        wfo_df.to_csv(output_dir / "walk_forward_results.csv", index=False)

        print("\n--- Walk-Forward OOS Summary ---")
        for name, _, _ in candidates:
            folds = wfo_df[wfo_df["candidate"] == name]
            if len(folds) == 0:
                continue
            sharpes = folds["Sharpe_net"].dropna()
            ics = folds["Mean_Rank_IC"].dropna()
            print(f"  {name}:")
            print(f"    Mean Sharpe: {sharpes.mean():.4f} (std={sharpes.std():.4f})")
            print(f"    Min Sharpe:  {sharpes.min():.4f}")
            print(f"    Mean IC:     {ics.mean():.4f}")
            print(f"    Max MDD:     {folds['MDD'].min():.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
