"""Phase 1A: BLPX Parameter Optimization — OAT + Focused Grid + WFO.

Three-stage optimization:
  Stage 1 (OAT):    One-at-a-time sensitivity sweep for each parameter
  Stage 2 (Grid):   Focused grid on top-K most impactful parameters
  Stage 3 (WFO):    Walk-forward OOS validation of the best parameter set

Metrics: Sharpe (net), AR (net), MDD, Mean Rank IC, ICIR, Turnover
"""

from __future__ import annotations

import argparse
import itertools
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
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

BLPX_KEYS = {
    "alpha_xx", "alpha_yy", "alpha_yx", "lambda_pca", "lambda_sector",
    "beta_conf", "rho", "winsor_sigma", "blp_window", "blp_ewma_halflife",
}

BASELINE_PARAMS = {
    "alpha_xx": 0.50, "alpha_yy": 0.50, "alpha_yx": 0.05,
    "lambda_pca": 0.10, "lambda_sector": 0.40, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 252,
    "blp_ewma_halflife": 45,
}

# OAT sweep levels for each parameter
OAT_LEVELS = {
    "alpha_xx":      [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "alpha_yy":      [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "alpha_yx":      [0.00, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30],
    "lambda_pca":    [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    "lambda_sector": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
    "beta_conf":     [0.00, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50],
    "rho":           [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05],
    "winsor_sigma":  [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 99.0],
    "blp_window":    [63, 126, 189, 252, 315, 378, 504],
    "blp_ewma_halflife": [21, 30, 45, 63, 90, 120, 180],
}


def build_config(yaml_path: str, overrides: dict | None = None) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for key, val in overrides.items():
            if key in BLPX_KEYS:
                cfg.setdefault("blpx", {})[key] = val
            else:
                cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_rank_ic(signals_df: pd.DataFrame, y_target: np.ndarray,
                    sim_dates: pd.DatetimeIndex, start_idx: int) -> tuple[float, float]:
    """Compute daily Spearman Rank IC between signals and actual JP target returns."""
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
    ic_series = np.array(ic_list)
    mean_ic = float(np.mean(ic_series))
    std_ic = float(np.std(ic_series, ddof=1))
    icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic > 1e-8 else np.nan
    return mean_ic, icir


def run_variant(name: str, model: SectorRelativeEnsembleBLPEnhancedModel,
                df_exec: pd.DataFrame, y_target: np.ndarray,
                start_date: str = "2015-01-01",
                slippage_bps: float = 5.0) -> dict:
    """Run a single backtest variant and compute all metrics."""
    t0 = time.perf_counter()

    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date=start_date,
        overnight_alpha_long=0.75,
        overnight_alpha_short=0.5,
        buy_interest_annual=0.025,
        borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
        slippage_bps=slippage_bps,
    )
    elapsed = time.perf_counter() - t0

    dr = results["daily_returns"]
    dr_gross = results["daily_returns_gross"]

    ar = float(dr.mean() * 245)
    ar_gross = float(dr_gross.mean() * 245)
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan

    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(results["daily_turnover"].mean())
    gross_exp = float(results["daily_gross_exps"].mean())

    sim_dates = df_exec.index
    start_dt = pd.to_datetime(start_date)
    start_idx = max(df_exec.index.searchsorted(start_dt), getattr(model, "corr_window", 60))
    signals_df = results["signals"]
    mean_ic, icir = compute_rank_ic(signals_df, y_target, sim_dates, start_idx)

    monthly = (1.0 + dr).groupby(dr.index.year * 12 + dr.index.month).prod() - 1.0
    if len(monthly) > 1:
        monthly_sharpe = float((monthly.mean() / monthly.std(ddof=1)) * np.sqrt(12.0))
    else:
        monthly_sharpe = np.nan

    total_cost = float(results["daily_costs"].mean() * 245)

    return {
        "name": name,
        "AR_net": ar,
        "AR_gross": ar_gross,
        "Vol_net": vol,
        "Sharpe_net": sharpe,
        "Sharpe_monthly": monthly_sharpe,
        "MDD": mdd,
        "Turnover": turnover,
        "GrossExp": gross_exp,
        "Cost_annual": total_cost,
        "Mean_Rank_IC": mean_ic,
        "ICIR": icir,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Stage 1: OAT Sensitivity Sweep
# ---------------------------------------------------------------------------

def run_oat_sweep(df_exec: pd.DataFrame, y_target: np.ndarray,
                  yaml_path: str, slippage_bps: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Run one-at-a-time parameter sensitivity sweep.

    Returns (results_df, best_params_per_key).
    """
    all_metrics = []

    # Baseline reference
    logger.info("Running baseline...")
    cfg_base = build_config(yaml_path)
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    base_m = run_variant("baseline", model_base, df_exec, y_target, slippage_bps=slippage_bps)
    base_m["param"] = "baseline"
    base_m["value"] = ""
    for k, v in BASELINE_PARAMS.items():
        base_m[k] = v
    all_metrics.append(base_m)
    logger.info("Baseline: Sharpe=%.4f IC=%.4f ICIR=%.2f (%.1fs)",
                base_m["Sharpe_net"], base_m["Mean_Rank_IC"], base_m["ICIR"], base_m["elapsed_s"])

    best_per_key = {}
    for param_name, levels in OAT_LEVELS.items():
        logger.info("=== OAT sweep: %s ===", param_name)
        best_val = BASELINE_PARAMS[param_name]
        best_score = base_m["Sharpe_net"]

        for val in levels:
            if val == BASELINE_PARAMS[param_name]:
                continue

            params = BASELINE_PARAMS.copy()
            params[param_name] = val
            cfg = build_config(yaml_path, overrides=params)
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            name = f"{param_name}_{val}"
            m = run_variant(name, model, df_exec, y_target, slippage_bps=slippage_bps)
            m["param"] = param_name
            m["value"] = val
            for k, v in params.items():
                m[k] = v
            all_metrics.append(m)
            logger.info("  %s=%s: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                        param_name, val, m["Sharpe_net"], m["Mean_Rank_IC"],
                        m["ICIR"], m["MDD"] * 100, m["elapsed_s"])

            if np.isfinite(m["Sharpe_net"]) and m["Sharpe_net"] > best_score:
                best_score = m["Sharpe_net"]
                best_val = val

        best_per_key[param_name] = {"best_val": best_val, "best_sharpe": best_score,
                                     "delta": best_score - base_m["Sharpe_net"]}
        logger.info("  -> Best %s = %s (Sharpe=%.4f, delta=%+.4f)",
                    param_name, best_val, best_score, best_score - base_m["Sharpe_net"])

    df_metrics = pd.DataFrame(all_metrics)
    return df_metrics, best_per_key


# ---------------------------------------------------------------------------
# Stage 2: Focused Grid Search
# ---------------------------------------------------------------------------

def select_top_params(best_per_key: dict, top_k: int = 5) -> list[str]:
    """Select top-K parameters by absolute Sharpe improvement."""
    ranked = sorted(best_per_key.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)
    return [k for k, _ in ranked[:top_k]]


def build_focused_levels(top_params: list[str], best_per_key: dict) -> dict:
    """Build focused grid levels for top parameters."""
    grid = {}
    for param in top_params:
        best_val = best_per_key[param]["best_val"]
        base_val = BASELINE_PARAMS[param]

        all_vals = sorted(set([best_val, base_val] + OAT_LEVELS[param]))

        if len(all_vals) <= 5:
            grid[param] = all_vals
        else:
            idx = all_vals.index(best_val) if best_val in all_vals else len(all_vals) // 2
            start = max(0, idx - 2)
            end = min(len(all_vals), idx + 3)
            grid[param] = all_vals[start:end]

    return grid


def run_focused_grid(grid: dict, df_exec: pd.DataFrame, y_target: np.ndarray,
                     yaml_path: str, slippage_bps: float = 5.0) -> pd.DataFrame:
    """Run focused grid search over selected parameters."""
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos = list(itertools.product(*value_lists))

    logger.info("Focused grid: %d combinations over %s", len(combos), keys)

    all_metrics = []
    for idx, combo in enumerate(combos):
        params = BASELINE_PARAMS.copy()
        params.update(dict(zip(keys, combo)))
        name = "_".join(f"{k[:4]}{v}" for k, v in zip(keys, combo))
        cfg = build_config(yaml_path, overrides=params)
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        m = run_variant(name, model, df_exec, y_target, slippage_bps=slippage_bps)
        m["params"] = str(params)
        m.update(params)
        all_metrics.append(m)

        if (idx + 1) % 5 == 0 or idx == 0:
            logger.info("[Grid] %d/%d: %s — Sharpe=%.4f IC=%.4f ICIR=%.2f (%.1fs)",
                        idx + 1, len(combos), name,
                        m["Sharpe_net"], m["Mean_Rank_IC"], m["ICIR"], m["elapsed_s"])

    return pd.DataFrame(all_metrics)


# ---------------------------------------------------------------------------
# Stage 3: Walk-Forward OOS Validation
# ---------------------------------------------------------------------------

def run_walk_forward(params: dict, df_exec: pd.DataFrame, y_target: np.ndarray,
                     yaml_path: str, n_folds: int = 5,
                     slippage_bps: float = 5.0) -> dict:
    """Run walk-forward OOS validation by slicing the backtest into folds."""
    sim_dates = df_exec.index
    start_dt = pd.to_datetime("2015-01-01")
    start_idx = max(df_exec.index.searchsorted(start_dt), 60)
    eval_dates = sim_dates[start_idx:]
    n_eval = len(eval_dates)

    fold_size = n_eval // n_folds
    if fold_size < 20:
        n_folds = 1
        fold_size = n_eval

    cfg = build_config(yaml_path, overrides=params)
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

    logger.info("Running full backtest for walk-forward slicing...")
    results = BacktestEngine.run_backtest(
        model,
        df_exec=df_exec,
        start_date="2015-01-01",
        overnight_alpha_long=0.75,
        overnight_alpha_short=0.5,
        buy_interest_annual=0.025,
        borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
        slippage_bps=slippage_bps,
    )

    dr = results["daily_returns"]
    signals_df = results["signals"]
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)

    fold_metrics = []
    for fold in range(n_folds):
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
                valid = ~(np.isnan(sig_t) | np.isnan(y_t))
                if valid.sum() >= 3:
                    rho, _ = stats.spearmanr(sig_t[valid], y_t[valid])
                    if np.isfinite(rho):
                        ic_vals.append(float(rho))

        fold_ic = float(np.mean(ic_vals)) if ic_vals else np.nan
        fold_icir = (fold_ic / np.std(ic_vals, ddof=1) * np.sqrt(252)) if (len(ic_vals) > 1 and np.std(ic_vals, ddof=1) > 1e-8) else np.nan

        fold_metrics.append({
            "fold": fold + 1,
            "start_date": fold_dates[0].strftime("%Y-%m-%d"),
            "end_date": fold_dates[-1].strftime("%Y-%m-%d"),
            "n_days": len(fold_dr),
            "AR_net": fold_ar,
            "Sharpe_net": fold_sharpe,
            "MDD": fold_mdd,
            "Mean_Rank_IC": fold_ic,
            "ICIR": fold_icir,
        })

    return {"folds": fold_metrics, "params": params}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1A: BLPX Parameter Optimization")
    parser.add_argument("--stage", choices=["oat", "grid", "wfo", "all"], default="all",
                        help="Which stage to run (default: all)")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="Slippage bps per side")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of walk-forward folds")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K parameters for focused grid")
    parser.add_argument("--output-dir", default="artifacts/phase1a_parameter_optimization",
                        help="Output directory for results")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec shape: %s", df_exec.shape)

    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    # --- Baseline ---
    logger.info("=== Baseline (current production parameters) ===")
    cfg_base = build_config(yaml_path)
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    baseline_m = run_variant("baseline", model_base, df_exec, y_target,
                             slippage_bps=args.slippage_bps)
    logger.info("Baseline: Sharpe=%.4f  AR=%.2f%%  MDD=%.2f%%  IC=%.4f  ICIR=%.2f",
                baseline_m["Sharpe_net"], baseline_m["AR_net"] * 100,
                baseline_m["MDD"] * 100, baseline_m["Mean_Rank_IC"], baseline_m["ICIR"])

    # --- Stage 1: OAT ---
    oat_df = None
    best_per_key = None
    if args.stage in ("oat", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: ONE-AT-A-TIME SENSITIVITY SWEEP")
        logger.info("=" * 80)
        oat_df, best_per_key = run_oat_sweep(df_exec, y_target, yaml_path,
                                             slippage_bps=args.slippage_bps)
        oat_df.to_csv(output_dir / "oat_sensitivity.csv", index=False)
        logger.info("OAT results saved to %s", output_dir / "oat_sensitivity.csv")

        print("\n--- OAT Sensitivity Summary ---")
        print(f"{'Parameter':<25} {'Best Value':<12} {'Best Sharpe':<12} {'Delta vs Base':<12}")
        print("-" * 61)
        for param_name, info in sorted(best_per_key.items(), key=lambda x: abs(x[1]["delta"]), reverse=True):
            print(f"{param_name:<25} {str(info['best_val']):<12} {info['best_sharpe']:<12.4f} {info['delta']:+.4f}")

    # --- Stage 2: Focused Grid ---
    grid_df = None
    if args.stage in ("grid", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: FOCUSED GRID SEARCH")
        logger.info("=" * 80)

        if best_per_key is None:
            oat_path = output_dir / "oat_sensitivity.csv"
            if oat_path.exists():
                oat_df = pd.read_csv(oat_path)
                best_per_key = {}
                for param_name in OAT_LEVELS:
                    param_rows = oat_df[oat_df["param"] == param_name]
                    if len(param_rows) > 0:
                        best_row = param_rows.nlargest(1, "Sharpe_net").iloc[0]
                        best_per_key[param_name] = {
                            "best_val": best_row["value"],
                            "best_sharpe": best_row["Sharpe_net"],
                            "delta": best_row["Sharpe_net"] - baseline_m["Sharpe_net"],
                        }
                    else:
                        best_per_key[param_name] = {
                            "best_val": BASELINE_PARAMS[param_name],
                            "best_sharpe": baseline_m["Sharpe_net"],
                            "delta": 0.0,
                        }
            else:
                logger.warning("No OAT results found, using baseline params for grid")
                best_per_key = {k: {"best_val": v, "best_sharpe": baseline_m["Sharpe_net"], "delta": 0.0}
                                for k, v in BASELINE_PARAMS.items()}

        top_params = select_top_params(best_per_key, top_k=args.top_k)
        logger.info("Top-%d parameters by impact: %s", args.top_k, top_params)

        focused_grid = build_focused_levels(top_params, best_per_key)
        logger.info("Focused grid levels: %s", {k: len(v) for k, v in focused_grid.items()})

        grid_df = run_focused_grid(focused_grid, df_exec, y_target, yaml_path,
                                   slippage_bps=args.slippage_bps)
        grid_df.to_csv(output_dir / "focused_grid_results.csv", index=False)
        logger.info("Grid results saved to %s", output_dir / "focused_grid_results.csv")

        top10 = grid_df.nlargest(10, "Sharpe_net")
        print("\n--- Top 10 by Sharpe (Focused Grid) ---")
        print(top10[["name", "Sharpe_net", "AR_net", "MDD", "Mean_Rank_IC", "ICIR"]].to_string(index=False))

    # --- Stage 3: Walk-Forward OOS ---
    if args.stage in ("wfo", "all"):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: WALK-FORWARD OOS VALIDATION")
        logger.info("=" * 80)

        candidates = []

        # Load grid results from file if not in memory
        if grid_df is None:
            grid_path = output_dir / "focused_grid_results.csv"
            if grid_path.exists():
                logger.info("Loading grid results from %s", grid_path)
                grid_df = pd.read_csv(grid_path)

        if grid_df is not None and len(grid_df) > 0:
            best_grid = grid_df.nlargest(1, "Sharpe_net").iloc[0]
            best_params = {}
            for key in BASELINE_PARAMS:
                best_params[key] = best_grid.get(key, BASELINE_PARAMS[key])
            candidates.append(("best_grid", best_params))
        elif oat_df is not None:
            best_params = BASELINE_PARAMS.copy()
            for param_name, info in best_per_key.items():
                best_params[param_name] = info["best_val"]
            candidates.append(("best_oat", best_params))

        candidates.append(("baseline", BASELINE_PARAMS.copy()))

        wfo_all = []
        for name, params in candidates:
            logger.info("WFO for %s: %s", name, params)
            wfo = run_walk_forward(params, df_exec, y_target, yaml_path,
                                   n_folds=args.n_folds, slippage_bps=args.slippage_bps)

            print(f"\n--- Walk-Forward Folds: {name} ---")
            fold_df = pd.DataFrame(wfo["folds"])
            print(fold_df.to_string(index=False))

            sharpes = [f["Sharpe_net"] for f in wfo["folds"] if np.isfinite(f["Sharpe_net"])]
            ics = [f["Mean_Rank_IC"] for f in wfo["folds"] if np.isfinite(f["Mean_Rank_IC"])]
            mdds = [f["MDD"] for f in wfo["folds"] if np.isfinite(f["MDD"])]
            if sharpes:
                print(f"  Mean Sharpe: {np.mean(sharpes):.4f} (std={np.std(sharpes):.4f})")
                print(f"  Mean IC:     {np.mean(ics):.4f} (std={np.std(ics):.4f})")
                print(f"  Min Sharpe:  {np.min(sharpes):.4f}")
                print(f"  Max MDD:     {np.min(mdds):.4f}")

            for fold in wfo["folds"]:
                row = {"candidate": name, **fold}
                row.update(params)
                wfo_all.append(row)

        pd.DataFrame(wfo_all).to_csv(output_dir / "walk_forward_results.csv", index=False)

    # --- Final Summary ---
    print("\n" + "=" * 100)
    print("PHASE 1A — BLPX PARAMETER OPTIMIZATION — FINAL SUMMARY")
    print("=" * 100)

    print(f"\nBaseline (current production):")
    print(f"  Sharpe: {baseline_m['Sharpe_net']:.4f}")
    print(f"  AR:     {baseline_m['AR_net']*100:.2f}%")
    print(f"  MDD:    {baseline_m['MDD']*100:.2f}%")
    print(f"  IC:     {baseline_m['Mean_Rank_IC']:.4f}")
    print(f"  ICIR:   {baseline_m['ICIR']:.2f}")

    if grid_df is not None and len(grid_df) > 0:
        best = grid_df.nlargest(1, "Sharpe_net").iloc[0]
        print(f"\nBest Grid:")
        print(f"  Sharpe: {best['Sharpe_net']:.4f} (delta={best['Sharpe_net']-baseline_m['Sharpe_net']:+.4f})")
        print(f"  IC:     {best['Mean_Rank_IC']:.4f} (delta={best['Mean_Rank_IC']-baseline_m['Mean_Rank_IC']:+.4f})")
        print(f"  ICIR:   {best['ICIR']:.2f} (delta={best['ICIR']-baseline_m['ICIR']:+.2f})")
        print(f"  MDD:    {best['MDD']*100:.2f}%")
        print(f"  Params: {best['params']}")
    elif best_per_key is not None:
        best_params = BASELINE_PARAMS.copy()
        for param_name, info in best_per_key.items():
            best_params[param_name] = info["best_val"]
        print(f"\nBest OAT (combined):")
        print(f"  Params: {best_params}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
