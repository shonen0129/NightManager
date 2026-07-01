"""Phase 1D: Walk-Forward OOS validation for top sector prior candidates."""

from __future__ import annotations

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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PHASE1A_BEST = {
    "alpha_xx": 0.20, "alpha_yy": 0.50, "alpha_yx": 0.15,
    "lambda_pca": 0.10, "lambda_sector": 0.60, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 504,
    "blp_ewma_halflife": 120,
}

PHASE1B_WEIGHTS = {
    "raw_pca": {"enabled": True, "weight": 0.2},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": True, "weight": 0.8},
    "residual_blpx": {"enabled": False, "weight": 0.0},
}

# Top candidates from OAT + focused grid
CANDIDATES = [
    {"name": "baseline", "sector_eta": 0.0, "sector_gamma": 2.0, "lambda_sector": 0.60},
    {"name": "oat_best_eta0.6", "sector_eta": 0.6, "sector_gamma": 2.0, "lambda_sector": 0.60},
    {"name": "grid_best_eta0.7_g2_ls0.8", "sector_eta": 0.7, "sector_gamma": 2.0, "lambda_sector": 0.80},
    {"name": "grid_eta0.6_g2_ls0.7", "sector_eta": 0.6, "sector_gamma": 2.0, "lambda_sector": 0.70},
    {"name": "grid_eta0.5_g4_ls0.60", "sector_eta": 0.5, "sector_gamma": 4.0, "lambda_sector": 0.60},
    {"name": "grid_eta0.8_g4_ls0.60", "sector_eta": 0.8, "sector_gamma": 4.0, "lambda_sector": 0.60},
]


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        for k, v in blpx_overrides.items():
            cfg.setdefault("blpx", {})[k] = v
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


def main():
    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / "artifacts" / "phase1d_sector_prior"
    output_dir.mkdir(parents=True, exist_ok=True)
    n_folds = 5

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    eval_dates = sim_dates[start_idx:]
    n_eval = len(eval_dates)
    fold_size = n_eval // n_folds

    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)

    wfo_all = []
    full_metrics = []

    for cand in CANDIDATES:
        name = cand["name"]
        logger.info("WFO for %s (eta=%s, gamma=%s, ls=%s)",
                    name, cand["sector_eta"], cand["sector_gamma"], cand["lambda_sector"])

        overrides = PHASE1A_BEST.copy()
        overrides["sector_eta"] = cand["sector_eta"]
        overrides["sector_gamma"] = cand["sector_gamma"]
        overrides["lambda_sector"] = cand["lambda_sector"]
        cfg = build_config(yaml_path, blpx_overrides=overrides,
                           signal_components=PHASE1B_WEIGHTS)
        _BLP_CORR_CACHE.clear()
        model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

        t0 = time.perf_counter()
        results = BacktestEngine.run_backtest(
            model, df_exec=df_exec, start_date="2015-01-01",
            overnight_alpha_long=0.75, overnight_alpha_short=0.5,
            buy_interest_annual=0.025, borrow_fee_annual=0.0115,
            reverse_fee_bps=2.0, slippage_bps=5.0,
        )
        elapsed = time.perf_counter() - t0

        dr = results["daily_returns"]
        ar = float(dr.mean() * 245)
        vol = float(dr.std(ddof=1) * np.sqrt(245))
        sharpe = ar / vol if vol > 0 else np.nan
        wealth = (1.0 + dr).cumprod()
        mdd = float(((wealth / wealth.cummax()) - 1.0).min())

        signals_df = results["signals"]
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
        mean_ic = float(np.mean(ic_list)) if ic_list else np.nan
        std_ic = float(np.std(ic_list, ddof=1)) if len(ic_list) > 1 else np.nan
        icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic and std_ic > 1e-8 else np.nan

        full_metrics.append({
            "name": name, "eta": cand["sector_eta"], "gamma": cand["sector_gamma"],
            "lambda_sector": cand["lambda_sector"],
            "Sharpe_net": sharpe, "AR_net": ar, "MDD": mdd,
            "Mean_Rank_IC": mean_ic, "ICIR": icir, "elapsed_s": elapsed,
        })
        logger.info("  Full: Sharpe=%.4f IC=%.4f ICIR=%.2f MDD=%.2f%% (%.1fs)",
                    sharpe, mean_ic, icir, mdd * 100, elapsed)

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

    full_df = pd.DataFrame(full_metrics)
    full_df.to_csv(output_dir / "wfo_full_metrics.csv", index=False)

    # Print summary
    print("\n" + "=" * 100)
    print("PHASE 1D — WALK-FORWARD OOS VALIDATION")
    print("=" * 100)

    print("\n--- Full Backtest Metrics ---")
    print(f"{'Candidate':<35} {'eta':<5} {'gamma':<6} {'ls':<5} {'Sharpe':<10} {'IC':<10} {'ICIR':<8} {'MDD':<8}")
    print("-" * 87)
    for r in full_metrics:
        print(f"{r['name']:<35} {r['eta']:<5} {r['gamma']:<6} {r['lambda_sector']:<5} {r['Sharpe_net']:<10.4f} {r['Mean_Rank_IC']:<10.4f} {r['ICIR']:<8.2f} {r['MDD']*100:<8.2f}")

    print("\n--- Walk-Forward Fold Summary ---")
    for cand in CANDIDATES:
        name = cand["name"]
        folds = wfo_df[wfo_df["candidate"] == name]
        if len(folds) == 0:
            continue
        sharpes = folds["Sharpe_net"].dropna()
        ics = folds["Mean_Rank_IC"].dropna()
        icirs = folds["ICIR"].dropna()
        print(f"\n  {name}:")
        print(f"    Sharpe: mean={sharpes.mean():.4f} std={sharpes.std():.4f} min={sharpes.min():.4f} max={sharpes.max():.4f}")
        print(f"    IC:     mean={ics.mean():.4f} std={ics.std():.4f}")
        print(f"    ICIR:   mean={icirs.mean():.2f}")
        print(f"    MDD:    max={folds['MDD'].min():.4f}")
        for _, f in folds.iterrows():
            print(f"      Fold {int(f['fold'])}: {f['start_date']} ~ {f['end_date']} Sharpe={f['Sharpe_net']:.4f} IC={f['Mean_Rank_IC']:.4f} MDD={f['MDD']*100:.2f}%")

    # Consistency check: how many folds beat baseline?
    base_folds = wfo_df[wfo_df["candidate"] == "baseline"]["Sharpe_net"].values
    print("\n--- Consistency vs Baseline ---")
    for cand in CANDIDATES[1:]:
        name = cand["name"]
        cand_folds = wfo_df[wfo_df["candidate"] == name]["Sharpe_net"].values
        n_better = sum(1 for a, b in zip(cand_folds, base_folds) if a > b)
        print(f"  {name}: {n_better}/{len(cand_folds)} folds better than baseline")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
