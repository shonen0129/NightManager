"""Phase 1D: Focused 2D grid on sector_eta x sector_gamma and sector_eta x lambda_sector.

Tests gamma sensitivity with eta > 0 (gamma has no effect when eta=0).
"""

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


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        for k, v in blpx_overrides.items():
            cfg.setdefault("blpx", {})[k] = v
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


def run_variant(name, model, df_exec, y_target, slippage_bps=5.0):
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
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
    return {
        "name": name, "Sharpe_net": sharpe, "Mean_Rank_IC": mean_ic,
        "ICIR": icir, "MDD": mdd, "elapsed_s": elapsed,
    }


def main():
    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / "artifacts" / "phase1d_sector_prior"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)

    results = []

    # Grid 1: eta x gamma (lambda_sector=0.60)
    eta_vals = [0.4, 0.5, 0.6, 0.7, 0.8]
    gamma_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    print("=== Grid 1: eta x gamma (lambda_sector=0.60) ===")
    for eta in eta_vals:
        for gamma in gamma_vals:
            overrides = PHASE1A_BEST.copy()
            overrides["sector_eta"] = eta
            overrides["sector_gamma"] = gamma
            cfg = build_config(yaml_path, blpx_overrides=overrides,
                               signal_components=PHASE1B_WEIGHTS)
            _BLP_CORR_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            name = f"eta{eta}_g{gamma}_ls0.60"
            m = run_variant(name, model, df_exec, y_target)
            m["eta"] = eta
            m["gamma"] = gamma
            m["lambda_sector"] = 0.60
            results.append(m)
            print(f"  {name}: Sharpe={m['Sharpe_net']:.4f} IC={m['Mean_Rank_IC']:.4f} ICIR={m['ICIR']:.2f} ({m['elapsed_s']:.1f}s)")

    # Grid 2: eta x lambda_sector (gamma=2.0)
    lambda_vals = [0.5, 0.6, 0.7, 0.8]

    print("\n=== Grid 2: eta x lambda_sector (gamma=2.0) ===")
    for eta in [0.5, 0.6, 0.7]:
        for ls in lambda_vals:
            overrides = PHASE1A_BEST.copy()
            overrides["sector_eta"] = eta
            overrides["sector_gamma"] = 2.0
            overrides["lambda_sector"] = ls
            cfg = build_config(yaml_path, blpx_overrides=overrides,
                               signal_components=PHASE1B_WEIGHTS)
            _BLP_CORR_CACHE.clear()
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
            name = f"eta{eta}_g2.0_ls{ls}"
            m = run_variant(name, model, df_exec, y_target)
            m["eta"] = eta
            m["gamma"] = 2.0
            m["lambda_sector"] = ls
            results.append(m)
            print(f"  {name}: Sharpe={m['Sharpe_net']:.4f} IC={m['Mean_Rank_IC']:.4f} ICIR={m['ICIR']:.2f} ({m['elapsed_s']:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "focused_grid.csv", index=False)

    print("\n=== TOP 5 by Sharpe ===")
    top5 = df.nlargest(5, "Sharpe_net")
    for _, r in top5.iterrows():
        print(f"  {r['name']}: Sharpe={r['Sharpe_net']:.4f} IC={r['Mean_Rank_IC']:.4f} ICIR={r['ICIR']:.2f} MDD={r['MDD']*100:.2f}%")

    print(f"\nResults saved to {output_dir / 'focused_grid.csv'}")


if __name__ == "__main__":
    main()
