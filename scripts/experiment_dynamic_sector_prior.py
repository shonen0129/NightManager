"""Experiment: Dynamic Sector Prior alternatives vs static M_sector.

Tests whether replacing the fixed 17×15 sector mapping matrix with
more adaptive approaches improves backtest performance.

Variants:
  1. baseline       — Current static M_sector (lambda_sector=0.40)
  2. rolling_corr   — Rolling cross-correlation as M_sector
  3. blend_50       — 50% static + 50% rolling correlation
  4. blend_75       — 75% static + 25% rolling correlation
  5. no_sector      — lambda_sector=0.0 (pure BLP + PCA prior)
  6. pure_blp       — lambda_sector=0.0, lambda_pca=0.0 (pure BLP regression)
  7. ridge_dynamic  — Rolling ridge regression coefficients as M_sector
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic Sector Prior Model Variants
# ---------------------------------------------------------------------------


class RollingCorrSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Use rolling cross-correlation (C_YX block) as sector prior."""

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        C_YX = corr[self.n_u:, :self.n_u]
        if C_YX.shape == B_blp.shape:
            return C_YX.copy()
        return np.zeros(B_blp.shape)


class BlendSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Blend static M_sector with rolling cross-correlation."""

    def __init__(self, config, blend_alpha=0.5):
        super().__init__(config)
        self.blend_alpha = blend_alpha

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        C_YX = corr[self.n_u:, :self.n_u]
        if C_YX.shape != B_blp.shape:
            return np.zeros(B_blp.shape)
        static = self.M_sector
        if static.shape != C_YX.shape:
            return C_YX.copy()
        return (1.0 - self.blend_alpha) * static + self.blend_alpha * C_YX


class RidgeDynamicSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Use rolling ridge regression coefficients as sector prior.

    At each time step, regress JP target returns on US returns over the
    BLP window with a small ridge penalty, producing a data-driven mapping.
    """

    def __init__(self, config, ridge_rho=0.05):
        super().__init__(config)
        self.ridge_rho = ridge_rho

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        window_start = max(0, current_index - self.blp_window)
        W = all_returns[window_start:current_index]
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        X = W[:, :self.n_u]
        Y = W[:, self.n_u:]
        # Ridge: B = Y^T X (X^T X + rho*I)^{-1}
        XtX = X.T @ X
        ridge = self.ridge_rho * np.mean(np.diag(XtX)) * np.eye(self.n_u)
        try:
            A_inv = np.linalg.inv(XtX + ridge)
            B_ridge = Y.T @ X @ A_inv
        except Exception:
            B_ridge = np.zeros((self.n_j, self.n_u))
        if B_ridge.shape == B_blp.shape:
            return B_ridge
        return np.zeros(B_blp.shape)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(yaml_path: str, overrides: dict | None = None) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for key, val in overrides.items():
            if key in ("lambda_pca", "lambda_sector"):
                cfg.setdefault("blpx", {})[key] = val
            elif key in ("alpha_xx", "alpha_yx", "alpha_yy", "rho"):
                cfg.setdefault("blpx", {})[key] = val
            else:
                cfg[key] = val
    return cfg


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_variant(name, model, df_exec, start_date="2015-01-01"):
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
    )
    elapsed = time.perf_counter() - t0

    dr = results["daily_returns"]
    dr_gross = results["daily_returns_gross"]
    T = len(dr)

    # Annualized return (simple)
    ar = float(dr.mean() * 245)
    ar_gross = float(dr_gross.mean() * 245)
    # Annualized volatility
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    vol_gross = float(dr_gross.std(ddof=1) * np.sqrt(245))
    # Sharpe
    sharpe = ar / vol if vol > 0 else np.nan
    sharpe_gross = ar_gross / vol_gross if vol_gross > 0 else np.nan
    # Max drawdown
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    # Turnover
    turnover = float(results["daily_turnover"].mean())
    # Gross exposure
    gross_exp = float(results["daily_gross_exps"].mean())

    # Monthly Sharpe (matching production metrics)
    monthly = (1.0 + dr).groupby(dr.index.year * 12 + dr.index.month).prod() - 1.0
    if len(monthly) > 1:
        monthly_sharpe = float(
            (monthly.mean() / monthly.std(ddof=1)) * np.sqrt(12.0)
        )
    else:
        monthly_sharpe = np.nan

    metrics = {
        "name": name,
        "n_days": T,
        "AR_net": ar,
        "AR_gross": ar_gross,
        "Vol_net": vol,
        "Sharpe_net": sharpe,
        "Sharpe_gross": sharpe_gross,
        "Sharpe_monthly": monthly_sharpe,
        "MDD": mdd,
        "Turnover": turnover,
        "GrossExp": gross_exp,
        "elapsed_s": elapsed,
    }
    return metrics, results


def main():
    yaml_path = str(ROOT / "configs" / "production.yaml")

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    logger.info("df_exec shape: %s", df_exec.shape)

    variants = []

    # 1. Baseline (static M_sector)
    cfg_base = build_config(yaml_path)
    variants.append(("baseline_static", cfg_base, None))

    # 2. Rolling correlation as M_sector
    cfg_rc = build_config(yaml_path)
    variants.append(("rolling_corr", cfg_rc, ("rolling", None)))

    # 3. Blend 50/50
    cfg_b50 = build_config(yaml_path)
    variants.append(("blend_50", cfg_b50, ("blend", 0.50)))

    # 4. Blend 75/25 (more static)
    cfg_b75 = build_config(yaml_path)
    variants.append(("blend_75", cfg_b75, ("blend", 0.25)))

    # 5. No sector prior (lambda_sector=0, keep lambda_pca=0.10)
    cfg_ns = build_config(yaml_path, overrides={"lambda_sector": 0.0})
    variants.append(("no_sector", cfg_ns, None))

    # 6. Pure BLP (lambda_sector=0, lambda_pca=0)
    cfg_pb = build_config(yaml_path, overrides={"lambda_sector": 0.0, "lambda_pca": 0.0})
    variants.append(("pure_blp", cfg_pb, None))

    # 7. Ridge dynamic sector prior
    cfg_rd = build_config(yaml_path)
    variants.append(("ridge_dynamic", cfg_rd, ("ridge", 0.05)))

    all_metrics = []
    all_results = {}

    for name, cfg, sector_mode in variants:
        logger.info("=== Running variant: %s ===", name)
        if sector_mode is None:
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
        elif sector_mode[0] == "rolling":
            model = RollingCorrSectorModel(cfg)
        elif sector_mode[0] == "blend":
            model = BlendSectorModel(cfg, blend_alpha=sector_mode[1])
        elif sector_mode[0] == "ridge":
            model = RidgeDynamicSectorModel(cfg, ridge_rho=sector_mode[1])
        else:
            model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

        metrics, results = run_variant(name, model, df_exec)
        all_metrics.append(metrics)
        all_results[name] = results
        logger.info(
            "%s: Sharpe(net)=%.4f  Sharpe(gross)=%.4f  AR=%.2f%%  MDD=%.2f%%  Turnover=%.4f  [%.1fs]",
            name,
            metrics["Sharpe_net"],
            metrics["Sharpe_gross"],
            metrics["AR_net"] * 100,
            metrics["MDD"] * 100,
            metrics["Turnover"],
            metrics["elapsed_s"],
        )

    # Summary table
    print("\n" + "=" * 120)
    print("DYNAMIC SECTOR PRIOR EXPERIMENT — RESULTS SUMMARY")
    print("=" * 120)
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.set_index("name")
    for col in ["AR_net", "AR_gross", "Vol_net", "Sharpe_net", "Sharpe_gross",
                 "Sharpe_monthly", "MDD", "Turnover", "GrossExp", "elapsed_s"]:
        if col in df_metrics.columns:
            df_metrics[col] = df_metrics[col].astype(float)

    # Format for display
    display = df_metrics.copy()
    display["AR_net"] = (display["AR_net"] * 100).round(2)
    display["AR_gross"] = (display["AR_gross"] * 100).round(2)
    display["Vol_net"] = (display["Vol_net"] * 100).round(2)
    display["Sharpe_net"] = display["Sharpe_net"].round(4)
    display["Sharpe_gross"] = display["Sharpe_gross"].round(4)
    display["Sharpe_monthly"] = display["Sharpe_monthly"].round(4)
    display["MDD"] = (display["MDD"] * 100).round(2)
    display["Turnover"] = display["Turnover"].round(4)
    display["GrossExp"] = display["GrossExp"].round(4)
    display["elapsed_s"] = display["elapsed_s"].round(1)

    print(display.to_string())
    print()

    # Save results
    output_dir = ROOT / "artifacts" / "sector_prior_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_dir / "sector_prior_comparison.csv")

    # Save daily returns for each variant
    returns_df = pd.DataFrame(
        {name: res["daily_returns"] for name, res in all_results.items()}
    )
    returns_df.to_csv(output_dir / "daily_returns_by_variant.csv")

    print(f"Results saved to {output_dir}")

    # Highlight best variant
    best_sharpe = df_metrics["Sharpe_net"].idxmax()
    best_ar = df_metrics["AR_net"].idxmax()
    print(f"\nBest Sharpe (net): {best_sharpe} ({df_metrics.loc[best_sharpe, 'Sharpe_net']:.4f})")
    print(f"Best AR (net):     {best_ar} ({df_metrics.loc[best_ar, 'AR_net']*100:.2f}%)")
    print(f"Baseline Sharpe:   {df_metrics.loc['baseline_static', 'Sharpe_net']:.4f}")
    delta_sharpe = df_metrics.loc[best_sharpe, 'Sharpe_net'] - df_metrics.loc['baseline_static', 'Sharpe_net']
    print(f"Sharpe improvement: {delta_sharpe:+.4f}")


if __name__ == "__main__":
    main()
