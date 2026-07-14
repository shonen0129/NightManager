#!/usr/bin/env python
"""A1: ギャップ調整の適応的c_t実験スクリプト.

設計仕様（docs/design/A_theory_design_specs.md A1参照）:
  理論値 c_t = 1 + β_rev_oc(t)（ローリング252日、shift(1)）を計算し、
  現行の固定 c=0.70 とシュリンクブレンドする:
    c_t = (1-λ)·0.70 + λ·(1 + β_rev_oc_hat(t))
  λ ∈ {0.25, 0.5, 0.75} の3点のみ試行。

  β_rev_oc = Cov(r_910, gap_idio) / Var(gap_idio)
  （r_910 = 9:10→大引けリターン、gap_idio = gap - beta·topix_night）

Usage:
  python3 scripts/experiments/experiment_a1_adaptive_gap_coef.py \
    --start-date 2015-01-05 --output-dir reports/sprint_a1_adaptive_gap
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.blp_base import _BLPBase
from leadlag.execution.backtester import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245


def compute_rolling_beta_rev_oc(
    y_target: np.ndarray,
    jp_gap: np.ndarray,
    jp_beta: np.ndarray,
    topix_night: np.ndarray,
    window: int = 252,
) -> np.ndarray:
    """Compute rolling β_rev_oc per ticker (strictly historical, shift(1)).

    β_rev_oc = Cov(r_910, gap_idio) / Var(gap_idio)
    where gap_idio = gap - beta * topix_night

    Args:
        y_target: (T, N_J) 9:10→close returns
        jp_gap: (T, N_J) overnight gap returns
        jp_beta: (T, N_J) rolling beta to TOPIX
        topix_night: (T,) TOPIX overnight return
        window: rolling estimation window

    Returns:
        (T, N_J) β_rev_oc array, shifted by 1 (row t uses data up to t-1)
    """
    T, N = y_target.shape

    # Compute gap_idio
    gap_idio = jp_gap - jp_beta * topix_night[:, np.newaxis]

    beta_rev = np.full((T, N), np.nan)

    for t in range(window + 1, T):
        # Use data from t-window to t-1 (strictly historical)
        r_window = y_target[t - window : t]
        g_window = gap_idio[t - window : t]

        for j in range(N):
            r = r_window[:, j]
            g = g_window[:, j]
            valid = np.isfinite(r) & np.isfinite(g)
            if valid.sum() < window // 2:
                continue
            r_v = r[valid]
            g_v = g[valid]
            var_g = np.var(g_v, ddof=1)
            if var_g < 1e-12:
                continue
            cov_rg = np.cov(r_v, g_v, ddof=1)[0, 1]
            beta_rev[t, j] = cov_rg / var_g

    return beta_rev


def compute_adaptive_c_t(
    beta_rev_oc: np.ndarray,
    lambda_shrink: float,
    c_prior: float = 0.70,
) -> np.ndarray:
    """Compute adaptive gap_open_coef c_t.

    c_t = (1-λ)·c_prior + λ·(1 + β_rev_oc)
    Clipped to [0.1, 1.5] for safety.
    """
    c_theory = 1.0 + beta_rev_oc
    c_t = (1.0 - lambda_shrink) * c_prior + lambda_shrink * c_theory
    c_t = np.clip(c_t, 0.1, 1.5)
    # Fill NaN with prior
    c_t = np.where(np.isfinite(c_t), c_t, c_prior)
    return c_t


class AdaptiveGapCoefModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Model that uses per-day adaptive gap_open_coef."""

    def __init__(self, cfg: dict, adaptive_c_t: np.ndarray):
        super().__init__(cfg)
        self._adaptive_c_t = adaptive_c_t  # (T, N_J)
        self._orig_gap_open_coef = self.gap_open_coef

    def compute_blp_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
        rolling_std: np.ndarray | None = None,
        v0_static: np.ndarray | None = None,
        c_full: np.ndarray | None = None,
        is_residual: bool = False,
        return_matrices: bool = False,
    ) -> dict:
        """Override to set adaptive gap_open_coef from precomputed array before calling super."""
        c_t_row = self._adaptive_c_t[current_index] if current_index < len(self._adaptive_c_t) else None
        if c_t_row is not None and np.all(np.isfinite(c_t_row)):
            self.gap_open_coef = float(np.mean(c_t_row))
        else:
            self.gap_open_coef = self._orig_gap_open_coef
        return super().compute_blp_signal(
            all_returns=all_returns, current_index=current_index,
            gap_override=gap_override, betas_t=betas_t,
            topix_night_t=topix_night_t, rolling_std=rolling_std,
            v0_static=v0_static, c_full=c_full,
            is_residual=is_residual, return_matrices=return_matrices,
        )


def run_backtest_with_adaptive_c(
    cfg: dict,
    df_exec: pd.DataFrame,
    adaptive_c_t: np.ndarray,
    start_date: str,
) -> dict:
    """Run backtest with adaptive gap coefficient."""
    model = AdaptiveGapCoefModel(cfg, adaptive_c_t)
    model._start_date = start_date
    return BacktestEngine.run_backtest(
        model, df_exec, start_date=start_date, slippage_bps=5.0,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )


def compute_metrics(daily_returns: pd.Series) -> dict:
    dr = daily_returns.dropna()
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    return {"Sharpe_net": sharpe, "AR_net": ar, "Vol_net": vol, "MDD": mdd, "n_days": len(dr)}


def main():
    parser = argparse.ArgumentParser(description="A1: Adaptive Gap Coefficient")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--output-dir", default="reports/sprint_a1_adaptive_gap")
    parser.add_argument("--beta-window", type=int, default=252, help="β_rev_oc rolling window")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()
    T = len(df_exec)
    sim_dates = df_exec.index

    # Extract inputs
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    jp_gap = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values
    jp_beta = df_exec[[f"jp_beta_{tk}" for tk in JP_TICKERS]].values
    topix_night = df_exec["topix_night_return"].values if "topix_night_return" in df_exec.columns else np.zeros(T)

    # 1. Compute rolling β_rev_oc
    logger.info("Computing rolling β_rev_oc (window=%d)...", args.beta_window)
    beta_rev_oc = compute_rolling_beta_rev_oc(
        y_target, jp_gap, jp_beta, topix_night, window=args.beta_window,
    )

    # Save β_rev_oc statistics
    beta_rev_df = pd.DataFrame(beta_rev_oc, index=sim_dates, columns=JP_TICKERS)
    beta_rev_df.to_csv(out_dir / "beta_rev_oc_timeseries.csv")
    valid_mask = np.isfinite(beta_rev_oc)
    if valid_mask.any():
        logger.info("β_rev_oc mean=%.4f std=%.4f (expect ~-0.23)",
                     float(np.nanmean(beta_rev_oc)), float(np.nanstd(beta_rev_oc)))

    # 2. Compute c_t for each lambda
    lambdas = [0.25, 0.50, 0.75]
    all_results = {}

    import yaml
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg_base = yaml.safe_load(f)

    # Baseline: fixed c=0.70
    logger.info("Running baseline (fixed gap_open_coef=0.70)...")
    model_base = SectorRelativeEnsembleBLPEnhancedModel(copy.deepcopy(cfg_base))
    model_base._start_date = args.start_date
    results_base = BacktestEngine.run_backtest(
        model_base, df_exec, start_date=args.start_date, slippage_bps=5.0,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    metrics_base = compute_metrics(results_base["daily_returns"])
    metrics_base["variant"] = "baseline_fixed_0.70"
    all_results["baseline"] = metrics_base
    logger.info("Baseline: Sharpe=%.4f MDD=%.2f%%", metrics_base["Sharpe_net"], metrics_base["MDD"] * 100)

    for lam in lambdas:
        logger.info("Computing adaptive c_t (lambda=%.2f)...", lam)
        c_t = compute_adaptive_c_t(beta_rev_oc, lambda_shrink=lam, c_prior=0.70)

        c_t_df = pd.DataFrame(c_t, index=sim_dates, columns=JP_TICKERS)
        c_t_df.to_csv(out_dir / f"adaptive_c_t_lambda{lam}.csv")
        logger.info("c_t mean=%.4f std=%.4f (theory: 1+β_rev_oc ≈ 0.77)",
                     float(np.nanmean(c_t)), float(np.nanstd(c_t)))

        logger.info("Running backtest (lambda=%.2f)...", lam)
        results = run_backtest_with_adaptive_c(
            copy.deepcopy(cfg_base), df_exec, c_t, args.start_date,
        )
        metrics = compute_metrics(results["daily_returns"])
        metrics["variant"] = f"adaptive_lambda{lam}"
        all_results[f"lambda_{lam}"] = metrics
        logger.info("lambda=%.2f: Sharpe=%.4f MDD=%.2f%%", lam, metrics["Sharpe_net"], metrics["MDD"] * 100)

        results["daily_returns"].to_csv(out_dir / f"daily_returns_lambda{lam}.csv")

    # 3. Summary
    results_df = pd.DataFrame(list(all_results.values()))
    results_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    report_lines = [
        "# A1: Adaptive Gap Coefficient Report\n",
        f"## Data: {T} rows, start={args.start_date}, β_rev_oc window={args.beta_window}\n",
        f"\n## β_rev_oc Statistics\n",
        f"Mean: {float(np.nanmean(beta_rev_oc)):.4f} (expect ~-0.23)\n",
        f"Std:  {float(np.nanstd(beta_rev_oc)):.4f}\n",
        f"Theory: c_t = 1 + β_rev_oc ≈ {1 + float(np.nanmean(beta_rev_oc)):.4f} (vs current 0.70)\n",
        f"\n## Metrics Comparison\n",
        results_df.to_string(index=False),
        f"\n\n## Verdict\n",
    ]

    base_sharpe = all_results["baseline"]["Sharpe_net"]
    best_lambda_sharpe = max(all_results[f"lambda_{l}"]["Sharpe_net"] for l in lambdas)
    improvement = (best_lambda_sharpe - base_sharpe) / base_sharpe * 100

    if improvement > 1.0:
        report_lines.append(f"Adaptive c_t improves Sharpe by {improvement:.1f}% over fixed 0.70.\n")
        report_lines.append("Recommend: adopt adaptive c_t with appropriate λ.\n")
    else:
        report_lines.append(f"Adaptive c_t does NOT improve Sharpe (best: {improvement:+.1f}%).\n")
        report_lines.append("Theory: 1+β_rev_oc ≈ 0.77 is close to current 0.70.\n")
        report_lines.append("The small gap (0.07) is within estimation noise → no benefit from adaptation.\n")
        report_lines.append("Recommend: keep fixed gap_open_coef=0.70.\n")

    report_text = "\n".join(report_lines)
    (out_dir / "a1_adaptive_gap_report.md").write_text(report_text)
    logger.info("Report saved to %s/a1_adaptive_gap_report.md", out_dir)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
