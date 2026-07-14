#!/usr/bin/env python
"""A4: アンサンブル重み IC最適化実験スクリプト.

設計仕様（docs/design/A_theory_design_specs.md A4参照）:
  1. 各成分シグナル（Raw-PCA, Residual-PCA, Raw-BLPX, Residual-BLPX）の日次ICを計算
  2. ローリング504日（shift(1)）で μ_IC, Σ_e を推定
  3. 最適重み w* ∝ Σ_e^{-1} μ_IC を計算（Ledoit-Wolfシュリンク）
  4. シュリンク δ ∈ {0.25, 0.5} で静的重みとブレンド
  5. baseline（静的重み）とOOS比較

Usage:
  python3 scripts/experiments/experiment_a4_ensemble_ic_optimization.py \
    --start-date 2015-01-05 --output-dir reports/sprint_a4_ensemble_ic
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 245


def compute_daily_rank_ic(signals: np.ndarray, targets: np.ndarray, start_idx: int) -> np.ndarray:
    """Compute daily Spearman rank IC between signals and targets.

    Args:
        signals: (T, N) signal array
        targets: (T, N) target return array
        start_idx: first index to compute IC

    Returns:
        (T,) array of daily IC values (NaN before start_idx)
    """
    T = signals.shape[0]
    ic = np.full(T, np.nan)
    for t in range(start_idx, T):
        s = signals[t]
        y = targets[t]
        valid = np.isfinite(s) & np.isfinite(y)
        if valid.sum() >= 5 and np.std(s[valid]) > 1e-8 and np.std(y[valid]) > 1e-8:
            ic[t] = float(spearmanr(s[valid], y[valid])[0])
    return ic


def compute_optimal_weights_rolling(
    ic_matrix: np.ndarray,
    static_weights: np.ndarray,
    window: int = 504,
    delta: float = 0.5,
) -> np.ndarray:
    """Compute rolling IC-optimal ensemble weights.

    w_t = (1-delta) * w_static + delta * normalize(Sigma_e^{-1} mu_IC)

    Uses simple inverse (4x4 matrix, no conditioning issues expected).
    Sigma_e is the covariance of IC across components.

    Args:
        ic_matrix: (T, n_components) daily IC per component
        static_weights: (n_components,) static ensemble weights
        window: rolling window for IC statistics
        delta: shrinkage towards static weights

    Returns:
        (T, n_components) daily weight matrix
    """
    T, n_comp = ic_matrix.shape
    weights = np.zeros((T, n_comp))

    for t in range(T):
        if t < window:
            weights[t] = static_weights
            continue

        ic_window = ic_matrix[t - window : t]
        valid_mask = np.all(np.isfinite(ic_window), axis=1)
        ic_valid = ic_window[valid_mask]

        if len(ic_valid) < window // 2:
            weights[t] = static_weights
            continue

        mu_ic = np.nanmean(ic_valid, axis=0)
        Sigma_e = np.cov(ic_valid, rowvar=False)

        if np.any(np.diag(Sigma_e) < 1e-10):
            weights[t] = static_weights
            continue

        try:
            Sigma_e_inv = np.linalg.inv(Sigma_e)
            w_opt = Sigma_e_inv @ mu_ic
            w_opt = w_opt / (np.sum(np.abs(w_opt)) + 1e-12)
        except np.linalg.LinAlgError:
            weights[t] = static_weights
            continue

        w_blended = (1.0 - delta) * static_weights + delta * w_opt
        w_blended = np.maximum(w_blended, 0.0)
        denom = np.sum(w_blended)
        if denom > 1e-12:
            w_blended = w_blended / denom
        else:
            w_blended = static_weights

        weights[t] = w_blended

    return weights


def run_backtest_with_custom_weights(
    cfg: dict,
    df_exec: pd.DataFrame,
    component_weights: np.ndarray,
    component_names: list[str],
    start_date: str,
) -> dict:
    """Run backtest with per-day custom ensemble weights.

    This creates a subclass that overrides predict_signals to use the precomputed weights.
    """
    T = len(df_exec)
    sim_dates = df_exec.index

    # Build a model that uses precomputed weights
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    model._start_date = start_date

    # We need to override predict_signals to use our weights
    # Instead of subclassing, we'll compute signals once and combine with our weights
    pred = model.predict_signals(df_exec)

    # Get individual component signals
    raw_pca = pred["raw_pca_signals"].values
    residual_pca = pred["residual_pca_signals"].values
    raw_blpx = pred["raw_blpx_signals"].values
    residual_blpx = pred["residual_blpx_signals"].values

    component_signals = {
        "raw_pca": raw_pca,
        "residual_pca": residual_pca,
        "raw_blpx": raw_blpx,
        "residual_blpx": residual_blpx,
    }

    # Normalize each component
    for name in component_names:
        for t in range(T):
            component_signals[name][t] = model.normalize_signals(
                component_signals[name][t], model.normalization_method
            )

    # Combine with custom weights
    combined = np.zeros((T, model.n_j))
    for t in range(T):
        s = np.zeros(model.n_j)
        for i, name in enumerate(component_names):
            s += component_weights[t, i] * component_signals[name][t]
        combined[t] = s

    # Replace signals in pred
    pred["signals"] = pd.DataFrame(combined, index=sim_dates, columns=JP_TICKERS)
    pred["normalized_signals"] = pd.DataFrame(
        np.array([model.normalize_signals(combined[t], model.normalization_method) for t in range(T)]),
        index=sim_dates, columns=JP_TICKERS,
    )

    # Now run backtest with the modified signals
    # We need to use BacktestEngine but with our precomputed signals
    # The engine calls model.predict_signals internally, so we need a wrapper
    class CustomWeightModel(SectorRelativeEnsembleBLPEnhancedModel):
        _custom_pred = None

        def predict_signals(self, df_exec):
            return self._custom_pred

    custom_model = CustomWeightModel(cfg)
    custom_model._start_date = start_date
    custom_model._custom_pred = pred

    results = BacktestEngine.run_backtest(
        custom_model, df_exec, start_date=start_date, slippage_bps=5.0,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    return results


def compute_metrics(daily_returns: pd.Series) -> dict:
    """Compute standard metrics from daily returns."""
    dr = daily_returns.dropna()
    ar = float(dr.mean() * TRADING_DAYS)
    vol = float(dr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    return {
        "Sharpe_net": sharpe,
        "AR_net": ar,
        "Vol_net": vol,
        "MDD": mdd,
        "n_days": len(dr),
    }


def main():
    parser = argparse.ArgumentParser(description="A4: Ensemble IC Optimization")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--output-dir", default="reports/sprint_a4_ensemble_ic")
    parser.add_argument("--ic-window", type=int, default=504, help="Rolling IC window")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading df_exec...")
    df_exec = load_df_exec_from_local_cache()
    T = len(df_exec)
    sim_dates = df_exec.index

    import yaml
    with open(ROOT / "configs" / "production" / "production.yaml") as f:
        cfg_base = yaml.safe_load(f)

    # Enable all 4 components for this experiment
    cfg_all = copy.deepcopy(cfg_base)
    cfg_all["signal_components"] = {
        "raw_pca": {"enabled": True, "weight": 0.25},
        "residual_pca": {"enabled": True, "weight": 0.25},
        "raw_blpx": {"enabled": True, "weight": 0.25},
        "residual_blpx": {"enabled": True, "weight": 0.25},
    }

    component_names = ["raw_pca", "residual_pca", "raw_blpx", "residual_blpx"]
    static_weights = np.array([0.25, 0.25, 0.25, 0.25])

    # 1. Run model with all components to get individual signals
    logger.info("Computing component signals (all 4 enabled, equal weight)...")
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg_all)
    model._start_date = args.start_date
    pred = model.predict_signals(df_exec)

    # 2. Compute targets
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime(args.start_date)), 60)

    # 3. Compute daily IC per component
    logger.info("Computing daily rank IC per component...")
    ic_matrix = np.full((T, 4), np.nan)
    for i, name in enumerate(component_names):
        sig_key = f"{name}_signals"
        if sig_key not in pred:
            logger.warning("Component %s not found in pred keys: %s", name, list(pred.keys()))
            continue
        sigs = pred[sig_key].values
        ic_matrix[:, i] = compute_daily_rank_ic(sigs, y_target, start_idx)

    ic_df = pd.DataFrame(ic_matrix, index=sim_dates, columns=component_names)
    ic_df.to_csv(out_dir / "daily_ic_per_component.csv")
    logger.info("Mean IC per component:\n%s", ic_df.mean().to_string())

    # 4. Compute IC-optimal weights for each delta
    deltas = [0.25, 0.50]
    all_results = {}

    # Baseline: static equal weights
    logger.info("Running baseline backtest (static equal weights)...")
    results_base = run_backtest_with_custom_weights(
        cfg_all, df_exec,
        np.tile(static_weights, (T, 1)),
        component_names, args.start_date,
    )
    metrics_base = compute_metrics(results_base["daily_returns"])
    metrics_base["variant"] = "baseline_static_equal"
    all_results["baseline"] = metrics_base
    logger.info("Baseline: Sharpe=%.4f MDD=%.2f%%", metrics_base["Sharpe_net"], metrics_base["MDD"] * 100)

    for delta in deltas:
        logger.info("Computing IC-optimal weights (delta=%.2f)...", delta)
        opt_weights = compute_optimal_weights_rolling(
            ic_matrix, static_weights, window=args.ic_window, delta=delta,
        )

        # Save weight time series
        w_df = pd.DataFrame(opt_weights, index=sim_dates, columns=component_names)
        w_df.to_csv(out_dir / f"optimal_weights_delta{delta}.csv")

        # Run backtest
        logger.info("Running backtest (delta=%.2f)...", delta)
        results = run_backtest_with_custom_weights(
            cfg_all, df_exec, opt_weights, component_names, args.start_date,
        )
        metrics = compute_metrics(results["daily_returns"])
        metrics["variant"] = f"ic_optimal_delta{delta}"
        all_results[f"delta_{delta}"] = metrics
        logger.info("delta=%.2f: Sharpe=%.4f MDD=%.2f%%", delta, metrics["Sharpe_net"], metrics["MDD"] * 100)

        # Save daily returns
        results["daily_returns"].to_csv(out_dir / f"daily_returns_delta{delta}.csv")

    # Also run with production config (residual_blpx only) as reference
    logger.info("Running production baseline (residual_blpx only)...")
    model_prod = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    model_prod._start_date = args.start_date
    results_prod = BacktestEngine.run_backtest(
        model_prod, df_exec, start_date=args.start_date, slippage_bps=5.0,
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0,
    )
    metrics_prod = compute_metrics(results_prod["daily_returns"])
    metrics_prod["variant"] = "production_residual_blpx_only"
    all_results["production"] = metrics_prod
    logger.info("Production: Sharpe=%.4f MDD=%.2f%%", metrics_prod["Sharpe_net"], metrics_prod["MDD"] * 100)

    # 5. Summary report
    results_df = pd.DataFrame(list(all_results.values()))
    results_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    report_lines = [
        "# A4: Ensemble IC Optimization Report\n",
        f"## Data: {T} rows, start={args.start_date}, IC window={args.ic_window}\n",
        f"\n## Mean Daily IC per Component\n",
        ic_df.mean().to_string(),
        f"\n\n## Metrics Comparison\n",
        results_df.to_string(index=False),
        f"\n\n## Verdict\n",
    ]

    base_sharpe = all_results["baseline"]["Sharpe_net"]
    prod_sharpe = all_results["production"]["Sharpe_net"]
    best_delta_sharpe = max(all_results[f"delta_{d}"]["Sharpe_net"] for d in deltas)
    improvement = (best_delta_sharpe - base_sharpe) / base_sharpe * 100

    if improvement > 1.0:
        report_lines.append(f"IC-optimal weights improve Sharpe by {improvement:.1f}% over equal-weight baseline.\n")
        report_lines.append("Recommend: adopt IC-optimal with appropriate delta.\n")
    else:
        report_lines.append(f"IC-optimal weights do NOT improve Sharpe (best delta: {improvement:+.1f}%).\n")
        report_lines.append("Recommend: keep static weights. Meta-learning deprecate candidate.\n")

    report_lines.append(f"\nNote: Production (residual_blpx only) Sharpe={prod_sharpe:.4f} is the real baseline.\n")
    report_lines.append("This experiment uses equal-weight 4-component as comparison baseline.\n")

    report_text = "\n".join(report_lines)
    (out_dir / "a4_ensemble_ic_report.md").write_text(report_text)
    logger.info("Report saved to %s/a4_ensemble_ic_report.md", out_dir)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
