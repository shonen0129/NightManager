"""runner/backtest.py — full backtesting runner.

Provides ``run_production()`` which downloads data, runs the V2 production
strategy over the full history, and saves performance artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from leadlag.core.risk import compute_var_es
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.helpers import build_output_dir, save_summary_files
from leadlag.reporting.metrics import calculate_metrics, generate_report

logger = logging.getLogger(__name__)


def _resolve_v2_config(config_path: str | Path | None) -> tuple[dict[str, Any], Path, Path]:
    """Load and return the V2 production config dict, project root, and resolved path."""
    project_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        resolved = project_root / "configs" / "production" / "production.yaml"
    else:
        resolved = Path(config_path)
        if not resolved.is_absolute():
            resolved = project_root / resolved
    with open(resolved, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, project_root, resolved


def _resolve_gap_input_dir(
    gap_input_dir: str | Path | None,
    cfg: dict[str, Any],
    project_root: Path,
) -> Path | None:
    """Resolve V2 gap input directory, returning None if it does not exist."""
    if gap_input_dir is None:
        gap_input_dir = cfg.get("gap_distribution", {}).get(
            "dir", "live/pipeline_data/gap_adjusted_distribution/latest"
        )
    gap_dir = Path(gap_input_dir)
    if not gap_dir.is_absolute():
        gap_dir = project_root / gap_dir
    if not gap_dir.exists():
        logger.warning(
            "Gap input dir not found: %s. V2 backtest will fall back to flat positions.",
            gap_dir,
        )
        return None
    return gap_dir


def run_production(
    start_date: str,
    output_root: str,
    run_tag: str | None,
    skip_chart: bool,
    slippage_bps: float | None = None,
    n_jobs: int = 1,
    config_path: str | Path | None = None,
    gap_input_dir: str | Path | None = None,
) -> str:
    """Run the full V2 production backtest and save performance artifacts.

    Args:
        start_date: Backtest start date.
        output_root: Directory root where outputs are written.
        run_tag: Optional run tag.
        skip_chart: Skip chart generation if True.
        slippage_bps: Override slippage bps. If None, use YAML default.
        n_jobs: Parallel workers for V2 weight generation.
        config_path: Path to V2 production YAML config.
        gap_input_dir: Directory containing mu_gap/omega_gap .npy files.

    Returns:
        Path to the output directory
    """
    cfg, project_root, resolved = _resolve_v2_config(config_path)
    app_config = load_config_from_yaml(resolved)

    output_dir = build_output_dir(output_root, run_tag, run_name="production_backtest")

    residual_cfg = app_config.strategy
    beta_window = residual_cfg.beta_window
    beta_ewma_halflife = residual_cfg.beta_ewma_halflife
    beta_shrinkage = residual_cfg.beta_shrinkage
    beta_winsor_sigma = residual_cfg.beta_winsor_sigma

    logger.info("[1/4] Downloading/loading market data...")
    data = download_data(beta_window=beta_window)

    logger.info("[2/4] Preprocessing aligned execution dataset...")
    df_exec = preprocess_data(
        data,
        beta_window=beta_window,
        beta_ewma_halflife=beta_ewma_halflife,
        beta_shrinkage=beta_shrinkage,
        beta_winsor_sigma=beta_winsor_sigma,
    )

    logger.info("[3/4] Running V2 production backtest...")
    resolved_slippage = slippage_bps if slippage_bps is not None else app_config.strategy.slippage_bps
    logger.info(
        "Slippage: %.1f bps one-way (round-trip = 2 x %.1f bps x gross_exposure/day)",
        resolved_slippage,
        resolved_slippage,
    )

    gap_dir = _resolve_gap_input_dir(gap_input_dir, cfg, project_root)

    results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_dir,
        df_exec=df_exec,
        start_date=start_date,
        slippage_bps=resolved_slippage,
        n_jobs=n_jobs,
    )

    valid_returns = results["daily_returns"]
    if "daily_fallback" in results:
        valid_returns = valid_returns[~results["daily_fallback"]]

    metrics = calculate_metrics(valid_returns)
    risk_cfg = app_config.risk
    var_es_result = compute_var_es(
        results["daily_returns"],
        confidence=risk_cfg.var_confidence,
        window=risk_cfg.var_window,
        var_method=risk_cfg.var_method,
    )

    if not skip_chart:
        # Generate chart report using df structured for graphing
        graph_df = pd.DataFrame(
            {"daily_return": results["daily_returns"]}, index=results["daily_returns"].index
        )
        generate_report(graph_df, output_dir)

    logger.info("[4/4] Writing production artifacts...")
    # Wrap results for save_summary_files
    summary_results_df = pd.DataFrame(
        {"daily_return": results["daily_returns"]}, index=results["daily_returns"].index
    )
    save_summary_files(summary_results_df, metrics, app_config.strategy, output_dir)

    # Print summary metrics to log
    print("=== Backtest Performance Metrics ===")
    for key, v in metrics.items():
        if key in ["AR", "RISK", "MDD", "Total Return"]:
            logger.info("  %s: %.2f%%", key, v * 100)
        elif key == "Sharpe":
            logger.info("  %s: %.4f", key, v)
        else:
            logger.info("  %s: %.2f", key, v)

    if var_es_result.available:
        logger.info(
            "VaR/ES snapshot (99%%,250d): VaR=%.4f%%, ES=%.4f%%",
            var_es_result.var_loss * 100,
            var_es_result.es_loss * 100,
        )
    else:
        logger.info(
            "VaR/ES snapshot skipped: history=%d < window=%d",
            var_es_result.samples,
            var_es_result.window,
        )

    logger.info("Artifacts saved in: %s", output_dir)
    return output_dir
