"""runner/backtest.py — full backtesting runner.

Provides ``run_production()`` which downloads data, runs the strategy
over the full history, and saves performance artifacts.
"""

from __future__ import annotations

import logging

import pandas as pd

from leadlag.core.risk import compute_var_es
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.execution.config import StrategyConfig as ProductionConfig
from leadlag.execution.helpers import build_output_dir, build_strategy, save_summary_files
from leadlag.reporting.metrics import calculate_metrics, generate_report

logger = logging.getLogger(__name__)


def run_production(
    start_date: str,
    output_root: str,
    run_tag: str | None,
    skip_chart: bool,
    slippage_bps: float | None = None,
) -> str:
    """Run the full backtest and save performance artifacts.

    Returns:
        Path to the output directory
    """
    if slippage_bps is not None:
        config = ProductionConfig(start_date=start_date, slippage_bps=float(slippage_bps))
    else:
        config = ProductionConfig(start_date=start_date)

    output_dir = build_output_dir(output_root, run_tag, run_name="production_backtest")

    logger.info("[1/4] Downloading/loading market data...")
    data = download_data(beta_window=config.beta_window)

    logger.info("[2/4] Preprocessing aligned execution dataset...")
    df_exec = preprocess_data(data, beta_window=config.beta_window)

    logger.info("[3/4] Running production strategy...")
    logger.info(
        "Slippage: %.1f bps one-way (round-trip = 2 x %.1f bps x gross_exposure/day)",
        config.slippage_bps,
        config.slippage_bps,
    )
    model = build_strategy(config, df_exec)

    from leadlag.execution.backtester import BacktestEngine
    results = BacktestEngine.run_backtest(model, df_exec=df_exec, start_date=config.start_date)

    metrics = calculate_metrics(results["daily_returns"])
    var_es_result = compute_var_es(
        results["daily_returns"],
        confidence=config.var_confidence,
        window=config.var_window,
    )

    if not skip_chart:
        # Generate chart report using df structured for graphing
        # SRE backtest result dict contains daily_returns and equity_curve keys
        # We need to build a DataFrame matching results_df structure: index=date, daily_return
        graph_df = pd.DataFrame(
            {"daily_return": results["daily_returns"]}, index=results["daily_returns"].index
        )
        generate_report(graph_df, output_dir)

    logger.info("[4/4] Writing production artifacts...")
    # Wrap results for save_summary_files
    summary_results_df = pd.DataFrame(
        {"daily_return": results["daily_returns"]}, index=results["daily_returns"].index
    )
    save_summary_files(summary_results_df, metrics, config, output_dir)

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
