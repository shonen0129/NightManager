"""runner/backtest.py — full backtesting runner.

Provides ``run_production()`` which downloads data, runs the strategy
over the full history, and saves performance artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

from data_loader import download_data, preprocess_data
from domain.risk.metrics import compute_var_es
from performance import calculate_metrics
from runner.config import ProductionConfig
from runner.helpers import build_output_dir, build_strategy, save_summary_files
from services.formatting import print_metrics as _print_metrics

logger = logging.getLogger(__name__)


def run_production(
    start_date: str,
    output_root: str,
    run_tag: Optional[str],
    skip_chart: bool,
    slippage_bps: Optional[float] = None,
) -> str:
    """Run the full backtest and save performance artifacts.

    Steps:
    1. Download / load OHLC market data
    2. Preprocess into the execution DataFrame (df_exec)
    3. Run strategy backtest over ``start_date`` to latest available date
    4. Write daily_results.csv, metrics.csv, run_summary.json

    Args:
        start_date: Backtest start date (YYYY-MM-DD)
        output_root: Root directory for run artifacts
        run_tag: Optional tag appended to the output directory name
        skip_chart: If True, skip HTML / PNG chart generation
        slippage_bps: Optional override for one-way slippage in basis points.
            If None, uses the value from ProductionConfig (default: 5.0 bps).

    Returns:
        Path to the output directory
    """
    config = ProductionConfig(start_date=start_date)
    if slippage_bps is not None:
        config = replace(config, slippage_bps=float(slippage_bps))

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
    strategy = build_strategy(config, df_exec)
    results = strategy.run_backtest(start_date=config.start_date)
    metrics = calculate_metrics(results["daily_return"])
    var_es_result = compute_var_es(
        results["daily_return"],
        confidence=config.var_confidence,
        window=config.var_window,
    )

    if not skip_chart:
        from performance import generate_report
        generate_report(results, output_dir)

    logger.info("[4/4] Writing production artifacts...")
    save_summary_files(results, metrics, config, output_dir)
    _print_metrics(metrics)

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
