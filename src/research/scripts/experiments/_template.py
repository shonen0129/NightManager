#!/usr/bin/env python3
"""Minimal V2 experiment template.

This script demonstrates the canonical V2 backtest hook:
load a Pydantic AppConfig, run BacktestEngine.run_v2_backtest,
and record the experiment to the registry.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache  # noqa: E402
from leadlag.execution.backtester import BacktestEngine  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from research.experiment_registry import Decision  # noqa: E402
from research.experiment_utils import record_backtest_experiment  # noqa: E402


def main() -> int:
    config_path = ROOT / "configs" / "production" / "production.yaml"
    gap_input_dir = ROOT / "var" / "live" / "pipeline_data" / "gap_adjusted_distribution" / "latest"

    app_config = load_config_from_yaml(config_path)
    df_exec = load_df_exec_from_local_cache()

    results = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        start_date="2015-01-05",
        end_date="latest",
        n_jobs=1,
    )

    returns = results["daily_returns"]
    valid = returns[~results["daily_fallback"]]
    sharpe = 0.0
    if len(valid) > 1 and np.std(valid, ddof=1) > 1e-12:
        sharpe = float(np.mean(valid) / np.std(valid, ddof=1) * np.sqrt(252))

    logger.info("V2 template backtest: n_days=%d  sharpe=%.4f", len(returns), sharpe)

    record_backtest_experiment(
        name=Path(__file__).stem,
        hypothesis="Minimal V2 experiment template: canonical backtest + registry recording.",
        app_config=app_config,
        results=results,
        decision=Decision.PENDING,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
