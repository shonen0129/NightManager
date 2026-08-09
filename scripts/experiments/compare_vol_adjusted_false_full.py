#!/usr/bin/env python3
"""Full period: vol_adjusted_target=false vs baseline."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml


def _sharpe(s: pd.Series) -> float:
    if s.std(ddof=1) < 1e-12:
        return 0.0
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252))


def _max_dd(ec: pd.Series) -> float:
    return float((ec / ec.cummax() - 1.0).min())


def print_metrics(label: str, res: dict) -> None:
    dr = res["daily_returns"]
    logger.info(
        "[%s] total=%.2f%%  sharpe=%.4f  maxdd=%.2f%%  turnover=%.4f  fallback=%.4f",
        label,
        dr.sum() * 100,
        _sharpe(dr),
        _max_dd(res["equity_curve"]) * 100,
        float(res["daily_turnover"].mean()),
        float(res["daily_fallback"].mean()),
    )


def main() -> int:
    base_gap_dir = ROOT / "outputs/long_period/gap_adjusted_true/20260730_111327"
    new_gap_dir = ROOT / "outputs/long_period/gap_adjusted_false/20260730_111327"

    df_exec = load_df_exec_from_local_cache()
    app_config = load_config_from_yaml(ROOT / "configs/production/production.yaml")

    base_res = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=base_gap_dir,
        df_exec=df_exec,
        start_date="2015-01-05",
        end_date="latest",
        n_jobs=-1,
    )
    new_res = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=new_gap_dir,
        df_exec=df_exec,
        start_date="2015-01-05",
        end_date="latest",
        n_jobs=-1,
    )

    print_metrics("baseline", base_res)
    print_metrics("vol_adjusted_false", new_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
