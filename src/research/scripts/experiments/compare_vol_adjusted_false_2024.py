#!/usr/bin/env python3
"""2024限定: vol_adjusted_target=false の mu_gap を baseline と比較."""
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

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.config import load_config_from_yaml
from research.experiment_registry import Decision
from research.experiment_utils import record_backtest_experiment


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
    base_gap_dir = ROOT / "var/results/gap_adjusted_distribution/20260615_004113"
    new_gap_dir = ROOT / "var/outputs/gap_adjusted_vol_false/20260730_090946"

    df_exec = load_df_exec_from_local_cache()
    app_config = load_config_from_yaml(ROOT / "configs/production/production.yaml")

    base_res = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=base_gap_dir,
        df_exec=df_exec,
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_jobs=-1,
    )
    new_res = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=new_gap_dir,
        df_exec=df_exec,
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_jobs=-1,
    )

    print_metrics("baseline", base_res)
    print_metrics("vol_adjusted_false", new_res)

    record_backtest_experiment(
        name=f"{Path(__file__).stem}_baseline",
        hypothesis="2024 comparison: baseline gap-adjusted distribution.",
        app_config=app_config,
        results=base_res,
        decision=Decision.PENDING,
    )
    record_backtest_experiment(
        name=f"{Path(__file__).stem}_vol_adjusted_false",
        hypothesis="2024 comparison: vol_adjusted_target=false mu_gap.",
        app_config=app_config,
        results=new_res,
        decision=Decision.PENDING,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
