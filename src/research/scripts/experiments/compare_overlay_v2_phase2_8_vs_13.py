#!/usr/bin/env python3
"""Compare V2 backtest with phase2_8 vs phase2_13 overlay models."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
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
    print(
        f"[{label}] total={dr.sum() * 100:.2f}%  sharpe={_sharpe(dr):.4f}  "
        f"maxdd={_max_dd(res['equity_curve']) * 100:.2f}%  "
        f"turnover={float(res['daily_turnover'].mean()):.4f}  "
        f"fallback={float(res['daily_fallback'].mean()):.4f}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gap-input-dir", required=True)
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--end-date", default="latest")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    df_exec = load_df_exec_from_local_cache()
    app_config = load_config_from_yaml(ROOT / "configs/production/production.yaml")

    base_res = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=args.gap_input_dir,
        df_exec=df_exec,
        start_date=args.start_date,
        end_date=args.end_date,
        n_jobs=args.n_jobs,
    )
    p8_res = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=args.gap_input_dir,
        df_exec=df_exec,
        start_date=args.start_date,
        end_date=args.end_date,
        n_jobs=args.n_jobs,
        overlay_model_dir=ROOT / "models/ml_order_overlay/phase2_8",
    )
    p13_res = BacktestEngine.run_v2_backtest(
        cfg=app_config,
        gap_input_dir=args.gap_input_dir,
        df_exec=df_exec,
        start_date=args.start_date,
        end_date=args.end_date,
        n_jobs=args.n_jobs,
        overlay_model_dir=ROOT / "models/ml_order_overlay/phase2_13",
    )

    print_metrics("baseline_v2", base_res)
    print_metrics("overlay_phase2_8", p8_res)
    print_metrics("overlay_phase2_13", p13_res)

    record_backtest_experiment(
        name=f"{Path(__file__).stem}_baseline",
        hypothesis="V2 backtest baseline without overlay.",
        app_config=app_config,
        results=base_res,
        decision=Decision.PENDING,
    )
    record_backtest_experiment(
        name=f"{Path(__file__).stem}_phase2_8",
        hypothesis="V2 backtest with phase2_8 ML order overlay.",
        app_config=app_config,
        results=p8_res,
        decision=Decision.PENDING,
    )
    record_backtest_experiment(
        name=f"{Path(__file__).stem}_phase2_13",
        hypothesis="V2 backtest with phase2_13 ML order overlay.",
        app_config=app_config,
        results=p13_res,
        decision=Decision.PENDING,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
