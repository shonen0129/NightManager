#!/usr/bin/env python3
"""Minimal V2 experiment template.

This script demonstrates the canonical V2 backtest hook:
load a Pydantic AppConfig, run BacktestEngine.run_v2_backtest,
and record the experiment to the registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.config import default_registry_path
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution.backtester import BacktestEngine  # noqa: E402
from leadlag.execution.config import load_config_from_yaml  # noqa: E402
from leadlag.experiment_registry import Decision, ExperimentRegistry  # noqa: E402
from leadlag.reporting.metrics import MetricsSpec
from research.experiment_utils import (  # noqa: E402
    record_backtest_experiment,
    record_simple_experiment,
    research_parameters,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-plan",
        type=Path,
        required=True,
        help="JSON: study_id, hypothesis, candidates, protocol, history_complete, historical_trials_lower_bound",
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--code-hash", required=True, help="SHA-256 of the complete evaluated code snapshot"
    )
    parser.add_argument("--registry", type=Path, default=default_registry_path())
    args = parser.parse_args()
    plan = json.loads(args.study_plan.read_text())
    registry = ExperimentRegistry(args.registry)
    try:
        existing = registry.get_study(plan["study_id"])
    except KeyError:
        registry.register_study(**plan)
    else:
        for key in (
            "hypothesis",
            "candidates",
            "protocol",
            "history_complete",
            "historical_trials_lower_bound",
        ):
            if existing[key] != plan[key]:
                raise ValueError("Study plan differs from its immutable registration")
    config_path = ROOT / "configs" / "production" / "production.yaml"
    gap_input_dir = ROOT / "var" / "live" / "pipeline_data" / "gap_adjusted_distribution" / "latest"

    app_config = load_config_from_yaml(config_path)
    df_exec = load_df_exec_from_local_cache()

    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(df_exec, index=True).values.tobytes())
    fingerprint.update(json.dumps(list(df_exec.columns)).encode())
    # Gap distributions are additional evaluated data, not only a config path.
    for path in sorted(gap_input_dir.rglob("*")):
        if path.is_file():
            fingerprint.update(path.relative_to(gap_input_dir).as_posix().encode())
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    fingerprint.update(block)
    data_hash = fingerprint.hexdigest()
    period = plan["protocol"]["oos_period"]
    spec = MetricsSpec(**plan["protocol"]["metrics_spec"])
    trial_id = registry.start_trial(
        plan["study_id"],
        args.candidate_id,
        parameters=research_parameters(app_config),
        code_hash=args.code_hash,
        data_hash=data_hash,
    )
    try:
        results = BacktestEngine.run_v2_backtest(
            cfg=app_config,
            gap_input_dir=gap_input_dir,
            df_exec=df_exec,
            start_date=period["start"],
            end_date=period["end"],
            n_jobs=1,
        )

    except BaseException as exc:
        record_simple_experiment(
            args.candidate_id,
            plan["hypothesis"],
            research_parameters(app_config),
            {"failure_type": type(exc).__name__},
            registry_path=registry.path,
            study_id=plan["study_id"],
            trial_id=trial_id,
            trial_status="aborted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "failed",
        )
        raise

    record = record_backtest_experiment(
        name=args.candidate_id,
        hypothesis=plan["hypothesis"],
        registry_path=registry.path,
        study_id=plan["study_id"],
        trial_id=trial_id,
        app_config=app_config,
        results=results,
        metrics_spec=spec,
        decision=Decision.PENDING,
    )
    logger.info(
        "V2 template backtest: n_days=%s sharpe=%s",
        record.metrics.get("n_observations"),
        record.metrics.get("net_sharpe"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
