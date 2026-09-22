"""Compare real local gap-cache and on-demand distributions at the same PIT input."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.paths import project_root
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.v2.distribution_source import OnDemandDistributionSource
from leadlag.models.v2.fallback_policy import FallbackPolicy
from leadlag.runner.model_factory import build_blpx_model
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from research.diagnostics.gap_inputs import attach_topix_trade_returns, load_gap_execution_inputs


def _deadline(_signum, _frame):
    raise TimeoutError("real-path comparison exceeded 240 seconds")


def main() -> None:
    signal.signal(signal.SIGALRM, _deadline)
    signal.alarm(240)
    root = project_root()
    gap_dir = Path(os.environ.get(
        "R3_GAP_DIR",
        str(root / "var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite"),
    ))
    dates = [pd.Timestamp("2026-08-14"), pd.Timestamp("2026-08-17")]
    frame_pickle = os.environ.get("R3_FRAME_PICKLE")
    if frame_pickle:
        df_exec = pd.read_pickle(frame_pickle)
    elif os.environ.get("R3_FRAME_SOURCE", "local") == "research":
        research_inputs = load_gap_execution_inputs()
        df_exec = attach_topix_trade_returns(research_inputs.df_exec, research_inputs.raw_data)
    else:
        df_exec = load_df_exec_from_local_cache()
    lake = PITDataLake(df_exec)
    config = load_config_from_yaml(root / "configs/production/production.yaml")
    # Distribution parity does not depend on the ML order overlay.  Build the
    # canonical BLPX/decision model directly so an unavailable legacy overlay
    # artifact cannot turn a cache validation run into an unrelated failure.
    model = ProductionV2Model(
        config.v2,
        blpx_model=build_blpx_model(config, clear_cache=True),
    )
    open_910 = build_open_910_returns(df_exec, JP_TICKERS)
    policy = FallbackPolicy.default(model, use_file_cache=True, gap_input_dir=gap_dir)
    rows: list[dict] = []
    for date in dates:
        if date not in df_exec.index:
            rows.append({"date": date.strftime("%Y-%m-%d"), "status": "SKIP", "reason": "date_missing"})
            continue
        as_of = date + pd.Timedelta(hours=9, minutes=10)
        snapshot = lake.get_snapshot(as_of)
        current = dict(snapshot.current_prices)
        historical_owned = HistoricalInputs(
            df_exec,
            source="r3_real_paths",
            open_910_returns=open_910,
        )
        historical = historical_owned.calculation_frame(as_of)
        result: dict = {"date": date.strftime("%Y-%m-%d"), "status": "OK", "horizons": {}}
        for horizon in (1, 3, 5):
            cached = policy.resolve(
                date.strftime("%Y-%m-%d"), historical, current, horizon=horizon,
                snapshot=snapshot, open_910_returns=open_910, allow_implicit_io=False,
            )
            ondemand = OnDemandDistributionSource(model, gap_input_dir=gap_dir).resolve(
                date.strftime("%Y-%m-%d"), historical, current, horizon=horizon,
                snapshot=snapshot, open_910_returns=open_910, allow_implicit_io=False,
            )
            row = {
                "cache_status": cached.status.value,
                "cache_source": cached.source,
                "cache_reason": cached.reason.value if cached.reason else None,
                "on_demand_status": ondemand.status.value,
                "on_demand_source": ondemand.source,
                "on_demand_reason": ondemand.reason.value if ondemand.reason else None,
                "cache_metadata": cached.metadata,
                "on_demand_metadata": ondemand.metadata,
            }
            if cached.mu_gap is not None and ondemand.mu_gap is not None:
                row["mu_max_abs_diff"] = float(np.max(np.abs(cached.mu_gap - ondemand.mu_gap)))
                row["mu_l2_diff"] = float(np.linalg.norm(cached.mu_gap - ondemand.mu_gap))
            if cached.Omega_gap is not None and ondemand.Omega_gap is not None:
                row["omega_max_abs_diff"] = float(np.max(np.abs(cached.Omega_gap - ondemand.Omega_gap)))
                row["omega_l2_diff"] = float(np.linalg.norm(cached.Omega_gap - ondemand.Omega_gap))
            result["horizons"][str(horizon)] = row
        rows.append(result)
    out = root / "reports/20260920_structural_completion/evidence/r3_real_paths.json"
    out.write_text(
        json.dumps(
            {
                "dates": rows,
                "gap_dir": str(gap_dir),
                "frame_source": os.environ.get("R3_FRAME_PICKLE", os.environ.get("R3_FRAME_SOURCE", "local")),
                "df_exec_fingerprint": dataframe_fingerprint(df_exec),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dates": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
