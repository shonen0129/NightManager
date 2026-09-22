from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import leadlag.execution.v2_bridge as bridge

accepted = pd.read_pickle(Path("var/results/20260922_production_acceptance/inputs/df_exec.pkl"))
bridge._load_df_exec = lambda app_config, data_source: accepted.copy(deep=True)  # type: ignore[attr-defined]
result = bridge.run_v2_decision(
    config_path="configs/production/production.yaml",
    gap_input_dir="var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite",
    live_dir="var/live/production_residual_blpx",
    trade_date="2026-08-14",
    api_enable=False,
    api_dry_run=True,
    capital_from_wallet=False,
    text_output=True,
    output_root="reports/20260922_production_acceptance/promoted_production_dry_run",
    run_tag="promoted_production_dry_run",
    dry_run=True,
)
print(json.dumps({"result_path": result, "dry_run": True, "orders_submitted": False}, ensure_ascii=False))
