#!/usr/bin/env python3
"""Publish the production V2 gap cache without importing research packages."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from leadlag.config.paths import project_root
from leadlag.core.market_calendar import is_trading_day
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.data.quote_snapshot import load_frozen_quote_snapshot
from leadlag.execution.config import load_config_from_yaml
from leadlag.pipeline.gap_publisher import publish_gap_cache

ROOT = project_root()
DEFAULT_GAP_STORE = ROOT / "var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite"
DEFAULT_QUOTE_DIR = ROOT / "var/shadow_runs/ml_overlay_value/microstructure"


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/production/production.yaml")
    parser.add_argument("--trade-date", default=None, help="YYYY-MM-DD; default is today in JST")
    parser.add_argument("--gap-store", type=Path, default=DEFAULT_GAP_STORE)
    quote_default = os.environ.get("LEADLAG_CAPTURE_OUTPUT_DIR", str(DEFAULT_QUOTE_DIR))
    parser.add_argument("--quote-dir", type=Path, default=Path(quote_default))
    parser.add_argument("--acceptance-output", type=Path, default=None)
    args = parser.parse_args(argv)

    trade_date = args.trade_date or pd.Timestamp.now(tz="Asia/Tokyo").date().isoformat()
    trade_ts = pd.Timestamp(trade_date)
    if not is_trading_day(trade_ts.date()):
        print(json.dumps({"trade_date": trade_date, "status": "MARKET_CLOSED"}))
        return 0

    app_config = load_config_from_yaml(_absolute(args.config), strict=True)
    df_exec = load_df_exec_from_local_cache(max_stale_bdays=1)
    if trade_ts.normalize() not in df_exec.index:
        raise RuntimeError(f"df_exec does not contain required trade date {trade_date}")

    frozen = load_frozen_quote_snapshot(_absolute(args.quote_dir), trade_date=trade_date)
    result = publish_gap_cache(
        app_config=app_config,
        df_exec=df_exec,
        frozen_snapshot=frozen,
        gap_store=_absolute(args.gap_store),
        trade_date=trade_date,
    )
    payload = {"status": "PASS", **asdict(result)}
    if args.acceptance_output is not None:
        output = _absolute(args.acceptance_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
