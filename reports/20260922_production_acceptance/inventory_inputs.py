"""Inspect canonical input coverage and save an owned local snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.paths import market_data
from leadlag.data.adr_features import DEFAULT_ADR_FEATURES_PATH, load_adr_features
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.market_data_cache import load_df_exec_from_local_cache, load_intraday_cache
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "var/results/20260922_production_acceptance/inputs"


def describe(frame):
    if frame is None:
        return None
    return {"rows": len(frame), "columns": len(frame.columns),
            "first": str(frame.index.min()), "last": str(frame.index.max())}


def main():
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    frame = load_df_exec_from_local_cache()
    bars = load_intraday_cache("5m")
    adr = load_adr_features()
    intraday = build_open_910_returns(frame, JP_TICKERS, df_5m=bars)
    for name, value in (("df_exec", frame), ("bars_5m", bars), ("open_910_returns", intraday), ("adr", adr)):
        if value is not None:
            value.to_pickle(SNAPSHOT / f"{name}.pkl")
    coverage = []
    for year in range(2010, int(frame.index.max().year) + 1):
        rows = intraday.loc[intraday.index.year == year]
        coverage.append({"year": year, "days": len(rows),
                         "all_17_at_0910": int(np.isfinite(rows).all(axis=1).sum()),
                         "cells_0910": int(np.isfinite(rows).sum().sum()),
                         "cells_open_fallback": int(rows.isna().sum().sum())})
    metadata = {}
    for filename, keys in (("df_exec.sqlite", ["df_exec_meta"]), ("etf_prices.sqlite", ["raw_ohlc_meta"])):
        path = market_data(filename)
        if path.exists():
            store = SqliteCacheStore(path)
            metadata[filename] = {key: store.get(key) for key in keys}
    artifacts = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in SNAPSHOT.glob("*.pkl")}
    payload = {"df_exec": describe(frame), "bars": describe(bars), "adr": describe(adr),
               "adr_path": str(DEFAULT_ADR_FEATURES_PATH), "coverage": coverage,
               "metadata": metadata, "snapshot_hashes": artifacts,
               "effective_model_config": cfg.v2.model_dump(mode="json"),
               "effective_risk_config": cfg.risk.model_dump(mode="json"),
               "gap_store": str(cfg.gap_distribution_dir)}
    (OUT / "input_inventory.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({k: payload[k] for k in ("df_exec", "bars", "adr", "coverage", "gap_store")}, indent=2))


if __name__ == "__main__":
    main()
