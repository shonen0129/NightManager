"""Read-only probe for historical 09:10 coverage and target fallback use."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.intraday_inputs import (
    build_open_910_returns,
    compute_jp_target_returns,
    has_valid_open_910_returns,
)
from leadlag.data.tickers import JP_TICKERS


def _read_local(path: Path, key: str):
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        blob = connection.execute(
            "SELECT value FROM cache_store WHERE key = ?", (key,)
        ).fetchone()[0]
    return SqliteCacheStore._restore_from_storage_static(json.loads(blob))


def main() -> None:
    frame = _read_local(Path("var/market_data/df_exec.sqlite"), "df_exec")
    bars = _read_local(Path("var/market_data/etf_prices.sqlite"), "intraday_5m")
    open_910 = build_open_910_returns(frame, JP_TICKERS, df_5m=bars)
    finite = np.isfinite(open_910.to_numpy(dtype=float))
    complete = finite.all(axis=1)
    latest_complete = frame.index[complete][-1]
    history = frame.loc[:latest_complete]
    history_open_910 = open_910.reindex(history.index)
    targets = compute_jp_target_returns(
        history,
        JP_TICKERS,
        open_910_returns=history_open_910,
        allow_implicit_io=False,
        required_index=[latest_complete],
    )
    jp_oc = history[[f"jp_oc_{ticker}" for ticker in JP_TICKERS]].to_numpy(dtype=float)
    missing_with_label = ~np.isfinite(history_open_910.to_numpy(dtype=float)) & np.isfinite(jp_oc)
    fallback_equal = np.isclose(targets, jp_oc, equal_nan=False) & missing_with_label
    output = {
        "history_rows": len(frame),
        "complete_910_rows": int(complete.sum()),
        "missing_cells": int((~finite).sum()),
        "latest_complete_date": str(latest_complete),
        "current_row_strict_valid": has_valid_open_910_returns(
            history_open_910,
            history,
            JP_TICKERS,
            required_index=[latest_complete],
        ),
        "historical_strict_valid": has_valid_open_910_returns(
            history_open_910, history, JP_TICKERS
        ),
        "missing_cells_with_jp_oc_label": int(missing_with_label.sum()),
        "missing_cells_falling_back_exactly_to_jp_oc": int(fallback_equal.sum()),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
