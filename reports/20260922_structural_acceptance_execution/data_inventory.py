"""Inventory fixed local inputs without changing any cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def frame_info(frame: pd.DataFrame) -> dict[str, object]:
    index = pd.DatetimeIndex(frame.index)
    names = [str(c) for c in frame.columns]
    return {
        "rows": len(frame),
        "columns": len(names),
        "first_date": str(index.min()) if len(index) else None,
        "last_date": str(index.max()) if len(index) else None,
        "has_open_910_columns": sum(name.startswith("jp_open_910_") for name in names),
        "has_open_trade_columns": sum(name.startswith("jp_open_trade_") for name in names),
        "has_close_columns": sum(name.startswith("jp_close_") for name in names),
        "timezone": str(index.tz) if index.tz is not None else None,
    }


def main() -> None:
    path = ROOT / "var/live/pipeline_data/cache/preprocessed/20260728/preprocessed_data.pkl"
    payload = pd.read_pickle(path)
    frame = payload.get("df_exec") if isinstance(payload, dict) else payload
    result = {
        "path": str(path),
        "payload_keys": sorted(payload) if isinstance(payload, dict) else None,
        "df_exec": frame_info(frame),
        "cache_files": sorted(
            str(p.relative_to(ROOT))
            for p in (ROOT / "var/live/pipeline_data/cache").rglob("*")
            if p.is_file()
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
