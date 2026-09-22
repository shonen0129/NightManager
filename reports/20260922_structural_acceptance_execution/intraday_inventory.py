from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.market_data_cache import load_intraday_cache


def main() -> None:
    bars = load_intraday_cache("5m")
    result: dict[str, object] = {"present": bars is not None, "rows": 0}
    if bars is not None:
        result.update({
            "rows": len(bars),
            "first": str(bars.index.min()),
            "last": str(bars.index.max()),
            "columns": len(bars.columns),
            "has_2026_08_14_0910": str(pd.Timestamp("2026-08-14 09:10:00")) in {str(x) for x in bars.index},
        })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
