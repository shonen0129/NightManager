#!/usr/bin/env python3
"""Inspect existing expanded subsector data assets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    # cap panel
    cap_path = ROOT / "var" / "research" / "subsector" / "cap_panel.parquet"
    if cap_path.exists():
        cap = pd.read_parquet(cap_path)
        print(f"cap_panel: shape={cap.shape}, tickers={len(cap.columns)}")
        print(f"cap_panel columns sample: {list(cap.columns[:10])}")
    else:
        print("cap_panel not found")

    # panels
    for name in ["oc", "cc"]:
        p = ROOT / "var" / "research" / "subsector" / f"panel_subsector_{name}_expanded_vw.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            print(f"{p.name}: shape={df.shape}, columns={len(df.columns)}, sample={list(df.columns[:5])}")
        else:
            print(f"{p.name} not found")

    # expanded mapping
    mpath = ROOT / "configs" / "research" / "subsector_mapping_expanded.yaml"
    with open(mpath, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    print("mapping meta:", m.get("meta"))
    print("stock_mapping length:", len(m["stock_mapping"]))
    print("A matrix rows/cols:", len(m["aggregation_matrix_A"]["rows"]), len(m["aggregation_matrix_A"]["cols"]))
    row_sums = [sum(row) for row in m["aggregation_matrix_A"]["data"]]
    print("A row sums (max/min):", max(row_sums), min(row_sums))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
