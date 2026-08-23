#!/usr/bin/env python3
"""Check alignment between expanded panel columns and A matrix columns."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    oc = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_expanded_vw.parquet")
    with open(ROOT / "configs" / "research" / "subsector_mapping_expanded.yaml", "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)
    cols = m["aggregation_matrix_A"]["cols"]
    print(f"panel columns: {len(oc.columns)}")
    print(f"A matrix columns: {len(cols)}")
    print(f"same order: {list(oc.columns) == cols}")
    print(f"same set: {set(oc.columns) == set(cols)}")
    if list(oc.columns) != cols:
        diff = [i for i, (a, b) in enumerate(zip(oc.columns, cols)) if a != b]
        print(f"first diffs at indexes: {diff[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
