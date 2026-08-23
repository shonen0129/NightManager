#!/usr/bin/env python3
"""Inspect existing subsector panel files and raw OHLC structure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.tickers import JP_TICKERS


def main() -> int:
    panel_dir = ROOT / "var" / "research" / "subsector"

    files = [
        "panel_subsector_oc_vw.parquet",
        "panel_subsector_oc.parquet",
        "panel_subsector_oc_expanded_vw.parquet",
    ]
    for fname in files:
        path = panel_dir / fname
        if not path.exists():
            print(f"--- {fname}: not found ---")
            continue
        df = pd.read_parquet(path)
        print(f"--- {fname} ---")
        print(f"shape: {df.shape}")
        print(f"index: {df.index[0]} to {df.index[-1]}")
        print(f"columns (first 5): {list(df.columns[:5])}")
        print(f"sample row:\n{df.iloc[-1, :5]}")
        print()

    # Raw OHLC structure
    raw_dir = panel_dir / "raw_ohlc"
    sample_files = sorted(raw_dir.glob("*.parquet"))[:3]
    for p in sample_files:
        df = pd.read_parquet(p)
        print(f"--- raw_ohlc/{p.name} ---")
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")
        print(f"index: {df.index[0]} to {df.index[-1]}")
        print(df.head(3))
        print()

    # Mapping file
    mapping_path = ROOT / "configs" / "research" / "subsector_mapping_generated.yaml"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    n = len(mapping.get("stock_mapping", []))
    print(f"--- mapping: {n} tickers ---")
    if n:
        print(mapping["stock_mapping"][0])
        print(mapping["stock_mapping"][-1])

    # Expanded mapping
    expanded_path = ROOT / "configs" / "research" / "subsector_mapping_expanded.yaml"
    with open(expanded_path, "r", encoding="utf-8") as f:
        expanded = yaml.safe_load(f)
    print(f"--- expanded A matrix: rows={len(expanded.get('aggregation_matrix_A',{}).get('rows',[]))} cols={len(expanded.get('aggregation_matrix_A',{}).get('cols',[]))} ---")

    # Check expanded vs canonical taxonomy overlap
    tax_path = ROOT / "configs" / "taxonomy_subsectors.yaml"
    with open(tax_path, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    canonical_tickers = set()
    for _, tickers in taxonomy.items():
        canonical_tickers.update(tickers)
    mapped_tickers = {m["ticker"] for m in mapping.get("stock_mapping", [])}
    print(f"--- taxonomy canonical tickers: {len(canonical_tickers)}; mapped: {len(mapped_tickers)}; overlap: {len(canonical_tickers & mapped_tickers)} ---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
