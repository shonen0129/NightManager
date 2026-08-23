#!/usr/bin/env python3
"""Build subsector -> TOPIX-17 aggregation matrix (value-weighted)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from research.experiments.subsector.taxonomy_map import (
    build_aggregation_matrix,
    build_coverage,
    load_mapping,
    save_aggregation,
    validate_convexity,
)


def main() -> int:
    mapping_path = ROOT / "configs" / "research" / "subsector_mapping_generated.yaml"
    cap_path = ROOT / "var" / "research" / "subsector" / "cap_panel.parquet"
    jpx_path = ROOT / "var" / "research" / "subsector" / "jpx_master" / "jpx_master.csv"
    out_path = ROOT / "configs" / "research" / "subsector_aggregation_vw.yaml"

    mapping = load_mapping(mapping_path)
    cap_panel = pd.read_parquet(cap_path)
    stock_mapping = mapping["stock_mapping"]

    A, rows, cols = build_aggregation_matrix(stock_mapping, cap_panel)
    coverage = build_coverage(stock_mapping, cap_panel, jpx_path)

    print(f"A matrix shape: {A.shape}")
    print(f"Row sums (non-empty): {A.sum(axis=1).tolist()}")
    print(f"All sectors covered: {all(A.sum(axis=1) > 0.9)}")
    print(f"Convex: {validate_convexity(A)}")

    meta = {
        "taxonomy_file": "configs/taxonomy_subsectors.yaml",
        "mapping_file": str(mapping_path),
        "cap_panel": str(cap_path),
        "jpx_master": str(jpx_path) if jpx_path.exists() else None,
        "n_tickers": len(stock_mapping),
        "n_sectors": len(rows),
        "n_subsectors": len(cols),
        "convex": bool(validate_convexity(A)),
    }

    cov_dict = {s: v for s, v in coverage.items()}
    save_aggregation(out_path, A, rows, cols, cov_dict, meta)
    print(f"Saved aggregation to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
