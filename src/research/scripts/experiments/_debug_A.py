#!/usr/bin/env python3
"""Debug A matrix build."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys_path = str(ROOT / "src")
import sys
sys.path.insert(0, sys_path)

from research.experiments.subsector.taxonomy_map import build_aggregation_matrix, load_mapping

mapping = load_mapping(ROOT / "configs" / "research" / "subsector_mapping_generated.yaml")
cap = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "cap_panel.parquet")
A, rows, cols = build_aggregation_matrix(mapping["stock_mapping"], cap)
print("A shape:", A.shape)
print("Row sums:", A.sum(axis=1).tolist())
i = rows.index("1629.T")
print("1629.T row:", A[i].tolist())
print("1629.T non-zero cols:", [cols[j] for j in range(len(cols)) if A[i, j] > 0])
