#!/usr/bin/env python3
"""Debug quality validation - column matching."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data

data = download_data(start_date="2009-01-01", end_date="2026-12-31", force=False)
df_exec = preprocess_data(data)
panel = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_vw.parquet")

with open(ROOT / "configs" / "research" / "subsector_aggregation_vw.yaml", "r", encoding="utf-8") as f:
    agg = yaml.safe_load(f)
cols = agg["aggregation_matrix_A"]["cols"]
print("panel columns count:", len(panel.columns))
print("cols count:", len(cols))
print("panel columns == cols (sorted):", set(panel.columns) == set(cols))
print("first panel col:", repr(panel.columns[0]))
print("first cols:", repr(cols[0]))
print("match:", panel.columns[0] == cols[0])

# Check if panel columns are exactly equal to cols (same order)
print("same order:", list(panel.columns) == list(cols))

# Try to get one column
sub = cols[0]
print("panel[sub] non-null:", panel[sub].notna().sum())
