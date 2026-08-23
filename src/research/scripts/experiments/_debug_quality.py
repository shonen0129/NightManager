#!/usr/bin/env python3
"""Debug quality validation."""
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
print("df_exec columns sample:", [c for c in df_exec.columns if c.startswith("jp_oc_")][:5])
print("jp_oc_1617.T head:")
print(df_exec["jp_oc_1617.T"].head())
print("jp_oc_1617.T describe:", df_exec["jp_oc_1617.T"].describe())

panel = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_vw.parquet")
print("panel head:")
print(panel.head())

with open(ROOT / "configs" / "research" / "subsector_aggregation_vw.yaml", "r", encoding="utf-8") as f:
    agg = yaml.safe_load(f)
A = np.array(agg["aggregation_matrix_A"]["data"])
cols = agg["aggregation_matrix_A"]["cols"]
rows = agg["aggregation_matrix_A"]["rows"]
print("A col sums:", A.sum(axis=0).tolist()[:10])
print("A first row:", A[0].tolist()[:10])

for k, sub in enumerate(cols[:5]):
    print(sub, "dominant:", rows[np.argmax(A[:, k])], A[:, k].max())
