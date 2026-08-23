#!/usr/bin/env python3
"""Debug dom_etf mapping."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

with open(ROOT / "configs" / "research" / "subsector_aggregation_vw.yaml", "r", encoding="utf-8") as f:
    agg = yaml.safe_load(f)
A = np.array(agg["aggregation_matrix_A"]["data"])
rows = agg["aggregation_matrix_A"]["rows"]
cols = agg["aggregation_matrix_A"]["cols"]

n_none = 0
n_zero = 0
for k, sub in enumerate(cols):
    s_idx = int(np.argmax(A[:, k]))
    if A[s_idx, k] <= 0:
        n_none += 1
        print(f"None: {sub}, max={A[s_idx, k]}, col_sum={A[:, k].sum()}")
    else:
        if k < 5:
            print(f"OK: {sub} -> {rows[s_idx]} (weight={A[s_idx, k]:.4f})")
    if A[:, k].sum() == 0:
        n_zero += 1

print(f"n_none={n_none}, n_zero={n_zero}")
