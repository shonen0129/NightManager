#!/usr/bin/env python3
"""Debug A matrix build in detail."""
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(ROOT / "src"))

mapping = yaml.safe_load(open(ROOT / "configs" / "research" / "subsector_mapping_generated.yaml", "r", encoding="utf-8"))
cap_panel = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "cap_panel.parquet")
cap = cap_panel.iloc[-1].copy()

stock_mapping = mapping["stock_mapping"]
subsectors = sorted({m["subsector"] for m in stock_mapping})
sectors = [f"{1617 + i}.T" for i in range(17)]
subsector_index = {s: i for i, s in enumerate(subsectors)}

A = np.zeros((17, len(subsectors)))
for m in stock_mapping:
    tk = m["ticker"]
    sub = m["subsector"]
    etf = m["topix17_etf"]
    s_idx = int(etf.replace(".T", "")) - 1617
    k_idx = subsector_index[sub]
    if etf == "1629.T":
        print(f"tk={tk} sub={sub} etf={etf} s_idx={s_idx} k_idx={k_idx} cap={cap.get(tk, 0)}")
    A[s_idx, k_idx] += cap.get(tk, 0.0)

print("1629.T raw sum:", A[12].sum())
print("1629.T row sum after norm:", (A[12] / A[12].sum()).sum() if A[12].sum() > 0 else 0)
