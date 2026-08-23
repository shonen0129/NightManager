#!/usr/bin/env python3
"""Debug 1629.T aggregation row."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
with open(ROOT / "configs" / "research" / "subsector_mapping_generated.yaml", "r", encoding="utf-8") as f:
    mapping = yaml.safe_load(f)
cap = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "cap_panel.parquet")

sec1629 = [m for m in mapping["stock_mapping"] if m["topix17_etf"] == "1629.T"]
print(f"1629.T mapping count: {len(sec1629)}")
for m in sec1629[:5]:
    tk = m["ticker"]
    print(tk, m["subsector"], cap.iloc[-1].get(tk, "MISSING"))
