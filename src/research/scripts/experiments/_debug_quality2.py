#!/usr/bin/env python3
"""Debug quality validation - index alignment."""
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

print("df_exec index name:", df_exec.index.name, "type:", type(df_exec.index))
print("panel index name:", panel.index.name, "type:", type(panel.index))
print("df_exec index[0]:", df_exec.index[0])
print("panel index[0]:", panel.index[0])

common = panel.index.intersection(df_exec.index)
print("common len:", len(common), "first:", common[0], "last:", common[-1])

etf = "1620.T"
col = f"jp_oc_{etf}"
print("etf_ret non-null:", df_exec[col].notna().sum())

sub = "Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材)"
print("basket non-null:", panel[sub].notna().sum())

basket = panel[sub].loc[common]
etf_ret = df_exec[col].loc[common]
print("basket.loc[common] non-null:", basket.notna().sum())
print("etf_ret.loc[common] non-null:", etf_ret.notna().sum())
print("joined non-null:", pd.DataFrame({"a": basket, "b": etf_ret}).dropna().shape)
