#!/usr/bin/env python3
"""Debug _rho function."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data

def _rho(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    print("joined len:", len(df), "a std:", df["a"].std(), "b std:", df["b"].std())
    if len(df) < 30:
        return np.nan
    return float(df["a"].corr(df["b"]))

data = download_data(start_date="2009-01-01", end_date="2026-12-31", force=False)
df_exec = preprocess_data(data)
panel = pd.read_parquet(ROOT / "var" / "research" / "subsector" / "panel_subsector_oc_vw.parquet")

common = panel.index.intersection(df_exec.index)
sub = "Advanced Films, Functional Polymers & Materials (高機能フィルム・機能性樹脂・先端素材)"
etf = "1620.T"
col = f"jp_oc_{etf}"

basket = panel[sub]
etf_ret = df_exec[col].loc[common]
print("rho:", _rho(basket, etf_ret))
