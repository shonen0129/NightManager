#!/usr/bin/env python3
"""Check the date range and basic shape of df_exec from local cache."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache

df = load_df_exec_from_local_cache()
print(f"df_exec shape: {df.shape}")
print(f"index type: {type(df.index)}")
print(f"start: {df.index.min()}")
print(f"end: {df.index.max()}")
print(f"columns: {len(df.columns)}")
