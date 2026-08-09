#!/usr/bin/env python3
"""Check for duplicate dates in df_exec and print details."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache

df = load_df_exec_from_local_cache()
dups = df.index[df.index.duplicated(keep=False)]
print(f"Total rows: {len(df)}")
print(f"Unique index: {df.index.is_unique}")
print(f"Duplicate rows count: {len(dups)}")
if len(dups):
    counts = dups.value_counts().sort_index()
    print("Duplicate dates and counts:")
    print(counts.head(20).to_string())
    # Show first duplicated date
    first_dup = counts.index[0]
    print("\nFirst duplicated date rows:")
    print(df.loc[first_dup].to_string())
