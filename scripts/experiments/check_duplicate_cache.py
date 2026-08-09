#!/usr/bin/env python3
"""Check for duplicate rows in decision cache."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_decision_cache

df = load_decision_cache()
print("rows:", len(df))
print("unique index:", len(df.index.unique()))
dups = df.index[df.index.duplicated(keep=False)]
if len(dups) > 0:
    print("duplicate dates:", sorted(set(d.strftime("%Y-%m-%d") for d in dups)))
else:
    print("no duplicate index")
