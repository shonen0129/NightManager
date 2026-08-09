#!/usr/bin/env python3
"""Check the 5m intraday cache date range."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_intraday_cache

df = load_intraday_cache("5m")
print(f"shape: {df.shape}")
print(f"index min: {df.index.min()}")
print(f"index max: {df.index.max()}")
print(f"unique dates: {df.index.date[:5]} ... {df.index.date[-5:]}")
