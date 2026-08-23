#!/usr/bin/env python3
"""Inspect expanded panel build artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    panel_dir = ROOT / "var" / "research" / "subsector"
    for name in ["mask_subsector_expanded_vw", "mask_subsector_vw", "mask_subsector_expanded"]:
        p = panel_dir / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            print(f"{p.name}: shape={df.shape}, index={df.index[0]} to {df.index[-1]}")
            print(df.iloc[-1, :5])
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
