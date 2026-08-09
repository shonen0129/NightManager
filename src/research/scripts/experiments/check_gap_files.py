#!/usr/bin/env python3
"""Check which gap distribution mu_gap files are missing for a given gap dir."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache


def main():
    if len(sys.argv) < 2:
        print("Usage: check_gap_files.py <gap_dir>")
        sys.exit(1)

    gap_dir = Path(sys.argv[1])
    df_exec = load_df_exec_from_local_cache()
    sim = df_exec.index[df_exec.index >= "2015-01-05"]
    existing = {p.stem[7:15] for p in (gap_dir / "matrices").glob("mu_gap_*.npy") if p.stem[7:15].isdigit() and len(p.stem[7:15]) == 8}
    missing = [d for d in sim if d.strftime("%Y%m%d") not in existing]
    print(f"total: {len(sim)}, existing: {len(existing)}, missing: {len(missing)}")
    if missing:
        for d in missing:
            print(d.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()
