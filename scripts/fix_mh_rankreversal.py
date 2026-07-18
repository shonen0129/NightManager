#!/usr/bin/env python3
"""One-time fix: copy historical mh and rank_reversal matrices to latest directory."""
import shutil
from pathlib import Path

BASE = Path("/Users/shonen/日米ラグ/live/pipeline_data/gap_adjusted_distribution")
BATCH_DIR = BASE / "20260712_231014" / "matrices"
LATEST_DIR = BASE / "latest" / "matrices"

# Also check 20260712_225931 for mh matrices
MH_DIR = BASE / "20260712_225931" / "matrices"

if not LATEST_DIR.exists():
    print("ERROR: latest/matrices not found")
    exit(1)

# Copy rank_reversal matrices from batch
rr_count = 0
if BATCH_DIR.exists():
    for f in BATCH_DIR.glob("rank_reversal_2*.npy"):
        dest = LATEST_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            rr_count += 1
print(f"Copied {rr_count} rank_reversal matrices from batch")

# Copy mh matrices from 20260712_225931 (the only dir that has them)
mh_count = 0
if MH_DIR.exists():
    for f in MH_DIR.glob("mu_gap_h*_2*.npy"):
        dest = LATEST_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            mh_count += 1
    for f in MH_DIR.glob("omega_gap_h*_2*.npy"):
        dest = LATEST_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            mh_count += 1
print(f"Copied {mh_count} multi-horizon matrices from 20260712_225931")

# Verify
rr_files = list(LATEST_DIR.glob("rank_reversal_2*.npy"))
mh_files = list(LATEST_DIR.glob("*_h*_2*.npy"))
print(f"\nVerification:")
print(f"  rank_reversal files in latest: {len(rr_files)}")
if rr_files:
    dates = sorted([f.stem.split("_")[-1] for f in rr_files])
    print(f"  rank_reversal date range: {dates[0]} to {dates[-1]}")
print(f"  mh files in latest: {len(mh_files)}")
if mh_files:
    dates = sorted([f.stem.split("_")[-1] for f in mh_files])
    print(f"  mh date range: {dates[0]} to {dates[-1]}")
