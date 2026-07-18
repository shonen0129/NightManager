#!/usr/bin/env python3
"""One-time fix: merge batch diagnostics CSV into latest directory.

The live 'latest' directory only has 4 rows of diagnostics (last 3 days).
Copy the batch-generated diagnostics CSV (1476 rows) to the latest directory
so that RuleD PIT binning has sufficient history immediately.
"""
import shutil
from pathlib import Path

BASE = Path("/Users/shonen/日米ラグ/live/pipeline_data/gap_adjusted_distribution")
BATCH_DIAG = BASE / "20260712_231014" / "portfolio_gap_distribution_diagnostics.csv"
LATEST_DIAG = BASE / "latest" / "portfolio_gap_distribution_diagnostics.csv"

print(f"Batch diagnostics: {BATCH_DIAG}")
print(f"  exists: {BATCH_DIAG.exists()}")
print(f"Live diagnostics:  {LATEST_DIAG}")
print(f"  exists: {LATEST_DIAG.exists()}")

if BATCH_DIAG.exists() and LATEST_DIAG.exists():
    import pandas as pd
    old = pd.read_csv(BATCH_DIAG)
    new = pd.read_csv(LATEST_DIAG)
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset='trade_date', keep='last')
    combined = combined.sort_values('trade_date').reset_index(drop=True)
    combined.to_csv(LATEST_DIAG, index=False)
    print(f"Merged: {len(old)} batch + {len(new)} live -> {len(combined)} total rows")
    print(f"Date range: {combined['trade_date'].min()} to {combined['trade_date'].max()}")
elif BATCH_DIAG.exists():
    shutil.copy2(BATCH_DIAG, LATEST_DIAG)
    print(f"Copied batch diagnostics to latest ({len(pd.read_csv(BATCH_DIAG))} rows)")
else:
    print("ERROR: Batch diagnostics not found")
