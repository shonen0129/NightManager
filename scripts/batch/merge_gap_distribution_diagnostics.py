#!/usr/bin/env python3
"""Merge previous gap-distribution diagnostics CSV into the current run.

Replaces the previous inline python3 -c invocation in run_gap_distribution.sh.
RuleD PIT binning needs 252+ days of history, but compute_gap_adjusted_distribution.py
only computes the requested date range, so we merge the previous diagnostics CSV
with the newly generated one (newer rows win on duplicate trade_date).
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def merge_diagnostics(prev_path: str, new_path: str, output_path: str | None = None) -> int:
    prev_file = Path(prev_path)
    new_file = Path(new_path)
    out_file = Path(output_path) if output_path else new_file

    if not prev_file.exists():
        print(f"Previous diagnostics not found: {prev_file}")
        return 1

    if not new_file.exists():
        # Copy previous diagnostics if the new run did not produce one.
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(prev_file, out_file)
        except OSError as e:
            print(f"Failed to copy previous diagnostics: {e}")
            return 1
        print(f"Copied previous diagnostics: {out_file}")
        return 0

    old = pd.read_csv(prev_file)
    new = pd.read_csv(new_file)
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="trade_date", keep="last")
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])
    combined = combined.set_index("trade_date").sort_index().reset_index()
    combined["trade_date"] = combined["trade_date"].dt.strftime("%Y-%m-%d")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_file, index=False)
    print(f"Merged diagnostics: {len(old)} old + {len(new)} new -> {len(combined)} total (saved to {out_file})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge gap distribution diagnostics CSVs")
    parser.add_argument("prev", help="Path to previous diagnostics CSV")
    parser.add_argument("new", help="Path to current (new) diagnostics CSV")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: overwrite new)")
    args = parser.parse_args()
    sys.exit(merge_diagnostics(args.prev, args.new, args.output))
