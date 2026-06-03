import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: compare_runs.py <run_dir_a> <run_dir_b>")
    sys.exit(2)

a = Path(sys.argv[1])
b = Path(sys.argv[2])

ra = pd.read_csv(a / "daily_results.csv", index_col=0, parse_dates=True)
rb = pd.read_csv(b / "daily_results.csv", index_col=0, parse_dates=True)

if len(ra) != len(rb):
    print(f"Row counts differ: {len(ra)} vs {len(rb)}")

# Compare numeric columns
num_cols = [c for c in ra.columns if ra[c].dtype.kind in 'fiu']

diffs = (ra[num_cols] - rb[num_cols]).abs()

# Rows with any non-zero diff beyond tolerance
tol = 1e-12
rows_diff = diffs.max(axis=1) > tol
n_diff = rows_diff.sum()
print(f"Rows with any numeric difference > {tol}: {n_diff} / {len(ra)}")

if n_diff > 0:
    sample = diffs[rows_diff].head(10)
    print("Sample differences (first 10 rows):")
    print(sample)
else:
    print("No numeric differences detected within tolerance.")

# Also check config differences
import json

a_conf = json.load(open(a / "run_summary.json", encoding='utf-8'))["config"]
b_conf = json.load(open(b / "run_summary.json", encoding='utf-8'))["config"]

for k in sorted(set(a_conf.keys()) | set(b_conf.keys())):
    if a_conf.get(k) != b_conf.get(k):
        print(f"Config {k} differs: {a_conf.get(k)} vs {b_conf.get(k)}")
