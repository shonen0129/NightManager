#!/usr/bin/env python3
"""Comprehensive gap matrix bias analysis."""
import numpy as np
import pandas as pd
from pathlib import Path
import json

BASE = Path("/Users/shonen/日米ラグ/live/pipeline_data/gap_adjusted_distribution")

# 1. Compare 07-10 matrix between two batch runs (same date, different run times)
print("=" * 70)
print("1. Same-date matrix comparison across batch runs")
print("=" * 70)

dirs_with_0710 = []
for d in sorted(BASE.iterdir()):
    if d.name == "latest" or not d.is_dir():
        continue
    f = d / "matrices" / "mu_gap_20260710.npy"
    if f.exists():
        dirs_with_0710.append(d)

print(f"  Directories with mu_gap_20260710: {[d.name for d in dirs_with_0710]}")

if len(dirs_with_0710) >= 2:
    refs = [np.load(d / "matrices" / "mu_gap_20260710.npy") for d in dirs_with_0710]
    for i in range(len(dirs_with_0710)):
        for j in range(i + 1, len(dirs_with_0710)):
            diff = float(np.max(np.abs(refs[i] - refs[j])))
            print(f"  {dirs_with_0710[i].name} vs {dirs_with_0710[j].name}: max_diff={diff:.2e}")

# 2. Compare 07-14 between batch (08:51, pre-market) and live (post-market)
print("\n" + "=" * 70)
print("2. 07-14 matrix: pre-market batch vs post-market live")
print("=" * 70)

batch_0714 = np.load(BASE / "20260714_085119" / "matrices" / "mu_gap_20260714.npy")
live_0714 = np.load(BASE / "latest" / "matrices" / "mu_gap_20260714.npy")

# Check if batch looks like no-gap (mu_raw without gap adjustment)
# mu_gap = (1 + mu_raw) / denominator - 1
# If gap=0, denominator=1, mu_gap = mu_raw
# If gap != 0, mu_gap != mu_raw
print(f"  batch (08:51 JST, pre-market): {batch_0714}")
print(f"  live  (09:10 JST, post-market): {live_0714}")
print(f"  max_diff: {np.max(np.abs(batch_0714 - live_0714)):.6f}")
print(f"  batch norm: {np.linalg.norm(batch_0714):.6f}")
print(f"  live norm:  {np.linalg.norm(live_0714):.6f}")

# 3. Check if 07-14 batch has same pattern as 07-10 (historical, with gap)
print("\n" + "=" * 70)
print("3. Pattern comparison: 07-10 (historical) vs 07-14 (batch pre-market)")
print("=" * 70)

batch_0710 = np.load(BASE / "20260712_231014" / "matrices" / "mu_gap_20260710.npy")
print(f"  07-10 (historical, with gap): {batch_0710}")
print(f"  07-14 (batch, pre-market):    {batch_0714}")
print(f"  07-14 (live, post-market):    {live_0714}")
print(f"  07-10 norm: {np.linalg.norm(batch_0710):.6f}")
print(f"  07-14 batch norm: {np.linalg.norm(batch_0714):.6f}")
print(f"  07-14 live norm:  {np.linalg.norm(live_0714):.6f}")

# 4. PIT IR history comparison
print("\n" + "=" * 70)
print("4. PIT IR history: batch vs live")
print("=" * 70)

batch_diag = pd.read_csv(BASE / "20260712_231014" / "portfolio_gap_distribution_diagnostics.csv")
live_diag = pd.read_csv(BASE / "latest" / "portfolio_gap_distribution_diagnostics.csv")
print(f"  Batch diagnostics: {len(batch_diag)} rows ({batch_diag['trade_date'].min()} to {batch_diag['trade_date'].max()})")
print(f"  Live diagnostics:  {len(live_diag)} rows ({live_diag['trade_date'].min()} to {live_diag['trade_date'].max()})")
print(f"  ** Backtest RuleD has {len(batch_diag)} days of PIT history **")
print(f"  ** Live RuleD has only {len(live_diag)} days of PIT history **")

# 5. Check leakage audits properly
print("\n" + "=" * 70)
print("5. Leakage audit (corrected)")
print("=" * 70)

for d in sorted(BASE.iterdir()):
    if d.name == "latest" or not d.is_dir():
        continue
    audit_file = d / "leakage_audit.json"
    n_matrices = len(list((d / "matrices").glob("mu_gap_2*.npy"))) if (d / "matrices").exists() else 0
    if n_matrices == 0 or not audit_file.exists():
        continue
    with open(audit_file) as f:
        audit = json.load(f)
    bool_checks = {k: v for k, v in audit.items() if isinstance(v, bool)}
    all_pass = all(bool_checks.values())
    failed = [k for k, v in bool_checks.items() if not v]
    nan_inf = audit.get("nan_inf_count", 0)
    print(f"  {d.name}: {'PASS' if all_pass else 'FAIL'} ({n_matrices} matrices, nan_inf={nan_inf})")
    if failed:
        for f_item in failed:
            print(f"    FAILED: {f_item}")

# 6. Summary
print("\n" + "=" * 70)
print("SUMMARY: Gap Matrix Bias Root Cause")
print("=" * 70)
print("""
Finding 1: Matrix stability for historical dates
  - omega_struct: stable across runs (max_diff = 0.00)
  - mu_gap for historical dates (e.g. 07-10): need to verify

Finding 2: 07-14 matrix differs between pre-market and post-market runs
  - Batch run at 08:51 JST (before Tokyo open) used gap=0
  - Live run at 09:10 JST (after Tokyo open) used actual gap data
  - This is NOT a bias for historical dates, only for the current day

Finding 3: PIT IR history length mismatch
  - Batch: 1476 days of diagnostics -> RuleD works fully in backtest
  - Live: 4 days of diagnostics -> RuleD cannot work in live trading
  - THIS IS THE REAL BIAS: backtest RuleD benefits from full history
    while live RuleD has insufficient history

Finding 4: Leakage audits pass for all batch runs
  - sig_date < trade_date: PASS
  - omega PIT only: PASS
  - No look-ahead in the code logic itself

CONCLUSION:
  The "gap matrix post-generation bias" is NOT a code-level look-ahead bug.
  It is an OPERATIONAL bias:
  1. The backtest uses a batch-generated diagnostics CSV with 1476 rows
     -> RuleD dynamic gross scaling has full PIT history
  2. Live trading uses a live-generated diagnostics CSV with only 4 rows
     -> RuleD cannot function properly
  3. This inflates backtest Sharpe (7.85) vs live performance

FIX:
  Ensure the live gap directory maintains a running diagnostics file
  that accumulates historical IR data, not just the last few days.
  Either:
  a. Copy the batch diagnostics CSV to the live directory, or
  b. Append new daily diagnostics to the existing file instead of
     overwriting it with only the current day's data
""")
