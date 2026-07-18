#!/usr/bin/env python3
"""Compare gap matrices between batch and live directories."""
import numpy as np
from pathlib import Path

BASE = Path("/Users/shonen/日米ラグ/live/pipeline_data/gap_adjusted_distribution")

batch = BASE / "20260714_085119" / "matrices"
live = BASE / "latest" / "matrices"

print("=== mu_gap 20260714: batch vs live ===")
mu_b = np.load(batch / "mu_gap_20260714.npy")
mu_l = np.load(live / "mu_gap_20260714.npy")
print(f"  batch: {mu_b}")
print(f"  live:  {mu_l}")
print(f"  diff:  {mu_b - mu_l}")
print(f"  max_diff: {np.max(np.abs(mu_b - mu_l)):.6f}")
print(f"  batch all zero? {np.allclose(mu_b, 0)}")
print(f"  live all zero?  {np.allclose(mu_l, 0)}")

print()
print("=== omega_gap 20260714: batch vs live ===")
om_b = np.load(batch / "omega_gap_20260714.npy")
om_l = np.load(live / "omega_gap_20260714.npy")
print(f"  batch diag: {np.diag(om_b)}")
print(f"  live diag:  {np.diag(om_l)}")
print(f"  max_diff: {np.max(np.abs(om_b - om_l)):.6f}")

# Also check 07-15, 07-16 in live
for dt in ["20260715", "20260716", "20260717"]:
    f = live / f"mu_gap_{dt}.npy"
    if f.exists():
        mu = np.load(f)
        print(f"\n=== mu_gap {dt} (live only) ===")
        print(f"  values: {mu}")
        print(f"  all zero? {np.allclose(mu, 0)}")
