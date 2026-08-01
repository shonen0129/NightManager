#!/usr/bin/env python3
"""Inspect mu_gap and sigma_gap values for a given date."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

date = sys.argv[1] if len(sys.argv) > 1 else "20200106"
gap_dir = ROOT / "live/pipeline_data/gap_adjusted_distribution/20260731_024303"
mu = np.load(gap_dir / "matrices" / f"mu_gap_{date}.npy")
omega = np.load(gap_dir / "matrices" / f"omega_gap_{date}.npy")
print(f"date={date}")
print(f"mu_gap: min={mu.min():.6f} max={mu.max():.6f} mean={mu.mean():.6f} std={mu.std(ddof=1):.6f}")
print(f"mu_gap: {mu}")
print(f"sigma_gap (sqrt diag Omega): {np.sqrt(np.diag(omega))}")
print(f"Omega diag: {np.diag(omega)}")
