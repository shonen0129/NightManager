#!/usr/bin/env python3
"""Quick timing profile for copula vs plain correlation."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.correlation import compute_correlation, estimate_t_copula

# Simulate BLP window returns: 504 days x 32 assets (15 US + 17 JP)
np.random.seed(42)
returns = np.random.randn(504, 32) * 0.01

# Time plain Pearson correlation
n = 10
t0 = time.perf_counter()
for _ in range(n):
    mu, sigma, corr = compute_correlation(returns, ewma_half_life=45.0, use_copula=False)
plain_time = (time.perf_counter() - t0) / n

# Time copula estimation (one call, slow)
t0 = time.perf_counter()
estimate_t_copula(returns[:60], nu_init=5.0)  # smaller T to keep it fast
copula_time_60 = time.perf_counter() - t0

t0 = time.perf_counter()
estimate_t_copula(returns[:252], nu_init=5.0)
copula_time_252 = time.perf_counter() - t0

t0 = time.perf_counter()
estimate_t_copula(returns, nu_init=5.0)
copula_time_504 = time.perf_counter() - t0

print(f"plain_corr (n={n}): {plain_time*1000:.2f} ms/call")
print(f"copula T=60:  {copula_time_60*1000:.2f} ms")
print(f"copula T=252: {copula_time_252*1000:.2f} ms")
print(f"copula T=504: {copula_time_504*1000:.2f} ms")
