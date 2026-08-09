#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.correlation import (
    _make_psd_correlation,
    _t_copula_neg_loglik,
    empirical_cdf_transform,
    estimate_t_copula,
)


def estimate_optimized(returns, nu_init=5.0, max_outer_iter=5):
    rows, cols = returns.shape
    if rows < 10 or cols < 2:
        return np.eye(cols), float(nu_init)
    u = np.clip(empirical_cdf_transform(returns), 1e-6, 1.0 - 1e-6)
    corr = np.eye(cols)
    for i in range(cols):
        for j in range(i + 1, cols):
            mask = np.isfinite(u[:, i]) & np.isfinite(u[:, j])
            if np.sum(mask) > 3:
                tau, _ = kendalltau(u[mask, i], u[mask, j])
                if np.isfinite(tau):
                    corr[i, j] = corr[j, i] = np.sin(np.pi * tau / 2.0)
    corr = _make_psd_correlation(corr)
    nu = float(nu_init)
    z = student_t.ppf(u, df=nu)
    z = np.nan_to_num(np.clip(z, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)
    for _ in range(max_outer_iter):
        jitter = 1e-6
        chol = None
        for _attempt in range(5):
            try:
                matrix = corr + jitter * np.eye(cols)
                chol = np.linalg.cholesky(matrix)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if chol is None:
            def objective(value):
                return _t_copula_neg_loglik(z, corr, value)
        else:
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            solved = np.linalg.solve(matrix, z.T).T
            quadratic = np.sum(z * solved, axis=1)
            z_sq = z ** 2

            def objective(value):
                half = value / 2.0
                from scipy.special import gammaln
                constant = (
                    gammaln((value + cols) / 2.0)
                    + (cols - 1) * gammaln(half)
                    - cols * gammaln((value + 1.0) / 2.0)
                    - 0.5 * log_det
                )
                values = (
                    constant
                    - ((value + cols) / 2.0) * np.log1p(quadratic / value)
                    + ((value + 1.0) / 2.0) * np.sum(np.log1p(z_sq / value), axis=1)
                )
                total = float(np.sum(values))
                return -total if np.isfinite(total) else 1e15

        result = minimize_scalar(objective, bounds=(2.5, 30.0), method="bounded")
        if result.success:
            nu = float(result.x)
        z = student_t.ppf(u, df=nu)
        z = np.nan_to_num(np.clip(z, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0)
        corr_new = np.nan_to_num(np.corrcoef(z.T), nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr_new, 1.0)
        corr = _make_psd_correlation(corr_new)
    return corr, nu


def run_case(name, returns):
    start = time.perf_counter()
    expected_corr, expected_nu = estimate_t_copula(returns)
    old_time = time.perf_counter() - start
    start = time.perf_counter()
    actual_corr, actual_nu = estimate_optimized(returns)
    new_time = time.perf_counter() - start
    print(
        name,
        f"old={old_time:.4f}s",
        f"new={new_time:.4f}s",
        f"speedup={old_time/new_time:.2f}x",
        f"corr_diff={np.max(np.abs(expected_corr-actual_corr)):.3e}",
        f"nu_diff={abs(expected_nu-actual_nu):.3e}",
    )


rng = np.random.default_rng(42)
run_case("normal", rng.normal(0.0, 0.01, (504, 32)))
returns = rng.standard_t(4.0, (504, 32)) * 0.01
returns[:, 1:] += returns[:, :1] * 0.4
run_case("heavy_tail", returns)
returns = np.round(rng.normal(0.0, 0.01, (504, 32)), 3)
run_case("ties", returns)
