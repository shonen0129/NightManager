"""Unit tests verifying volatility adjustment and leakage-free properties."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.core import signal as signals
from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
)


def _build_test_returns(n_days: int = 100, n_us: int = 15, n_jp: int = 17) -> np.ndarray:
    np.random.seed(42)
    return np.random.normal(0.0001, 0.015, (n_days, n_us + n_jp))


def test_compute_signal_vol_adjustment():
    """Verify that vol_adjusted_target=True applies 20-day standard deviation scaling."""
    n_days = 100
    n_us = 15
    n_jp = 17
    all_returns = _build_test_returns(n_days, n_us, n_jp)
    date_index = pd.date_range("2010-01-01", periods=n_days).values

    corr_window = 60
    ewma_half_life = 45.0
    k = 6
    lambda_reg = 0.75
    lambda_lw = 0.5
    lw_target = "equicorrelation"

    v0_static = build_v3_static(n_us, n_jp, include_v4=True)
    base_vectors = build_base_vectors(n_us, n_jp)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    # Test index
    test_idx = 80

    # Case A: vol_adjusted_target=False
    sig_raw = signals.compute_signal(
        all_returns,
        test_idx,
        n_us,
        corr_window,
        compute_baseline_correlation(all_returns, date_index, ewma_half_life),
        v0_static,
        v1,
        v2,
        k,
        lambda_reg,
        lambda_lw,
        lw_target,
        ewma_half_life,
        v3_dynamic=False,
        vol_adjusted_target=False,
    )

    # Case B: vol_adjusted_target=True
    sig_adj = signals.compute_signal(
        all_returns,
        test_idx,
        n_us,
        corr_window,
        compute_baseline_correlation(all_returns, date_index, ewma_half_life),
        v0_static,
        v1,
        v2,
        k,
        lambda_reg,
        lambda_lw,
        lw_target,
        ewma_half_life,
        v3_dynamic=False,
        vol_adjusted_target=True,
    )

    # Calculate 20-day standard deviation manually
    jp_returns_20 = all_returns[test_idx - 20 : test_idx, n_us:]
    sigma_20 = np.std(jp_returns_20, axis=0, ddof=1)
    sigma_20 = np.maximum(sigma_20, 1e-8)

    # Verify: r_hat_jp_cc (adjusted) = z_hat_j_t1 * sigma_20
    z_hat = sig_raw["z_hat_j_t1"]
    expected_r_hat = z_hat * sigma_20
    np.testing.assert_allclose(sig_adj["r_hat_jp_cc"], expected_r_hat, rtol=1e-7, atol=1e-7)


def test_vol_adjusted_signal_no_leakage():
    """Verify that compute_signal with vol_adjusted_target=True does not look ahead."""
    n_days = 100
    n_us = 15
    n_jp = 17
    all_returns = _build_test_returns(n_days, n_us, n_jp)
    date_index = pd.date_range("2010-01-01", periods=n_days).values

    corr_window = 60
    ewma_half_life = 45.0
    k = 6
    lambda_reg = 0.75
    lambda_lw = 0.5
    lw_target = "equicorrelation"

    v0_static = build_v3_static(n_us, n_jp, include_v4=True)
    base_vectors = build_base_vectors(n_us, n_jp)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    test_idx = 75
    c_full = compute_baseline_correlation(all_returns, date_index, ewma_half_life)

    # 1. Original calculation
    res_orig = signals.compute_signal(
        all_returns,
        test_idx,
        n_us,
        corr_window,
        c_full,
        v0_static,
        v1,
        v2,
        k,
        lambda_reg,
        lambda_lw,
        lw_target,
        ewma_half_life,
        v3_dynamic=False,
        vol_adjusted_target=True,
    )

    # 2. Corrupt target returns at test_idx (which corresponds to Japanese D_t+1 returns)
    # and all subsequent rows. Signals computed at test_idx must NOT change.
    all_returns_corrupted = all_returns.copy()
    all_returns_corrupted[test_idx, n_us:] = 999.0
    all_returns_corrupted[test_idx + 1 :] = 999.0

    res_corr = signals.compute_signal(
        all_returns_corrupted,
        test_idx,
        n_us,
        corr_window,
        c_full,
        v0_static,
        v1,
        v2,
        k,
        lambda_reg,
        lambda_lw,
        lw_target,
        ewma_half_life,
        v3_dynamic=False,
        vol_adjusted_target=True,
    )

    # Signals must be identical (no leakage of future data)
    np.testing.assert_allclose(res_orig["signal"], res_corr["signal"], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(res_orig["r_hat_jp_cc"], res_corr["r_hat_jp_cc"], rtol=1e-7, atol=1e-7)
