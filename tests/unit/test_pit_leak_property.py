"""Property-based / randomized tests for PIT and leakage invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
)
from leadlag.core.pit import PITMatrixView
from leadlag.core.signal import compute_signal


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.leak
def test_compute_signal_is_invariant_to_future_corruption():
    """Randomized: corrupting any future rows must not change the signal."""
    n_us = 15
    n_jp = 17
    n_days = 120
    corr_window = 60
    n_trials = 20

    base_vectors = build_base_vectors(n_us, n_jp)
    v0_static = build_v3_static(n_us, n_jp, include_v4=True)

    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        all_returns = rng.normal(0.0001, 0.015, (n_days, n_us + n_jp))
        date_index = pd.date_range("2010-01-01", periods=n_days).values
        c_full = compute_baseline_correlation(all_returns, date_index, ewma_half_life=45.0)

        for current_index in range(corr_window + 5, n_days - 5):
            res_orig = compute_signal(
                all_returns,
                current_index,
                n_us,
                corr_window,
                c_full,
                v0_static,
                base_vectors["v1"],
                base_vectors["v2"],
                k=4,
                lambda_reg=0.5,
                lambda_lw=0.3,
                lw_target="equicorrelation",
                ewma_half_life=30.0,
                vol_adjusted_target=True,
            )

            corrupted = all_returns.copy()
            corrupted[current_index, n_us:] = 999.0
            corrupted[current_index + 1 :] = 999.0

            res_corr = compute_signal(
                corrupted,
                current_index,
                n_us,
                corr_window,
                c_full,
                v0_static,
                base_vectors["v1"],
                base_vectors["v2"],
                k=4,
                lambda_reg=0.5,
                lambda_lw=0.3,
                lw_target="equicorrelation",
                ewma_half_life=30.0,
                vol_adjusted_target=True,
            )

            np.testing.assert_allclose(
                res_orig["signal"], res_corr["signal"], rtol=1e-12, atol=1e-12
            )


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.leak
def test_pit_matrix_view_blocks_future_access():
    """Randomized: any slice ending after as_of must raise PITAccessError."""
    from leadlag.core.pit import PITAccessError

    n_trials = 50
    rng = np.random.default_rng(123)
    for _ in range(n_trials):
        n_rows = rng.integers(20, 200)
        n_cols = rng.integers(2, 10)
        as_of = rng.integers(1, n_rows - 1)
        data = rng.standard_normal((n_rows, n_cols))
        view = PITMatrixView(data, as_of=as_of)

        # as_of row is fine
        view.asof_row()

        # historical windows ending at or before as_of are fine
        end = rng.integers(0, as_of + 1)
        _ = view.historical_range(max(0, end - 20), end)

        # any end > as_of raises
        if as_of < n_rows - 1:
            bad_end = rng.integers(as_of + 1, n_rows)
            with pytest.raises(PITAccessError):
                view.historical_range(0, bad_end)
