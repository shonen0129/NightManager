"""Unit tests for the point-in-time (PIT) data access layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
)
from leadlag.core.pit import PITAccessError, PITMatrixView, maybe_as_pit
from leadlag.core.signal import compute_signal


def _random_returns(n_days: int = 50, n_us: int = 15, n_jp: int = 17) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.02, (n_days, n_us + n_jp))


def test_historical_slice_excludes_as_of():
    data = _random_returns(20, 3, 4)
    view = PITMatrixView(data, as_of=10, name="rets")
    window = view.historical_slice(5)
    assert window.shape == (5, 7)
    np.testing.assert_array_equal(window, data[5:10])


def test_historical_slice_short_window_at_start():
    data = _random_returns(20, 3, 4)
    view = PITMatrixView(data, as_of=2, name="rets")
    window = view.historical_slice(5)
    assert window.shape == (2, 7)
    np.testing.assert_array_equal(window, data[0:2])


def test_asof_row_returns_last_allowed_row():
    data = _random_returns(20, 3, 4)
    view = PITMatrixView(data, as_of=10, name="rets")
    row = view.asof_row()
    np.testing.assert_array_equal(row, data[10])


def test_historical_range_blocks_future():
    data = _random_returns(20, 3, 4)
    view = PITMatrixView(data, as_of=10, name="rets")
    with pytest.raises(PITAccessError, match="as-of row is 10"):
        view.historical_range(0, 11)


def test_future_range_access_raises():
    data = _random_returns(20, 3, 4)
    view = PITMatrixView(data, as_of=10, name="rets")
    with pytest.raises(PITAccessError, match="as-of row is 10"):
        view.historical_range(5, 11)


def test_as_of_out_of_bounds_raises():
    data = _random_returns(20, 3, 4)
    # as_of equals length is out of bounds for asof_row
    with pytest.raises(IndexError):
        PITMatrixView(data, as_of=20).asof_row()


def test_maybe_as_pit_wraps_and_reuses_view():
    data = _random_returns(20, 3, 4)
    view = PITMatrixView(data, as_of=10, name="rets")
    wrapped = maybe_as_pit(view, as_of=10)
    assert wrapped is view

    wrapped_array = maybe_as_pit(data, as_of=10)
    assert isinstance(wrapped_array, PITMatrixView)


def test_compute_signal_backward_compatible_and_leakage_free():
    n_days = 80
    n_us = 15
    n_jp = 17
    all_returns = _random_returns(n_days, n_us, n_jp)
    date_index = pd.date_range("2010-01-01", periods=n_days).values

    base_vectors = build_base_vectors(n_us, n_jp)
    v0_static = build_v3_static(n_us, n_jp, include_v4=True)
    c_full = compute_baseline_correlation(all_returns, date_index, ewma_half_life=45.0)

    result = compute_signal(
        all_returns,
        current_index=60,
        n_u=n_us,
        corr_window=40,
        c_full=c_full,
        v0_static=v0_static,
        v1=base_vectors["v1"],
        v2=base_vectors["v2"],
        k=4,
        lambda_reg=0.5,
        lambda_lw=0.3,
        lw_target="equicorrelation",
        ewma_half_life=30.0,
        vol_adjusted_target=True,
    )
    assert "signal" in result
    assert result["signal"].shape == (n_jp,)

    # Corrupting future target returns must not change the signal.
    corrupted = all_returns.copy()
    corrupted[60, n_us:] = 999.0
    corrupted[61:] = 999.0
    result2 = compute_signal(
        corrupted,
        current_index=60,
        n_u=n_us,
        corr_window=40,
        c_full=c_full,
        v0_static=v0_static,
        v1=base_vectors["v1"],
        v2=base_vectors["v2"],
        k=4,
        lambda_reg=0.5,
        lambda_lw=0.3,
        lw_target="equicorrelation",
        ewma_half_life=30.0,
        vol_adjusted_target=True,
    )
    np.testing.assert_allclose(result["signal"], result2["signal"], rtol=1e-12, atol=1e-12)


def test_historical_range_rejects_negative_start():
    view = PITMatrixView(np.arange(50.0).reshape(10, 5), as_of=5, name="test")
    with pytest.raises(PITAccessError):
        view.historical_range(-2, 5)


def test_maybe_as_pit_rejects_asof_mismatch():
    view = PITMatrixView(np.arange(50.0).reshape(10, 5), as_of=5, name="test")
    with pytest.raises(ValueError):
        maybe_as_pit(view, as_of=7)
