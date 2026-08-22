"""tests/unit/test_macro.py

Unit tests for macro confidence (Factor-Specific Kappa) functionality.

Covers:
- compute_macro_surprise: lookahead safety, shape, z-score properties
- compute_factor_kappa_scale: scale >= 1.0, shape, zero surprise, sensitivity
- download_macro_prices: timeout behavior, cache behavior
- Integration: signal direction preservation under scaling
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.macro import (
    MACRO_NAMES,
    MACRO_SENS_MATRIX,
    N_MACRO,
    clear_macro_cache,
    compute_factor_kappa_scale,
    compute_macro_surprise,
    download_macro_prices,
)

# ---------------------------------------------------------------------------
# compute_macro_surprise tests
# ---------------------------------------------------------------------------


def test_surprise_shape():
    """Surprise output should have shape (T, n_macro)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, (100, N_MACRO))
    surprise = compute_macro_surprise(returns)
    assert surprise.shape == (100, N_MACRO)


def test_surprise_lookahead_safety():
    """z_t should use only data from t-1 and earlier.

    Verify that changing returns at time t does not affect surprise at t-1.
    """
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.01, (50, N_MACRO))
    surprise_orig = compute_macro_surprise(returns)

    # Perturb returns at t=30
    returns_perturbed = returns.copy()
    returns_perturbed[30] = rng.normal(0.0, 0.01, N_MACRO)
    surprise_perturbed = compute_macro_surprise(returns_perturbed)

    # Surprise at t < 30 should be identical
    np.testing.assert_allclose(surprise_orig[:30], surprise_perturbed[:30])
    # Surprise at t=30 should differ (it uses r_t which changed)
    assert not np.allclose(surprise_orig[30], surprise_perturbed[30])


def test_surprise_first_row_equals_input():
    """At t=0, surprise equals the raw return (ewma_mean=0, sigma_safe=1.0)."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, (20, N_MACRO))
    surprise = compute_macro_surprise(returns)
    # At t=0: ewma_mean=0, ewma_var=0, sigma_safe=1.0, so surprise = vals[0]
    np.testing.assert_allclose(surprise[0], returns[0])


def test_surprise_dataframe_input():
    """Should accept DataFrame with MACRO_NAMES columns."""
    dates = pd.date_range("2020-01-01", periods=50)
    rng = np.random.default_rng(99)
    df = pd.DataFrame(rng.normal(0.0, 0.01, (50, N_MACRO)), columns=MACRO_NAMES, index=dates)
    surprise = compute_macro_surprise(df)
    assert surprise.shape == (50, N_MACRO)


def test_surprise_constant_returns_decreasing():
    """With constant returns, surprise magnitude should decrease over time.

    EWMA mean converges to the constant value, so the surprise (return - mean)
    shrinks. The variance also decays, but the net effect is decreasing |surprise|.
    """
    returns = np.ones((200, N_MACRO)) * 0.001
    surprise = compute_macro_surprise(returns)
    # Later surprise magnitudes should be smaller than early ones
    early_mag = np.mean(np.abs(surprise[10:20]))
    late_mag = np.mean(np.abs(surprise[180:190]))
    assert late_mag < early_mag


# ---------------------------------------------------------------------------
# compute_factor_kappa_scale tests
# ---------------------------------------------------------------------------


def test_kappa_scale_shape():
    """Scale output should have shape (T, n_j)."""
    T = 50
    n_j = MACRO_SENS_MATRIX.shape[0]
    surprise = np.random.default_rng(42).normal(0.0, 1.0, (T, N_MACRO))
    kappas = np.array([3.0, 0.5, 0.5])
    scales = compute_factor_kappa_scale(surprise, kappas)
    assert scales.shape == (T, n_j)


def test_kappa_scale_geq_one():
    """All scale values should be >= 1.0 (by construction)."""
    T = 100
    surprise = np.random.default_rng(55).normal(0.0, 3.0, (T, N_MACRO))
    kappas = np.array([3.0, 0.5, 0.5])
    scales = compute_factor_kappa_scale(surprise, kappas)
    assert np.all(scales >= 1.0 - 1e-12)


def test_kappa_scale_zero_surprise():
    """With zero surprise, all scales should be exactly 1.0."""
    T = 30
    surprise = np.zeros((T, N_MACRO))
    kappas = np.array([3.0, 0.5, 0.5])
    scales = compute_factor_kappa_scale(surprise, kappas)
    np.testing.assert_allclose(scales, 1.0)


def test_kappa_scale_sensitivity_effect():
    """Sectors with higher sensitivity should get larger scales."""
    surprise = np.array([[2.0, 0.0, 0.0]])  # Only USDJPY surprise
    kappas = np.array([3.0, 0.0, 0.0])  # Only USDJPY kappa
    scales = compute_factor_kappa_scale(surprise, kappas)

    # Find sectors with highest and lowest USDJPY sensitivity
    usdjpy_sens = np.abs(MACRO_SENS_MATRIX[:, 0])
    high_idx = np.argmax(usdjpy_sens)
    low_idx = np.argmin(usdjpy_sens)

    # Scale = 1 + kappa * |surprise| * |sensitivity|
    expected_high = 1.0 + 3.0 * 2.0 * usdjpy_sens[high_idx]
    expected_low = 1.0 + 3.0 * 2.0 * usdjpy_sens[low_idx]

    assert scales[0, high_idx] > scales[0, low_idx]
    np.testing.assert_allclose(scales[0, high_idx], expected_high)
    np.testing.assert_allclose(scales[0, low_idx], expected_low)


def test_kappa_scale_custom_sens_matrix():
    """Should accept a custom sensitivity matrix."""
    n_j_custom = 3
    T = 10
    custom_sens = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ])
    surprise = np.random.default_rng(77).normal(0.0, 1.0, (T, N_MACRO))
    kappas = np.array([1.0, 1.0, 0.0])
    scales = compute_factor_kappa_scale(surprise, kappas, sens_matrix=custom_sens)
    assert scales.shape == (T, n_j_custom)
    assert np.all(scales >= 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# download_macro_prices tests (mocked yfinance)
# ---------------------------------------------------------------------------


def _make_mock_yf_download(close_data: pd.DataFrame):
    """Create a mock for yf.download that returns the given close data."""
    def _mock_download(tickers, start=None, end=None, period=None, progress=False, auto_adjust=False):
        if isinstance(close_data.columns, pd.MultiIndex):
            return close_data
        # Simulate MultiIndex structure that yfinance returns
        multi_df = pd.DataFrame(
            close_data.values,
            index=close_data.index,
            columns=pd.MultiIndex.from_product([["Close"], close_data.columns]),
        )
        return multi_df
    return _mock_download


def test_download_macro_prices_cache():
    """download_macro_prices should cache results and avoid re-downloading."""
    clear_macro_cache()
    dates = pd.date_range("2020-01-01", periods=50)
    mock_close = pd.DataFrame(
        np.random.default_rng(1).uniform(90, 110, (50, N_MACRO)),
        columns=MACRO_NAMES,
        index=dates,
    )

    call_count = 0

    def counting_download(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _make_mock_yf_download(mock_close)(*args, **kwargs)

    with patch("yfinance.download", side_effect=counting_download):
        result1 = download_macro_prices(start="2020-01-01", end="2020-02-20")
        result2 = download_macro_prices(start="2020-01-01", end="2020-02-20")

    assert call_count == 1  # Only downloaded once
    pd.testing.assert_frame_equal(result1, result2)
    clear_macro_cache()


def test_download_macro_prices_timeout():
    """download_macro_prices should raise TimeoutError when yfinance hangs."""
    clear_macro_cache()

    def hanging_download(*args, **kwargs):
        import time
        time.sleep(100)

    with patch("yfinance.download", side_effect=hanging_download):
        with pytest.raises(TimeoutError):
            download_macro_prices(start="2020-01-01", end="2020-02-20", timeout=0.5)

    clear_macro_cache()


def test_download_macro_prices_column_names():
    """Returned DataFrame should have MACRO_NAMES as columns."""
    clear_macro_cache()
    dates = pd.date_range("2020-01-01", periods=30)
    mock_close = pd.DataFrame(
        np.random.default_rng(2).uniform(90, 110, (30, N_MACRO)),
        columns=MACRO_NAMES,
        index=dates,
    )

    with patch("yfinance.download", side_effect=_make_mock_yf_download(mock_close)):
        result = download_macro_prices(start="2020-01-01", end="2020-01-30")

    assert list(result.columns) == MACRO_NAMES
    clear_macro_cache()


def test_download_macro_prices_yfinance_column_reordering():
    """Columns returned by yfinance in a different order must be mapped by ticker, not position.

    yfinance 1.5.2 returns the Close panel with columns sorted alphabetically
    (['CL=F', 'JPY=X', '^TNX']) rather than in request order.  The function must
    still label JPY=X data as USDJPY, CL=F data as CLF, and ^TNX data as TNX.
    """
    clear_macro_cache()
    dates = pd.date_range("2020-01-01", periods=30)
    # Build a DataFrame whose columns are in yfinance's alphabetical order.
    yf_order = ["CL=F", "JPY=X", "^TNX"]
    requested_order = ["JPY=X", "CL=F", "^TNX"]
    mock_data = {
        "CL=F": np.arange(30, dtype=float) + 50.0,
        "JPY=X": np.arange(30, dtype=float) + 100.0,
        "^TNX": np.arange(30, dtype=float) + 1.0,
    }
    mock_close_yf_order = pd.DataFrame(mock_data, index=dates)

    # Build a MultiIndex DataFrame in yfinance's sorted order.
    mock_multi = pd.DataFrame(
        mock_close_yf_order.values,
        index=dates,
        columns=pd.MultiIndex.from_product([["Close"], yf_order]),
    )

    # A second mock returns in request order to catch position-based assumptions.
    mock_close_request_order = pd.DataFrame(
        {t: mock_data[t] for t in requested_order}, index=dates
    )
    mock_multi_request = pd.DataFrame(
        mock_close_request_order.values,
        index=dates,
        columns=pd.MultiIndex.from_product([["Close"], requested_order]),
    )

    for mock_raw in [mock_multi, mock_multi_request]:
        with patch("yfinance.download", side_effect=_make_mock_yf_download(mock_raw)):
            result = download_macro_prices(start="2020-01-01", end="2020-01-30")

        assert list(result.columns) == MACRO_NAMES
        np.testing.assert_allclose(result["USDJPY"].values, mock_data["JPY=X"])
        np.testing.assert_allclose(result["CLF"].values, mock_data["CL=F"])
        np.testing.assert_allclose(result["TNX"].values, mock_data["^TNX"])

    clear_macro_cache()


# ---------------------------------------------------------------------------
# Integration: signal direction preservation
# ---------------------------------------------------------------------------


def test_signal_direction_preserved_under_scaling():
    """Scaling should preserve signal direction (sign of signal unchanged)."""
    rng = np.random.default_rng(88)
    n_j = MACRO_SENS_MATRIX.shape[0]
    T = 50

    # Generate random signals
    signals = rng.normal(0.0, 1.0, (T, n_j))

    # Generate surprise and scales
    surprise = rng.normal(0.0, 1.0, (T, N_MACRO))
    kappas = np.array([3.0, 0.5, 0.5])
    scales = compute_factor_kappa_scale(surprise, kappas)

    # Apply scaling: signal / scale
    scaled_signals = signals / scales
    scaled_signals = np.nan_to_num(scaled_signals, nan=0.0, posinf=0.0, neginf=0.0)

    # Direction (sign) should be preserved
    np.testing.assert_array_equal(np.sign(signals), np.sign(scaled_signals))


def test_scaling_reduces_signal_magnitude():
    """Scaling should reduce (or maintain) signal magnitude since scale >= 1."""
    rng = np.random.default_rng(66)
    n_j = MACRO_SENS_MATRIX.shape[0]
    T = 50

    signals = rng.normal(0.0, 1.0, (T, n_j))
    surprise = rng.normal(0.0, 2.0, (T, N_MACRO))
    kappas = np.array([3.0, 0.5, 0.5])
    scales = compute_factor_kappa_scale(surprise, kappas)

    scaled_signals = signals / scales

    # Magnitude should be <= original (since scale >= 1)
    assert np.all(np.abs(scaled_signals) <= np.abs(signals) + 1e-12)
