"""Unit tests for leadlag.core.signal min-var weight construction."""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.core.signal import build_weights_minvar


def _make_signal_and_cov(n_j: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Return a 6-asset signal/covariance pair with well-conditioned min-var weights."""
    signal = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    # Diagonal values chosen so that min-var basket weights stay below the
    # 0.35 max_abs_weight cap and do not need clipping.
    Sigma = np.diag([1.4, 1.5, 1.5, 1.4, 1.5, 1.5])
    return signal, Sigma


def test_build_weights_minvar_alpha_one_is_min_variance():
    """With alpha=1 the weights must solve the constrained min-var problem."""
    signal, Sigma = _make_signal_and_cov()

    w = build_weights_minvar(
        signal=signal,
        q=0.5,
        n_j=6,
        Sigma_YY=Sigma,
        alpha=1.0,
        enforce_sign=False,
    )

    # Long basket (indices 0, 1, 2): weights ~ [1/1.4, 1/1.5, 1/1.5] / sum
    # Expected: [0.3488..., 0.3256..., 0.3256...]
    long_expected = np.array([1.0 / 1.4, 1.0 / 1.5, 1.0 / 1.5])
    long_expected = long_expected / long_expected.sum()
    assert np.allclose(w[[0, 1, 2]], long_expected, atol=1e-10)
    # Short basket mirrored.
    assert np.allclose(w[[3, 4, 5]], -long_expected, atol=1e-10)
    assert np.sum(w) == pytest.approx(0.0, abs=1e-10)
    assert np.sum(np.abs(w)) == pytest.approx(2.0, abs=1e-10)


def test_build_weights_minvar_alpha_zero_is_signal_proportional():
    """With alpha=0 the weights are proportional to the centered signal."""
    signal, Sigma = _make_signal_and_cov()

    w = build_weights_minvar(
        signal=signal,
        q=0.5,
        n_j=6,
        Sigma_YY=Sigma,
        alpha=0.0,
        enforce_sign=False,
    )

    # Long basket raw = [1, 2, 3] (indices 0,1,2) -> normalized [1/6, 2/6, 3/6]
    long_expected = np.array([1.0, 2.0, 3.0]) / 6.0
    assert np.allclose(w[[0, 1, 2]], long_expected, atol=1e-10)
    # Short basket mirrored.
    assert np.allclose(w[[3, 4, 5]], -long_expected, atol=1e-10)


def test_build_weights_minvar_handles_singular_covariance():
    """A singular Sigma should fall back to signal-proportional weights."""
    signal = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    # Rank-1 covariance makes solve fail.
    Sigma = np.ones((6, 6))

    w = build_weights_minvar(
        signal=signal,
        q=0.5,
        n_j=6,
        Sigma_YY=Sigma,
        alpha=1.0,
        enforce_sign=False,
    )

    # Fallback is the centered, non-negative signal weights (alpha=0 behavior).
    assert not np.isnan(w).any()
    assert np.isfinite(w).all()
    assert np.sum(w) == pytest.approx(0.0, abs=1e-10)
    assert np.sum(np.abs(w)) == pytest.approx(2.0, abs=1e-10)
