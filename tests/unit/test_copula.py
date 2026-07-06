"""Unit tests for copula-based correlation estimation."""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.core.correlation import (
    blend_correlation,
    compute_correlation,
    compute_stress_weight,
    empirical_cdf_transform,
    estimate_t_copula,
    _make_psd_correlation,
)


# ---------------------------------------------------------------------------
# empirical_cdf_transform
# ---------------------------------------------------------------------------


class TestEmpiricalCDFTransform:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(size=(100, 5))
        u = empirical_cdf_transform(returns)
        assert u.shape == (100, 5)
        assert np.all(u > 0.0) and np.all(u < 1.0)

    def test_uniform_marginals(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(size=(200, 3))
        u = empirical_cdf_transform(returns)
        for k in range(3):
            hist, _ = np.histogram(u[:, k], bins=10, range=(0, 1))
            assert hist.std() / hist.mean() < 0.3

    def test_handles_nans(self):
        returns = np.array([[1.0, np.nan], [2.0, 0.5], [3.0, 1.0], [4.0, 0.2]])
        u = empirical_cdf_transform(returns)
        assert u.shape == (4, 2)
        assert np.all(np.isfinite(u[:, 0]))


# ---------------------------------------------------------------------------
# _make_psd_correlation
# ---------------------------------------------------------------------------


class TestMakePSDCorrelation:
    def test_already_psd(self):
        R = np.eye(3)
        result = _make_psd_correlation(R)
        np.testing.assert_allclose(result, np.eye(3))

    def test_non_psd_fixed(self):
        R = np.array([[1.0, 2.0], [2.0, 1.0]])
        result = _make_psd_correlation(R)
        eigvals = np.linalg.eigvalsh(result)
        assert eigvals.min() >= -1e-10
        np.testing.assert_allclose(np.diag(result), [1.0, 1.0])

    def test_symmetry(self):
        R = np.array([[1.0, 0.5, 0.9], [0.3, 1.0, 0.2], [0.1, 0.4, 1.0]])
        result = _make_psd_correlation(R)
        np.testing.assert_allclose(result, result.T)


# ---------------------------------------------------------------------------
# estimate_t_copula
# ---------------------------------------------------------------------------


class TestEstimateTCopula:
    def test_returns_correlation_matrix(self):
        rng = np.random.default_rng(42)
        returns = rng.multivariate_normal(
            mean=np.zeros(5),
            cov=np.eye(5),
            size=200,
        )
        R, nu = estimate_t_copula(returns, nu_init=5.0, max_outer_iter=3)
        assert R.shape == (5, 5)
        assert 2.5 <= nu <= 30.0
        np.testing.assert_allclose(np.diag(R), [1.0] * 5)
        np.testing.assert_allclose(R, R.T)

    def test_psd_output(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(size=(150, 10))
        R, _ = estimate_t_copula(returns, nu_init=5.0, max_outer_iter=3)
        eigvals = np.linalg.eigvalsh(R)
        assert eigvals.min() >= -1e-8

    def test_correlated_data(self):
        rng = np.random.default_rng(42)
        cov = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ])
        returns = rng.multivariate_normal(mean=np.zeros(3), cov=cov, size=300)
        R, _ = estimate_t_copula(returns, nu_init=5.0, max_outer_iter=3)
        assert R[0, 1] > R[0, 2]
        assert R[1, 2] > R[0, 2]

    def test_small_input_returns_identity(self):
        returns = np.random.normal(size=(5, 3))
        R, nu = estimate_t_copula(returns, nu_init=5.0)
        np.testing.assert_allclose(R, np.eye(3))

    def test_tail_dependence_increases_correlation(self):
        rng = np.random.default_rng(42)
        n = 500
        # Generate correlated t-distributed data via Gaussian copula + chi-squared
        cov = np.array([[1.0, 0.7], [0.7, 1.0]])
        z = rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n)
        chi2 = rng.chisquare(df=3, size=n)
        returns = z / np.sqrt(chi2 / 3)[:, None] * 0.01
        R, nu = estimate_t_copula(returns, nu_init=5.0, max_outer_iter=5)
        # t-distributed data with df=3 should yield nu < 30
        assert nu < 30.0
        # Copula correlation should capture the underlying correlation
        assert R[0, 1] > 0.3


# ---------------------------------------------------------------------------
# blend_correlation
# ---------------------------------------------------------------------------


class TestBlendCorrelation:
    def test_weight_zero_returns_pearson(self):
        pearson = np.array([[1.0, 0.5], [0.5, 1.0]])
        copula = np.array([[1.0, 0.8], [0.8, 1.0]])
        result = blend_correlation(pearson, copula, 0.0)
        np.testing.assert_allclose(result, pearson)

    def test_weight_one_returns_copula(self):
        pearson = np.array([[1.0, 0.5], [0.5, 1.0]])
        copula = np.array([[1.0, 0.8], [0.8, 1.0]])
        result = blend_correlation(pearson, copula, 1.0)
        np.testing.assert_allclose(result, copula)

    def test_weight_half_is_average(self):
        pearson = np.array([[1.0, 0.5], [0.5, 1.0]])
        copula = np.array([[1.0, 0.8], [0.8, 1.0]])
        result = blend_correlation(pearson, copula, 0.5)
        expected = 0.5 * pearson + 0.5 * copula
        np.testing.assert_allclose(result, expected)

    def test_diagonal_is_one(self):
        pearson = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        copula = np.array([[1.0, 0.7, 0.4], [0.7, 1.0, 0.3], [0.4, 0.3, 1.0]])
        result = blend_correlation(pearson, copula, 0.3)
        np.testing.assert_allclose(np.diag(result), [1.0, 1.0, 1.0])

    def test_weight_clipped(self):
        pearson = np.array([[1.0, 0.5], [0.5, 1.0]])
        copula = np.array([[1.0, 0.8], [0.8, 1.0]])
        result_high = blend_correlation(pearson, copula, 2.0)
        result_low = blend_correlation(pearson, copula, -1.0)
        np.testing.assert_allclose(result_high, copula)
        np.testing.assert_allclose(result_low, pearson)


# ---------------------------------------------------------------------------
# compute_stress_weight
# ---------------------------------------------------------------------------


class TestComputeStressWeight:
    def test_calm_period_low_weight(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(scale=0.01, size=(252, 5))
        w = compute_stress_weight(returns)
        assert 0.0 <= w <= 1.0
        # In calm periods the ratio is near 1.0; sharper sigmoid gives ~0.02
        assert w < 0.05

    def test_stress_period_high_weight(self):
        rng = np.random.default_rng(42)
        calm = rng.normal(scale=0.01, size=(232, 5))
        stressed = rng.normal(scale=0.03, size=(20, 5))
        returns = np.vstack([calm, stressed])
        w = compute_stress_weight(returns)
        assert w > 0.5

    def test_short_window_returns_zero(self):
        returns = np.random.normal(size=(15, 5))
        w = compute_stress_weight(returns)
        assert w == 0.0

    def test_custom_threshold(self):
        rng = np.random.default_rng(42)
        calm = rng.normal(scale=0.01, size=(232, 5))
        mild = rng.normal(scale=0.015, size=(20, 5))
        returns = np.vstack([calm, mild])
        w_low_threshold = compute_stress_weight(returns, threshold=1.2)
        w_high_threshold = compute_stress_weight(returns, threshold=2.0)
        assert w_low_threshold > w_high_threshold


# ---------------------------------------------------------------------------
# compute_correlation with copula
# ---------------------------------------------------------------------------


class TestComputeCorrelationWithCopula:
    def test_copula_disabled_returns_pearson(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(size=(100, 5))
        mu1, sig1, corr1 = compute_correlation(returns, ewma_half_life=45)
        mu2, sig2, corr2 = compute_correlation(
            returns, ewma_half_life=45, use_copula=False
        )
        np.testing.assert_allclose(corr1, corr2)
        np.testing.assert_allclose(mu1, mu2)
        np.testing.assert_allclose(sig1, sig2)

    def test_copula_enabled_changes_correlation(self):
        rng = np.random.default_rng(42)
        z = rng.standard_t(df=3, size=(200, 5))
        returns = z * 0.01
        _, _, corr_pearson = compute_correlation(
            returns, ewma_half_life=45, use_copula=False
        )
        _, _, corr_blended = compute_correlation(
            returns, ewma_half_life=45,
            use_copula=True, copula_blend_weight=0.5,
        )
        assert not np.allclose(corr_pearson, corr_blended)

    def test_copula_preserves_shape(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(size=(100, 10))
        mu, sigma, corr = compute_correlation(
            returns, ewma_half_life=45,
            use_copula=True, copula_blend_weight=0.3,
        )
        assert mu.shape == (10,)
        assert sigma.shape == (10,)
        assert corr.shape == (10, 10)
        np.testing.assert_allclose(np.diag(corr), [1.0] * 10)

    def test_copula_psd_output(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(size=(150, 8))
        _, _, corr = compute_correlation(
            returns, ewma_half_life=45,
            use_copula=True, copula_blend_weight=0.5,
        )
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals.min() >= -1e-8
