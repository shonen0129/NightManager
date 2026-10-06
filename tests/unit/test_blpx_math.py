"""Explicit-input shared BLPX linear algebra contracts."""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.core.blpx_math import (
    apply_confidence_weighting,
    compute_pca_prior,
    safe_solve_inverse,
    solve_blp_coefficients,
    solve_tikhonov,
)


def test_ridge_blocks_and_rank_are_explicit():
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    blp, xx, yx, yy, condition, fallback = solve_blp_coefficients(
        corr, n_u=2, n_j=1, alpha_xx=0.5, alpha_yx=0.25, alpha_yy=0.5, rho=0.1, rank="full"
    )
    np.testing.assert_array_equal(xx, [[1.0, 0.15], [0.15, 1.0]])
    np.testing.assert_allclose(yx, [[0.15, 0.3]], rtol=0, atol=1e-16)
    np.testing.assert_array_equal(yy, [[1.0]])
    np.testing.assert_allclose(blp @ np.array([[1.1, 0.15], [0.15, 1.1]]), yx, atol=1e-15)
    assert condition == pytest.approx(1.25 / 0.95)
    assert not fallback
    assert corr[0, 1] == 0.3


@pytest.mark.parametrize(
    "matrix", [np.zeros((2, 2)), np.full((2, 2), np.nan), np.full((2, 2), np.inf)]
)
def test_singular_and_nonfinite_inverse_falls_back_finitely(matrix):
    result, inverse, fallback = safe_solve_inverse(matrix, np.ones((1, 2)))
    assert fallback
    assert np.isfinite(result).all() and np.isfinite(inverse).all()
    np.testing.assert_array_equal(result, [[0.0, 0.0]])


def test_tikhonov_caps_prior_sum_and_scales_frobenius_without_mutating_inputs():
    pca = np.array([[2.0, 0.0]])
    sector = np.array([[0.0, 4.0]])
    result, inverse = solve_tikhonov(
        np.eye(2),
        np.zeros((1, 2)),
        pca,
        sector,
        1.0,
        B_blp=np.array([[3.0, 4.0]]),
        n_u=2,
        rho=0.1,
        lambda_pca=1.0,
        lambda_sector=2.0,
        frobenius_scale_priors=True,
    )
    # Prior sum capped at .75 in a 1:2 ratio. Each norm becomes ||B_blp|| = 5.
    np.testing.assert_allclose(result, [[1.25 / 1.85, 2.5 / 1.85]], atol=1e-15)
    np.testing.assert_allclose(inverse, np.eye(2) / 1.85)
    np.testing.assert_array_equal(pca, [[2.0, 0.0]])
    np.testing.assert_array_equal(sector, [[0.0, 4.0]])


def test_confidence_floor_and_clip():
    weighted, variance, floored = apply_confidence_weighting(
        np.array([1.0, -1.0]), np.eye(2), np.eye(2), np.eye(2) * 2, 0.5
    )
    np.testing.assert_array_equal(variance, [0.0, 0.0])
    np.testing.assert_array_equal(weighted, [5.0, -5.0])
    assert floored == 2


def test_pca_prior_uses_explicit_universe_dimensions():
    kwargs = dict(n_u=2, n_j=3, k=2, lambda_reg=0.75, lambda_lw=0.5, lw_target="identity")
    result = compute_pca_prior(np.eye(5), np.eye(5)[:, :2], np.eye(5), **kwargs)
    assert result.shape == (3, 2) and np.isfinite(result).all()
    np.testing.assert_array_equal(
        compute_pca_prior(np.eye(4), np.eye(5)[:, :2], np.eye(5), **kwargs), np.zeros((3, 2))
    )
