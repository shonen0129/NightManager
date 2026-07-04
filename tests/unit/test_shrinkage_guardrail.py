"""Unit tests for two-stage shrinkage attenuation guardrail."""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.core.correlation import (
    build_lw_target_correlation,
    effective_raw_weight,
    regularize_correlation,
)


@pytest.fixture
def sample_corr() -> np.ndarray:
    """A simple 4x4 sample correlation matrix."""
    c = np.array([
        [1.0, 0.6, 0.2, -0.1],
        [0.6, 1.0, 0.3, 0.0],
        [0.2, 0.3, 1.0, 0.4],
        [-0.1, 0.0, 0.4, 1.0],
    ])
    return c


@pytest.fixture
def prior_corr() -> np.ndarray:
    """A simple 4x4 prior correlation matrix (equicorrelation-like)."""
    c = np.full((4, 4), 0.3)
    np.fill_diagonal(c, 1.0)
    return c


class TestEffectiveRawWeight:
    def test_no_shrinkage(self):
        assert effective_raw_weight(0.0, 0.0) == 1.0

    def test_full_lw_shrinkage(self):
        assert effective_raw_weight(1.0, 0.0) == 0.0

    def test_full_reg_shrinkage(self):
        assert effective_raw_weight(0.0, 1.0) == 0.0

    def test_old_defaults(self):
        """Old defaults lambda_lw=0.5, lambda_reg=0.75 → 12.5%."""
        assert abs(effective_raw_weight(0.5, 0.75) - 0.125) < 1e-10

    def test_new_defaults(self):
        """New defaults lambda_lw=0.3, lambda_reg=0.30 → 49%."""
        assert abs(effective_raw_weight(0.3, 0.30) - 0.49) < 1e-10


class TestGuardrail:
    def test_guardrail_activates_when_raw_weight_below_threshold(
        self, sample_corr, prior_corr
    ):
        """With lambda_lw=0.5, lambda_reg=0.75, raw weight=12.5% < 30% guardrail.
        The guardrail should rescale lambda_reg so effective raw weight >= 30%.
        """
        c_reg = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.75, lambda_lw=0.5,
            lw_target="equicorrelation",
            min_raw_weight=0.30,
        )
        # The guardrail should have kicked in: lambda_reg adjusted to
        # 1 - 0.30 / (1 - 0.5) = 1 - 0.60 = 0.40
        # Effective raw weight = 0.5 * 0.6 = 0.30
        c_reg_no_guard = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.75, lambda_lw=0.5,
            lw_target="equicorrelation",
            min_raw_weight=0.0,
        )
        # The guarded result should differ from the unguarded one
        assert not np.allclose(c_reg, c_reg_no_guard)

    def test_guardrail_not_activated_when_raw_weight_above_threshold(
        self, sample_corr, prior_corr
    ):
        """With lambda_lw=0.3, lambda_reg=0.30, raw weight=49% > 30% guardrail.
        Guardrail should not change anything.
        """
        c_reg_guarded = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.30, lambda_lw=0.3,
            lw_target="equicorrelation",
            min_raw_weight=0.30,
        )
        c_reg_unguarded = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.30, lambda_lw=0.3,
            lw_target="equicorrelation",
            min_raw_weight=0.0,
        )
        assert np.allclose(c_reg_guarded, c_reg_unguarded)

    def test_guardrail_preserves_minimum_raw_weight(self, sample_corr, prior_corr):
        """Verify that the guarded result has at least min_raw_weight of c_t."""
        min_rw = 0.30
        c_reg = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.90, lambda_lw=0.5,
            lw_target="equicorrelation",
            min_raw_weight=min_rw,
        )
        # Reconstruct what lambda_reg was actually used
        # c_reg = (1-lam_lw)*(1-lam_reg_eff)*c_t + lam_lw*(1-lam_reg_eff)*C_LW + lam_reg_eff*c_0
        # With guardrail: lam_reg_eff = 1 - min_rw / (1 - lam_lw) = 1 - 0.30/0.5 = 0.40
        lam_reg_eff = 1.0 - min_rw / (1.0 - 0.5)
        expected = (1 - 0.5) * (1 - lam_reg_eff) * sample_corr + \
                   0.5 * (1 - lam_reg_eff) * build_lw_target_correlation(sample_corr, "equicorrelation") + \
                   lam_reg_eff * prior_corr
        expected = 0.5 * (expected + expected.T)
        np.fill_diagonal(expected, 1.0)
        assert np.allclose(c_reg, expected, atol=1e-10)

    def test_guardrail_with_full_lw_shrinkage(self, sample_corr, prior_corr):
        """When lambda_lw=1.0, all info goes to LW target, guardrail can't help."""
        # (1 - lambda_lw) = 0, so guardrail condition (1-lambda_lw) > 1e-10 is False
        # Should not crash and should just apply original params
        c_reg = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.75, lambda_lw=1.0,
            lw_target="equicorrelation",
            min_raw_weight=0.30,
        )
        lw_mat = build_lw_target_correlation(sample_corr, "equicorrelation")
        expected = 0.25 * lw_mat + 0.75 * prior_corr
        expected = 0.5 * (expected + expected.T)
        np.fill_diagonal(expected, 1.0)
        assert np.allclose(c_reg, expected, atol=1e-10)

    def test_output_is_valid_correlation_matrix(self, sample_corr, prior_corr):
        """Output should be symmetric with unit diagonal."""
        c_reg = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.75, lambda_lw=0.5,
            lw_target="equicorrelation",
            min_raw_weight=0.30,
        )
        assert np.allclose(c_reg, c_reg.T)
        assert np.allclose(np.diag(c_reg), 1.0)

    def test_guardrail_makes_result_closer_to_sample_than_unguarded(
        self, sample_corr, prior_corr
    ):
        """The guarded result should be closer to c_t than the unguarded one."""
        c_reg_guarded = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.75, lambda_lw=0.5,
            lw_target="equicorrelation",
            min_raw_weight=0.30,
        )
        c_reg_unguarded = regularize_correlation(
            sample_corr, prior_corr,
            lambda_reg=0.75, lambda_lw=0.5,
            lw_target="equicorrelation",
            min_raw_weight=0.0,
        )
        d_guarded = np.linalg.norm(c_reg_guarded - sample_corr)
        d_unguarded = np.linalg.norm(c_reg_unguarded - sample_corr)
        assert d_guarded < d_unguarded
