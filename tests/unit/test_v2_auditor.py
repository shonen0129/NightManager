"""Unit tests for leadlag.compliance.v2_auditor numerical audit."""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.compliance.v2_auditor import run_numerical_audit


def test_run_numerical_audit_passes_for_valid_inputs():
    """A valid weight/covariance pair should pass all checks."""
    w = np.array([0.6, -0.6, 0.4, -0.4])
    scores = np.zeros(4)
    Omega = np.eye(4) * 0.01
    result = run_numerical_audit(w, scores, Omega)
    assert result["status"] == "PASSED"
    assert result["gross_exposure_within_limit"] is True
    assert result["covariance_psd"] is True
    assert result["gross_exposure_value"] == pytest.approx(2.0, abs=1e-10)


def test_run_numerical_audit_fails_on_excess_gross():
    """Gross exposure above the limit must fail."""
    w = np.array([1.5, -1.0, 0.5, -0.5])
    scores = np.zeros(4)
    Omega = np.eye(4) * 0.01
    result = run_numerical_audit(w, scores, Omega)
    assert result["gross_exposure_value"] == pytest.approx(3.5, abs=1e-10)
    assert result["gross_exposure_within_limit"] is False
    assert result["status"] == "FAILED"


def test_run_numerical_audit_fails_on_non_psd_covariance():
    """A non-positive-semi-definite covariance matrix must fail."""
    w = np.array([0.25, -0.25, 0.25, -0.25])
    scores = np.zeros(4)
    # Symmetric but has a negative eigenvalue.
    Omega = np.array(
        [[1.0, 0.5, 0.0, 0.0],
         [0.5, 1.0, 0.0, 0.0],
         [0.0, 0.0, -0.01, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )
    result = run_numerical_audit(w, scores, Omega)
    assert result["covariance_psd"] is False
    assert result["status"] == "FAILED"


def test_run_numerical_audit_fails_on_non_finite():
    """NaN/Inf in scores or weights must fail."""
    w = np.array([0.2, np.nan, -0.2, 0.0])
    scores = np.zeros(4)
    Omega = np.eye(4) * 0.01
    result = run_numerical_audit(w, scores, Omega)
    assert result["weights_finite"] is False
    assert result["status"] == "FAILED"
