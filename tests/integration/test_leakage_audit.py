"""Automated leakage and compliance audit tests.

These tests ensure that the ComplianceAuditor and v2_auditor correctly
detect leakage, numerical issues, and configuration problems.  They act
as regression guards — if someone introduces a lookahead bug or breaks
the audit logic, CI will catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.compliance.auditor import ComplianceAuditor
from leadlag.models.base import AuditContext


# ---------------------------------------------------------------------------
# v2_auditor: run_leakage_audit
# ---------------------------------------------------------------------------

class TestLeakageAudit:
    def test_signal_before_trade_passes(self):
        result = run_leakage_audit("2026-06-12", "2026-06-16")
        assert result["status"] == "PASSED"
        assert result["signal_date_strictly_before_trade_date"] is True

    def test_signal_equals_trade_fails(self):
        result = run_leakage_audit("2026-06-16", "2026-06-16")
        assert result["status"] == "FAILED"
        assert result["signal_date_strictly_before_trade_date"] is False

    def test_signal_after_trade_fails(self):
        result = run_leakage_audit("2026-06-17", "2026-06-16")
        assert result["status"] == "FAILED"
        assert result["signal_date_strictly_before_trade_date"] is False

    def test_post_open_timing_respected_when_gap_loaded(self):
        result = run_leakage_audit("2026-06-12", "2026-06-16", gap_data_loaded=True)
        assert result["post_open_timing_respected"] is True
        assert result["status"] == "PASSED"

    def test_post_open_timing_fails_when_gap_not_loaded(self):
        result = run_leakage_audit("2026-06-12", "2026-06-16", gap_data_loaded=False)
        assert result["post_open_timing_respected"] is False
        assert result["status"] == "FAILED"

    def test_pit_binning_historical_when_all_before_trade(self):
        dates = np.array(["2026-06-10", "2026-06-11", "2026-06-12"], dtype="datetime64[ns]")
        result = run_leakage_audit("2026-06-12", "2026-06-16", pit_history_trade_dates=dates)
        assert result["pit_binning_strictly_historical"] is True
        assert result["status"] == "PASSED"

    def test_pit_binning_fails_when_trade_date_in_history(self):
        dates = np.array(["2026-06-10", "2026-06-16", "2026-06-15"], dtype="datetime64[ns]")
        result = run_leakage_audit("2026-06-12", "2026-06-16", pit_history_trade_dates=dates)
        assert result["pit_binning_strictly_historical"] is False
        assert result["status"] == "FAILED"

    def test_pit_binning_vacuously_true_when_none(self):
        result = run_leakage_audit("2026-06-12", "2026-06-16", pit_history_trade_dates=None)
        assert result["pit_binning_strictly_historical"] is True

    def test_realized_returns_not_used_when_dates_ok(self):
        result = run_leakage_audit("2026-06-12", "2026-06-16")
        assert result["realized_returns_not_used_in_signal"] is True

    def test_realized_returns_flagged_when_dates_equal(self):
        result = run_leakage_audit("2026-06-16", "2026-06-16")
        assert result["realized_returns_not_used_in_signal"] is False

    def test_gap_freshness_passes_within_10_days(self):
        result = run_leakage_audit("2026-06-12", "2026-06-15")
        assert result["gap_data_freshness_ok"] is True
        assert result["status"] == "PASSED"

    def test_gap_freshness_fails_if_too_old(self):
        result = run_leakage_audit("2026-06-04", "2026-06-15")
        assert result["gap_data_freshness_ok"] is False
        assert result["status"] == "FAILED"


# ---------------------------------------------------------------------------
# v2_auditor: run_numerical_audit
# ---------------------------------------------------------------------------

class TestNumericalAudit:
    def test_valid_weights_pass(self):
        n = 17
        w = np.zeros(n)
        w[:5] = 0.2
        w[12:] = -0.2
        scores = np.random.randn(n)
        Omega = np.eye(n) * 0.01
        result = run_numerical_audit(w, scores, Omega)
        assert result["status"] == "PASSED"
        assert result["weights_finite"] is True
        assert result["scores_finite"] is True
        assert result["net_exposure_near_zero"] is True
        assert result["covariance_diag_nonneg"] is True
        assert result["covariance_symmetric"] is True

    def test_nan_weights_fail(self):
        n = 17
        w = np.zeros(n)
        w[0] = np.nan
        result = run_numerical_audit(w, np.ones(n), np.eye(n))
        assert result["status"] == "FAILED"
        assert result["weights_finite"] is False

    def test_inf_scores_fail(self):
        n = 17
        w = np.zeros(n)
        scores = np.ones(n) * np.inf
        result = run_numerical_audit(w, scores, np.eye(n))
        assert result["status"] == "FAILED"
        assert result["scores_finite"] is False

    def test_nonzero_net_exposure_fails(self):
        n = 17
        w = np.ones(n) * 0.1  # net = 1.7, not zero
        result = run_numerical_audit(w, np.ones(n), np.eye(n))
        assert result["status"] == "FAILED"
        assert result["net_exposure_near_zero"] is False

    def test_non_symmetric_covariance_fails(self):
        n = 5
        w = np.zeros(n)
        w[:2] = 0.1
        w[3:] = -0.1
        Omega = np.eye(n) * 0.01
        Omega[0, 1] = 0.5  # asymmetric
        result = run_numerical_audit(w, np.ones(n), Omega)
        assert result["status"] == "FAILED"
        assert result["covariance_symmetric"] is False

    def test_negative_diag_covariance_fails(self):
        n = 5
        w = np.zeros(n)
        w[:2] = 0.1
        w[3:] = -0.1
        Omega = np.eye(n) * 0.01
        Omega[0, 0] = -0.01  # negative diagonal
        result = run_numerical_audit(w, np.ones(n), Omega)
        assert result["status"] == "FAILED"
        assert result["covariance_diag_nonneg"] is False


# ---------------------------------------------------------------------------
# ComplianceAuditor.run_audit (integration with synthetic data)
# ---------------------------------------------------------------------------

class TestComplianceAuditorIntegration:
    """End-to-end audit with synthetic model and results."""

    def _make_synthetic_model(self):
        """Create a minimal mock model with get_audit_context."""
        class MockModel:
            n_u = 15
            n_j = 17
            us_res_enabled = False
            us_res_beta_shift = 1
            us_res_beta_window = 60
            us_res_gamma = 0.5
            prior_variant = None
            raw_pca_weight = 0.5
            residual_pca_weight = 0.5
            p4_weight = 0.0
            raw_blpx_weight = 0.0
            residual_blpx_weight = 0.0

            def get_audit_context(self) -> AuditContext:
                return AuditContext(
                    n_u=self.n_u,
                    n_j=self.n_j,
                    us_res_enabled=self.us_res_enabled,
                    us_res_beta_shift=self.us_res_beta_shift,
                    us_res_beta_window=self.us_res_beta_window,
                    us_res_gamma=self.us_res_gamma,
                    prior_variant=self.prior_variant,
                    raw_pca_weight=self.raw_pca_weight,
                    residual_pca_weight=self.residual_pca_weight,
                    p4_weight=self.p4_weight,
                    raw_blpx_weight=self.raw_blpx_weight,
                    residual_blpx_weight=self.residual_blpx_weight,
                )

        return MockModel()

    def _make_synthetic_results(self, n_days=60, n_j=17):
        dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
        signals = pd.DataFrame(
            np.random.randn(n_days, n_j) * 0.01,
            index=dates,
            columns=[f"JP{i}" for i in range(n_j)],
        )
        weights = np.zeros((n_days, n_j))
        weights[:, :5] = 0.2
        weights[:, 12:] = -0.2
        weights_df = pd.DataFrame(weights, index=dates, columns=signals.columns)
        daily_gross = np.random.randn(n_days) * 0.001
        daily_costs = np.abs(daily_gross) * 0.1
        daily_net = daily_gross - daily_costs
        return {
            "signals": signals,
            "weights": weights_df,
            "daily_returns_gross": daily_gross,
            "daily_returns": daily_net,
            "daily_costs": daily_costs,
        }

    def _make_synthetic_df_exec(self, n_days=60):
        dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
        sig_dates = dates - pd.Timedelta(days=1)
        return pd.DataFrame(
            {"sig_date": sig_dates},
            index=dates,
        )

    def test_audit_all_passed(self, tmp_path):
        model = self._make_synthetic_model()
        df_exec = self._make_synthetic_df_exec()
        results = self._make_synthetic_results()
        audit_res = ComplianceAuditor.run_audit(
            model, df_exec, results, tmp_path, max_net_limit=0.051, max_gross_limit=2.01
        )
        assert audit_res["all_passed"] is True
        assert audit_res["no_lookahead_detected"] is True
        assert audit_res["ensemble_weights_sum_to_one"] is True
        assert audit_res["no_nan_inf_in_signals"] is True
        assert audit_res["no_nan_inf_in_weights"] is True
        assert audit_res["cost_consistency_passed"] is True
        # Audit files written
        assert (tmp_path / "audit.json").exists()
        assert (tmp_path / "audit_summary.csv").exists()

    def test_audit_detects_lookahead(self, tmp_path):
        model = self._make_synthetic_model()
        df_exec = self._make_synthetic_df_exec()
        # Make sig_date >= trade_date for some rows
        df_exec["sig_date"] = df_exec.index  # sig_date == trade_date → lookahead
        results = self._make_synthetic_results()
        audit_res = ComplianceAuditor.run_audit(
            model, df_exec, results, tmp_path
        )
        assert audit_res["no_lookahead_detected"] is False
        assert audit_res["all_passed"] is False

    def test_audit_detects_bad_weights(self, tmp_path):
        model = self._make_synthetic_model()
        df_exec = self._make_synthetic_df_exec()
        results = self._make_synthetic_results()
        # Inject NaN into weights
        results["weights"].iloc[0, 0] = np.nan
        audit_res = ComplianceAuditor.run_audit(
            model, df_exec, results, tmp_path
        )
        assert audit_res["no_nan_inf_in_weights"] is False
        assert audit_res["all_passed"] is False

    def test_audit_detects_cost_inconsistency(self, tmp_path):
        model = self._make_synthetic_model()
        df_exec = self._make_synthetic_df_exec()
        results = self._make_synthetic_results()
        # Break cost consistency: gross - costs != net
        results["daily_costs"] = results["daily_returns_gross"] * 0.5  # wrong
        audit_res = ComplianceAuditor.run_audit(
            model, df_exec, results, tmp_path
        )
        assert audit_res["cost_consistency_passed"] is False
        assert audit_res["all_passed"] is False

    def test_audit_detects_ensemble_weight_mismatch(self, tmp_path):
        model = self._make_synthetic_model()
        model.raw_pca_weight = 0.3  # Not 0.5
        model.residual_pca_weight = 0.3  # Not 0.5
        df_exec = self._make_synthetic_df_exec()
        results = self._make_synthetic_results()
        audit_res = ComplianceAuditor.run_audit(
            model, df_exec, results, tmp_path
        )
        # Weights sum to 0.6, not 1.0
        assert audit_res["ensemble_weights_sum_to_one"] is False
        assert audit_res["raw_pca_weight_ok"] is False
        assert audit_res["all_passed"] is False

    def test_audit_with_blpx_weights(self, tmp_path):
        """Test audit passes when BLPX weights are included."""
        model = self._make_synthetic_model()
        model.raw_pca_weight = 0.25
        model.residual_pca_weight = 0.25
        model.raw_blpx_weight = 0.25
        model.residual_blpx_weight = 0.25
        df_exec = self._make_synthetic_df_exec()
        results = self._make_synthetic_results()
        audit_res = ComplianceAuditor.run_audit(
            model, df_exec, results, tmp_path
        )
        assert audit_res["ensemble_weights_sum_to_one"] is True
