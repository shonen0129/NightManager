"""Unit tests for EWMA beta estimation and Bayesian shrinkage in preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.data.preprocessor import (
    _apply_beta_shrinkage,
    _compute_ewma_betas,
    _winsorize_rolling,
)


@pytest.fixture
def synthetic_gap_topix_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic gap returns and TOPIX night returns with known beta."""
    np.random.seed(42)
    T = 300
    n_jp = 5

    topix_night = pd.Series(np.random.randn(T) * 0.01, index=pd.date_range("2020-01-01", periods=T, freq="B"))

    true_betas = np.array([0.8, 1.0, 1.2, 0.5, 1.5])
    idio_vol = 0.005

    gap_data = {}
    for j, tk in enumerate(["A", "B", "C", "D", "E"]):
        idio = np.random.randn(T) * idio_vol
        gap_data[tk] = true_betas[j] * topix_night.values + idio

    ret_jp_gap = pd.DataFrame(gap_data, index=topix_night.index)
    return ret_jp_gap, topix_night


class TestComputeEwmaBetas:
    def test_returns_dataframe_with_correct_columns(self, synthetic_gap_topix_data):
        ret_jp_gap, topix_night = synthetic_gap_topix_data
        beta_df = _compute_ewma_betas(ret_jp_gap, topix_night, beta_window=60, ewma_halflife=45)

        assert isinstance(beta_df, pd.DataFrame)
        assert list(beta_df.columns) == list(ret_jp_gap.columns)
        assert beta_df.shape == ret_jp_gap.shape

    def test_warmup_period_is_nan(self, synthetic_gap_topix_data):
        ret_jp_gap, topix_night = synthetic_gap_topix_data
        beta_window = 60
        beta_df = _compute_ewma_betas(ret_jp_gap, topix_night, beta_window=beta_window, ewma_halflife=45)

        warmup = beta_df.iloc[:beta_window]
        assert warmup.isna().all().all()

    def test_betas_converge_to_true_values(self, synthetic_gap_topix_data):
        ret_jp_gap, topix_night = synthetic_gap_topix_data
        beta_df = _compute_ewma_betas(ret_jp_gap, topix_night, beta_window=60, ewma_halflife=45)

        true_betas = np.array([0.8, 1.0, 1.2, 0.5, 1.5])
        last_betas = beta_df.iloc[-1].values

        assert np.all(np.isfinite(last_betas))
        np.testing.assert_allclose(last_betas, true_betas, atol=0.15)

    def test_no_inf_values(self, synthetic_gap_topix_data):
        ret_jp_gap, topix_night = synthetic_gap_topix_data
        beta_df = _compute_ewma_betas(ret_jp_gap, topix_night, beta_window=60, ewma_halflife=45)

        finite_part = beta_df.iloc[60:]
        assert not np.any(np.isinf(finite_part.values))


class TestApplyBetaShrinkage:
    def test_zero_shrinkage_is_identity(self):
        beta_df = pd.DataFrame({"A": [0.5, 1.2, 1.8], "B": [0.9, 1.1, 1.3]})
        result = _apply_beta_shrinkage(beta_df, shrinkage=0.0)
        pd.testing.assert_frame_equal(result, beta_df)

    def test_full_shrinkage_returns_one(self):
        beta_df = pd.DataFrame({"A": [0.5, 1.2, 1.8], "B": [0.9, 1.1, 1.3]})
        result = _apply_beta_shrinkage(beta_df, shrinkage=1.0)
        expected = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_partial_shrinkage_blends_correctly(self):
        beta_df = pd.DataFrame({"A": [0.5, 1.5]})
        result = _apply_beta_shrinkage(beta_df, shrinkage=0.2)
        expected = pd.DataFrame({"A": [0.5 * 0.8 + 1.0 * 0.2, 1.5 * 0.8 + 1.0 * 0.2]})
        pd.testing.assert_frame_equal(result, expected)

    def test_shrinkage_clipped_to_valid_range(self):
        beta_df = pd.DataFrame({"A": [0.5, 1.5]})
        result = _apply_beta_shrinkage(beta_df, shrinkage=-0.5)
        pd.testing.assert_frame_equal(result, beta_df)

        result2 = _apply_beta_shrinkage(beta_df, shrinkage=2.0)
        expected = pd.DataFrame({"A": [1.0, 1.0]})
        pd.testing.assert_frame_equal(result2, expected)

    def test_nan_preserved(self):
        beta_df = pd.DataFrame({"A": [0.5, np.nan, 1.5]})
        result = _apply_beta_shrinkage(beta_df, shrinkage=0.3)
        assert np.isnan(result["A"].iloc[1])
        assert result["A"].iloc[0] == pytest.approx(0.5 * 0.7 + 1.0 * 0.3)


class TestWinsorizeRolling:
    def test_clips_extreme_values(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(30) * 0.01)
        s.iloc[-1] = 0.5  # extreme outlier
        result = _winsorize_rolling(s, window=20, n_sigma=3.0)
        assert result.iloc[-1] < 0.5

    def test_preserves_normal_values(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(100) * 0.01)
        result = _winsorize_rolling(s, window=20, n_sigma=3.0)
        eval_part = result.iloc[20:]
        orig_part = s.iloc[20:]
        close_mask = np.abs(eval_part - orig_part) < 1e-10
        assert close_mask.sum() >= 75

    def test_warmup_preserves_original(self):
        s = pd.Series([0.01, 0.02, 0.03])
        result = _winsorize_rolling(s, window=5, n_sigma=3.0)
        pd.testing.assert_series_equal(result, s)

    def test_works_with_dataframe(self):
        df = pd.DataFrame({"A": np.random.randn(50) * 0.01, "B": np.random.randn(50) * 0.02})
        result = _winsorize_rolling(df, window=20, n_sigma=3.0)
        assert result.shape == df.shape
        assert result.iloc[20:].notna().all().all()
