"""Regression tests for the shared pure gap-distribution calculation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from leadlag.pipeline.gap_distribution import compute_gap_distribution, select_gap_coefficients
from leadlag.pipeline.gap_reporting import (
    build_gap_diagnostic_frames,
    write_gap_diagnostic_frames,
)


def _blpx_result(n_j: int = 3) -> dict:
    sigma = np.array([0.02, 0.03, 0.04])[:n_j]
    sigma_y = np.array([0.001, -0.002, 0.003])[:n_j]
    z = np.array([0.5, -0.25, 0.75])[:n_j]
    corr = np.eye(n_j) * 1.0
    corr += (np.ones((n_j, n_j)) - np.eye(n_j)) * 0.1
    return {
        "z_hat_j_t1": z,
        "sigma_Y_denorm": sigma,
        "sigma_Y": sigma,
        "mu_Y": sigma_y,
        "Sigma_XX": np.zeros((1, 1)),
        "Sigma_YX": np.zeros((n_j, 1)),
        "Sigma_YY": corr,
        "B_struct": np.zeros((n_j, 1)),
    }


def test_shared_calculation_matches_gap_adjustment_algebra() -> None:
    result = _blpx_result()
    gap = np.array([0.02, -0.01, 0.03])
    betas = np.array([0.8, 1.1, 0.6])
    topix = 0.01
    c = 0.7
    b = 0.6

    computed = compute_gap_distribution(
        result,
        gap_override=gap,
        betas_t=betas,
        topix_night_t=topix,
        vol_adjusted_target=False,
        gap_open_coef=c,
        topix_beta_coef=b,
    )

    mu_raw = result["mu_Y"] + result["sigma_Y"] * result["z_hat_j_t1"]
    gap_syst = betas * topix
    gap_filt = c * (gap - gap_syst) + (c - b) * gap_syst
    denominator = 1.0 + gap_filt
    denominator_floored = np.maximum(denominator, 0.1)
    omega_raw = np.diag(result["sigma_Y_denorm"]) @ result["Sigma_YY"] @ np.diag(
        result["sigma_Y_denorm"]
    )

    np.testing.assert_allclose(computed.mu_raw, mu_raw)
    np.testing.assert_allclose(computed.omega_raw, omega_raw)
    np.testing.assert_allclose(computed.gap_filt, gap_filt)
    np.testing.assert_allclose(computed.denominator, denominator)
    np.testing.assert_allclose(computed.denominator_floored, denominator_floored)
    np.testing.assert_allclose(computed.mu_gap, (1.0 + mu_raw) / denominator_floored - 1.0)
    np.testing.assert_allclose(
        computed.omega_gap,
        np.diag(1.0 / denominator_floored) @ omega_raw @ np.diag(1.0 / denominator_floored),
    )


def test_shared_calculation_does_not_mutate_inputs() -> None:
    result = _blpx_result()
    gap = np.array([0.02, -0.01, 0.03])
    betas = np.array([0.8, 1.1, 0.6])
    before = {key: value.copy() for key, value in result.items() if isinstance(value, np.ndarray)}
    gap_before = gap.copy()
    betas_before = betas.copy()

    compute_gap_distribution(
        result,
        gap_override=gap,
        betas_t=betas,
        topix_night_t=0.01,
        vol_adjusted_target=True,
        gap_open_coef=0.7,
        topix_beta_coef=0.6,
    )

    for key, value in before.items():
        np.testing.assert_array_equal(result[key], value)
    np.testing.assert_array_equal(gap, gap_before)
    np.testing.assert_array_equal(betas, betas_before)


def test_step1_covariance_is_an_explicit_override() -> None:
    result = _blpx_result()
    step1_structure = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.15],
            [0.0, 0.15, 1.0],
        ]
    )
    computed = compute_gap_distribution(
        result,
        gap_override=np.zeros(3),
        betas_t=np.zeros(3),
        topix_night_t=0.0,
        vol_adjusted_target=True,
        gap_open_coef=0.7,
        topix_beta_coef=0.6,
        omega_struct=step1_structure,
    )

    expected = np.diag(result["sigma_Y_denorm"]) @ step1_structure @ np.diag(
        result["sigma_Y_denorm"]
    )
    np.testing.assert_allclose(computed.omega_raw, expected)


def test_directional_gap_coefficients_match_blpx_regime() -> None:
    model = SimpleNamespace(
        gap_open_coef=0.7,
        topix_beta_coef=0.6,
        gap_open_coef_neg=0.9,
        topix_beta_coef_neg=0.4,
    )
    positive = {"z_U_t": np.array([0.1, 0.2])}
    negative = {"z_U_t": np.array([-0.1, 0.02])}

    assert select_gap_coefficients(model, positive) == (0.7, 0.6)
    assert select_gap_coefficients(model, negative) == (0.9, 0.4)


def test_directional_gap_coefficients_keep_legacy_result_compatibility() -> None:
    model = SimpleNamespace(gap_open_coef=0.7, topix_beta_coef=0.6)
    assert select_gap_coefficients(model, {}) == (0.7, 0.6)


def test_gap_diagnostic_frames_are_published_with_stable_names(tmp_path) -> None:
    class Accumulator:
        omega_gap_ticker_records = {
            "A": [{"omega_raw_diag": 1.0, "omega_gap_diag": 2.0}],
            "B": [],
        }
        gap_long_records = [{"trade_date": "2024-01-02"}]
        gap_daily_records = [{"trade_date": "2024-01-02"}]
        dist_long_records = [{"trade_date": "2024-01-02"}]
        dist_daily_records = [{"trade_date": "2024-01-02"}]
        omega_gap_daily_records = [{"trade_date": "2024-01-02"}]

    frames = build_gap_diagnostic_frames(Accumulator(), tickers=("A", "B"))
    assert list(frames.ticker_summary["ticker"]) == ["A", "B"]
    assert frames.ticker_summary.loc[0, "mean_ratio"] == 2.0
    assert isinstance(frames.gap_daily, pd.DataFrame)

    write_gap_diagnostic_frames(frames, tmp_path)
    assert (tmp_path / "gap_components_daily.csv").exists()
    assert (tmp_path / "omega_gap_summary_by_ticker.csv").exists()
