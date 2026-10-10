"""Fixed-input parity checks for the production and research BLPX callers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from leadlag.models.blpx.model import ProductionBLPXModel
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

BASELINE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "issue44_blpx_signal_baseline.json"
MODEL_CLASSES = {
    "production": ProductionBLPXModel,
    "research": SectorRelativeEnsembleBLPEnhancedModel,
}
DIAGNOSTIC_KEYS = (
    "cond_num",
    "b_norm",
    "b_pca_norm",
    "b_sector_norm",
    "b_struct_norm",
    "sigma_xx_trace",
    "sigma_yx_norm",
    "sigma_yy_trace",
    "min_pred_var",
    "max_pred_var",
    "num_pred_var_floored",
    "pinv_fallback",
    "num_training_samples",
)
MATRIX_KEYS = (
    "sigma_Y_cov",
    "Sigma_XX",
    "Sigma_YX",
    "Sigma_YY",
    "inv_A",
    "B_blp",
    "B_pca_prior",
    "B_sector_prior",
    "B_struct",
    "z_U_t",
    "pred_var_vec",
    "sigma_X",
    "sigma_Y",
    "sigma_Y_denorm",
    "mu_X",
    "mu_Y",
)


def _fixed_inputs() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261008)
    returns = rng.normal(0.0001, 0.012, (640, 32))
    cases = {
        "finite_psd": returns.copy(),
        "nonfinite_window": returns.copy(),
        "window_prefix_changed": returns.copy(),
        "covariance_asymmetry": returns.copy(),
    }
    current_index = 560
    cases["nonfinite_window"][current_index - 120 : current_index - 80, 2] = np.nan
    cases["nonfinite_window"][current_index - 80 : current_index - 40, 17] = np.inf
    cases["window_prefix_changed"][: current_index - 504] += 100.0
    return cases, rng.normal(0.0, 0.1, (32, 6)), np.eye(32)


def _run_signal(model, all_returns: np.ndarray, v0_static: np.ndarray, c_full: np.ndarray):
    return model.compute_blp_signal(
        all_returns,
        current_index=560,
        gap_override=np.linspace(-0.015, 0.015, 17),
        betas_t=np.linspace(0.2, 0.8, 17),
        topix_night_t=0.004,
        v0_static=v0_static,
        c_full=c_full,
        return_matrices=True,
    )


def test_production_and_research_signals_match_pre_refactor_fixture(residual_blpx_prod_config):
    """Preserve unaffected/research baselines while changing production labels only."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    cases, v0_static, c_full = _fixed_inputs()
    actual_by_case = {name: {} for name in MODEL_CLASSES}

    for case_name, all_returns in cases.items():
        config = copy.deepcopy(residual_blpx_prod_config)
        if case_name == "covariance_asymmetry":
            config["blpx"]["asymmetry_mode"] = "covariance"

        for model_name, model_class in MODEL_CLASSES.items():
            model_config = config["blpx"] if model_name == "production" else config
            model = model_class(copy.deepcopy(model_config))
            actual = _run_signal(model, all_returns, v0_static, c_full)
            actual_by_case[model_name][case_name] = actual
            expected = baseline[case_name][model_name]

            if case_name == "nonfinite_window" and model_name == "production":
                # Issue #63 intentionally changes only the production JP-label
                # boundary: unavailable realized labels are excluded instead of
                # zero-filled. Legacy research baselines remain unchanged.
                assert actual["num_training_samples"] == (
                    expected["diagnostics"]["num_training_samples"] - 40
                )
                # All 40 unavailable-label rows must be excluded from the
                # estimation, including their otherwise usable US predictors.
                excluded_predictors_changed = all_returns.copy()
                excluded_predictors_changed[480:520, :15] += 100.0
                changed = _run_signal(
                    model_class(copy.deepcopy(model_config)),
                    excluded_predictors_changed,
                    v0_static,
                    c_full,
                )
                for key in ("signal", "z_hat_j_t1", *DIAGNOSTIC_KEYS, *MATRIX_KEYS):
                    np.testing.assert_allclose(
                        changed[key], actual[key], rtol=0.0, atol=1e-12, equal_nan=True
                    )
            else:
                np.testing.assert_allclose(
                    actual["signal"], expected["signal"], rtol=0.0, atol=1e-12
                )
                np.testing.assert_allclose(
                    actual["z_hat_j_t1"], expected["z_hat_j_t1"], rtol=0.0, atol=1e-12
                )
                for key in DIAGNOSTIC_KEYS:
                    np.testing.assert_allclose(
                        actual[key], expected["diagnostics"][key], rtol=0.0, atol=1e-12
                    )
                for key in MATRIX_KEYS:
                    np.testing.assert_allclose(
                        actual[key],
                        expected["matrix_outputs"][key],
                        rtol=0.0,
                        atol=1e-12,
                        equal_nan=True,
                    )

            if case_name == "finite_psd":
                window = model._prepare_window_returns(all_returns, 560, None)
                corr = model._estimate_correlation(window, 560, False)[2]
                min_eigenvalue = float(np.linalg.eigvalsh(corr).min())
                assert min_eigenvalue >= -1e-10
                np.testing.assert_allclose(
                    min_eigenvalue, expected["corr_min_eigenvalue"], rtol=0.0, atol=1e-12
                )
            if case_name == "nonfinite_window":
                assert np.isfinite(actual["signal"]).all()
                for key in MATRIX_KEYS:
                    assert np.isfinite(actual[key]).all()

    for model_name in MODEL_CLASSES:
        finite = actual_by_case[model_name]["finite_psd"]
        prefix_changed = actual_by_case[model_name]["window_prefix_changed"]
        for key in ("signal", "z_hat_j_t1", *DIAGNOSTIC_KEYS, *MATRIX_KEYS):
            np.testing.assert_allclose(
                prefix_changed[key], finite[key], rtol=0.0, atol=1e-12, equal_nan=True
            )

    future_changed = cases["finite_psd"].copy()
    future_changed[561:] += 100.0
    for model_name, model_class in MODEL_CLASSES.items():
        config = copy.deepcopy(residual_blpx_prod_config)
        model_config = config["blpx"] if model_name == "production" else config
        model = model_class(model_config)
        actual = _run_signal(model, future_changed, v0_static, c_full)
        expected = baseline["finite_psd"][model_name]
        np.testing.assert_allclose(actual["signal"], expected["signal"], rtol=0.0, atol=1e-12)
        finite = actual_by_case[model_name]["finite_psd"]
        for key in ("z_hat_j_t1", *DIAGNOSTIC_KEYS, *MATRIX_KEYS):
            np.testing.assert_allclose(
                actual[key], finite[key], rtol=0.0, atol=1e-12, equal_nan=True
            )


def test_model_sector_prior_hooks_remain_active(residual_blpx_prod_config):
    """Keep prior policy overrideable while sharing its default calculation."""
    cases, v0_static, c_full = _fixed_inputs()
    all_returns = cases["finite_psd"]

    for model_name, model_class in MODEL_CLASSES.items():
        config = copy.deepcopy(residual_blpx_prod_config)
        model_config = config["blpx"] if model_name == "production" else config
        model = model_class(model_config)
        hook_calls = []

        def custom_sector_prior(current_index, returns, corr, B_blp):
            hook_calls.append((current_index, returns, corr))
            return np.zeros_like(B_blp)

        model._get_sector_prior = custom_sector_prior
        actual = _run_signal(model, all_returns, v0_static, c_full)

        assert len(hook_calls) == 1
        assert actual["b_sector_norm"] == 0.0
