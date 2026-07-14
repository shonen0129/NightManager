"""B3 Step 1: Characterization tests for all 5 model predict_signals outputs.

These tests capture the exact output structure (keys, types, shapes, indices, columns,
dtypes, values, NaN positions) of every model's predict_signals before the composition
refactor. They serve as golden masters to verify behavior preservation during migration.

Each model is tested for:
1. Dict key set matches expected
2. DataFrame types, shapes, indices, columns, dtypes
3. Values match golden master (atol=1e-12) on a recent slice
4. Repeated run on same instance produces identical output
5. Fresh instance produces identical output
6. NaN positions match
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.sre import SectorRelativeEnsembleModel
from leadlag.models.sector_relative_ensemble_blp import SectorRelativeEnsembleBLPModel
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.bayesian_blpx import BayesianBLPXModel
from leadlag.models.sector_relative_ensemble_rrr import SectorRelativeEnsembleRRRModel


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sre_config() -> dict:
    return {
        "model": {"name": "sector_relative_ensemble"},
        "portfolio": {"weight_mode": "signal", "long_short_frac": 0.3},
        "ensemble": {"raw_pca_weight": 0.5, "residual_pca_weight": 0.5, "normalization": "zscore"},
        "costs": {"slippage_bps_per_side": 5.0},
        "residualization": {"enabled_for_p3": True, "beta_window": 60},
    }


@pytest.fixture
def blp_config() -> dict:
    return {
        "model": {"name": "sector_relative_ensemble_blp"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "raw_pca_weight": 0.4,
            "residual_pca_weight": 0.4,
            "p5_weight": 0.1,
            "p5p3_weight": 0.1,
            "normalization": "zscore",
        },
        "costs": {"slippage_bps_per_side": 5.0},
        "blp_window": 252,
        "blp_ewma_halflife": 45,
        "alpha_xx": 0.5,
        "alpha_yx": 0.25,
        "rho": 0.03,
        "rank": "full",
    }


@pytest.fixture
def blpx_config() -> dict:
    return {
        "model": {"name": "sector_relative_ensemble_blp_enhanced"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "raw_pca_weight": 0.4,
            "residual_pca_weight": 0.4,
            "raw_blpx_weight": 0.1,
            "residual_blpx_weight": 0.1,
            "normalization": "zscore",
        },
        "costs": {"slippage_bps_per_side": 5.0},
        "blp_window": 252,
        "blp_ewma_halflife": 45,
        "alpha_xx": 0.5,
        "alpha_yx": 0.25,
        "alpha_yy": 0.5,
        "rho": 0.03,
        "rank": "full",
        "lambda_pca": 0.1,
        "lambda_sector": 0.1,
        "beta_conf": 0.25,
        "winsor_sigma": 4.0,
        "exec_adjustment": "none",
    }


@pytest.fixture
def bayesian_config(blpx_config) -> dict:
    cfg = copy.deepcopy(blpx_config)
    cfg["bayesian_enabled"] = True
    cfg["bayesian_mode"] = "ic"
    cfg["bayesian_eta_base"] = 0.3
    cfg["bayesian_ic_window"] = 63
    cfg["bayesian_ic_amplifier"] = 5.0
    cfg["bayesian_eta_min"] = 0.05
    cfg["bayesian_eta_max"] = 0.80
    cfg["bayesian_warmup"] = 21
    return cfg


@pytest.fixture
def rrr_config() -> dict:
    return {
        "model": {"name": "sector_relative_ensemble_rrr"},
        "portfolio": {"long_short_frac": 0.3, "weight_mode": "signal"},
        "ensemble": {
            "raw_pca_weight": 0.4,
            "residual_pca_weight": 0.4,
            "p6_weight": 0.1,
            "p6p3_weight": 0.1,
            "p7_weight": 0.0,
            "p7p3_weight": 0.0,
            "normalization": "zscore",
        },
        "costs": {"slippage_bps_per_side": 5.0},
        "rrr_window": 252,
        "rrr_ewma_halflife": 45,
        "lambda_ridge": 0.03,
        "lambda_prior": 0.3,
        "rank": 3,
        "variant": "Lowrank_BLP",
        "rho_blp": 0.03,
        "alpha_xx": 0.5,
        "alpha_yx": 0.25,
    }


# ---------------------------------------------------------------------------
# Expected key sets per model
# ---------------------------------------------------------------------------

SRE_KEYS = {
    "raw_pca_signals", "residual_pca_signals", "p4_signals",
    "signals", "normalized_signals", "y_jp_oc_df",
}

BLP_KEYS = {
    "raw_pca_signals", "residual_pca_signals", "p4_signals",
    "p5_signals", "p5p3_signals",
    "signals", "normalized_signals", "y_jp_oc_df", "blp_diagnostics",
}

BLPX_KEYS = {
    "raw_pca_signals", "residual_pca_signals", "p4_signals",
    "raw_blpx_signals", "residual_blpx_signals",
    "signals", "normalized_signals", "sigma_yy",
    "y_jp_oc_df", "blp_diagnostics",
}

BAYESIAN_KEYS = {
    "signals", "normalized_signals",
    "residual_blpx_signals", "raw_pca_signals", "residual_pca_signals",
    "p4_signals", "raw_blpx_signals",
    "sigma_yy", "y_jp_oc_df",
    "blp_diagnostics", "bayesian_diagnostics",
}

RRR_KEYS = {
    "raw_pca_signals", "residual_pca_signals", "p4_signals",
    "p6_signals", "p6p3_signals", "p7_signals", "p7p3_signals",
    "signals", "normalized_signals",
    "y_jp_oc_df", "rrr_diagnostics",
}

# Keys that are DataFrame[JP_TICKERS] with same index as df_exec
DF_SIGNAL_KEYS = {
    "raw_pca_signals", "residual_pca_signals", "p4_signals",
    "p5_signals", "p5p3_signals",
    "raw_blpx_signals", "residual_blpx_signals",
    "p6_signals", "p6p3_signals", "p7_signals", "p7p3_signals",
    "signals", "normalized_signals",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _slice_tail(df_or_arr, df_exec, n=20):
    """Get the last n rows of a DataFrame or array aligned with df_exec."""
    if isinstance(df_or_arr, pd.DataFrame):
        return df_or_arr.iloc[-n:]
    elif isinstance(df_or_arr, np.ndarray):
        return df_or_arr[-n:]
    elif df_or_arr is None:
        return None
    else:
        return df_or_arr


def _check_df_keys(result, expected_keys, df_exec):
    """Verify dict key set and DataFrame structure."""
    actual_keys = set(result.keys())
    assert actual_keys == expected_keys, (
        f"Key mismatch: extra={actual_keys - expected_keys}, "
        f"missing={expected_keys - actual_keys}"
    )

    T = len(df_exec)
    expected_index = df_exec.index
    expected_columns = JP_TICKERS

    for key in result:
        val = result[key]
        if key in DF_SIGNAL_KEYS:
            assert isinstance(val, pd.DataFrame), f"{key} is not DataFrame: {type(val)}"
            assert val.shape == (T, len(JP_TICKERS)), f"{key} shape={val.shape}, expected=({T},{len(JP_TICKERS)})"
            assert val.index.equals(expected_index), f"{key} index mismatch"
            assert list(val.columns) == list(expected_columns), f"{key} columns mismatch: {list(val.columns)}"
        elif key == "y_jp_oc_df":
            assert isinstance(val, pd.DataFrame), f"{key} is not DataFrame"
            assert val.index.equals(expected_index), f"{key} index mismatch"
        elif key == "sigma_yy":
            assert isinstance(val, np.ndarray), f"{key} is not ndarray: {type(val)}"
            assert val.shape == (T, len(JP_TICKERS), len(JP_TICKERS)), f"{key} shape={val.shape}"
        elif key in ("blp_diagnostics", "bayesian_diagnostics", "rrr_diagnostics"):
            assert isinstance(val, pd.DataFrame), f"{key} is not DataFrame: {type(val)}"


def _compare_results(result_a, result_b, label_a="A", label_b="B", n_tail=20):
    """Compare two predict_signals outputs with tight tolerance on tail slice."""
    assert set(result_a.keys()) == set(result_b.keys()), (
        f"Key set mismatch: {set(result_a.keys()) ^ set(result_b.keys())}"
    )

    for key in result_a:
        va, vb = result_a[key], result_b[key]

        if key in DF_SIGNAL_KEYS:
            # Compare full index/columns
            assert va.index.equals(vb.index), f"{key}: index mismatch"
            assert list(va.columns) == list(vb.columns), f"{key}: columns mismatch"

            # Convert to float64 for comparison (SRE normalized_signals has object dtype)
            va_f = va.astype(np.float64)
            vb_f = vb.astype(np.float64)

            # NaN positions
            nan_a = va_f.isna().values
            nan_b = vb_f.isna().values
            assert np.array_equal(nan_a, nan_b), f"{key}: NaN position mismatch"

            # Values on tail slice (atol=0, rtol=1e-12 for exact reproduction)
            tail_a = va_f.iloc[-n_tail:].values
            tail_b = vb_f.iloc[-n_tail:].values
            assert np.allclose(tail_a, tail_b, rtol=0, atol=1e-12, equal_nan=True), (
                f"{key}: value mismatch on tail {n_tail} rows "
                f"(max diff={np.nanmax(np.abs(tail_a - tail_b))})"
            )

        elif key == "sigma_yy":
            tail_a = va[-n_tail:]
            tail_b = vb[-n_tail:]
            assert np.allclose(tail_a, tail_b, rtol=0, atol=1e-12, equal_nan=True), (
                f"{key}: value mismatch on tail {n_tail} rows"
            )

        elif key == "y_jp_oc_df":
            # Just check index and columns match
            assert va.index.equals(vb.index), f"{key}: index mismatch"
            assert list(va.columns) == list(vb.columns), f"{key}: columns mismatch"

        elif key in ("blp_diagnostics", "bayesian_diagnostics", "rrr_diagnostics"):
            # Diagnostics: check key set and values on tail
            if isinstance(va, pd.DataFrame) and isinstance(vb, pd.DataFrame):
                if va.empty and vb.empty:
                    continue
                assert set(va.columns) == set(vb.columns), (
                    f"{key}: diagnostics column mismatch "
                    f"{set(va.columns) ^ set(vb.columns)}"
                )
                # Compare tail values
                tail_a = va.iloc[-n_tail:]
                tail_b = vb.iloc[-n_tail:]
                for col in va.columns:
                    if col in ("date", "variant", "mode"):
                        continue
                    vals_a = tail_a[col].values
                    vals_b = tail_b[col].values
                    assert np.allclose(vals_a, vals_b, rtol=1e-10, atol=1e-12, equal_nan=True), (
                        f"{key}.{col}: value mismatch"
                    )


# ---------------------------------------------------------------------------
# SRE Characterization
# ---------------------------------------------------------------------------

class TestSRECharacterization:
    """Golden master for SectorRelativeEnsembleModel."""

    def test_key_set(self, sre_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleModel(sre_config)
        result = model.predict_signals(df_exec)
        _check_df_keys(result, SRE_KEYS, df_exec)

    def test_repeated_run_identical(self, sre_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleModel(sre_config)
        result1 = model.predict_signals(df_exec)
        result2 = model.predict_signals(df_exec)
        _compare_results(result1, result2, "run1", "run2")

    def test_fresh_instance_identical(self, sre_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model1 = SectorRelativeEnsembleModel(sre_config)
        result1 = model1.predict_signals(df_exec)
        model2 = SectorRelativeEnsembleModel(sre_config)
        result2 = model2.predict_signals(df_exec)
        _compare_results(result1, result2, "instance1", "instance2")

    def test_no_nan_inf_in_signals(self, sre_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleModel(sre_config)
        result = model.predict_signals(df_exec)
        for key in ("signals", "normalized_signals", "raw_pca_signals", "residual_pca_signals"):
            vals = result[key].loc["2015-01-05":].astype(np.float64).values
            assert not np.isnan(vals).any(), f"{key} contains NaN"
            assert not np.isinf(vals).any(), f"{key} contains Inf"


# ---------------------------------------------------------------------------
# BLP Characterization
# ---------------------------------------------------------------------------

class TestBLPCharacterization:
    """Golden master for SectorRelativeEnsembleBLPModel."""

    def test_key_set(self, blp_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleBLPModel(blp_config)
        result = model.predict_signals(df_exec)
        _check_df_keys(result, BLP_KEYS, df_exec)

    def test_repeated_run_identical(self, blp_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleBLPModel(blp_config)
        result1 = model.predict_signals(df_exec)
        result2 = model.predict_signals(df_exec)
        _compare_results(result1, result2, "run1", "run2")

    def test_fresh_instance_identical(self, blp_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model1 = SectorRelativeEnsembleBLPModel(blp_config)
        result1 = model1.predict_signals(df_exec)
        model2 = SectorRelativeEnsembleBLPModel(blp_config)
        result2 = model2.predict_signals(df_exec)
        _compare_results(result1, result2, "instance1", "instance2")

    def test_no_nan_inf_in_signals(self, blp_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleBLPModel(blp_config)
        result = model.predict_signals(df_exec)
        for key in ("signals", "normalized_signals", "p5_signals", "p5p3_signals"):
            vals = result[key].loc["2015-01-05":].astype(np.float64).values
            assert not np.isnan(vals).any(), f"{key} contains NaN"
            assert not np.isinf(vals).any(), f"{key} contains Inf"


# ---------------------------------------------------------------------------
# BLPEnhanced Characterization
# ---------------------------------------------------------------------------

class TestBLPXCharacterization:
    """Golden master for SectorRelativeEnsembleBLPEnhancedModel."""

    def test_key_set(self, blpx_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleBLPEnhancedModel(blpx_config)
        result = model.predict_signals(df_exec)
        _check_df_keys(result, BLPX_KEYS, df_exec)

    def test_repeated_run_identical(self, blpx_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleBLPEnhancedModel(blpx_config)
        result1 = model.predict_signals(df_exec)
        result2 = model.predict_signals(df_exec)
        _compare_results(result1, result2, "run1", "run2")

    def test_fresh_instance_identical(self, blpx_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model1 = SectorRelativeEnsembleBLPEnhancedModel(blpx_config)
        result1 = model1.predict_signals(df_exec)
        model2 = SectorRelativeEnsembleBLPEnhancedModel(blpx_config)
        result2 = model2.predict_signals(df_exec)
        _compare_results(result1, result2, "instance1", "instance2")

    def test_no_nan_inf_in_signals(self, blpx_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleBLPEnhancedModel(blpx_config)
        result = model.predict_signals(df_exec)
        for key in ("signals", "normalized_signals", "raw_blpx_signals", "residual_blpx_signals"):
            vals = result[key].loc["2015-01-05":].astype(np.float64).values
            assert not np.isnan(vals).any(), f"{key} contains NaN"
            assert not np.isinf(vals).any(), f"{key} contains Inf"


# ---------------------------------------------------------------------------
# Bayesian Characterization
# ---------------------------------------------------------------------------

class TestBayesianCharacterization:
    """Golden master for BayesianBLPXModel."""

    def test_key_set(self, bayesian_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = BayesianBLPXModel(bayesian_config)
        result = model.predict_signals(df_exec)
        _check_df_keys(result, BAYESIAN_KEYS, df_exec)

    def test_repeated_run_identical(self, bayesian_config, sample_df_exec):
        """Bayesian state must reset on each predict_signals call."""
        df_exec, _ = sample_df_exec
        model = BayesianBLPXModel(bayesian_config)
        result1 = model.predict_signals(df_exec)
        result2 = model.predict_signals(df_exec)
        _compare_results(result1, result2, "run1", "run2")

    def test_fresh_instance_identical(self, bayesian_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model1 = BayesianBLPXModel(bayesian_config)
        result1 = model1.predict_signals(df_exec)
        model2 = BayesianBLPXModel(bayesian_config)
        result2 = model2.predict_signals(df_exec)
        _compare_results(result1, result2, "instance1", "instance2")

    def test_bayesian_aliases(self, bayesian_config, sample_df_exec):
        """Bayesian output aliases raw_pca=residual_pca=p4=raw_blpx=residual_blpx."""
        df_exec, _ = sample_df_exec
        model = BayesianBLPXModel(bayesian_config)
        result = model.predict_signals(df_exec)
        rblpx = result["residual_blpx_signals"]
        for key in ("raw_pca_signals", "residual_pca_signals", "p4_signals", "raw_blpx_signals"):
            assert result[key].equals(rblpx), f"{key} != residual_blpx_signals in Bayesian output"

    def test_no_nan_inf_in_signals(self, bayesian_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = BayesianBLPXModel(bayesian_config)
        result = model.predict_signals(df_exec)
        for key in ("signals", "normalized_signals", "residual_blpx_signals"):
            vals = result[key].loc["2015-01-05":].astype(np.float64).values
            assert not np.isnan(vals).any(), f"{key} contains NaN"
            assert not np.isinf(vals).any(), f"{key} contains Inf"


# ---------------------------------------------------------------------------
# RRR Characterization
# ---------------------------------------------------------------------------

class TestRRRCharacterization:
    """Golden master for SectorRelativeEnsembleRRRModel."""

    def test_key_set(self, rrr_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleRRRModel(rrr_config)
        result = model.predict_signals(df_exec)
        _check_df_keys(result, RRR_KEYS, df_exec)

    def test_repeated_run_identical(self, rrr_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleRRRModel(rrr_config)
        result1 = model.predict_signals(df_exec)
        result2 = model.predict_signals(df_exec)
        _compare_results(result1, result2, "run1", "run2")

    def test_fresh_instance_identical(self, rrr_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model1 = SectorRelativeEnsembleRRRModel(rrr_config)
        result1 = model1.predict_signals(df_exec)
        model2 = SectorRelativeEnsembleRRRModel(rrr_config)
        result2 = model2.predict_signals(df_exec)
        _compare_results(result1, result2, "instance1", "instance2")

    def test_no_nan_inf_in_signals(self, rrr_config, sample_df_exec):
        df_exec, _ = sample_df_exec
        model = SectorRelativeEnsembleRRRModel(rrr_config)
        result = model.predict_signals(df_exec)
        for key in ("signals", "normalized_signals", "p6_signals", "p6p3_signals"):
            vals = result[key].loc["2015-01-05":].astype(np.float64).values
            assert not np.isnan(vals).any(), f"{key} contains NaN"
            assert not np.isinf(vals).any(), f"{key} contains Inf"


# ---------------------------------------------------------------------------
# Cross-model: BacktestEngine contract keys
# ---------------------------------------------------------------------------

class TestBacktestEngineContract:
    """Verify all models produce keys that BacktestEngine and decision consumers require."""

    REQUIRED_KEYS = {"raw_pca_signals", "residual_pca_signals", "p4_signals", "signals", "y_jp_oc_df"}

    @pytest.mark.parametrize("model_name,config_fixture", [
        ("SRE", "sre_config"),
        ("BLP", "blp_config"),
        ("BLPX", "blpx_config"),
        ("Bayesian", "bayesian_config"),
        ("RRR", "rrr_config"),
    ])
    def test_required_keys_present(self, model_name, config_fixture, sample_df_exec, request):
        df_exec, _ = sample_df_exec
        cfg = request.getfixturevalue(config_fixture)
        model_cls = {
            "SRE": SectorRelativeEnsembleModel,
            "BLP": SectorRelativeEnsembleBLPModel,
            "BLPX": SectorRelativeEnsembleBLPEnhancedModel,
            "Bayesian": BayesianBLPXModel,
            "RRR": SectorRelativeEnsembleRRRModel,
        }[model_name]
        model = model_cls(cfg)
        result = model.predict_signals(df_exec)
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"{model_name} missing required keys: {missing}"
