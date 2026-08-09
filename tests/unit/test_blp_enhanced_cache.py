"""Regression tests for SectorRelativeEnsembleBLPEnhancedModel per-run caches.

Pins the P1-1 fix (2026-07): ``predict_signals`` must clear ``_raw_pca_cache``,
``_residual_pca_cache`` and ``_blp_corr_cache`` at entry so that a reused model
instance (e.g. walk-forward) cannot read stale results computed from an older
df_exec slice. Also guards the P1-2 fix: universe dimensions must be derived
from ``self.n_u + self.n_j`` instead of hardcoded ``(32, 32)`` / ``(32, 6)``.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import leadlag.core.correlation as corr_mod
from leadlag.data.tickers import SENSITIVITY_LABELS
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)


def _minimal_config() -> dict:
    return {
        "model": {"name": "sector_relative_ensemble_blp_enhanced"},
        "portfolio": {"weight_mode": "signal"},
    }


class TestCacheClearingOnPredictSignals:
    def test_caches_cleared_at_entry(self, monkeypatch):
        """Stale cache entries from a previous run must be wiped at predict_signals entry."""
        model = SectorRelativeEnsembleBLPEnhancedModel(_minimal_config())
        model._raw_pca_cache["stale_key"] = np.ones((model.n_j, model.n_u))
        model._residual_pca_cache["stale_key"] = np.ones((model.n_j, model.n_u))
        model._blp_corr_cache["stale_key"] = (None, None, None)

        class _Sentinel(Exception):
            pass

        def _boom(df_exec):
            raise _Sentinel

        monkeypatch.setattr(model, "_prepare_common_inputs", _boom)

        df_exec = pd.DataFrame(index=pd.DatetimeIndex(["2026-01-05"]))
        with pytest.raises(_Sentinel):
            model.predict_signals(df_exec)

        assert model._raw_pca_cache == {}
        assert model._residual_pca_cache == {}
        assert model._blp_corr_cache == {}

    def test_clearing_happens_before_common_inputs(self, monkeypatch):
        """Cache clearing must precede any signal computation (ordering guard)."""
        model = SectorRelativeEnsembleBLPEnhancedModel(_minimal_config())
        model._blp_corr_cache["stale_key"] = (None, None, None)

        observed_cache_sizes: list[int] = []

        def _spy(df_exec):
            observed_cache_sizes.append(len(model._blp_corr_cache))
            raise RuntimeError("stop after observing")

        monkeypatch.setattr(model, "_prepare_common_inputs", _spy)

        df_exec = pd.DataFrame(index=pd.DatetimeIndex(["2026-01-05"]))
        with pytest.raises(RuntimeError, match="stop after observing"):
            model.predict_signals(df_exec)

        assert observed_cache_sizes == [0]


class TestPcaPriorDerivedDimensions:
    """P1-2: _compute_pca_prior must use n_u + n_j, not hardcoded (32, 32)/(32, 6)."""

    def _valid_inputs(self, model, n: int):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(n, 200))
        corr = np.corrcoef(a)
        v0_static = rng.normal(size=(n, 6))
        c_full = np.eye(n)
        return corr, v0_static, c_full

    def test_pca_prior_computed_with_derived_dims(self):
        model = SectorRelativeEnsembleBLPEnhancedModel(_minimal_config())
        n = model.n_u + model.n_j
        corr, v0_static, c_full = self._valid_inputs(model, n)

        b_pca = model._compute_pca_prior(corr, v0_static, c_full)

        assert b_pca.shape == (model.n_j, model.n_u)
        assert np.any(b_pca != 0.0)

    def test_pca_prior_skipped_on_shape_mismatch(self):
        model = SectorRelativeEnsembleBLPEnhancedModel(_minimal_config())
        n = model.n_u + model.n_j
        corr, v0_static, c_full = self._valid_inputs(model, n)

        b_pca = model._compute_pca_prior(corr, v0_static[:-1], c_full)
        assert np.all(b_pca == 0.0)

        b_pca = model._compute_pca_prior(np.eye(n - 1), v0_static, c_full)
        assert np.all(b_pca == 0.0)


class TestSensitivityLabelUniverseValidation:
    """P1-2: hardcoded sensitivity labels must fail loudly on universe change."""

    def test_labels_match_current_universe(self):
        labels = corr_mod.get_static_sensitivity_labels()
        expected = len(corr_mod.US_TICKERS) + len(corr_mod.JP_TICKERS)
        for name in ("w3", "w4", "w5", "w6"):
            assert labels[name].shape == (expected,)

    def test_labels_raise_on_universe_mismatch(self, monkeypatch):
        # Sensitivity labels now live in the ticker registry. The failure mode
        # is a ticker existing in the universe but missing from SENSITIVITY_LABELS.
        missing_ticker = corr_mod.US_TICKERS[-1]
        pruned_labels = copy.deepcopy(SENSITIVITY_LABELS)
        pruned_labels.pop(missing_ticker, None)
        monkeypatch.setattr(corr_mod, "SENSITIVITY_LABELS", pruned_labels)
        with pytest.raises(ValueError, match="Missing sensitivity labels"):
            corr_mod.get_static_sensitivity_labels()
