"""Production BLPX model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from leadlag.core.macro import MACRO_SENS_MATRIX
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.blp_base import _BLPBase
from leadlag.models.blpx.blp_solver import (
    _apply_confidence_weighting,
    _build_blp_diagnostics,
    _compute_pca_prior,
    _safe_solve_inv,
    _solve_blp_coefficients,
    _solve_tikhonov,
)
from leadlag.models.blpx.correlation import (
    _estimate_correlation,
    _prepare_window_returns,
)
from leadlag.models.blpx.model_meta import BLPXMetaMixin
from leadlag.models.blpx.model_predict import BLPXPredictMixin
from leadlag.models.blpx.prior_builder import (
    _SECTOR_MAPPING_STRUCTURE,
    _build_sector_prior,
    _get_sector_prior,
    _load_macro_returns,
)
from leadlag.models.blpx.signal_computer import compute_blp_signal

logger = logging.getLogger("leadlag.models.blpx")


class ProductionBLPXModel(BLPXPredictMixin, BLPXMetaMixin, _BLPBase):
    """Production BLPX model (migrated from research package)."""

    _config_sections = ["model", "ensemble", "portfolio", "costs", "residualization", "blpx"]
    _config_aliases = {
        "blp_ewma_halflife": ["ewma_halflife"],
        "exec_adjustment": [
            "execution_target_cost_adjustment",
            "execution_target_cost_adjustment_mode",
        ],
    }

    _ZERO_BLP_DIAGNOSTICS: dict[str, Any] = {
        "signal": None,  # set per-call to np.zeros(n_j)
        "cond_num": 0.0,
        "b_norm": 0.0,
        "b_pca_norm": 0.0,
        "b_sector_norm": 0.0,
        "b_struct_norm": 0.0,
        "sigma_xx_trace": 0.0,
        "sigma_yx_norm": 0.0,
        "sigma_yy_trace": 0.0,
        "min_pred_var": 0.0,
        "max_pred_var": 0.0,
        "num_pred_var_floored": 0,
        "pinv_fallback": 0,
        "num_training_samples": 0,
    }

    # Bound helpers from split modules
    _build_sector_prior = _build_sector_prior
    _load_macro_returns = _load_macro_returns
    _get_sector_prior = _get_sector_prior
    _prepare_window_returns = _prepare_window_returns
    _estimate_correlation = _estimate_correlation
    _safe_solve_inv = staticmethod(_safe_solve_inv)
    _solve_blp_coefficients = _solve_blp_coefficients
    _compute_pca_prior = _compute_pca_prior
    _solve_tikhonov = _solve_tikhonov
    _apply_confidence_weighting = staticmethod(_apply_confidence_weighting)
    _build_blp_diagnostics = staticmethod(_build_blp_diagnostics)
    compute_blp_signal = compute_blp_signal
    _SECTOR_MAPPING_STRUCTURE = _SECTOR_MAPPING_STRUCTURE

    def __init__(self, cfg: Any) -> None:
        """Initialize ProductionBLPXModel.

        Args:
            cfg: ``BLPXConfig`` or a dict compatible with ``BLPXConfig`` validation.
        """
        super().__init__(cfg)

        # Universe dimensions (configurable, but defaults to the canonical ticker lists)
        self.n_u = int(getattr(self.cfg, "n_u", len(US_TICKERS)))
        self.n_j = int(getattr(self.cfg, "n_j", len(JP_TICKERS)))

        # Model metadata
        self.model_name = str(getattr(self.cfg, "model_name", "ProductionBLPXModel"))
        self.param_set = str(getattr(self.cfg, "param_set", "default"))

        # PCA / core parameters
        self.k = int(getattr(self.cfg, "k", 6))
        self.q = float(getattr(self.cfg, "q", 0.3))
        self.weight_mode = str(getattr(self.cfg, "weight_mode", "signal"))
        self.normalization_method = str(getattr(self.cfg, "normalization", "zscore"))
        self.rank = getattr(self.cfg, "rank", "full")

        # Window / prior parameters
        self.blp_window = int(getattr(self.cfg, "blp_window", 252))
        self.corr_window = int(getattr(self.cfg, "corr_window", 60))
        self.corr_min_periods = int(getattr(self.cfg, "corr_min_periods", self.corr_window))
        self.beta_window = int(getattr(self.cfg, "beta_window", 60))
        self.beta_floor = float(getattr(self.cfg, "beta_floor", 0.0))
        self.prior_variant = getattr(self.cfg, "prior_variant", None)

        # EWMA / shrinkage
        self.blp_ewma_halflife = float(getattr(self.cfg, "blp_ewma_halflife", self.ewma_halflife))
        self.ewma_half_life = self.ewma_halflife
        self.lambda_reg = float(getattr(self.cfg, "lambda_reg", 0.75))
        self.lambda_lw = float(getattr(self.cfg, "lambda_lw", 0.5))
        self.lw_target = str(getattr(self.cfg, "lw_target", "equicorrelation"))
        self.include_v4_prior = bool(getattr(self.cfg, "include_v4_prior", True))

        # BLP coefficient regularization
        self.rho = float(getattr(self.cfg, "rho", 0.003))
        self.alpha_xx = float(getattr(self.cfg, "alpha_xx", 0.75))
        self.alpha_yx = float(getattr(self.cfg, "alpha_yx", 0.0))
        self.alpha_yy = float(getattr(self.cfg, "alpha_yy", 0.5))

        # Structured prior / Tikhonov
        self.lambda_pca = float(getattr(self.cfg, "lambda_pca", 0.0))
        self.lambda_sector = float(getattr(self.cfg, "lambda_sector", 0.0))
        self.beta_conf = float(getattr(self.cfg, "beta_conf", 0.0))
        self.frobenius_scale_priors = bool(getattr(self.cfg, "frobenius_scale_priors", False))
        self.sector_eta = float(getattr(self.cfg, "sector_eta", 0.0))
        self.sector_gamma = float(getattr(self.cfg, "sector_gamma", 2.0))

        # Robust winsorization
        winsor_val = getattr(self.cfg, "winsor_sigma", None)
        self.winsor_sigma: float | None
        if winsor_val is not None and str(winsor_val).lower() != "none":
            self.winsor_sigma = float(winsor_val)
        else:
            self.winsor_sigma = None

        # Execution / gap adjustment
        self.exec_adjustment = str(getattr(self.cfg, "exec_adjustment", "none"))
        self.gap_open_coef = float(getattr(self.cfg, "gap_open_coef", 0.70))
        self.topix_beta_coef = float(getattr(self.cfg, "topix_beta_coef", 0.6))
        self.vol_adjusted_target = bool(getattr(self.cfg, "vol_adjusted_target", True))
        self.target = str(getattr(self.cfg, "target", "topix_residual"))
        self.use_raw_target = bool(getattr(self.cfg, "use_raw_target", False))

        gap_neg = getattr(self.cfg, "gap_open_coef_neg", None)
        self.gap_open_coef_neg = (
            float(gap_neg) if gap_neg is not None and str(gap_neg).lower() != "none" else None
        )

        beta_neg = getattr(self.cfg, "topix_beta_coef_neg", None)
        self.topix_beta_coef_neg = (
            float(beta_neg) if beta_neg is not None and str(beta_neg).lower() != "none" else None
        )

        # Asymmetric propagation
        self.asymmetry_delta = float(getattr(self.cfg, "asymmetry_delta", 0.0))
        self.asymmetry_mode = str(getattr(self.cfg, "asymmetry_mode", "scalar"))
        self.asymmetry_post_gap_delta = float(getattr(self.cfg, "asymmetry_post_gap_delta", 0.0))
        self.asymmetry_post_gap_mode = str(getattr(self.cfg, "asymmetry_post_gap_mode", "signal_split"))

        # Ensemble / signal component weights
        raw_pca_val = getattr(self.cfg, "raw_pca_weight", None)
        residual_pca_val = getattr(self.cfg, "residual_pca_weight", None)
        raw_blpx_val = getattr(self.cfg, "raw_blpx_weight", None) or getattr(self.cfg, "p5_weight", None)
        residual_blpx_val = getattr(self.cfg, "residual_blpx_weight", None) or getattr(
            self.cfg, "p5p3_weight", None
        )

        if (
            raw_pca_val is not None
            or residual_pca_val is not None
            or raw_blpx_val is not None
            or residual_blpx_val is not None
        ):
            self.raw_pca_weight = float(raw_pca_val) if raw_pca_val is not None else 0.0
            self.residual_pca_weight = float(residual_pca_val) if residual_pca_val is not None else 0.0
            self.raw_blpx_weight = float(raw_blpx_val) if raw_blpx_val is not None else 0.0
            self.residual_blpx_weight = float(residual_blpx_val) if residual_blpx_val is not None else 0.0
        else:
            sig_comps = getattr(self.cfg, "signal_components", None)
            if isinstance(sig_comps, dict):
                self.raw_pca_weight = (
                    float(sig_comps.get("raw_pca", {}).get("weight", 0.0))
                    if sig_comps.get("raw_pca", {}).get("enabled", False)
                    else 0.0
                )
                self.residual_pca_weight = (
                    float(sig_comps.get("residual_pca", {}).get("weight", 0.0))
                    if sig_comps.get("residual_pca", {}).get("enabled", False)
                    else 0.0
                )
                self.raw_blpx_weight = (
                    float(sig_comps.get("raw_blpx", {}).get("weight", 0.0))
                    if sig_comps.get("raw_blpx", {}).get("enabled", False)
                    else 0.0
                )
                self.residual_blpx_weight = (
                    float(sig_comps.get("residual_blpx", {}).get("weight", 0.0))
                    if sig_comps.get("residual_blpx", {}).get("enabled", False)
                    else 0.0
                )
            else:
                self.raw_pca_weight = 0.4
                self.residual_pca_weight = 0.4
                self.raw_blpx_weight = 0.1
                self.residual_blpx_weight = 0.1

        self.p4_weight = float(getattr(self.cfg, "p4_weight", 0.0))

        # Min-variance weight optimization
        self.minvar_enabled = bool(getattr(self.cfg, "minvar_enabled", False))
        self.minvar_alpha = float(getattr(self.cfg, "minvar_alpha", 0.5))

        # Copula correlation blending
        self.copula_enabled = bool(getattr(self.cfg, "copula_enabled", False))
        self.copula_blend_weight = float(getattr(self.cfg, "copula_blend_weight", 0.3))
        self.copula_dynamic_blend = bool(getattr(self.cfg, "copula_dynamic_blend", True))
        self.copula_stress_threshold = float(getattr(self.cfg, "copula_stress_threshold", 1.5))
        self.copula_nu_init = float(getattr(self.cfg, "copula_nu_init", 5.0))
        self.copula_marginal_method = str(getattr(self.cfg, "copula_marginal_method", "empirical"))

        # Macro confidence (Factor-Specific Kappa)
        self.macro_confidence_enabled = bool(getattr(self.cfg, "macro_confidence_enabled", False))
        self.macro_kappa_enabled = bool(
            getattr(self.cfg, "macro_kappa_enabled", self.macro_confidence_enabled)
        )
        self.macro_direction_enabled = bool(getattr(self.cfg, "macro_direction_enabled", False))
        self.macro_sigma_yy_inflation_enabled = bool(
            getattr(self.cfg, "macro_sigma_yy_inflation_enabled", False)
        )
        self.macro_kappas = getattr(self.cfg, "macro_kappas", None)
        if self.macro_kappas is not None and not isinstance(self.macro_kappas, (list, tuple, np.ndarray)):
            self.macro_kappas = None
        if isinstance(self.macro_kappas, (list, tuple)):
            self.macro_kappas = np.array(self.macro_kappas, dtype=float)
        self.macro_surprise_halflife_mean = float(getattr(self.cfg, "macro_surprise_halflife_mean", 20.0))
        self.macro_surprise_halflife_vol = float(getattr(self.cfg, "macro_surprise_halflife_vol", 60.0))
        self._macro_surprise_raw: np.ndarray | None = None
        self._macro_scales: np.ndarray | None = None
        self._macro_direction_adj: np.ndarray | None = None

        # Sensitivity matrix override (for experimentation); defaults to MACRO_SENS_MATRIX
        _sens_override = getattr(self.cfg, "macro_sens_matrix", None)
        if _sens_override == "derived":
            from leadlag.core.macro import MACRO_SENS_MATRIX_DERIVED

            self._macro_sens_matrix = MACRO_SENS_MATRIX_DERIVED
        else:
            self._macro_sens_matrix = MACRO_SENS_MATRIX

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_slippage_bps()

        # Meta-learning parameters
        self.meta_enabled = bool(getattr(self.cfg, "meta_learning_enabled", False))
        self.meta_model_type = str(getattr(self.cfg, "meta_learning_model_type", "logistic_regression"))
        self.meta_train_window = int(getattr(self.cfg, "meta_learning_train_window", 252))
        self.meta_smooth_factor = float(getattr(self.cfg, "meta_learning_smooth_factor", 1.0))

        # Precompute the fixed Sector Mapping matrix M_sector
        self.M_sector = self._build_sector_prior()
        self._M_sector_fixed = self.M_sector.copy()

        # Precompute sector mapping indices to avoid list.index lookups in hot loops
        self._sector_mapping_indices = {}
        for us_tk, jp_tks in self._SECTOR_MAPPING_STRUCTURE.items():
            if us_tk in US_TICKERS:
                u_idx = US_TICKERS.index(us_tk)
                j_indices = []
                for jp_tk in jp_tks:
                    if jp_tk in JP_TICKERS:
                        j_indices.append(JP_TICKERS.index(jp_tk))
                self._sector_mapping_indices[u_idx] = j_indices

        # Per-instance caches aggregated under the CacheManager from _BLPBase.
        self._raw_pca_cache = self._cache_manager.namespace("raw_pca")
        self._residual_pca_cache = self._cache_manager.namespace("residual_pca")
        self._blp_corr_cache = self._cache_manager.namespace("blp_corr")
        self._macro_price_cache = self._cache_manager.namespace("macro_price")
