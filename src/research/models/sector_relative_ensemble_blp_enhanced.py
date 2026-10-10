"""Sector Relative Ensemble with Enhanced Regularized Block BLP (PCA-BLPX Ensemble) Model.

Implements PCA-BLPX Ensemble which integrates standard Production signals (Raw-PCA, Residual-PCA)
with Enhanced Regularized Block BLP signals (Raw-BLPX, Residual-BLPX) incorporating structured shrinkage,
conditional confidence weighting, winsorized robust covariance, and execution cost adjustment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from leadlag.config import safe_config_copy
from leadlag.core.blpx_math import (
    BLPXSignalParameters,
    build_fixed_sector_prior,
    compute_blp_signal_math,
    compute_sector_prior,
    solve_blp_coefficients,
)
from leadlag.core.correlation import (
    compute_correlation,
    compute_stress_weight,
)
from leadlag.core.macro import (
    MACRO_SENS_MATRIX,
    compute_factor_kappa_scale,
    compute_macro_surprise,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS, US_TO_JP_SECTOR_MAPPING
from research.models.blp_base import _BLPBase

logger = logging.getLogger(__name__)

# Module-level caches removed — moved to per-instance attributes for thread safety


class SectorRelativeEnsembleBLPEnhancedModel(_BLPBase):
    """Sector Relative Ensemble with Enhanced Regularized Block BLP (PCA-BLPX Ensemble) Model."""

    _config_sections = ["model", "ensemble", "portfolio", "costs", "residualization", "blpx"]
    _config_aliases = {
        "blp_ewma_halflife": ["ewma_halflife"],
        "exec_adjustment": ["execution_target_cost_adjustment", "execution_target_cost_adjustment_mode"],
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

    def __init__(self, config: dict | object):
        """Initialize SectorRelativeEnsembleBLPEnhancedModel.

        Args:
            config: Dict or object containing configuration options.
        """
        self.config = safe_config_copy(config)
        self.n_u = len(US_TICKERS)
        self.n_j = len(JP_TICKERS)

        # Config resolution
        self.model_name = self._resolve_val("model_name", "sector_relative_ensemble_blp_enhanced")
        self.k = self._resolve_val("k", 6)
        self.lambda_reg = self._resolve_val("lambda_reg", 0.75)
        self.q = self._resolve_val("q", 0.3)
        self.weight_mode = self._resolve_val("weight_mode", "signal")
        self.ewma_half_life = self._resolve_val("ewma_half_life", 45)
        self.lambda_lw = self._resolve_val("lambda_lw", 0.5)
        self.lw_target = self._resolve_val("lw_target", "equicorrelation")
        self.corr_window = self._resolve_val("corr_window", 60)
        self.include_v4_prior = self._resolve_val("include_v4_prior", True)
        self.gap_open_coef = self._resolve_val("gap_open_coef", 0.70)
        self.topix_beta_coef = self._resolve_val("topix_beta_coef", 0.6)
        self.beta_window = self._resolve_val("beta_window", 60)
        self.vol_adjusted_target = self._resolve_val("vol_adjusted_target", True)
        self.min_raw_weight = self._resolve_val("min_raw_weight", 0.0)
        self.normalization_method = self._resolve_val("normalization", "zscore")

        # BLP Parameters
        self.blp_window = int(self._resolve_val("blp_window", 252))
        self.blp_ewma_halflife = self._resolve_val("blp_ewma_halflife", 45)
        self.alpha_xx = float(self._resolve_val("alpha_xx", 0.75))
        self.alpha_yx = float(self._resolve_val("alpha_yx", 0.0))
        self.rho = float(self._resolve_val("rho", 0.003))
        self.rank = self._resolve_val("rank", "full")

        # Enhanced BLP variant parameters
        self.alpha_yy = float(self._resolve_val("alpha_yy", 0.5))
        self.lambda_pca = float(self._resolve_val("lambda_pca", 0.0))
        self.lambda_sector = float(self._resolve_val("lambda_sector", 0.0))
        self.beta_conf = float(self._resolve_val("beta_conf", 0.0))
        self.frobenius_scale_priors = bool(self._resolve_val("frobenius_scale_priors", False))

        winsor_val = self._resolve_val("winsor_sigma", None)
        if winsor_val is not None and str(winsor_val).lower() != "none":
            self.winsor_sigma = float(winsor_val)
        else:
            self.winsor_sigma = None

        self.exec_adjustment = self._resolve_val("exec_adjustment", "none")

        # Ensemble weights
        # Support both direct ensemble parameters (common in tests/legacy config) and signal_components (production yaml config)
        raw_pca_val = self._resolve_val("raw_pca_weight", None)
        residual_pca_val = self._resolve_val("residual_pca_weight", None)
        raw_blpx_val = self._resolve_val("raw_blpx_weight", None) or self._resolve_val("p5_weight", None)
        residual_blpx_val = self._resolve_val("residual_blpx_weight", None) or self._resolve_val("p5p3_weight", None)

        if raw_pca_val is not None or residual_pca_val is not None or raw_blpx_val is not None or residual_blpx_val is not None:
            self.raw_pca_weight = float(raw_pca_val) if raw_pca_val is not None else 0.0
            self.residual_pca_weight = float(residual_pca_val) if residual_pca_val is not None else 0.0
            self.raw_blpx_weight = float(raw_blpx_val) if raw_blpx_val is not None else 0.0
            self.residual_blpx_weight = float(residual_blpx_val) if residual_blpx_val is not None else 0.0
        else:
            sig_comps = self._resolve_val("signal_components", None)
            if isinstance(sig_comps, dict):
                self.raw_pca_weight = float(sig_comps.get("raw_pca", {}).get("weight", 0.0)) if sig_comps.get("raw_pca", {}).get("enabled", False) else 0.0
                self.residual_pca_weight = float(sig_comps.get("residual_pca", {}).get("weight", 0.0)) if sig_comps.get("residual_pca", {}).get("enabled", False) else 0.0
                self.raw_blpx_weight = float(sig_comps.get("raw_blpx", {}).get("weight", 0.0)) if sig_comps.get("raw_blpx", {}).get("enabled", False) else 0.0
                self.residual_blpx_weight = float(sig_comps.get("residual_blpx", {}).get("weight", 0.0)) if sig_comps.get("residual_blpx", {}).get("enabled", False) else 0.0
            else:
                self.raw_pca_weight = 0.4
                self.residual_pca_weight = 0.4
                self.raw_blpx_weight = 0.1
                self.residual_blpx_weight = 0.1

        # Continuous M_sector parameters
        self.sector_eta = float(self._resolve_val("sector_eta", 0.0))
        self.sector_gamma = float(self._resolve_val("sector_gamma", 2.0))

        # Precompute the fixed Sector Mapping matrix M_sector
        self.M_sector = build_fixed_sector_prior(
            self._SECTOR_MAPPING_STRUCTURE,
            US_TICKERS,
            JP_TICKERS,
            n_u=self.n_u,
            n_j=self.n_j,
        )
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

        # Copula parameters
        self.copula_enabled = bool(self._resolve_val("copula_enabled", False))
        self.copula_blend_weight = float(self._resolve_val("copula_blend_weight", 0.3))
        self.copula_dynamic_blend = bool(self._resolve_val("copula_dynamic_blend", True))
        self.copula_stress_threshold = float(self._resolve_val("copula_stress_threshold", 1.5))
        self.copula_nu_init = float(self._resolve_val("copula_nu_init", 5.0))
        self.copula_marginal_method = str(self._resolve_val("copula_marginal_method", "empirical"))

        # Covariance-aware weight optimization
        self.minvar_enabled = bool(self._resolve_val("minvar_enabled", False))
        self.minvar_alpha = float(self._resolve_val("minvar_alpha", 0.5))

        # Macro confidence (Factor-Specific Kappa) parameters
        self.macro_confidence_enabled = bool(self._resolve_val("macro_confidence_enabled", False))
        self.macro_kappas = self._resolve_val("macro_kappas", None)
        if self.macro_kappas is not None and not isinstance(self.macro_kappas, (list, tuple, np.ndarray)):
            self.macro_kappas = None
        if isinstance(self.macro_kappas, (list, tuple)):
            self.macro_kappas = np.array(self.macro_kappas, dtype=float)
        self.macro_surprise_halflife_mean = float(self._resolve_val("macro_surprise_halflife_mean", 20.0))
        self.macro_surprise_halflife_vol = float(self._resolve_val("macro_surprise_halflife_vol", 60.0))
        self._macro_surprise_raw: np.ndarray | None = None
        self._macro_scales: np.ndarray | None = None

        # Per-instance caches (replaces module-level globals)
        self._raw_pca_cache: dict = {}
        self._residual_pca_cache: dict = {}
        self._blp_corr_cache: dict = {}

        # Extension A: Directional adjustment (signed surprise * signed sensitivity)
        self.macro_direction_enabled = bool(self._resolve_val("macro_direction_enabled", False))
        self._macro_direction_adj: np.ndarray | None = None

        # Extension E: Sigma_YY inflation (adjust predictive covariance)
        self.macro_sigma_yy_inflation_enabled = bool(self._resolve_val("macro_sigma_yy_inflation_enabled", False))

        # Sensitivity matrix override (for experimentation); defaults to MACRO_SENS_MATRIX
        _sens_override = self._resolve_val("macro_sens_matrix", None)
        if _sens_override == "derived":
            from leadlag.core.macro import MACRO_SENS_MATRIX_DERIVED
            self._macro_sens_matrix = MACRO_SENS_MATRIX_DERIVED
        else:
            self._macro_sens_matrix = MACRO_SENS_MATRIX

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_slippage_bps()

        # Asymmetric propagation parameters
        self.asymmetry_delta = float(self._resolve_val("asymmetry_delta", 0.0))
        self.asymmetry_mode = str(self._resolve_val("asymmetry_mode", "scalar"))

        gap_neg = self._resolve_val("gap_open_coef_neg", None)
        self.gap_open_coef_neg = float(gap_neg) if gap_neg is not None and str(gap_neg).lower() != "none" else None

        beta_neg = self._resolve_val("topix_beta_coef_neg", None)
        self.topix_beta_coef_neg = float(beta_neg) if beta_neg is not None and str(beta_neg).lower() != "none" else None

        self.asymmetry_post_gap_delta = float(self._resolve_val("asymmetry_post_gap_delta", 0.0))
        self.asymmetry_post_gap_mode = str(self._resolve_val("asymmetry_post_gap_mode", "signal_split"))

        # Meta-learning parameters
        self.meta_enabled = bool(self._resolve_val("meta_learning_enabled", False))
        self.meta_model_type = str(self._resolve_val("meta_learning_model_type", "logistic_regression"))
        self.meta_train_window = int(self._resolve_val("meta_learning_train_window", 252))
        self.meta_smooth_factor = float(self._resolve_val("meta_learning_smooth_factor", 1.0))

    def _load_macro_returns(self, df_exec: pd.DataFrame) -> pd.DataFrame | None:
        """Load macro factor returns aligned to df_exec index.

        Downloads macro close prices (USDJPY, CLF, TNX) via yfinance,
        aligns them to the trading dates in df_exec with forward-fill,
        then computes daily returns. This ensures that non-trading days
        (e.g. JP market open but US market closed) produce zero returns
        rather than carrying forward the previous day's return.

        If download fails or the resulting data is too short, returns None.
        """
        try:
            from leadlag.core.macro import MACRO_NAMES
            from leadlag.data import macro as macro_data

            sim_dates = df_exec.index
            start = sim_dates[0].strftime("%Y-%m-%d")
            end = sim_dates[-1].strftime("%Y-%m-%d")

            close_prices = macro_data.load_macro_prices(start=start, end=end)
            if close_prices is None or len(close_prices) < 30:
                logger.warning("Macro data too short (%d rows); skipping.", len(close_prices) if close_prices is not None else 0)
                return None

            # Align prices to df_exec dates, forward-fill missing values
            prices_aligned = close_prices.reindex(sim_dates, method="ffill")
            prices_aligned = prices_aligned.ffill().fillna(0.0)

            # Compute returns AFTER alignment so non-trading days get zero return
            macro_returns = prices_aligned.pct_change()
            macro_returns = macro_returns.replace([np.inf, -np.inf], np.nan)
            macro_returns = macro_returns.fillna(0.0)
            return macro_returns[MACRO_NAMES]
        except Exception as e:
            logger.warning("Failed to load macro data: %s", e)
            return None

    # Keep the prior mapping overrideable per model while sourcing it once.
    _SECTOR_MAPPING_STRUCTURE = US_TO_JP_SECTOR_MAPPING

    def _get_sector_prior(
        self,
        current_index: int,
        all_returns: np.ndarray,
        corr: np.ndarray,
        B_blp: np.ndarray,
    ) -> np.ndarray:
        """Apply research model settings through the shared sector calculation."""
        return compute_sector_prior(
            corr,
            B_blp,
            self.M_sector,
            self._M_sector_fixed,
            self._sector_mapping_indices,
            n_u=self.n_u,
            n_j=self.n_j,
            sector_eta=self.sector_eta,
            sector_gamma=self.sector_gamma,
        )

    def _prepare_window_returns(
        self, all_returns: np.ndarray, current_index: int, rolling_std: np.ndarray | None
    ) -> np.ndarray:
        """Slice window returns, apply vol-scaling and winsorization."""
        window_start = max(0, current_index - self.blp_window)
        window_returns = all_returns[window_start:current_index].copy()

        if self.exec_adjustment == "vol_scale" and rolling_std is not None:
            vol_factors = rolling_std[window_start:current_index]
            window_returns[:, self.n_u:] /= vol_factors

        complete_rows = np.isfinite(window_returns[:, self.n_u :]).all(axis=1)
        window_returns = window_returns[complete_rows]
        if window_returns.shape[0] == 0:
            raise ValueError("No complete finite rows in BLPX training window")
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        if self.winsor_sigma is not None:
            mus = np.mean(window_returns, axis=0)
            stds = np.std(window_returns, axis=0)
            for c in range(window_returns.shape[1]):
                if stds[c] > 1e-8:
                    window_returns[:, c] = np.clip(
                        window_returns[:, c],
                        mus[c] - self.winsor_sigma * stds[c],
                        mus[c] + self.winsor_sigma * stds[c],
                    )
        return window_returns

    def _estimate_correlation(
        self, window_returns: np.ndarray, current_index: int, is_residual: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate rolling mean, std, and correlation with caching.

        When copula_enabled is True, the Pearson correlation is blended with
        a t-copula correlation matrix. The blend weight is either fixed
        (copula_dynamic_blend=False) or dynamically increased during stress
        periods (copula_dynamic_blend=True).
        """
        cache_key = (current_index, self.blp_window, self.winsor_sigma, self.exec_adjustment, self.blp_ewma_halflife, is_residual, self.copula_enabled)
        if cache_key in self._blp_corr_cache:
            return self._blp_corr_cache[cache_key]

        use_copula = False
        copula_weight = 0.0

        if self.copula_enabled and self.copula_blend_weight > 0.0:
            if self.copula_dynamic_blend:
                w_stress = compute_stress_weight(
                    window_returns,
                    threshold=self.copula_stress_threshold,
                )
                copula_weight = self.copula_blend_weight * w_stress
            else:
                copula_weight = self.copula_blend_weight

            if copula_weight > 0.05:
                use_copula = True

        mu, sigma, corr = compute_correlation(
            window_returns,
            self.blp_ewma_halflife,
            use_copula=use_copula,
            copula_blend_weight=copula_weight,
            copula_nu_init=self.copula_nu_init,
            use_cache=False,
        )
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)
        self._blp_corr_cache[cache_key] = (mu, sigma, corr)
        return mu, sigma, corr


    def _estimate_asymmetric_covariance(
        self, window_returns: np.ndarray, corr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate asymmetric covariance/correlation matrices based on US market factor sign.

        Returns:
            C_YX_pos, C_YX_neg, C_XX, C_YY
        """
        C_XX = corr[:self.n_u, :self.n_u]
        C_YY = corr[self.n_u:, self.n_u:]

        # US market factor: average of all US assets (first n_u columns)
        us_factor = np.mean(window_returns[:, :self.n_u], axis=1)
        pos_mask = us_factor >= 0.0
        neg_mask = us_factor < 0.0

        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)

        if n_pos < 30 or n_neg < 30:
            C_YX = corr[self.n_u:, :self.n_u]
            return C_YX.copy(), C_YX.copy(), C_XX, C_YY

        from leadlag.core.correlation import compute_correlation

        try:
            _, _, corr_pos = compute_correlation(
                window_returns[pos_mask], self.blp_ewma_halflife
            )
            C_YX_pos = corr_pos[self.n_u:, :self.n_u]
        except Exception as e:
            logger.warning(f"Failed to compute positive subset correlation: {e}. Falling back.")
            C_YX_pos = corr[self.n_u:, :self.n_u].copy()

        try:
            _, _, corr_neg = compute_correlation(
                window_returns[neg_mask], self.blp_ewma_halflife
            )
            C_YX_neg = corr_neg[self.n_u:, :self.n_u]
        except Exception as e:
            logger.warning(f"Failed to compute negative subset correlation: {e}. Falling back.")
            C_YX_neg = corr[self.n_u:, :self.n_u].copy()

        return C_YX_pos, C_YX_neg, C_XX, C_YY


    def compute_blp_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
        rolling_std: np.ndarray | None = None,
        v0_static: np.ndarray | None = None,
        c_full: np.ndarray | None = None,
        is_residual: bool = False,
        return_matrices: bool = False,
    ) -> dict[str, Any]:
        """Prepare model-specific statistics and delegate BLPX math."""
        window_returns = self._prepare_window_returns(all_returns, current_index, rolling_std)
        mu, sigma, corr = self._estimate_correlation(window_returns, current_index, is_residual)

        B_blp, Sigma_XX_reg, Sigma_YX_reg, Sigma_YY_reg, cond_num, pinv_fallback = (
            solve_blp_coefficients(
                corr,
                alpha_xx=self.alpha_xx,
                alpha_yx=self.alpha_yx,
                alpha_yy=self.alpha_yy,
                n_j=self.n_j,
                n_u=self.n_u,
                rank=self.rank,
                rho=self.rho,
            )
        )
        M_sector = self._get_sector_prior(current_index, all_returns, corr, B_blp)

        asymmetric_covariance = None
        if self.asymmetry_mode == "covariance":
            asymmetric_covariance = self._estimate_asymmetric_covariance(window_returns, corr)

        parameters = BLPXSignalParameters(
            n_u=self.n_u,
            n_j=self.n_j,
            alpha_xx=self.alpha_xx,
            alpha_yx=self.alpha_yx,
            alpha_yy=self.alpha_yy,
            rank=self.rank,
            rho=self.rho,
            k=self.k,
            lambda_lw=self.lambda_lw,
            lambda_reg=self.lambda_reg,
            lw_target=self.lw_target,
            min_raw_weight=getattr(self, "min_raw_weight", 0.0),
            frobenius_scale_priors=self.frobenius_scale_priors,
            lambda_pca=self.lambda_pca,
            lambda_sector=self.lambda_sector,
            asymmetry_delta=self.asymmetry_delta,
            asymmetry_mode=self.asymmetry_mode,
            beta_conf=self.beta_conf,
            vol_adjusted_target=self.vol_adjusted_target,
            gap_open_coef=self.gap_open_coef,
            topix_beta_coef=self.topix_beta_coef,
            gap_open_coef_neg=self.gap_open_coef_neg,
            topix_beta_coef_neg=self.topix_beta_coef_neg,
            asymmetry_post_gap_delta=self.asymmetry_post_gap_delta,
            asymmetry_post_gap_mode=self.asymmetry_post_gap_mode,
        )
        return compute_blp_signal_math(
            all_returns=all_returns,
            current_index=current_index,
            window_returns=window_returns,
            mu=mu,
            sigma=sigma,
            corr=corr,
            v0_static=v0_static,
            c_full=c_full,
            B_blp=B_blp,
            Sigma_XX_reg=Sigma_XX_reg,
            Sigma_YX_reg=Sigma_YX_reg,
            Sigma_YY_reg=Sigma_YY_reg,
            cond_num=cond_num,
            pinv_fallback=pinv_fallback,
            M_sector=M_sector,
            gap_override=gap_override,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            parameters=parameters,
            asymmetric_covariance=asymmetric_covariance,
            return_matrices=return_matrices,
        )

    def combine_signals(
        self, z0: np.ndarray, z3: np.ndarray, z_raw_blpx: np.ndarray, z_residual_blpx: np.ndarray
    ) -> np.ndarray:
        """Combine component signals with ensemble weights."""
        return (
            self.raw_pca_weight * z0
            + self.residual_pca_weight * z3
            + self.raw_blpx_weight * z_raw_blpx
            + self.residual_blpx_weight * z_residual_blpx
        )

    def _predict_meta_weight(
        self,
        i: int,
        us_dispersions: list[float],
        cond_nums: list[float],
        vix_vals: list[float],
        ic_blpx_vals: list[float],
        ic_pca_vals: list[float],
    ) -> float:
        """Fit a low-capacity meta-model and predict tomorrow's ensemble weight w_t.

        Uses expanding/rolling window up to day i-1.
        """
        # We need at least some minimum number of samples to train, say 100 days
        min_samples = 100

        # Let's collect training features and targets
        X_train = []
        Y_train = []

        start_train = max(self.corr_window + 10, i - self.meta_train_window)
        for j in range(start_train, i):
            if j - 10 < 0 or j >= len(ic_blpx_vals):
                continue
            rec_ic_blpx = np.nanmean(ic_blpx_vals[j - 10 : j])
            rec_ic_pca = np.nanmean(ic_pca_vals[j - 10 : j])

            if np.isnan(rec_ic_blpx) or np.isnan(rec_ic_pca):
                continue

            f_j = [
                us_dispersions[j],
                cond_nums[j],
                vix_vals[j],
                rec_ic_blpx,
                rec_ic_pca,
            ]

            target_j = ic_blpx_vals[j] - ic_pca_vals[j]
            if np.isnan(target_j):
                continue

            X_train.append(f_j)
            Y_train.append(target_j)

        if len(X_train) < min_samples:
            return 0.8  # Fallback to static weight

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Current features at day i (to predict for tomorrow)
        rec_ic_blpx_i = np.nanmean(ic_blpx_vals[i - 10 : i])
        rec_ic_pca_i = np.nanmean(ic_pca_vals[i - 10 : i])
        F_i = np.array([[
            us_dispersions[i],
            cond_nums[i],
            vix_vals[i],
            rec_ic_blpx_i,
            rec_ic_pca_i,
        ]])

        if np.isnan(F_i).any():
            return 0.8

        try:
            if self.meta_model_type == "logistic_regression":
                # Binary target: 1 if blpx outperformed, 0 otherwise
                Y_train_bin = (Y_train > 0).astype(int)
                if len(np.unique(Y_train_bin)) < 2:
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, Y_train)
                    y_pred = float(model.predict(F_i)[0])
                    w_t = np.clip(0.8 + y_pred, 0.6, 1.0)
                else:
                    model = LogisticRegression(C=1.0, solver="liblinear")
                    model.fit(X_train, Y_train_bin)
                    prob = float(model.predict_proba(F_i)[0, 1])
                    w_t = 0.6 + 0.4 * prob
            else:
                # Default: Ridge Regression
                model = Ridge(alpha=1.0)
                model.fit(X_train, Y_train)
                y_pred = float(model.predict(F_i)[0])
                w_t = np.clip(0.8 + y_pred, 0.6, 1.0)
        except Exception as e:
            logger.warning(f"Meta-model training failed at index {i}: {e}. Using static weight 0.8.")
            w_t = 0.8

        return w_t

    def predict_signals(self, df_exec: pd.DataFrame, n_jobs: int = 1) -> dict[str, Any]:
        """Generate component and ensemble signals for all rows in df_exec."""
        from leadlag.core.pipeline import (
            CallableComponent,
            CommonInputs,
            SignalPipeline,
        )
        from leadlag.core.pipeline_blpx import (
            BLPXCombiner,
            BLPXOutputAdapter,
        )

        # Clear per-run signal caches to prevent cross-run contamination
        self._raw_pca_cache.clear()
        self._residual_pca_cache.clear()
        self._blp_corr_cache.clear()

        T = len(df_exec)
        sim_dates = df_exec.index

        inputs = self._prepare_common_inputs(df_exec)
        all_returns_raw = inputs["all_returns_raw"]
        c_full = inputs["c_full"]
        c_full_p3 = inputs["c_full_p3"]
        v0_static = inputs["v0_static"]
        v1 = inputs["v1"]
        v2 = inputs["v2"]
        jp_gap = inputs["jp_gap"]
        jp_beta = inputs["jp_beta"]
        topix_night = inputs["topix_night"]
        jp_res_returns_p3 = inputs["jp_res_returns_p3"]
        y_jp_target = inputs["y_jp_target"]

        # Precompute target standard deviations if target cost adjustment is vol_scale
        rolling_std = None
        if self.exec_adjustment == "vol_scale":
            df_y = pd.DataFrame(y_jp_target)
            rolling_std = df_y.rolling(20).std(ddof=1).values
            overall_std = np.std(y_jp_target, axis=0, ddof=1)
            overall_std = np.maximum(overall_std, 1e-8)
            for col_idx in range(self.n_j):
                nan_mask = np.isnan(rolling_std[:, col_idx])
                rolling_std[nan_mask, col_idx] = overall_std[col_idx]
            rolling_std = np.maximum(rolling_std, 1e-8)

        # Precompute macro confidence scales if enabled
        if self.macro_confidence_enabled and self.macro_kappas is not None:
            macro_returns = self._load_macro_returns(df_exec)
            if macro_returns is not None:
                surprise_raw = compute_macro_surprise(
                    macro_returns,
                    halflife_mean=self.macro_surprise_halflife_mean,
                    halflife_vol=self.macro_surprise_halflife_vol,
                )
                self._macro_surprise_raw = surprise_raw
                self._macro_scales = compute_factor_kappa_scale(
                    surprise_raw, self.macro_kappas, self._macro_sens_matrix,
                )
                if self.macro_direction_enabled:
                    from leadlag.core.macro import compute_macro_direction_adjustment
                    self._macro_direction_adj = compute_macro_direction_adjustment(
                        surprise_raw, self.macro_kappas, self._macro_sens_matrix,
                    )
                logger.info(
                    "Macro confidence enabled: kappas=%s, halflife_mean=%.1f, halflife_vol=%.1f, "
                    "direction=%s, sigma_yy_inflation=%s",
                    self.macro_kappas.tolist(),
                    self.macro_surprise_halflife_mean,
                    self.macro_surprise_halflife_vol,
                    self.macro_direction_enabled,
                    self.macro_sigma_yy_inflation_enabled,
                )
            else:
                logger.warning("Macro confidence enabled but macro data unavailable; skipping.")
                self.macro_confidence_enabled = False

        # Load VIX if meta-learning is enabled
        vix_series = None
        if self.meta_enabled:
            from pathlib import Path
            macro_path = Path(__file__).resolve().parents[3] / "market_data" / "macro_data.pkl"
            if macro_path.exists():
                try:
                    macro_df = pd.read_pickle(macro_path)
                    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
                    vix_series = macro_df["^VIX"].reindex(sim_dates).ffill()
                    vix_series = vix_series.bfill()
                    logger.info("Successfully loaded VIX from macro_data.pkl for meta-learning.")
                except Exception as e:
                    logger.warning(f"Failed to load VIX from macro_data.pkl: {e}")
            if vix_series is None:
                vix_series = pd.Series(20.0, index=sim_dates)

        # Optimize: skip loop iterations before warmup if _start_date is specified
        start_date_str = getattr(self, "_start_date", None)
        if start_date_str is not None:
            start_dt = pd.to_datetime(start_date_str)
            start_idx_raw = df_exec.index.searchsorted(start_dt)
            start_idx = max(self.corr_window, start_idx_raw - self.blp_window)
        else:
            start_idx = self.corr_window
            start_idx_raw = self.corr_window

        # Determine which components to compute (skip zero-weight for speed)
        need_raw_pca = (self.raw_pca_weight > 0.0) or self.meta_enabled
        need_residual_pca = self.residual_pca_weight > 0.0
        need_raw_blpx = (self.raw_blpx_weight > 0.0) or self.meta_enabled
        need_residual_blpx = self.residual_blpx_weight > 0.0

        cache_key = (
            len(df_exec),
            df_exec.index[0],
            df_exec.index[-1],
            self.corr_window,
            self.k,
            self.lambda_reg,
            self.ewma_half_life,
            self.lambda_lw,
            self.lw_target,
            self.gap_open_coef,
            self.topix_beta_coef,
            self.vol_adjusted_target,
        )

        # PCA caching
        raw_pca_cached = False
        raw_pca_cache_arr = None
        if need_raw_pca:
            if cache_key in self._raw_pca_cache:
                raw_pca_cache_arr = self._raw_pca_cache[cache_key]
                raw_pca_cached = True
        residual_pca_cached = False
        residual_pca_cache_arr = None
        if need_residual_pca:
            if cache_key in self._residual_pca_cache:
                residual_pca_cache_arr = self._residual_pca_cache[cache_key]
                residual_pca_cached = True

        # Build CommonInputs
        common_inputs = CommonInputs(
            all_returns_raw=all_returns_raw,
            c_full=c_full,
            c_full_p3=c_full_p3,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            jp_gap=jp_gap,
            jp_beta=jp_beta,
            topix_night=topix_night,
            y_jp_oc_df=inputs["y_jp_oc_df"],
            jp_res_returns_p3=jp_res_returns_p3,
            y_jp_target=y_jp_target,
            n_u=self.n_u,
            n_j=self.n_j,
            dates=sim_dates,
            p4=None,
        )

        # Build component closures
        def _raw_pca_fn(ctx):
            i = ctx.i
            if not need_raw_pca:
                return {"signal": np.zeros(self.n_j)}
            if raw_pca_cached and raw_pca_cache_arr is not None:
                return {"signal": raw_pca_cache_arr[i]}
            inp = ctx.inputs
            sig = self.compute_production_signal(
                i, inp.c_full, inp.v0_static, inp.v1, inp.v2,
                inp.all_returns_raw, inp.jp_gap, inp.jp_beta, inp.topix_night,
            )
            return {"signal": sig}

        def _residual_pca_fn(ctx):
            i = ctx.i
            if not need_residual_pca:
                return {"signal": np.zeros(self.n_j)}
            if residual_pca_cached and residual_pca_cache_arr is not None:
                return {"signal": residual_pca_cache_arr[i]}
            inp = ctx.inputs
            sig = self.compute_residual_signal(
                inp.jp_res_returns_p3, i, inp.c_full_p3, inp.v0_static, inp.v1, inp.v2,
                inp.jp_gap, inp.jp_beta, inp.topix_night,
            )
            return {"signal": sig}

        def _raw_blpx_fn(ctx):
            i = ctx.i
            if not need_raw_blpx or i < start_idx_raw:
                return {**self._ZERO_BLP_DIAGNOSTICS, "signal": np.zeros(self.n_j)}
            inp = ctx.inputs
            gap_override = np.nan_to_num(inp.jp_gap[i], nan=0.0) if inp.jp_gap is not None else None
            betas_t = np.asarray(inp.jp_beta[i], dtype=float) if inp.jp_beta is not None else None
            topix_night_t = float(inp.topix_night[i]) if inp.topix_night is not None else None
            return self.compute_blp_signal(
                inp.all_returns_raw, i,
                gap_override=gap_override, betas_t=betas_t, topix_night_t=topix_night_t,
                rolling_std=rolling_std, v0_static=inp.v0_static, c_full=inp.c_full,
                is_residual=False,
            )

        def _residual_blpx_fn(ctx):
            i = ctx.i
            if not need_residual_blpx or i < start_idx_raw:
                return {**self._ZERO_BLP_DIAGNOSTICS, "signal": np.zeros(self.n_j)}
            inp = ctx.inputs
            gap_override = np.nan_to_num(inp.jp_gap[i], nan=0.0) if inp.jp_gap is not None else None
            betas_t = np.asarray(inp.jp_beta[i], dtype=float) if inp.jp_beta is not None else None
            topix_night_t = float(inp.topix_night[i]) if inp.topix_night is not None else None
            result = self.compute_blp_signal(
                inp.jp_res_returns_p3, i,
                gap_override=gap_override, betas_t=betas_t, topix_night_t=topix_night_t,
                rolling_std=rolling_std, v0_static=inp.v0_static, c_full=inp.c_full_p3,
                is_residual=True,
            )
            if self.minvar_enabled and "sigma_Y_cov" in result:
                sigma_yy_array[i] = result["sigma_Y_cov"]
            return result

        components = [
            CallableComponent("raw_pca", _raw_pca_fn),
            CallableComponent("residual_pca", _residual_pca_fn),
            CallableComponent("raw_blpx", _raw_blpx_fn),
            CallableComponent("residual_blpx", _residual_blpx_fn),
        ]

        # Prepare signal arrays for IC tracking
        raw_pca_signals_arr = np.zeros((T, self.n_j))
        raw_blpx_signals_arr = np.zeros((T, self.n_j))
        sigma_yy_array = np.zeros((T, self.n_j, self.n_j))

        combiner = BLPXCombiner(
            raw_pca_weight=self.raw_pca_weight,
            residual_pca_weight=self.residual_pca_weight,
            raw_blpx_weight=self.raw_blpx_weight,
            residual_blpx_weight=self.residual_blpx_weight,
            normalization_method=self.normalization_method,
            n_j=self.n_j,
            n_u=self.n_u,
            normalize_fn=self.normalize_signals,
            meta_enabled=self.meta_enabled,
            meta_train_window=self.meta_train_window,
            meta_smooth_factor=self.meta_smooth_factor,
            corr_window=self.corr_window,
            meta_predict_fn=self._predict_meta_weight if self.meta_enabled else None,
            macro_confidence_enabled=self.macro_confidence_enabled,
            macro_scales=self._macro_scales,
            macro_direction_adj=self._macro_direction_adj,
            vix_series=vix_series,
            y_jp_target=y_jp_target,
            all_returns_raw=all_returns_raw,
        )
        combiner._raw_pca_signals = raw_pca_signals_arr
        combiner._raw_blpx_signals = raw_blpx_signals_arr

        pipeline = SignalPipeline(components=components, combiner=combiner)
        pipeline_results = pipeline.run(common_inputs, start_idx=start_idx, T=T, start_idx_raw=start_idx_raw, n_jobs=n_jobs)

        # Update PCA caches
        if need_raw_pca and not raw_pca_cached:
            self._raw_pca_cache[cache_key] = pipeline_results["raw_pca"].copy()
        if need_residual_pca and not residual_pca_cached:
            self._residual_pca_cache[cache_key] = pipeline_results["residual_pca"].copy()

        # Extension E: Inflate Sigma_YY based on macro surprise
        if (self.macro_confidence_enabled and self.macro_sigma_yy_inflation_enabled
                and self._macro_surprise_raw is not None
                and np.any(sigma_yy_array)):
            from leadlag.core.macro import compute_sigma_yy_inflation
            sigma_yy_array = compute_sigma_yy_inflation(
                self._macro_surprise_raw,
                self.macro_kappas,
                self._macro_sens_matrix,
                sigma_yy_base=sigma_yy_array,
            )

        adapter = BLPXOutputAdapter(n_j=self.n_j, jp_tickers=JP_TICKERS)
        return adapter.adapt(pipeline_results, common_inputs, sigma_yy=sigma_yy_array)
