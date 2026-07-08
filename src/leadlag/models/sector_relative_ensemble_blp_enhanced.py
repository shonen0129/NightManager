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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

from leadlag.core.correlation import (
    build_c0_from_v0,
    compute_correlation,
    compute_stress_weight,
    regularize_correlation,
)
from leadlag.core.macro import (
    MACRO_NAMES,
    MACRO_SENS_MATRIX,
    compute_factor_kappa_scale,
    compute_macro_surprise,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.blp_base import _BLPBase

logger = logging.getLogger(__name__)

# Caches to speed up grid search backtesting by avoiding redundant calculations
_RAW_PCA_CACHE: dict = {}
_RESIDUAL_PCA_CACHE: dict = {}
_BLP_CORR_CACHE: dict = {}


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
        self.config = config
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

    def _build_sector_prior(self) -> np.ndarray:
        """Build the fixed 日米業種対応行列 M_sector of size (n_j x n_u).

        Weights are derived from _SECTOR_MAPPING_STRUCTURE with equal split,
        then column-normalized so each US ETF column sums to 1.0.
        """
        M = np.zeros((self.n_j, self.n_u))

        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk in self._SECTOR_MAPPING_STRUCTURE:
                jp_tickers = self._SECTOR_MAPPING_STRUCTURE[us_tk]
                w = 1.0 / len(jp_tickers)
                for jp_tk in jp_tickers:
                    if jp_tk in JP_TICKERS:
                        j_idx = JP_TICKERS.index(jp_tk)
                        M[j_idx, u_idx] = w

        # Column normalize (sum to 1.0)
        col_sums = np.sum(M, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 0:
                M[:, u_idx] /= col_sums[u_idx]

        return M

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
            from leadlag.core.macro import MACRO_NAMES, download_macro_prices

            sim_dates = df_exec.index
            start = sim_dates[0].strftime("%Y-%m-%d")
            end = sim_dates[-1].strftime("%Y-%m-%d")

            close_prices = download_macro_prices(start=start, end=end)
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

    # Structural mapping: which JP tickers relate to which US tickers
    _SECTOR_MAPPING_STRUCTURE = {
        "XLB": ["1620.T", "1623.T"],
        "XLC": ["1626.T"],
        "XLE": ["1618.T", "1627.T"],
        "XLF": ["1631.T", "1632.T"],
        "XLI": ["1624.T", "1622.T", "1626.T"],
        "XLK": ["1626.T", "1625.T"],
        "XLP": ["1617.T", "1630.T"],
        "XLRE": ["1633.T"],
        "XLU": ["1627.T"],
        "XLV": ["1621.T"],
        "XLY": ["1630.T", "1626.T", "1622.T"],
        "MTUM": ["1625.T", "1626.T"],
        "VLUE": ["1631.T", "1632.T", "1623.T", "1622.T"],
        "IUSG": ["1626.T", "1625.T"],
        "USMV": ["1617.T", "1621.T", "1627.T"],
    }

    def _get_sector_prior(
        self,
        current_index: int,
        all_returns: np.ndarray,
        corr: np.ndarray,
        B_blp: np.ndarray,
    ) -> np.ndarray:
        """Return the sector prior matrix M_sector (n_j x n_u).

        When sector_eta > 0, blends the fixed mapping with data-driven
        weights derived from the rolling cross-correlation:
          w_ji = max(0, corr(u, ji))^gamma / sum_k max(0, corr(u, jk))^gamma
          M_final = (1-eta) * M_fixed + eta * M_data

        Override in subclasses to provide a fully dynamic sector prior.
        """
        if self.sector_eta <= 0.0 or self._M_sector_fixed.shape != B_blp.shape:
            if self.M_sector.shape == B_blp.shape:
                return self.M_sector
            return np.zeros(B_blp.shape)

        if corr.shape != (self.n_u + self.n_j, self.n_u + self.n_j):
            return self._M_sector_fixed if self._M_sector_fixed.shape == B_blp.shape else np.zeros(B_blp.shape)

        c_xy = corr[: self.n_u, self.n_u:]  # (n_u, n_j) — US vs JP cross-corr

        M_data = np.zeros((self.n_j, self.n_u))
        for u_idx, j_indices in self._sector_mapping_indices.items():
            weights = []
            for j_idx in j_indices:
                raw_corr = c_xy[u_idx, j_idx]
                weights.append((j_idx, max(0.0, raw_corr) ** self.sector_gamma))
            if not weights:
                continue
            total = sum(w for _, w in weights)
            if total > 1e-10:
                for j_idx, w in weights:
                    M_data[j_idx, u_idx] = w / total

        M_blended = (1.0 - self.sector_eta) * self._M_sector_fixed + self.sector_eta * M_data

        col_sums = np.sum(M_blended, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 1e-10:
                M_blended[:, u_idx] /= col_sums[u_idx]

        if M_blended.shape == B_blp.shape:
            return M_blended
        return np.zeros(B_blp.shape)

    def _prepare_window_returns(
        self, all_returns: np.ndarray, current_index: int, rolling_std: np.ndarray | None
    ) -> np.ndarray:
        """Slice window returns, apply vol-scaling and winsorization."""
        window_start = max(0, current_index - self.blp_window)
        window_returns = all_returns[window_start:current_index].copy()

        if self.exec_adjustment == "vol_scale" and rolling_std is not None:
            vol_factors = rolling_std[window_start:current_index]
            window_returns[:, self.n_u:] /= vol_factors

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
        cache_key = (current_index, self.blp_window, self.winsor_sigma, self.exec_adjustment, self.blp_ewma_halflife, is_residual, id(window_returns), self.copula_enabled)
        if cache_key in _BLP_CORR_CACHE:
            return _BLP_CORR_CACHE[cache_key]

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
        )
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)
        _BLP_CORR_CACHE[cache_key] = (mu, sigma, corr)
        return mu, sigma, corr

    @staticmethod
    def _safe_solve_inv(A: np.ndarray, B: np.ndarray, label: str = "A") -> tuple[np.ndarray, np.ndarray, bool]:
        """Solve B @ inv(A) with pseudo-inverse fallback."""
        pinv_fallback = False
        try:
            if not np.isfinite(A).all():
                raise ValueError(f"{label} contains NaNs or Infs")
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                inv_A = np.linalg.inv(A)
                result = B @ inv_A
        except Exception:
            pinv_fallback = True
            try:
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    inv_A = np.linalg.pinv(A)
                    result = B @ inv_A
            except Exception:
                result = np.zeros((B.shape[0], A.shape[1]))
                inv_A = np.zeros((A.shape[0], A.shape[1]))
        return result, inv_A, pinv_fallback

    def _solve_blp_coefficients(
        self, corr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """Regularize correlation, solve for B_blp via ridge regression, and apply SVD rank reduction.

        Returns (B_blp, Sigma_XX_reg, Sigma_YX_reg, Sigma_YY_reg, cond_num, pinv_fallback).
        """
        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]
        C_YY = corr[self.n_u :, self.n_u :]

        Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
        Sigma_YX_reg = (1.0 - self.alpha_yx) * C_YX
        Sigma_YY_reg = (1.0 - self.alpha_yy) * C_YY + self.alpha_yy * np.eye(self.n_j)

        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
        ridge_matrix = self.rho * diag_mean * np.eye(self.n_u)
        A = Sigma_XX_reg + ridge_matrix

        try:
            singular_values = np.linalg.svd(A, compute_uv=False)
            cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
        except Exception:
            cond_num = np.nan

        B_blp, _, pinv_fallback = self._safe_solve_inv(A, Sigma_YX_reg, label="A")

        if self.rank != "full" and self.rank is not None:
            rank_val = int(self.rank)
            if rank_val < min(B_blp.shape):
                try:
                    U, S, Vt = np.linalg.svd(B_blp, full_matrices=False)
                    B_blp = U[:, :rank_val] @ np.diag(S[:rank_val]) @ Vt[:rank_val, :]
                except Exception as e:
                    logger.warning(f"SVD rank reduction failed: {e}")

        return B_blp, Sigma_XX_reg, Sigma_YX_reg, Sigma_YY_reg, cond_num, pinv_fallback

    def _compute_pca_prior(
        self, corr: np.ndarray, v0_static: np.ndarray | None, c_full: np.ndarray | None
    ) -> np.ndarray:
        """Compute PCA prior B_pca from eigen decomposition of regularized correlation."""
        B_pca = np.zeros((self.n_j, self.n_u))
        if v0_static is not None and c_full is not None and corr.shape == (32, 32) and v0_static.shape == (32, 6) and c_full.shape == (32, 32):
            c0_t = build_c0_from_v0(v0_static, c_full)
            c_t_reg = regularize_correlation(
                corr, c0_t, self.lambda_reg, self.lambda_lw, self.lw_target,
                getattr(self, "min_raw_weight", 0.0),
            )
            eigvals, eigvecs = np.linalg.eigh(c_t_reg)
            sort_idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sort_idx]

            v_t_k = eigvecs[:, : self.k]
            v_u_t_k = v_t_k[: self.n_u, :]
            v_j_t_k = v_t_k[self.n_u :, :]
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                B_pca = v_j_t_k @ v_u_t_k.T
        return B_pca

    def _solve_tikhonov(
        self,
        Sigma_XX_reg: np.ndarray,
        Sigma_YX_reg: np.ndarray,
        B_pca: np.ndarray,
        M_sector: np.ndarray,
        diag_mean: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-target Tikhonov regularization solve.

        Returns (B_struct, inv_A_tikh).
        """
        l_pca = self.lambda_pca
        l_sec = self.lambda_sector
        lambda_sum = l_pca + l_sec
        if lambda_sum > 0.75:
            l_pca = (self.lambda_pca / lambda_sum) * 0.75
            l_sec = (self.lambda_sector / lambda_sum) * 0.75

        lambda_tikh = self.rho * diag_mean + l_pca + l_sec
        A_tikh = Sigma_XX_reg + lambda_tikh * np.eye(self.n_u)
        rhs = Sigma_YX_reg + l_pca * B_pca + l_sec * M_sector

        B_struct, inv_A_tikh, _ = self._safe_solve_inv(A_tikh, rhs, label="A_tikh")
        return B_struct, inv_A_tikh

    @staticmethod
    def _apply_confidence_weighting(
        z_hat_j_t1: np.ndarray,
        Sigma_YY_reg: np.ndarray,
        Sigma_YX_reg: np.ndarray,
        inv_A_tikh: np.ndarray,
        beta_conf: float,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Apply confidence weighting based on conditional prediction variance.

        Returns (z_hat_j_t1_weighted, pred_var, num_floored).
        """
        Sigma_XY_reg = Sigma_YX_reg.T
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            Sigma_Y_given_X = Sigma_YY_reg - Sigma_YX_reg @ inv_A_tikh @ Sigma_XY_reg

        pred_var = np.maximum(np.diag(Sigma_Y_given_X), 0.0)
        var_floor = 1e-8
        pred_var_floored = np.maximum(pred_var, var_floor)
        num_floored = int(np.sum(pred_var < var_floor))

        if beta_conf > 0.0:
            z_hat_j_t1 = z_hat_j_t1 / (pred_var_floored ** beta_conf)
            z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)
            z_hat_j_t1 = np.clip(z_hat_j_t1, -5.0, 5.0)

        return z_hat_j_t1, pred_var, num_floored

    @staticmethod
    def _build_blp_diagnostics(
        signal: np.ndarray,
        z_hat_j_t1: np.ndarray,
        cond_num: float,
        B_blp: np.ndarray,
        B_pca: np.ndarray,
        M_sector: np.ndarray,
        B_struct: np.ndarray,
        C_XX: np.ndarray,
        C_YX: np.ndarray,
        C_YY: np.ndarray,
        pred_var: np.ndarray,
        num_floored: int,
        pinv_fallback: bool,
        num_training_samples: int,
        return_matrices: bool,
        A: np.ndarray | None = None,
        Sigma_XX_reg: np.ndarray | None = None,
        Sigma_YX_reg: np.ndarray | None = None,
        Sigma_YY_reg: np.ndarray | None = None,
        inv_A_tikh: np.ndarray | None = None,
        z_U_t: np.ndarray | None = None,
        mu: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        sigma_j_t: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build diagnostics dict for BLP signal."""
        diag = {
            "signal": signal,
            "z_hat_j_t1": z_hat_j_t1,
            "cond_num": cond_num,
            "b_norm": float(np.linalg.norm(B_blp, "fro")),
            "b_pca_norm": float(np.linalg.norm(B_pca)),
            "b_sector_norm": float(np.linalg.norm(M_sector)),
            "b_struct_norm": float(np.linalg.norm(B_struct)),
            "sigma_xx_trace": float(np.trace(C_XX)),
            "sigma_yx_norm": float(np.linalg.norm(C_YX)),
            "sigma_yy_trace": float(np.trace(C_YY)),
            "min_pred_var": float(np.min(pred_var)),
            "max_pred_var": float(np.max(pred_var)),
            "num_pred_var_floored": num_floored,
            "pinv_fallback": pinv_fallback,
            "num_training_samples": num_training_samples,
            "sigma_Y_cov": Sigma_YY_reg,
        }
        if return_matrices:
            diag.update({
                "Sigma_XX": A,
                "Sigma_YX": Sigma_YX_reg,
                "Sigma_YY": Sigma_YY_reg,
                "inv_A": inv_A_tikh,
                "B_blp": B_blp,
                "B_pca_prior": B_pca,
                "B_sector_prior": M_sector,
                "B_struct": B_struct,
                "z_U": z_U_t,
                "pred_var_vec": pred_var,
                "sigma_X": sigma[:len(sigma)//2] if sigma is not None else None,
                "sigma_Y": sigma[len(sigma)//2:] if sigma is not None else None,
                "sigma_Y_denorm": sigma_j_t,
                "mu_X": mu[:len(mu)//2] if mu is not None else None,
                "mu_Y": mu[len(mu)//2:] if mu is not None else None,
            })
        return diag

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

    def _solve_asymmetric_blp(
        self,
        C_YX_pos: np.ndarray,
        C_YX_neg: np.ndarray,
        C_XX: np.ndarray,
        C_YY: np.ndarray,
        B_pca: np.ndarray,
        M_sector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve BLP coefficients separately for positive and negative regimes.

        Returns:
            B_pos_struct, B_neg_struct, inv_A_avg, Sigma_YX_reg_avg
        """
        Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
        Sigma_YX_reg_pos = (1.0 - self.alpha_yx) * C_YX_pos
        Sigma_YX_reg_neg = (1.0 - self.alpha_yx) * C_YX_neg
        Sigma_YY_reg = (1.0 - self.alpha_yy) * C_YY + self.alpha_yy * np.eye(self.n_j)

        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))

        B_pos_struct, inv_A_pos = self._solve_tikhonov(
            Sigma_XX_reg, Sigma_YX_reg_pos, B_pca, M_sector, diag_mean
        )
        B_neg_struct, inv_A_neg = self._solve_tikhonov(
            Sigma_XX_reg, Sigma_YX_reg_neg, B_pca, M_sector, diag_mean
        )

        inv_A_avg = 0.5 * (inv_A_pos + inv_A_neg)
        Sigma_YX_reg_avg = 0.5 * (Sigma_YX_reg_pos + Sigma_YX_reg_neg)

        return B_pos_struct, B_neg_struct, inv_A_avg, Sigma_YX_reg_avg

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
        """Compute the Enhanced Regularized Block BLP signal for a single time step.

        Ensure Y_date <= signal_date by slicing up to current_index - 1.
        """
        # 1. Prepare window returns (vol-scaling + winsorization)
        window_returns = self._prepare_window_returns(all_returns, current_index, rolling_std)

        # 2. Estimate correlation
        mu, sigma, corr = self._estimate_correlation(window_returns, current_index, is_residual)

        # 3. Solve BLP coefficients
        B_blp, Sigma_XX_reg, Sigma_YX_reg, Sigma_YY_reg, cond_num, pinv_fallback = (
            self._solve_blp_coefficients(corr)
        )

        # 4. Structured shrinkage: PCA prior + sector prior + Tikhonov
        B_pca = self._compute_pca_prior(corr, v0_static, c_full)
        M_sector = self._get_sector_prior(current_index, all_returns, corr, B_blp)
        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
        B_struct, inv_A_tikh = self._solve_tikhonov(
            Sigma_XX_reg, Sigma_YX_reg, B_pca, M_sector, diag_mean
        )

        # 5. Predict standardized JP returns
        X_t = all_returns[current_index, : self.n_u]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu_X = mu[: self.n_u]
        sigma_X = sigma[: self.n_u]
        sigma_X_safe = np.where(sigma_X > 1e-8, sigma_X, 1.0)
        z_U_t = (X_t - mu_X) / sigma_X_safe

        # Step 5a: Input asymmetric propagation
        z_U_pos = np.maximum(z_U_t, 0.0)
        z_U_neg = np.minimum(z_U_t, 0.0)
        z_U_neg_scaled = (1.0 + self.asymmetry_delta) * z_U_neg

        if self.asymmetry_mode == "covariance":
            C_YX_pos, C_YX_neg, C_XX, C_YY = self._estimate_asymmetric_covariance(window_returns, corr)
            B_pos_struct, B_neg_struct, inv_A_tikh, Sigma_YX_reg = self._solve_asymmetric_blp(
                C_YX_pos, C_YX_neg, C_XX, C_YY, B_pca, M_sector
            )
            z_hat_j_t1 = B_pos_struct @ z_U_pos + B_neg_struct @ z_U_neg_scaled
            B_struct_diag = 0.5 * (B_pos_struct + B_neg_struct)
        else:
            z_U_asym = z_U_pos + z_U_neg_scaled
            z_hat_j_t1 = B_struct @ z_U_asym
            B_struct_diag = B_struct

        z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)

        # 6. Confidence weighting
        z_hat_j_t1, pred_var, num_floored = self._apply_confidence_weighting(
            z_hat_j_t1, Sigma_YY_reg, Sigma_YX_reg, inv_A_tikh, self.beta_conf
        )

        # 7. Denormalize and apply gap adjustment
        r_hat_jp_cc = self._denormalize_signal(
            z_hat_j_t1, mu, sigma, all_returns, current_index, self.n_u, self.vol_adjusted_target
        )
        if self.vol_adjusted_target and current_index >= 20:
            jp_returns_20 = all_returns[current_index - 20 : current_index, self.n_u :]
            jp_returns_20 = np.nan_to_num(jp_returns_20, nan=0.0, posinf=0.0, neginf=0.0)
            sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
            sigma_j_t = np.maximum(sigma_j_t, 1e-8)
        else:
            sigma_j_t = sigma[self.n_u :]

        # Determine US market direction
        us_market_mean = np.nanmean(z_U_t)
        us_negative = us_market_mean < 0.0

        gap_coef_override = None
        beta_coef_override = None
        if us_negative and self.gap_open_coef_neg is not None:
            gap_coef_override = self.gap_open_coef_neg
            beta_coef_override = self.topix_beta_coef_neg

        signal = self._apply_gap_adjustment(
            r_hat_jp_cc,
            z_hat_j_t1,
            gap_override,
            betas_t,
            topix_night_t,
            gap_open_coef_override=gap_coef_override,
            topix_beta_coef_override=beta_coef_override,
        )

        if self.asymmetry_post_gap_delta != 0.0:
            if self.asymmetry_post_gap_mode == "signal_split":
                signal = np.maximum(signal, 0.0) + (1.0 + self.asymmetry_post_gap_delta) * np.minimum(signal, 0.0)
            elif self.asymmetry_post_gap_mode == "us_direction":
                if us_negative:
                    signal = signal * (1.0 + self.asymmetry_post_gap_delta)

        # 8. Build diagnostics
        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]
        C_YY = corr[self.n_u :, self.n_u :]
        A = Sigma_XX_reg + self.rho * diag_mean * np.eye(self.n_u)

        return self._build_blp_diagnostics(
            signal=signal,
            z_hat_j_t1=z_hat_j_t1,
            cond_num=cond_num,
            B_blp=B_blp,
            B_pca=B_pca,
            M_sector=M_sector,
            B_struct=B_struct_diag,
            C_XX=C_XX,
            C_YX=C_YX,
            C_YY=C_YY,
            pred_var=pred_var,
            num_floored=num_floored,
            pinv_fallback=pinv_fallback,
            num_training_samples=len(window_returns),
            return_matrices=return_matrices,
            A=A,
            Sigma_XX_reg=Sigma_XX_reg,
            Sigma_YX_reg=Sigma_YX_reg,
            Sigma_YY_reg=Sigma_YY_reg,
            inv_A_tikh=inv_A_tikh,
            z_U_t=z_U_t,
            mu=mu,
            sigma=sigma,
            sigma_j_t=sigma_j_t,
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

    def predict_signals(self, df_exec: pd.DataFrame) -> dict[str, Any]:
        """Generate component and ensemble signals for all rows in df_exec."""
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
            macro_path = Path("/Users/shonen/日米ラグ/market_data/macro_data.pkl")
            if macro_path.exists():
                try:
                    macro_df = pd.read_pickle(macro_path)
                    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
                    # Safe ffill + bfill
                    vix_series = macro_df["^VIX"].reindex(sim_dates).ffill()
                    vix_series = vix_series.bfill()
                    logger.info("Successfully loaded VIX from macro_data.pkl for meta-learning.")
                except Exception as e:
                    logger.warning(f"Failed to load VIX from macro_data.pkl: {e}")
            if vix_series is None:
                vix_series = pd.Series(20.0, index=sim_dates)

        # Setup output arrays
        raw_pca_signals = np.zeros((T, self.n_j))
        residual_pca_signals = np.zeros((T, self.n_j))
        raw_blpx_signals = np.zeros((T, self.n_j))
        residual_blpx_signals = np.zeros((T, self.n_j))
        combined_signals = np.zeros((T, self.n_j))
        normalized_combined_signals = np.zeros((T, self.n_j))
        sigma_yy_array = np.zeros((T, self.n_j, self.n_j))

        # Setup arrays for meta-learning tracking
        us_dispersions = [0.0] * T
        cond_nums = [0.0] * T
        vix_vals = [20.0] * T
        ic_blpx_vals = [0.0] * T
        ic_pca_vals = [0.0] * T
        meta_weights = [0.8] * T

        # Track BLP diagnostics
        blp_diagnostics = []

        # Optimize: skip loop iterations before warmup if _start_date is specified
        start_date_str = getattr(self, "_start_date", None)
        if start_date_str is not None:
            start_dt = pd.to_datetime(start_date_str)
            start_idx_raw = df_exec.index.searchsorted(start_dt)
            start_idx = max(self.corr_window, start_idx_raw - self.blp_window)
        else:
            start_idx = self.corr_window
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

        raw_pca_cached = False
        if need_raw_pca:
            if cache_key in _RAW_PCA_CACHE:
                raw_pca_signals = _RAW_PCA_CACHE[cache_key]
                raw_pca_cached = True
            else:
                raw_pca_signals = np.zeros((T, self.n_j))
        else:
            raw_pca_signals = np.zeros((T, self.n_j))

        residual_pca_cached = False
        if need_residual_pca:
            if cache_key in _RESIDUAL_PCA_CACHE:
                residual_pca_signals = _RESIDUAL_PCA_CACHE[cache_key]
                residual_pca_cached = True
            else:
                residual_pca_signals = np.zeros((T, self.n_j))
        else:
            residual_pca_signals = np.zeros((T, self.n_j))

        for i in range(start_idx, T):
            # 1. Raw-PCA (skip if weight=0 and not cached)
            if need_raw_pca and not raw_pca_cached:
                raw_pca_sig = self.compute_production_signal(
                    i, c_full, v0_static, v1, v2, all_returns_raw, jp_gap, jp_beta, topix_night
                )
                raw_pca_signals[i] = raw_pca_sig
            else:
                raw_pca_sig = raw_pca_signals[i]

            # 2. Residual-PCA (skip if weight=0 and not cached)
            if need_residual_pca and not residual_pca_cached:
                residual_pca_sig = self.compute_residual_signal(
                    jp_res_returns_p3, i, c_full_p3, v0_static, v1, v2, jp_gap, jp_beta, topix_night
                )
                residual_pca_signals[i] = residual_pca_sig
            else:
                residual_pca_sig = residual_pca_signals[i]

            # 3. Raw-BLPX (skip if weight=0)
            gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
            betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
            topix_night_t = float(topix_night[i]) if topix_night is not None else None

            if need_raw_blpx:
                raw_blpx_res = self.compute_blp_signal(
                    all_returns_raw,
                    i,
                    gap_override=gap_override,
                    betas_t=betas_t,
                    topix_night_t=topix_night_t,
                    rolling_std=rolling_std,
                    v0_static=v0_static,
                    c_full=c_full,
                    is_residual=False,
                )
                raw_blpx_signals[i] = raw_blpx_res["signal"]
            else:
                raw_blpx_res = {**self._ZERO_BLP_DIAGNOSTICS, "signal": np.zeros(self.n_j)}

            # 4. Residual-BLPX (skip if weight=0)
            if need_residual_blpx:
                residual_blpx_res = self.compute_blp_signal(
                    jp_res_returns_p3,
                    i,
                    gap_override=gap_override,
                    betas_t=betas_t,
                    topix_night_t=topix_night_t,
                    rolling_std=rolling_std,
                    v0_static=v0_static,
                    c_full=c_full_p3,
                    is_residual=True,
                )
                residual_blpx_signals[i] = residual_blpx_res["signal"]
                if self.minvar_enabled and "sigma_Y_cov" in residual_blpx_res:
                    sigma_yy_array[i] = residual_blpx_res["sigma_Y_cov"]
            else:
                residual_blpx_res = {**self._ZERO_BLP_DIAGNOSTICS, "signal": np.zeros(self.n_j)}

            # Standard Z-score normalization of component signals
            z0 = self.normalize_signals(raw_pca_sig, self.normalization_method)
            z3 = self.normalize_signals(residual_pca_sig, self.normalization_method)
            z_raw_blpx = self.normalize_signals(raw_blpx_res["signal"], self.normalization_method)
            z_residual_blpx = self.normalize_signals(residual_blpx_res["signal"], self.normalization_method)

            # Store dispersion, condition number, VIX
            us_returns = all_returns_raw[i, :self.n_u]
            us_dispersions[i] = float(np.nanvar(us_returns))
            cond_nums[i] = float(raw_blpx_res["cond_num"])
            vix_vals[i] = float(vix_series.iloc[i]) if vix_series is not None else 20.0

            # Calculate actual daily ICs for row i-1
            if i - 1 >= start_idx:
                y_prev = y_jp_target[i-1]
                sig_blpx_prev = raw_blpx_signals[i-1]
                sig_pca_prev = raw_pca_signals[i-1]

                valid_blpx = np.isfinite(sig_blpx_prev) & np.isfinite(y_prev)
                if np.sum(valid_blpx) >= 5 and np.std(sig_blpx_prev[valid_blpx]) > 1e-8 and np.std(y_prev[valid_blpx]) > 1e-8:
                    ic_blpx_vals[i-1] = float(spearmanr(sig_blpx_prev[valid_blpx], y_prev[valid_blpx])[0])
                else:
                    ic_blpx_vals[i-1] = 0.0

                valid_pca = np.isfinite(sig_pca_prev) & np.isfinite(y_prev)
                if np.sum(valid_pca) >= 5 and np.std(sig_pca_prev[valid_pca]) > 1e-8 and np.std(y_prev[valid_pca]) > 1e-8:
                    ic_pca_vals[i-1] = float(spearmanr(sig_pca_prev[valid_pca], y_prev[valid_pca])[0])
                else:
                    ic_pca_vals[i-1] = 0.0

            # Predict daily ensemble weight w_t
            w_t = 0.8
            if self.meta_enabled:
                if i >= start_idx + self.meta_train_window:
                    w_t = self._predict_meta_weight(
                        i, us_dispersions, cond_nums, vix_vals, ic_blpx_vals, ic_pca_vals
                    )
                    if self.meta_smooth_factor < 1.0 and i - 1 >= start_idx:
                        w_prev_meta = meta_weights[i-1]
                        w_t = self.meta_smooth_factor * w_t + (1.0 - self.meta_smooth_factor) * w_prev_meta
                meta_weights[i] = w_t

            # Combined PCA-BLPX Ensemble signal
            if self.meta_enabled:
                pca_denom = self.raw_pca_weight + self.residual_pca_weight
                if pca_denom > 0.0:
                    s_pca = (self.raw_pca_weight * z0 + self.residual_pca_weight * z3) / pca_denom
                else:
                    s_pca = 0.5 * z0 + 0.5 * z3

                blpx_denom = self.raw_blpx_weight + self.residual_blpx_weight
                if blpx_denom > 0.0:
                    s_blpx = (self.raw_blpx_weight * z_raw_blpx + self.residual_blpx_weight * z_residual_blpx) / blpx_denom
                else:
                    s_blpx = 0.5 * z_raw_blpx + 0.5 * z_residual_blpx

                s_ens = (1.0 - w_t) * s_pca + w_t * s_blpx
            else:
                s_ens = self.combine_signals(z0, z3, z_raw_blpx, z_residual_blpx)

            # Apply macro confidence scaling (Factor-Specific Kappa)
            if self.macro_confidence_enabled and self._macro_scales is not None:
                scale_t = self._macro_scales[i]
                s_ens = s_ens / scale_t
                s_ens = np.nan_to_num(s_ens, nan=0.0, posinf=0.0, neginf=0.0)

                # Extension A: Directional adjustment (signed surprise * signed sensitivity)
                if self._macro_direction_adj is not None:
                    dir_adj_t = self._macro_direction_adj[i]
                    s_ens = s_ens * dir_adj_t
                    s_ens = np.nan_to_num(s_ens, nan=0.0, posinf=0.0, neginf=0.0)

            combined_signals[i] = s_ens
            normalized_combined_signals[i] = self.normalize_signals(
                s_ens, self.normalization_method
            )

            # Diagnostics recording
            date_str = sim_dates[i].strftime("%Y-%m-%d")
            blp_diagnostics.append(
                {
                    "date": date_str,
                    "raw_blpx_cond_num": raw_blpx_res["cond_num"],
                    "raw_blpx_b_norm": raw_blpx_res["b_norm"],
                    "raw_blpx_b_pca_norm": raw_blpx_res["b_pca_norm"],
                    "raw_blpx_b_sector_norm": raw_blpx_res["b_sector_norm"],
                    "raw_blpx_b_struct_norm": raw_blpx_res["b_struct_norm"],
                    "raw_blpx_sigma_xx_trace": raw_blpx_res["sigma_xx_trace"],
                    "raw_blpx_sigma_yx_norm": raw_blpx_res["sigma_yx_norm"],
                    "raw_blpx_sigma_yy_trace": raw_blpx_res["sigma_yy_trace"],
                    "raw_blpx_min_pred_var": raw_blpx_res["min_pred_var"],
                    "raw_blpx_max_pred_var": raw_blpx_res["max_pred_var"],
                    "raw_blpx_num_pred_var_floored": raw_blpx_res["num_pred_var_floored"],
                    "raw_blpx_pinv_fallback": int(raw_blpx_res["pinv_fallback"]),
                    "raw_blpx_num_training_samples": raw_blpx_res["num_training_samples"],
                    "meta_ensemble_weight": w_t,
                }
            )

        if need_raw_pca and not raw_pca_cached:
            _RAW_PCA_CACHE[cache_key] = raw_pca_signals.copy()
        if need_residual_pca and not residual_pca_cached:
            _RESIDUAL_PCA_CACHE[cache_key] = residual_pca_signals.copy()

        # Build DataFrames
        raw_pca_df = pd.DataFrame(raw_pca_signals, index=sim_dates, columns=JP_TICKERS)
        residual_pca_df = pd.DataFrame(residual_pca_signals, index=sim_dates, columns=JP_TICKERS)
        p4_signals = np.zeros((T, self.n_j))
        p4_df = pd.DataFrame(p4_signals, index=sim_dates, columns=JP_TICKERS)
        raw_blpx_df = pd.DataFrame(raw_blpx_signals, index=sim_dates, columns=JP_TICKERS)
        residual_blpx_df = pd.DataFrame(residual_blpx_signals, index=sim_dates, columns=JP_TICKERS)
        combined_df = pd.DataFrame(combined_signals, index=sim_dates, columns=JP_TICKERS)
        normalized_df = pd.DataFrame(
            normalized_combined_signals, index=sim_dates, columns=JP_TICKERS
        )

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

        return {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "raw_blpx_signals": raw_blpx_df,
            "residual_blpx_signals": residual_blpx_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "sigma_yy": sigma_yy_array,
            "y_jp_oc_df": inputs["y_jp_oc_df"],
            "blp_diagnostics": pd.DataFrame(blp_diagnostics).set_index("date"),
        }
