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

from leadlag.core.correlation import (
    build_c0_from_v0,
    compute_correlation,
    regularize_correlation,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.blp_base import _BLPBase

logger = logging.getLogger(__name__)

# Caches to speed up grid search backtesting by avoiding redundant calculations
_RAW_PCA_RESIDUAL_PCA_CACHE: dict = {}
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
        sig_comps = self._resolve_val("signal_components", None)
        if isinstance(sig_comps, dict):
            self.raw_pca_weight = float(sig_comps.get("raw_pca", {}).get("weight", 0.0)) if sig_comps.get("raw_pca", {}).get("enabled", False) else 0.0
            self.residual_pca_weight = float(sig_comps.get("residual_pca", {}).get("weight", 0.0)) if sig_comps.get("residual_pca", {}).get("enabled", False) else 0.0
            self.raw_blpx_weight = float(sig_comps.get("raw_blpx", {}).get("weight", 0.0)) if sig_comps.get("raw_blpx", {}).get("enabled", False) else 0.0
            self.residual_blpx_weight = float(sig_comps.get("residual_blpx", {}).get("weight", 0.0)) if sig_comps.get("residual_blpx", {}).get("enabled", False) else 0.0
        else:
            self.raw_pca_weight = float(self._resolve_val("raw_pca_weight", 0.4))
            self.residual_pca_weight = float(self._resolve_val("residual_pca_weight", 0.4))
            # Support both raw_blpx_weight and legacy p5_weight naming
            self.raw_blpx_weight = float(self._resolve_val("raw_blpx_weight", self._resolve_val("p5_weight", 0.1)))
            self.residual_blpx_weight = float(self._resolve_val("residual_blpx_weight", self._resolve_val("p5p3_weight", 0.1)))

        # Continuous M_sector parameters
        self.sector_eta = float(self._resolve_val("sector_eta", 0.0))
        self.sector_gamma = float(self._resolve_val("sector_gamma", 2.0))

        # Precompute the fixed Sector Mapping matrix M_sector
        self.M_sector = self._build_sector_prior()
        self._M_sector_fixed = self.M_sector.copy()

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_slippage_bps()

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
        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk not in self._SECTOR_MAPPING_STRUCTURE:
                continue
            jp_tickers = self._SECTOR_MAPPING_STRUCTURE[us_tk]
            weights = []
            for jp_tk in jp_tickers:
                if jp_tk in JP_TICKERS:
                    j_idx = JP_TICKERS.index(jp_tk)
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
            for idx_local in range(len(window_returns)):
                idx_global = window_start + idx_local
                vol_factor = rolling_std[idx_global]
                window_returns[idx_local, self.n_u:] /= vol_factor

        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        if self.winsor_sigma is not None:
            for c in range(window_returns.shape[1]):
                mu_c = np.mean(window_returns[:, c])
                std_c = np.std(window_returns[:, c])
                if std_c > 1e-8:
                    window_returns[:, c] = np.clip(
                        window_returns[:, c],
                        mu_c - self.winsor_sigma * std_c,
                        mu_c + self.winsor_sigma * std_c,
                    )
        return window_returns

    def _estimate_correlation(
        self, window_returns: np.ndarray, current_index: int, is_residual: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate rolling mean, std, and correlation with caching."""
        cache_key = (current_index, self.blp_window, self.winsor_sigma, self.exec_adjustment, self.blp_ewma_halflife, is_residual, id(window_returns))
        if cache_key in _BLP_CORR_CACHE:
            return _BLP_CORR_CACHE[cache_key]

        mu, sigma, corr = compute_correlation(window_returns, self.blp_ewma_halflife)
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
            inv_A = np.linalg.inv(A)
            result = B @ inv_A
        except Exception:
            pinv_fallback = True
            try:
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
                corr, c0_t, self.lambda_reg, self.lambda_lw, self.lw_target
            )
            eigvals, eigvecs = np.linalg.eigh(c_t_reg)
            sort_idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sort_idx]

            v_t_k = eigvecs[:, : self.k]
            v_u_t_k = v_t_k[: self.n_u, :]
            v_j_t_k = v_t_k[self.n_u :, :]
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

        z_hat_j_t1 = B_struct @ z_U_t
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

        signal = self._apply_gap_adjustment(
            r_hat_jp_cc, z_hat_j_t1, gap_override, betas_t, topix_night_t
        )

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
            B_struct=B_struct,
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

        # Setup output arrays
        raw_pca_signals = np.zeros((T, self.n_j))
        residual_pca_signals = np.zeros((T, self.n_j))
        raw_blpx_signals = np.zeros((T, self.n_j))
        residual_blpx_signals = np.zeros((T, self.n_j))
        combined_signals = np.zeros((T, self.n_j))
        normalized_combined_signals = np.zeros((T, self.n_j))

        # Track BLP diagnostics
        blp_diagnostics = []

        start_idx = self.corr_window
        cache_key_raw_pca_residual_pca = (
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
        if cache_key_raw_pca_residual_pca in _RAW_PCA_RESIDUAL_PCA_CACHE:
            raw_pca_signals, residual_pca_signals = _RAW_PCA_RESIDUAL_PCA_CACHE[cache_key_raw_pca_residual_pca]
            raw_pca_cached = True
        else:
            raw_pca_cached = False

        # Determine which components to compute (skip zero-weight for speed)
        need_raw_pca = self.raw_pca_weight > 0.0
        need_residual_pca = self.residual_pca_weight > 0.0
        need_raw_blpx = self.raw_blpx_weight > 0.0
        need_residual_blpx = self.residual_blpx_weight > 0.0

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
            if need_residual_pca and not raw_pca_cached:
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
            else:
                residual_blpx_res = {**self._ZERO_BLP_DIAGNOSTICS, "signal": np.zeros(self.n_j)}

            # Standard Z-score normalization of component signals
            z0 = self.normalize_signals(raw_pca_sig, self.normalization_method)
            z3 = self.normalize_signals(residual_pca_sig, self.normalization_method)
            z_raw_blpx = self.normalize_signals(raw_blpx_res["signal"], self.normalization_method)
            z_residual_blpx = self.normalize_signals(residual_blpx_res["signal"], self.normalization_method)

            # Combined PCA-BLPX Ensemble signal
            s_ens = self.combine_signals(z0, z3, z_raw_blpx, z_residual_blpx)
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
                }
            )

        if not raw_pca_cached:
            _RAW_PCA_RESIDUAL_PCA_CACHE[cache_key_raw_pca_residual_pca] = (raw_pca_signals.copy(), residual_pca_signals.copy())

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

        return {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "raw_blpx_signals": raw_blpx_df,
            "residual_blpx_signals": residual_blpx_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "y_jp_oc_df": inputs["y_jp_oc_df"],
            "blp_diagnostics": pd.DataFrame(blp_diagnostics).set_index("date"),
        }
