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

from leadlag.core import signal as signals
from leadlag.core.correlation import (
    build_base_vectors,
    build_c0_from_v0,
    build_v3_static,
    compute_baseline_correlation,
    compute_correlation,
    regularize_correlation,
)
from leadlag.core.residualize import compute_rolling_ols_betas
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.base import BaseModel
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)

# Caches to speed up grid search backtesting by avoiding redundant calculations
_RAW_PCA_RESIDUAL_PCA_CACHE: dict = {}
_BLP_CORR_CACHE: dict = {}


class SectorRelativeEnsembleBLPEnhancedModel(BaseModel):
    """Sector Relative Ensemble with Enhanced Regularized Block BLP (PCA-BLPX Ensemble) Model."""

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

        # Precompute the fixed Sector Mapping matrix M_sector
        self.M_sector = self._build_sector_prior()

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_val("slippage_bps", 5.0)
        if isinstance(config, dict):
            if "costs" in config and "slippage_bps_per_side" in config["costs"]:
                self.slippage_bps = float(config["costs"]["slippage_bps_per_side"])

    def _resolve_val(self, key: str, default: any) -> any:
        """Resolve value from config object or dict."""
        aliases = []
        if key == "blp_ewma_halflife":
            aliases.append("ewma_halflife")
        elif key == "exec_adjustment":
            aliases.append("execution_target_cost_adjustment")
            aliases.append("execution_target_cost_adjustment_mode")

        keys_to_try = [key] + aliases
        for k in keys_to_try:
            if hasattr(self.config, k):
                return getattr(self.config, k)
            if isinstance(self.config, dict):
                if k in self.config:
                    return self.config[k]
                for section in ["model", "ensemble", "portfolio", "costs", "residualization", "blpx"]:
                    if section in self.config and isinstance(self.config[section], dict) and k in self.config[section]:
                        return self.config[section][k]
                # Translations
                if k == "model_name" and "name" in self.config.get("model", {}):
                    return self.config["model"]["name"]
                if k == "k" and "k" in self.config.get("model", {}):
                    return self.config["model"]["k"]
                if k == "q" and "long_short_frac" in self.config.get("portfolio", {}):
                    return self.config["portfolio"]["long_short_frac"]
        return default

    def _build_sector_prior(self) -> np.ndarray:
        """Build the fixed日米業種対応行列 M_sector of size 17 x 15."""
        M = np.zeros((self.n_j, self.n_u))

        # Sector Mapping dictionary mapping US ETFs to JP tickers
        mapping = {
            "XLB": {"1620.T": 0.5, "1623.T": 0.5},
            "XLC": {"1626.T": 1.0},
            "XLE": {"1618.T": 0.5, "1627.T": 0.5},
            "XLF": {"1631.T": 0.5, "1632.T": 0.5},
            "XLI": {"1624.T": 0.33, "1622.T": 0.33, "1626.T": 0.34},
            "XLK": {"1626.T": 0.5, "1625.T": 0.5},
            "XLP": {"1617.T": 0.5, "1630.T": 0.5},
            "XLRE": {"1633.T": 1.0},
            "XLU": {"1627.T": 1.0},
            "XLV": {"1621.T": 1.0},
            "XLY": {"1630.T": 0.33, "1626.T": 0.33, "1622.T": 0.34},
            "MTUM": {"1625.T": 0.5, "1626.T": 0.5},
            "VLUE": {"1631.T": 0.25, "1632.T": 0.25, "1623.T": 0.25, "1622.T": 0.25},
            "IUSG": {"1626.T": 0.5, "1625.T": 0.5},
            "USMV": {"1617.T": 0.33, "1621.T": 0.33, "1627.T": 0.34}
        }

        for u_idx, us_tk in enumerate(US_TICKERS):
            if us_tk in mapping:
                for jp_tk, w in mapping[us_tk].items():
                    if jp_tk in JP_TICKERS:
                        j_idx = JP_TICKERS.index(jp_tk)
                        M[j_idx, u_idx] = w

        # Column normalize (sum to 1.0)
        col_sums = np.sum(M, axis=0)
        for u_idx in range(self.n_u):
            if col_sums[u_idx] > 0:
                M[:, u_idx] /= col_sums[u_idx]

        return M

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Prepare inputs common to signal computation (raw targets, residuals, betas)."""
        sim_dates = df_exec.index

        # Target returns for JP on trade_date (D_t+1)
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)

        # Build all_returns_raw: US columns are us_cc_* (on D_t), JP columns are y_jp_target (on D_t+1)
        us_returns_raw = df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].values
        all_returns_raw = np.column_stack([us_returns_raw, y_jp_target])

        # PCA baseline correlation
        c_full = compute_baseline_correlation(
            all_returns_raw, sim_dates.values, self.ewma_half_life
        )
        v0_static = build_v3_static(self.n_u, self.n_j, self.include_v4_prior)
        base_vectors = build_base_vectors(self.n_u, self.n_j)
        v1, v2 = base_vectors["v1"], base_vectors["v2"]

        # Parse gap, beta, and topix night
        jp_gap = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].values
        jp_beta = (
            df_exec[[f"jp_beta_{tk}" for tk in JP_TICKERS]].values
            if any(c.startswith("jp_beta_") for c in df_exec.columns)
            else None
        )
        topix_night = (
            df_exec["topix_night_return"].values
            if "topix_night_return" in df_exec.columns
            else None
        )

        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )

        # TOPIX trade return
        topix_cc_trade = (
            df_exec["topix_cc_trade"].values
            if "topix_cc_trade" in df_exec.columns
            else df_exec["topix_night_return"].values + df_exec["topix_oc_return"].values
        )

        # Rolling OLS residualization for Residual-PCA/Residual-BLPX
        betas_jp_p3 = compute_rolling_ols_betas(
            y_jp_target, topix_cc_trade.reshape(-1, 1), self.beta_window
        )
        y_residuals_p3 = y_jp_target - betas_jp_p3[:, :, 0] * topix_cc_trade.reshape(-1, 1)

        # Replace JP columns with residuals for TOPIX-residualized returns
        jp_res_returns_p3 = all_returns_raw.copy()
        jp_res_returns_p3[:, self.n_u :] = y_residuals_p3

        c_full_p3 = compute_baseline_correlation(
            jp_res_returns_p3, sim_dates.values, self.ewma_half_life
        )

        return {
            "all_returns_raw": all_returns_raw,
            "c_full": c_full,
            "c_full_p3": c_full_p3,
            "v0_static": v0_static,
            "v1": v1,
            "v2": v2,
            "jp_gap": jp_gap,
            "jp_beta": jp_beta,
            "topix_night": topix_night,
            "y_jp_oc_df": y_jp_oc_df,
            "jp_res_returns_p3": jp_res_returns_p3,
            "y_jp_target": y_jp_target,
        }

    def compute_production_signal(
        self,
        i: int,
        c_full: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        all_returns: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the Raw-PCA (Production PCA) signal at index i."""
        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(self.n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_res = signals.compute_signal(
            all_returns=all_returns,
            current_index=i,
            n_u=self.n_u,
            corr_window=self.corr_window,
            c_full=c_full,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            k=self.k,
            lambda_reg=self.lambda_reg,
            lambda_lw=self.lambda_lw,
            lw_target=self.lw_target,
            ewma_half_life=self.ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_t1,
            gap_open_coef=self.gap_open_coef,
            topix_beta_coef=self.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=self.vol_adjusted_target,
        )
        return np.asarray(sig_res["signal"], dtype=float)

    def compute_residual_signal(
        self,
        jp_res_returns_p3: np.ndarray,
        i: int,
        c_full_p3: np.ndarray,
        v0_static: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        jp_gap: np.ndarray,
        jp_beta: np.ndarray | None,
        topix_night: np.ndarray | None,
    ) -> np.ndarray:
        """Compute the Residual-PCA (Residual target PCA) signal at index i."""
        gap_t1 = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(self.n_j)
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[i]) if topix_night is not None else None

        sig_res = signals.compute_signal(
            all_returns=jp_res_returns_p3,
            current_index=i,
            n_u=self.n_u,
            corr_window=self.corr_window,
            c_full=c_full_p3,
            v0_static=v0_static,
            v1=v1,
            v2=v2,
            k=self.k,
            lambda_reg=self.lambda_reg,
            lambda_lw=self.lambda_lw,
            lw_target=self.lw_target,
            ewma_half_life=self.ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_t1,
            gap_open_coef=self.gap_open_coef,
            topix_beta_coef=self.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            vol_adjusted_target=self.vol_adjusted_target,
        )
        return np.asarray(sig_res["signal"], dtype=float)

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
        window_start = max(0, current_index - self.blp_window)
        # Slices from window_start to current_index-1 (contains historical X and Y)
        window_returns = all_returns[window_start:current_index].copy()

        # 1. Execution-aware target adjustment (vol-scaling target)
        if self.exec_adjustment == "vol_scale" and rolling_std is not None:
            for idx_local in range(len(window_returns)):
                idx_global = window_start + idx_local
                vol_factor = rolling_std[idx_global]
                # Scale JP returns (columns self.n_u onwards) by rolling volatility
                window_returns[idx_local, self.n_u:] /= vol_factor

        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Robust winsorization
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

        # Estimate rolling mean, std, and correlation
        cache_key_corr = (current_index, self.blp_window, self.winsor_sigma, self.exec_adjustment, self.blp_ewma_halflife, is_residual)
        if cache_key_corr in _BLP_CORR_CACHE:
            mu, sigma, corr = _BLP_CORR_CACHE[cache_key_corr]
        else:
            mu, sigma, corr = compute_correlation(window_returns, self.blp_ewma_halflife)
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            np.fill_diagonal(corr, 1.0)
            _BLP_CORR_CACHE[cache_key_corr] = (mu, sigma, corr)

        # Partition correlation matrix (scale-standardized covariance)
        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]
        C_YY = corr[self.n_u :, self.n_u :]

        # Regularize Sigma_XX, Sigma_YX, Sigma_YY
        Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
        Sigma_YX_reg = (1.0 - self.alpha_yx) * C_YX
        Sigma_YY_reg = (1.0 - self.alpha_yy) * C_YY + self.alpha_yy * np.eye(self.n_j)

        # Scale-type Ridge matrix
        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
        ridge_matrix = self.rho * diag_mean * np.eye(self.n_u)
        A = Sigma_XX_reg + ridge_matrix

        # Solve for B_blp with pseudo-inverse fallback
        try:
            singular_values = np.linalg.svd(A, compute_uv=False)
            cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
        except Exception:
            cond_num = np.nan

        pinv_fallback = False
        try:
            if not np.isfinite(A).all():
                raise ValueError("A contains NaNs or Infs")
            inv_A = np.linalg.inv(A)
            B_blp = Sigma_YX_reg @ inv_A
        except Exception:
            pinv_fallback = True
            try:
                inv_A = np.linalg.pinv(A)
                B_blp = Sigma_YX_reg @ inv_A
            except Exception:
                B_blp = np.zeros((self.n_j, self.n_u))
                inv_A = np.zeros((self.n_u, self.n_u))

        # Low rank SVD projection
        if self.rank != "full" and self.rank is not None:
            rank_val = int(self.rank)
            if rank_val < min(B_blp.shape):
                try:
                    U, S, Vt = np.linalg.svd(B_blp, full_matrices=False)
                    B_blp = U[:, :rank_val] @ np.diag(S[:rank_val]) @ Vt[:rank_val, :]
                except Exception as e:
                    logger.warning(f"SVD rank reduction failed: {e}")

        # 3. Structured Shrinkage
        # PCA Prior
        B_pca = np.zeros((self.n_j, self.n_u))
        if v0_static is not None and c_full is not None and corr.shape == (32, 32) and v0_static.shape == (32, 6) and c_full.shape == (32, 32):
            # Reconstruct c_t_reg from corr (the 32x32 correlation matrix)
            c0_t = build_c0_from_v0(v0_static, c_full)
            c_t_reg = regularize_correlation(
                corr, c0_t, self.lambda_reg, self.lambda_lw, self.lw_target
            )
            # Eigen decomposition to match Raw-PCA/Residual-PCA logic
            eigvals, eigvecs = np.linalg.eigh(c_t_reg)
            sort_idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sort_idx]

            # Re-use eigenvectors for K = 6 standard
            v_t_k = eigvecs[:, : self.k]
            v_u_t_k = v_t_k[: self.n_u, :]
            v_j_t_k = v_t_k[self.n_u :, :]
            B_pca = v_j_t_k @ v_u_t_k.T

        norm_B_blp = np.linalg.norm(B_blp, "fro")

        # PCA scaling
        norm_B_pca = np.linalg.norm(B_pca, "fro")
        scale_pca = norm_B_blp / (norm_B_pca + 1e-12)
        B_pca_scaled = scale_pca * B_pca

        # Sector prior scaling
        if self.M_sector.shape == B_blp.shape:
            M_sector = self.M_sector
        else:
            M_sector = np.zeros(B_blp.shape)
        norm_M_sector = np.linalg.norm(M_sector, "fro")
        scale_sector = norm_B_blp / (norm_M_sector + 1e-12)
        B_sector = scale_sector * M_sector

        # Mix priors subject to sum constraints
        lambda_sum = self.lambda_pca + self.lambda_sector
        l_pca = self.lambda_pca
        l_sec = self.lambda_sector
        if lambda_sum > 0.75:
            # Scale down to sum to 0.75 max
            l_pca = (self.lambda_pca / lambda_sum) * 0.75
            l_sec = (self.lambda_sector / lambda_sum) * 0.75
            lambda_sum = 0.75

        B_struct = (1.0 - lambda_sum) * B_blp + l_pca * B_pca_scaled + l_sec * B_sector

        # Standardize current US input
        X_t = all_returns[current_index, : self.n_u]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu_X = mu[: self.n_u]
        sigma_X = sigma[: self.n_u]
        sigma_X_safe = np.where(sigma_X > 1e-8, sigma_X, 1.0)
        z_U_t = (X_t - mu_X) / sigma_X_safe

        # Predict standardized JP returns
        z_hat_j_t1 = B_struct @ z_U_t
        z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Confidence Weighting
        # conditional variance: Sigma_Y_given_X = Sigma_YY_reg - Sigma_YX_reg @ inv_A @ Sigma_XY_reg
        Sigma_XY_reg = Sigma_YX_reg.T
        Sigma_Y_given_X = Sigma_YY_reg - Sigma_YX_reg @ inv_A @ Sigma_XY_reg

        pred_var = np.maximum(np.diag(Sigma_Y_given_X), 0.0)
        var_floor = 1e-8
        pred_var_floored = np.maximum(pred_var, var_floor)
        num_floored = int(np.sum(pred_var < var_floor))

        if self.beta_conf > 0.0:
            z_hat_j_t1 = z_hat_j_t1 / (pred_var_floored ** self.beta_conf)
            z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)
            # Winsorize/clip final confidence weighted signal to avoid extreme values
            z_hat_j_t1 = np.clip(z_hat_j_t1, -5.0, 5.0)

        # Vol adjustment / denormalization
        mu_jp = mu[self.n_u :]
        sigma_jp = sigma[self.n_u :]
        if self.vol_adjusted_target:
            if current_index >= 20:
                jp_returns_20 = all_returns[current_index - 20 : current_index, self.n_u :]
                jp_returns_20 = np.nan_to_num(jp_returns_20, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
                sigma_j_t = np.maximum(sigma_j_t, 1e-8)
            else:
                sigma_j_t = sigma_jp
            r_hat_jp_cc = z_hat_j_t1 * sigma_j_t
        else:
            r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
        r_hat_jp_cc = np.nan_to_num(r_hat_jp_cc, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply gap override
        if gap_override is not None:
            gap_vec = np.asarray(gap_override, dtype=float).reshape(-1)
            use_topix = False
            if betas_t is not None and topix_night_t is not None:
                betas_vec = np.asarray(betas_t, dtype=float).reshape(-1)
                if (
                    betas_vec.shape == gap_vec.shape
                    and np.all(np.isfinite(betas_vec))
                    and np.isfinite(float(topix_night_t))
                ):
                    use_topix = True

            if use_topix:
                gap_syst = betas_vec * float(topix_night_t)
                gap_idio = gap_vec - gap_syst
                gap_filt = (
                    self.gap_open_coef * gap_idio
                    + (self.gap_open_coef - self.topix_beta_coef) * gap_syst
                )
                denom = np.maximum(1.0 + gap_filt, 0.1)
                signal = (1.0 + r_hat_jp_cc) / denom - 1.0
            else:
                signal = r_hat_jp_cc - self.gap_open_coef * gap_vec
        else:
            signal = z_hat_j_t1

        # Diagnostic metrics
        if return_matrices:
            return {
                "signal": signal,
                "z_hat_j_t1": z_hat_j_t1,
                "cond_num": cond_num,
                "b_norm": norm_B_blp,
                "b_pca_norm": float(np.linalg.norm(B_pca)),
                "b_sector_norm": float(np.linalg.norm(self.M_sector)),
                "b_struct_norm": float(np.linalg.norm(B_struct)),
                "sigma_xx_trace": float(np.trace(C_XX)),
                "sigma_yx_norm": float(np.linalg.norm(C_YX)),
                "sigma_yy_trace": float(np.trace(C_YY)),
                "min_pred_var": float(np.min(pred_var)),
                "max_pred_var": float(np.max(pred_var)),
                "num_pred_var_floored": num_floored,
                "pinv_fallback": pinv_fallback,
                "num_training_samples": len(window_returns),
                # Raw matrices
                "Sigma_XX": A,
                "Sigma_YX": Sigma_YX_reg,
                "Sigma_YY": Sigma_YY_reg,
                "inv_A": inv_A,
                "B_blp": B_blp,
                "B_pca_scaled": B_pca_scaled,
                "B_sector_scaled": B_sector,
                "B_struct": B_struct,
                "z_U": z_U_t,
                "pred_var_vec": pred_var,
                "sigma_X": sigma[:self.n_u],
                "sigma_Y": sigma[self.n_u:],
                "sigma_Y_denorm": sigma_j_t,
                "mu_X": mu[:self.n_u],
                "mu_Y": mu[self.n_u:],
            }

        return {
            "signal": signal,
            "z_hat_j_t1": z_hat_j_t1,
            "cond_num": cond_num,
            "b_norm": norm_B_blp,
            "b_pca_norm": float(np.linalg.norm(B_pca)),
            "b_sector_norm": float(np.linalg.norm(self.M_sector)),
            "b_struct_norm": float(np.linalg.norm(B_struct)),
            "sigma_xx_trace": float(np.trace(C_XX)),
            "sigma_yx_norm": float(np.linalg.norm(C_YX)),
            "sigma_yy_trace": float(np.trace(C_YY)),
            "min_pred_var": float(np.min(pred_var)),
            "max_pred_var": float(np.max(pred_var)),
            "num_pred_var_floored": num_floored,
            "pinv_fallback": pinv_fallback,
            "num_training_samples": len(window_returns),
        }

    def normalize_signals(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Cross-sectionally normalize the signal values."""
        if method == "identity":
            return sig
        centered = sig - np.median(sig)
        if method == "zscore":
            std = np.std(centered)
            std_safe = std if std > 1e-8 else 1.0
            return centered / std_safe
        elif method == "rank_normalize":
            ranks = pd.Series(sig).rank(pct=True).values
            return (ranks - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")

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

    def build_weights(self, signal: np.ndarray, q: float | None = None) -> np.ndarray:
        """Construct portfolio weights from combined signal."""
        q_val = q if q is not None else self.q
        return signals.build_weights(
            signal=signal,
            q=q_val,
            n_j=self.n_j,
            weight_mode=self.weight_mode,
            enforce_sign=False,
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
        cache_key_raw_pca_residual_pca = (id(df_exec), self.corr_window, self.k, self.lambda_reg, self.ewma_half_life)
        if cache_key_raw_pca_residual_pca in _RAW_PCA_RESIDUAL_PCA_CACHE:
            raw_pca_signals, residual_pca_signals = _RAW_PCA_RESIDUAL_PCA_CACHE[cache_key_raw_pca_residual_pca]
            raw_pca_cached = True
        else:
            raw_pca_cached = False

        for i in range(start_idx, T):
            if not raw_pca_cached:
                # 1. Raw-PCA (Production PCA)
                raw_pca_sig = self.compute_production_signal(
                    i, c_full, v0_static, v1, v2, all_returns_raw, jp_gap, jp_beta, topix_night
                )
                raw_pca_signals[i] = raw_pca_sig

                # 2. Residual-PCA (Residual target PCA)
                residual_pca_sig = self.compute_residual_signal(
                    jp_res_returns_p3, i, c_full_p3, v0_static, v1, v2, jp_gap, jp_beta, topix_night
                )
                residual_pca_signals[i] = residual_pca_sig
            else:
                raw_pca_sig = raw_pca_signals[i]
                residual_pca_sig = residual_pca_signals[i]

            # 3. Raw-BLPX (Enhanced BLP Raw target)
            gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
            betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
            topix_night_t = float(topix_night[i]) if topix_night is not None else None

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

            # 4. Residual-BLPX (Enhanced BLP Residual target)
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
