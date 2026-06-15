"""Sector Relative Ensemble with Reduced-Rank Regression Model.

Implements the SRE-RRR model which integrates standard Production signals (P0, P3)
with Reduced-Rank Regression signals (P6, P6P3) and Lowrank BLP signals (P7, P7P3).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Any

from leadlag.core import signal as signals
from leadlag.core.correlation import (
    build_base_vectors,
    build_v3_static,
    compute_baseline_correlation,
    compute_correlation,
)
from leadlag.core.residualize import compute_rolling_ols_betas
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.base import BaseModel
from leadlag.models.sre import compute_jp_target_returns

logger = logging.getLogger(__name__)

_PRODUCTION_SIGNAL_CACHE = {}
_RESIDUAL_SIGNAL_CACHE = {}
_CORR_MATRIX_CACHE = {}
_PREPARE_COMMON_INPUTS_CACHE = {}




class SectorRelativeEnsembleRRRModel(BaseModel):
    """Sector Relative Ensemble with Reduced-Rank Regression (SRE-RRR) Model.

    Ensembles standard PCA signals (P0, P3) and RRR/Lowrank BLP signals.
    """

    def __init__(self, config: dict | object):
        """Initialize SectorRelativeEnsembleRRRModel.

        Args:
            config: Dict or object containing configuration options.
        """
        self.config = config
        self.n_u = len(US_TICKERS)
        self.n_j = len(JP_TICKERS)

        # Config resolution
        self.model_name = self._resolve_val("model_name", "sector_relative_ensemble_rrr")
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

        # RRR Parameters (with standard defaults)
        self.rrr_window = int(self._resolve_val("rrr_window", 252))
        self.rrr_ewma_halflife = self._resolve_val("rrr_ewma_halflife", 45)
        self.lambda_ridge = float(self._resolve_val("lambda_ridge", 0.03))
        self.lambda_prior = float(self._resolve_val("lambda_prior", 0.3))
        self.rank = self._resolve_val("rank", 3)
        self.variant = self._resolve_val("variant", "Lowrank_BLP")

        # BLP Prior parameters
        self.rho_blp = float(self._resolve_val("rho_blp", 0.03))
        self.alpha_xx = float(self._resolve_val("alpha_xx", 0.5))
        self.alpha_yx = float(self._resolve_val("alpha_yx", 0.25))

        # Ensemble weights
        self.p0_weight = float(self._resolve_val("p0_weight", 0.4))
        self.p3_weight = float(self._resolve_val("p3_weight", 0.4))
        self.p6_weight = float(self._resolve_val("p6_weight", 0.1))
        self.p6p3_weight = float(self._resolve_val("p6p3_weight", 0.1))
        self.p7_weight = float(self._resolve_val("p7_weight", 0.0))
        self.p7p3_weight = float(self._resolve_val("p7p3_weight", 0.0))

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_val("slippage_bps", 5.0)
        if isinstance(config, dict):
            if "costs" in config and "slippage_bps_per_side" in config["costs"]:
                self.slippage_bps = float(config["costs"]["slippage_bps_per_side"])

        # Instance-level caches to prevent multi-parameter grid search memory leaks
        self._rrr_signal_cache = {}
        self._pca_prior_matrix_cache = {}
        self._blp_prior_matrix_cache = {}

    def _resolve_val(self, key: str, default: any) -> any:
        """Resolve value from config object or dict."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        if isinstance(self.config, dict):
            if key in self.config:
                return self.config[key]
            for section in ["model", "ensemble", "portfolio", "costs", "residualization", "grids"]:
                if section in self.config and key in self.config[section]:
                    return self.config[section][key]
            # Translations
            if key == "model_name" and "name" in self.config.get("model", {}):
                return self.config["model"]["name"]
            if key == "k" and "k" in self.config.get("model", {}):
                return self.config["model"]["k"]
            if key == "q" and "long_short_frac" in self.config.get("portfolio", {}):
                return self.config["portfolio"]["long_short_frac"]
        return default

    def _prepare_common_inputs(self, df_exec: pd.DataFrame) -> dict:
        """Prepare inputs common to signal computation (raw targets, residuals, betas)."""
        cache_key = (
            len(df_exec),
            df_exec.index[0],
            df_exec.index[-1],
            self.ewma_half_life,
            self.beta_window,
            self.include_v4_prior,
        )
        global _PREPARE_COMMON_INPUTS_CACHE
        if cache_key in _PREPARE_COMMON_INPUTS_CACHE:
            return _PREPARE_COMMON_INPUTS_CACHE[cache_key]

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

        # Rolling OLS residualization for P3/P6P3/P7P3
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

        res = {
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
        }
        _PREPARE_COMMON_INPUTS_CACHE[cache_key] = res
        return res

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
        """Compute the P0 (Production PCA) signal at index i."""
        cache_key = (i, self.lambda_reg, self.k)
        global _PRODUCTION_SIGNAL_CACHE
        if cache_key in _PRODUCTION_SIGNAL_CACHE:
            return _PRODUCTION_SIGNAL_CACHE[cache_key].copy()

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
        sig = np.asarray(sig_res["signal"], dtype=float)
        _PRODUCTION_SIGNAL_CACHE[cache_key] = sig
        return sig


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
        """Compute the P3 (Residual target PCA) signal at index i."""
        cache_key = (i, self.lambda_reg, self.k)
        global _RESIDUAL_SIGNAL_CACHE
        if cache_key in _RESIDUAL_SIGNAL_CACHE:
            return _RESIDUAL_SIGNAL_CACHE[cache_key].copy()

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
        sig = np.asarray(sig_res["signal"], dtype=float)
        _RESIDUAL_SIGNAL_CACHE[cache_key] = sig
        return sig


    def compute_rrr_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        c_full_prior: np.ndarray,
        v0_static: np.ndarray | None = None,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
        variant_override: str | None = None,
    ) -> dict[str, Any]:
        """Compute the Reduced-Rank Regression signal for a single time step.

        Ensures lookahead-safety by slicing window_returns up to current_index - 1.
        """
        variant = variant_override if variant_override is not None else self.variant
        cache_key = (
            current_index,
            variant,
            self.rrr_window,
            self.rrr_ewma_halflife,
            self.rank,
            self.lambda_ridge,
            self.lambda_prior,
            self.rho_blp,
            self.alpha_xx,
            self.alpha_yx,
            id(c_full_prior),
        )
        if cache_key in self._rrr_signal_cache:
            cached = self._rrr_signal_cache[cache_key]
            return {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in cached.items()
            }


        window_start = max(0, current_index - self.rrr_window)
        window_returns = all_returns[window_start:current_index]
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Estimate rolling mean, std, and correlation
        corr_cache_key = (current_index, self.rrr_window, self.rrr_ewma_halflife)
        global _CORR_MATRIX_CACHE
        if corr_cache_key in _CORR_MATRIX_CACHE:
            mu, sigma, corr = _CORR_MATRIX_CACHE[corr_cache_key]
        else:
            mu, sigma, corr = compute_correlation(window_returns, self.rrr_ewma_halflife)
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
            sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            np.fill_diagonal(corr, 1.0)
            _CORR_MATRIX_CACHE[corr_cache_key] = (mu.copy(), sigma.copy(), corr.copy())

        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]

        diag_mean_XX = float(np.mean(np.diag(C_XX))) if len(C_XX) > 0 else 1.0

        # Compute priors if applicable
        B_prior = np.zeros((self.n_j, self.n_u))
        if variant == "PCA_prior_RRR" and v0_static is not None:
            pca_cache_key = (
                current_index,
                self.rrr_window,
                self.rrr_ewma_halflife,
                self.k,
                self.lambda_reg,
                self.lambda_lw,
                self.lw_target,
                id(c_full_prior),
            )
            if pca_cache_key in self._pca_prior_matrix_cache:
                B_prior = self._pca_prior_matrix_cache[pca_cache_key].copy()
            else:
                from leadlag.core.correlation import build_c0_from_v0, regularize_correlation
                c0_t = build_c0_from_v0(v0_static, c_full_prior)
                c_t_reg = regularize_correlation(
                    corr, c0_t, self.lambda_reg, self.lambda_lw, self.lw_target
                )
                try:
                    eigvals, eigvecs = np.linalg.eigh(c_t_reg)
                    sort_idx = np.argsort(eigvals)[::-1]
                    eigvecs = eigvecs[:, sort_idx]
                    v_t_k = eigvecs[:, : self.k]
                    v_u_t_k = v_t_k[: self.n_u, :]
                    v_j_t_k = v_t_k[self.n_u :, :]
                    B_prior = v_j_t_k @ v_u_t_k.T
                except Exception as e:
                    logger.warning(f"PCA prior SVD failed: {e}")
                self._pca_prior_matrix_cache[pca_cache_key] = B_prior.copy()
        elif variant in ["BLP_prior_RRR", "Lowrank_BLP"]:
            blp_cache_key = (
                current_index,
                self.rrr_window,
                self.rrr_ewma_halflife,
                self.rho_blp,
                self.alpha_xx,
                self.alpha_yx,
                id(c_full_prior),
            )
            if blp_cache_key in self._blp_prior_matrix_cache:
                B_prior = self._blp_prior_matrix_cache[blp_cache_key].copy()
            else:
                # Standard BLP calculation
                Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
                Sigma_YX_reg = (1.0 - self.alpha_yx) * C_YX
                diag_mean_blp = float(np.mean(np.diag(Sigma_XX_reg)))
                A_blp = Sigma_XX_reg + self.rho_blp * diag_mean_blp * np.eye(self.n_u)
                try:
                    inv_A_blp = np.linalg.inv(A_blp)
                    B_prior = Sigma_YX_reg @ inv_A_blp
                except Exception:
                    try:
                        inv_A_blp = np.linalg.pinv(A_blp)
                        B_prior = Sigma_YX_reg @ inv_A_blp
                    except Exception:
                        pass
                self._blp_prior_matrix_cache[blp_cache_key] = B_prior.copy()


        # Solve for B_base
        pinv_fallback = False
        cond_num = np.nan
        B_base = np.zeros((self.n_j, self.n_u))

        if variant == "RRR_pure":
            A = C_XX
            try:
                singular_values = np.linalg.svd(A, compute_uv=False)
                cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
            except Exception:
                pass
            try:
                inv_A = np.linalg.inv(A)
                B_base = C_YX @ inv_A
            except Exception:
                pinv_fallback = True
                try:
                    inv_A = np.linalg.pinv(A)
                    B_base = C_YX @ inv_A
                except Exception:
                    pass
        elif variant == "Ridge_RRR":
            A = C_XX + self.lambda_ridge * diag_mean_XX * np.eye(self.n_u)
            try:
                singular_values = np.linalg.svd(A, compute_uv=False)
                cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
            except Exception:
                pass
            try:
                inv_A = np.linalg.inv(A)
                B_base = C_YX @ inv_A
            except Exception:
                pinv_fallback = True
                try:
                    inv_A = np.linalg.pinv(A)
                    B_base = C_YX @ inv_A
                except Exception:
                    pass
        elif variant in ["PCA_prior_RRR", "BLP_prior_RRR"]:
            A = C_XX + (self.lambda_ridge + self.lambda_prior) * diag_mean_XX * np.eye(self.n_u)
            try:
                singular_values = np.linalg.svd(A, compute_uv=False)
                cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
            except Exception:
                pass
            try:
                inv_A = np.linalg.inv(A)
                B_base = (C_YX + self.lambda_prior * B_prior) @ inv_A
            except Exception:
                pinv_fallback = True
                try:
                    inv_A = np.linalg.pinv(A)
                    B_base = (C_YX + self.lambda_prior * B_prior) @ inv_A
                except Exception:
                    pass
        elif variant == "Lowrank_BLP":
            B_base = B_prior.copy()
            # Under Lowrank_BLP, A is the BLP A matrix for condition number diagnostics
            Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
            diag_mean_blp = float(np.mean(np.diag(Sigma_XX_reg)))
            A = Sigma_XX_reg + self.rho_blp * diag_mean_blp * np.eye(self.n_u)
            try:
                singular_values = np.linalg.svd(A, compute_uv=False)
                cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
            except Exception:
                pass

        # Apply SVD low rank approximation
        B_t = B_base.copy()
        effective_rank = self.n_u
        singular_values_B = np.zeros(self.n_u)
        if self.rank != "full" and self.rank is not None:
            rank_val = int(self.rank)
            if rank_val < min(B_t.shape):
                try:
                    U, S, Vt = np.linalg.svd(B_t, full_matrices=False)
                    singular_values_B = S
                    effective_rank = int(np.sum(S > 1e-7))
                    B_t = U[:, :rank_val] @ np.diag(S[:rank_val]) @ Vt[:rank_val, :]
                except Exception as e:
                    logger.warning(f"SVD rank reduction failed: {e}")
            else:
                try:
                    S = np.linalg.svd(B_t, compute_uv=False)
                    singular_values_B = S
                    effective_rank = int(np.sum(S > 1e-7))
                except Exception:
                    pass
        else:
            try:
                S = np.linalg.svd(B_t, compute_uv=False)
                singular_values_B = S
                effective_rank = int(np.sum(S > 1e-7))
            except Exception:
                pass

        # Standardize current US input
        X_t = all_returns[current_index, : self.n_u]
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu_X = mu[: self.n_u]
        sigma_X = sigma[: self.n_u]
        sigma_X_safe = np.where(sigma_X > 1e-8, sigma_X, 1.0)
        z_U_t = (X_t - mu_X) / sigma_X_safe

        # Predict standardized JP returns
        z_hat_j_t1 = B_t @ z_U_t
        z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)

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

        b_norm = float(np.linalg.norm(B_t))
        prior_norm = float(np.linalg.norm(B_prior))
        b_minus_prior_norm = float(np.linalg.norm(B_t - B_prior))

        res = {
            "signal": signal,
            "z_hat_j_t1": z_hat_j_t1,
            "cond_num": cond_num,
            "b_norm": b_norm,
            "prior_norm": prior_norm,
            "b_minus_prior_norm": b_minus_prior_norm,
            "pinv_fallback": pinv_fallback,
            "num_training_samples": len(window_returns),
            "effective_rank": effective_rank,
            "singular_values": singular_values_B,
            "B_prior": B_prior,
            "B_t": B_t,
        }
        self._rrr_signal_cache[cache_key] = {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in res.items()
        }
        return res


    def normalize_signals(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Cross-sectionally normalize the signal values."""
        if method == "identity":
            return sig
        centered = sig - np.median(sig)
        if method == "zscore":
            std = np.std(centered)
            # Safe std division
            std_safe = std if std > 1e-8 else 1.0
            return centered / std_safe
        elif method == "rank_normalize":
            ranks = pd.Series(sig).rank(pct=True).values
            return (ranks - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def combine_signals(
        self,
        z0: np.ndarray,
        z3: np.ndarray,
        z6: np.ndarray,
        z6p3: np.ndarray,
        z7: np.ndarray,
        z7p3: np.ndarray,
    ) -> np.ndarray:
        """Combine component signals with ensemble weights."""
        return (
            self.p0_weight * z0
            + self.p3_weight * z3
            + self.p6_weight * z6
            + self.p6p3_weight * z6p3
            + self.p7_weight * z7
            + self.p7p3_weight * z7p3
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

        # Setup output arrays
        p0_signals = np.zeros((T, self.n_j))
        p3_signals = np.zeros((T, self.n_j))
        p6_signals = np.zeros((T, self.n_j))
        p6p3_signals = np.zeros((T, self.n_j))
        p7_signals = np.zeros((T, self.n_j))
        p7p3_signals = np.zeros((T, self.n_j))
        combined_signals = np.zeros((T, self.n_j))
        normalized_combined_signals = np.zeros((T, self.n_j))

        # Diagnostics logs
        rrr_diagnostics = []

        start_idx = self.corr_window
        for i in range(start_idx, T):
            # 1. P0 (Production PCA)
            p0_sig = self.compute_production_signal(
                i, c_full, v0_static, v1, v2, all_returns_raw, jp_gap, jp_beta, topix_night
            )
            p0_signals[i] = p0_sig

            # 2. P3 (Residual target PCA)
            p3_sig = self.compute_residual_signal(
                jp_res_returns_p3, i, c_full_p3, v0_static, v1, v2, jp_gap, jp_beta, topix_night
            )
            p3_signals[i] = p3_sig

            # 3. P6 (RRR Raw target)
            gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
            betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
            topix_night_t = float(topix_night[i]) if topix_night is not None else None

            p6_res = self.compute_rrr_signal(
                all_returns_raw,
                i,
                c_full_prior=c_full,
                v0_static=v0_static,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
            )
            p6_signals[i] = p6_res["signal"]

            # 4. P6P3 (RRR Residual target)
            p6p3_res = self.compute_rrr_signal(
                jp_res_returns_p3,
                i,
                c_full_prior=c_full_p3,
                v0_static=v0_static,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
            )
            p6p3_signals[i] = p6p3_res["signal"]

            # 5. P7 (Lowrank_BLP Raw target)
            p7_res = self.compute_rrr_signal(
                all_returns_raw,
                i,
                c_full_prior=c_full,
                v0_static=v0_static,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
                variant_override="Lowrank_BLP",
            )
            p7_signals[i] = p7_res["signal"]

            # 6. P7P3 (Lowrank_BLP Residual target)
            p7p3_res = self.compute_rrr_signal(
                jp_res_returns_p3,
                i,
                c_full_prior=c_full_p3,
                v0_static=v0_static,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
                variant_override="Lowrank_BLP",
            )
            p7p3_signals[i] = p7p3_res["signal"]

            # Normalization
            z0 = self.normalize_signals(p0_sig, self.normalization_method)
            z3 = self.normalize_signals(p3_sig, self.normalization_method)
            z6 = self.normalize_signals(p6_res["signal"], self.normalization_method)
            z6p3 = self.normalize_signals(p6p3_res["signal"], self.normalization_method)
            z7 = self.normalize_signals(p7_res["signal"], self.normalization_method)
            z7p3 = self.normalize_signals(p7p3_res["signal"], self.normalization_method)

            # Combined SRE-RRR signal
            s_ens = self.combine_signals(z0, z3, z6, z6p3, z7, z7p3)
            combined_signals[i] = s_ens
            normalized_combined_signals[i] = self.normalize_signals(
                s_ens, self.normalization_method
            )

            # Diagnostics log daily
            date_str = sim_dates[i].strftime("%Y-%m-%d")
            s_vals = p6_res["singular_values"]
            s_top = float(s_vals[0]) if len(s_vals) > 0 else 0.0

            rrr_diagnostics.append(
                {
                    "date": date_str,
                    "variant": self.variant,
                    "rank": self.rank,
                    "effective_rank": p6_res["effective_rank"],
                    "singular_values_top": s_top,
                    "condition_number": p6_res["cond_num"],
                    "b_norm": p6_res["b_norm"],
                    "prior_norm": p6_res["prior_norm"],
                    "b_minus_prior_norm": p6_res["b_minus_prior_norm"],
                    "lambda_ridge": self.lambda_ridge,
                    "lambda_prior": self.lambda_prior,
                    "num_training_samples": p6_res["num_training_samples"],
                    "pinv_fallback": int(p6_res["pinv_fallback"]),
                }
            )

        p0_df = pd.DataFrame(p0_signals, index=sim_dates, columns=JP_TICKERS)
        p3_df = pd.DataFrame(p3_signals, index=sim_dates, columns=JP_TICKERS)
        p6_df = pd.DataFrame(p6_signals, index=sim_dates, columns=JP_TICKERS)
        p6p3_df = pd.DataFrame(p6p3_signals, index=sim_dates, columns=JP_TICKERS)
        p7_df = pd.DataFrame(p7_signals, index=sim_dates, columns=JP_TICKERS)
        p7p3_df = pd.DataFrame(p7p3_signals, index=sim_dates, columns=JP_TICKERS)
        combined_df = pd.DataFrame(combined_signals, index=sim_dates, columns=JP_TICKERS)
        normalized_df = pd.DataFrame(
            normalized_combined_signals, index=sim_dates, columns=JP_TICKERS
        )

        p4_df = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)

        return {
            "p0_signals": p0_df,
            "p3_signals": p3_df,
            "p4_signals": p4_df,
            "p6_signals": p6_df,
            "p6p3_signals": p6p3_df,
            "p7_signals": p7_df,
            "p7p3_signals": p7p3_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "y_jp_oc_df": inputs["y_jp_oc_df"],
            "rrr_diagnostics": pd.DataFrame(rrr_diagnostics).set_index("date") if len(rrr_diagnostics) > 0 else pd.DataFrame(),
        }
