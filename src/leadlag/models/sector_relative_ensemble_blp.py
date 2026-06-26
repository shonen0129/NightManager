"""Sector Relative Ensemble with Regularized Block BLP Model.

Implements the PCA-Ensemble-BLP model which integrates standard Production signals (Raw-PCA, Residual-PCA)
with Regularized Block BLP signals (P5, P5P3).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from leadlag.core.correlation import compute_correlation
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.blp_base import _BLPBase

logger = logging.getLogger(__name__)


class SectorRelativeEnsembleBLPModel(_BLPBase):
    """Sector Relative Ensemble with Regularized Block BLP (PCA-Ensemble-BLP) Model.

    Ensembles standard PCA signals (Raw-PCA, Residual-PCA) and Regularized Block BLP signals (P5, P5P3).
    """

    def __init__(self, config: dict | object):
        """Initialize SectorRelativeEnsembleBLPModel.

        Args:
            config: Dict or object containing configuration options.
        """
        self.config = config
        self.n_u = len(US_TICKERS)
        self.n_j = len(JP_TICKERS)

        # Config resolution
        self.model_name = self._resolve_val("model_name", "sector_relative_ensemble_blp")
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

        # BLP Parameters (with standard defaults)
        self.blp_window = int(self._resolve_val("blp_window", 252))
        self.blp_ewma_halflife = self._resolve_val("blp_ewma_halflife", 45)
        self.alpha_xx = float(self._resolve_val("alpha_xx", 0.5))
        self.alpha_yx = float(self._resolve_val("alpha_yx", 0.25))
        self.rho = float(self._resolve_val("rho", 0.03))
        self.rank = self._resolve_val("rank", "full")

        # Ensemble weights
        self.raw_pca_weight = float(self._resolve_val("raw_pca_weight", 0.4))
        self.residual_pca_weight = float(self._resolve_val("residual_pca_weight", 0.4))
        self.p5_weight = float(self._resolve_val("p5_weight", 0.1))
        self.p5p3_weight = float(self._resolve_val("p5p3_weight", 0.1))

        # Slippage cost parameter resolution
        self.slippage_bps = self._resolve_slippage_bps()

    def compute_blp_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
    ) -> dict[str, Any]:
        """Compute the Regularized Block BLP signal for a single time step.

        Ensure Y_date <= signal_date by slicing up to current_index - 1.
        """
        window_start = max(0, current_index - self.blp_window)
        # Slices from window_start to current_index-1 (contains historical X and Y)
        window_returns = all_returns[window_start:current_index]
        window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Estimate rolling mean, std, and correlation
        mu, sigma, corr = compute_correlation(window_returns, self.blp_ewma_halflife)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        # Partition correlation matrix (scale-standardized covariance)
        C_XX = corr[: self.n_u, : self.n_u]
        C_YX = corr[self.n_u :, : self.n_u]

        # Regularize Sigma_XX and Sigma_YX
        Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
        Sigma_YX_reg = (1.0 - self.alpha_yx) * C_YX

        # Scale-type Ridge matrix
        diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
        ridge_matrix = self.rho * diag_mean * np.eye(self.n_u)
        A = Sigma_XX_reg + ridge_matrix

        # Solve for B_t with pseudo-inverse fallback
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
            B_t = Sigma_YX_reg @ inv_A
        except Exception:
            pinv_fallback = True
            try:
                inv_A = np.linalg.pinv(A)
                B_t = Sigma_YX_reg @ inv_A
            except Exception:
                B_t = np.zeros((self.n_j, self.n_u))

        # Low rank SVD projection
        if self.rank != "full" and self.rank is not None:
            rank_val = int(self.rank)
            if rank_val < min(B_t.shape):
                try:
                    U, S, Vt = np.linalg.svd(B_t, full_matrices=False)
                    B_t = U[:, :rank_val] @ np.diag(S[:rank_val]) @ Vt[:rank_val, :]
                except Exception as e:
                    logger.warning(f"SVD rank reduction failed: {e}")

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
        r_hat_jp_cc = self._denormalize_signal(
            z_hat_j_t1, mu, sigma, all_returns, current_index, self.n_u, self.vol_adjusted_target
        )

        # Apply gap override
        signal = self._apply_gap_adjustment(
            r_hat_jp_cc, z_hat_j_t1, gap_override, betas_t, topix_night_t
        )

        b_norm = float(np.linalg.norm(B_t))
        sigma_xx_trace = float(np.trace(C_XX))
        sigma_yx_norm = float(np.linalg.norm(C_YX))

        return {
            "signal": signal,
            "z_hat_j_t1": z_hat_j_t1,
            "cond_num": cond_num,
            "b_norm": b_norm,
            "sigma_xx_trace": sigma_xx_trace,
            "sigma_yx_norm": sigma_yx_norm,
            "pinv_fallback": pinv_fallback,
            "num_training_samples": len(window_returns),
        }

    def combine_signals(
        self, z0: np.ndarray, z3: np.ndarray, z5: np.ndarray, z5p3: np.ndarray
    ) -> np.ndarray:
        """Combine component signals with ensemble weights."""
        return (
            self.raw_pca_weight * z0
            + self.residual_pca_weight * z3
            + self.p5_weight * z5
            + self.p5p3_weight * z5p3
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
        raw_pca_signals = np.zeros((T, self.n_j))
        residual_pca_signals = np.zeros((T, self.n_j))
        p5_signals = np.zeros((T, self.n_j))
        p5p3_signals = np.zeros((T, self.n_j))
        combined_signals = np.zeros((T, self.n_j))
        normalized_combined_signals = np.zeros((T, self.n_j))

        # Track BLP diagnostics
        blp_diagnostics = []

        start_idx = self.corr_window
        for i in range(start_idx, T):
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

            # 3. P5 (Raw target BLP)
            gap_override = np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else None
            betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
            topix_night_t = float(topix_night[i]) if topix_night is not None else None

            p5_res = self.compute_blp_signal(
                all_returns_raw,
                i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
            )
            p5_signals[i] = p5_res["signal"]

            # 4. P5P3 (Residual target BLP)
            p5p3_res = self.compute_blp_signal(
                jp_res_returns_p3,
                i,
                gap_override=gap_override,
                betas_t=betas_t,
                topix_night_t=topix_night_t,
            )
            p5p3_signals[i] = p5p3_res["signal"]

            # Standard Z-score normalization of component signals
            z0 = self.normalize_signals(raw_pca_sig, self.normalization_method)
            z3 = self.normalize_signals(residual_pca_sig, self.normalization_method)
            z5 = self.normalize_signals(p5_res["signal"], self.normalization_method)
            z5p3 = self.normalize_signals(p5p3_res["signal"], self.normalization_method)

            # Combined PCA-Ensemble-BLP signal
            s_ens = self.combine_signals(z0, z3, z5, z5p3)
            combined_signals[i] = s_ens
            normalized_combined_signals[i] = self.normalize_signals(
                s_ens, self.normalization_method
            )

            # Diagnostics recording (e.g. log daily BLP states)
            date_str = sim_dates[i].strftime("%Y-%m-%d")
            blp_diagnostics.append(
                {
                    "date": date_str,
                    "p5_cond_num": p5_res["cond_num"],
                    "p5_b_norm": p5_res["b_norm"],
                    "p5_sigma_xx_trace": p5_res["sigma_xx_trace"],
                    "p5_sigma_yx_norm": p5_res["sigma_yx_norm"],
                    "p5_pinv_fallback": int(p5_res["pinv_fallback"]),
                    "p5_num_training_samples": p5_res["num_training_samples"],
                    "p5p3_cond_num": p5p3_res["cond_num"],
                    "p5p3_b_norm": p5p3_res["b_norm"],
                    "p5p3_sigma_xx_trace": p5p3_res["sigma_xx_trace"],
                    "p5p3_sigma_yx_norm": p5p3_res["sigma_yx_norm"],
                    "p5p3_pinv_fallback": int(p5p3_res["pinv_fallback"]),
                    "p5p3_num_training_samples": p5p3_res["num_training_samples"],
                }
            )

        # Build DataFrames
        raw_pca_df = pd.DataFrame(raw_pca_signals, index=sim_dates, columns=JP_TICKERS)
        residual_pca_df = pd.DataFrame(residual_pca_signals, index=sim_dates, columns=JP_TICKERS)
        p4_signals = np.zeros((T, self.n_j))
        p4_df = pd.DataFrame(p4_signals, index=sim_dates, columns=JP_TICKERS)
        p5_df = pd.DataFrame(p5_signals, index=sim_dates, columns=JP_TICKERS)
        p5p3_df = pd.DataFrame(p5p3_signals, index=sim_dates, columns=JP_TICKERS)
        combined_df = pd.DataFrame(combined_signals, index=sim_dates, columns=JP_TICKERS)
        normalized_df = pd.DataFrame(
            normalized_combined_signals, index=sim_dates, columns=JP_TICKERS
        )

        return {
            "raw_pca_signals": raw_pca_df,
            "residual_pca_signals": residual_pca_df,
            "p4_signals": p4_df,
            "p5_signals": p5_df,
            "p5p3_signals": p5p3_df,
            "signals": combined_df,
            "normalized_signals": normalized_df,
            "y_jp_oc_df": inputs["y_jp_oc_df"],
            "blp_diagnostics": pd.DataFrame(blp_diagnostics).set_index("date"),
        }
