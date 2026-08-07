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
        self.min_raw_weight = self._resolve_val("min_raw_weight", 0.0)
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

    def predict_signals(self, df_exec: pd.DataFrame, n_jobs: int = 1) -> dict[str, Any]:
        """Generate component and ensemble signals for all rows in df_exec."""
        from leadlag.core.pipeline import (
            BLPCombiner,
            BLPOutputAdapter,
            CallableComponent,
            CommonInputs,
            PCAComponent,
            SignalPipeline,
            _SRERawPCAComponent,
            _SREResidualPCAComponent,
        )

        T = len(df_exec)

        inputs_dict = self._prepare_common_inputs(df_exec)
        inputs = CommonInputs(
            all_returns_raw=inputs_dict["all_returns_raw"],
            c_full=inputs_dict["c_full"],
            c_full_p3=inputs_dict["c_full_p3"],
            v0_static=inputs_dict["v0_static"],
            v1=inputs_dict["v1"],
            v2=inputs_dict["v2"],
            jp_gap=inputs_dict["jp_gap"],
            jp_beta=inputs_dict["jp_beta"],
            topix_night=inputs_dict["topix_night"],
            y_jp_oc_df=inputs_dict["y_jp_oc_df"],
            jp_res_returns_p3=inputs_dict["jp_res_returns_p3"],
            y_jp_target=inputs_dict["y_jp_target"],
            n_u=self.n_u,
            n_j=self.n_j,
            dates=df_exec.index,
            p4=None,
        )

        pca_comp = PCAComponent(
            name="pca",
            n_u=self.n_u, n_j=self.n_j,
            corr_window=self.corr_window, k=self.k,
            lambda_reg=self.lambda_reg, lambda_lw=self.lambda_lw,
            lw_target=self.lw_target, ewma_half_life=self.ewma_half_life,
            gap_open_coef=self.gap_open_coef, topix_beta_coef=self.topix_beta_coef,
            vol_adjusted_target=self.vol_adjusted_target,
            min_raw_weight=getattr(self, "min_raw_weight", 0.0),
        )

        def _p5_fn(ctx):
            i = ctx.i
            inp = ctx.inputs
            gap_override = np.nan_to_num(inp.jp_gap[i], nan=0.0) if inp.jp_gap is not None else None
            betas_t = np.asarray(inp.jp_beta[i], dtype=float) if inp.jp_beta is not None else None
            topix_night_t = float(inp.topix_night[i]) if inp.topix_night is not None else None
            return self.compute_blp_signal(
                inp.all_returns_raw, i,
                gap_override=gap_override, betas_t=betas_t, topix_night_t=topix_night_t,
            )

        def _p5p3_fn(ctx):
            i = ctx.i
            inp = ctx.inputs
            gap_override = np.nan_to_num(inp.jp_gap[i], nan=0.0) if inp.jp_gap is not None else None
            betas_t = np.asarray(inp.jp_beta[i], dtype=float) if inp.jp_beta is not None else None
            topix_night_t = float(inp.topix_night[i]) if inp.topix_night is not None else None
            return self.compute_blp_signal(
                inp.jp_res_returns_p3, i,
                gap_override=gap_override, betas_t=betas_t, topix_night_t=topix_night_t,
            )

        components = [
            _SRERawPCAComponent(pca_comp),
            _SREResidualPCAComponent(pca_comp),
            CallableComponent("p5", _p5_fn),
            CallableComponent("p5p3", _p5p3_fn),
        ]

        combiner = BLPCombiner(
            raw_pca_weight=self.raw_pca_weight,
            residual_pca_weight=self.residual_pca_weight,
            p5_weight=self.p5_weight,
            p5p3_weight=self.p5p3_weight,
            normalization_method=self.normalization_method,
            n_j=self.n_j,
            normalize_fn=self.normalize_signals,
        )

        pipeline = SignalPipeline(components=components, combiner=combiner)
        pipeline_results = pipeline.run(inputs, start_idx=self.corr_window, T=T, n_jobs=n_jobs)

        adapter = BLPOutputAdapter(n_j=self.n_j, jp_tickers=JP_TICKERS)
        return adapter.adapt(pipeline_results, inputs)
