"""BLPX signal computer helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from leadlag.core.blpx_math import (
    BLPXSignalParameters,
    compute_blp_signal_math,
    solve_blp_coefficients,
)

if TYPE_CHECKING:
    from leadlag.models.blpx.model import ProductionBLPXModel

def compute_blp_signal(
    self: ProductionBLPXModel,
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

    Returns at minimum ``signal`` (the gap-adjusted JP forecast). When
    ``return_matrices=True`` also returns the blocks needed to reconstruct
    the predictive distribution: ``Sigma_XX``, ``Sigma_YX``, ``Sigma_YY``,
    ``B_struct`` and ``z_U_t``.
    """
    # Window and correlation policies stay with the model instance.
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
