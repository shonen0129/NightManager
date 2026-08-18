"""BLPX signal computer helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

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

if TYPE_CHECKING:
    from leadlag.models.blpx.model import ProductionBLPXModel

logger = logging.getLogger("leadlag.models.blpx")

__all__ = [
    "_apply_confidence_weighting",
    "_build_blp_diagnostics",
    "_compute_pca_prior",
    "_estimate_correlation",
    "_prepare_window_returns",
    "_safe_solve_inv",
    "_solve_blp_coefficients",
    "_solve_tikhonov",
    "compute_blp_signal",
]


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
        Sigma_XX_reg, Sigma_YX_reg, B_pca, M_sector, diag_mean, B_blp
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
            C_YX_pos, C_YX_neg, C_XX, C_YY, B_pca, M_sector, B_blp
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
            signal = np.maximum(signal, 0.0) + (1.0 + self.asymmetry_post_gap_delta) * np.minimum(
                signal, 0.0
            )
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
