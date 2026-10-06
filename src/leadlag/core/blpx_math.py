"""Shared, explicit-input BLPX linear algebra for production and research."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from leadlag.core.correlation import build_c0_from_v0, regularize_correlation
from leadlag.data.tickers import US_TICKERS

logger = logging.getLogger(__name__)


def safe_solve_inverse(
    A: np.ndarray, B: np.ndarray, label: str = "A"
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve B @ inv(A) with pseudo-inverse fallback."""
    pinv_fallback = False
    try:
        if not np.isfinite(A).all():
            raise ValueError(f"{label} contains NaNs or Infs")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            inv_A = np.linalg.inv(A)
            result = B @ inv_A
        if not np.isfinite(inv_A).all() or not np.isfinite(result).all():
            raise ValueError(f"{label} solve is nonfinite")
    except (np.linalg.LinAlgError, ValueError):
        pinv_fallback = True
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                inv_A = np.linalg.pinv(A)
                result = B @ inv_A
            if not np.isfinite(inv_A).all() or not np.isfinite(result).all():
                raise ValueError(f"{label} pseudo-inverse solve is nonfinite")
        except (np.linalg.LinAlgError, ValueError):
            result = np.zeros((B.shape[0], A.shape[1]))
            inv_A = np.zeros((A.shape[0], A.shape[1]))
    return (result, inv_A, pinv_fallback)


def solve_blp_coefficients(
    corr: np.ndarray,
    *,
    alpha_xx: float,
    alpha_yx: float,
    alpha_yy: float,
    n_j: int,
    n_u: int,
    rank: Any,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Regularize correlation, solve for B_blp via ridge regression, and apply SVD rank reduction.

    Returns (B_blp, Sigma_XX_reg, Sigma_YX_reg, Sigma_YY_reg, cond_num, pinv_fallback).
    """
    C_XX = corr[:n_u, :n_u]
    C_YX = corr[n_u:, :n_u]
    C_YY = corr[n_u:, n_u:]
    Sigma_XX_reg = (1.0 - alpha_xx) * C_XX + alpha_xx * np.eye(n_u)
    Sigma_YX_reg = (1.0 - alpha_yx) * C_YX
    Sigma_YY_reg = (1.0 - alpha_yy) * C_YY + alpha_yy * np.eye(n_j)
    diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
    ridge_matrix = rho * diag_mean * np.eye(n_u)
    A = Sigma_XX_reg + ridge_matrix
    try:
        singular_values = np.linalg.svd(A, compute_uv=False)
        cond_num = float(singular_values[0] / np.maximum(singular_values[-1], 1e-12))
    except np.linalg.LinAlgError:
        cond_num = np.nan
    B_blp, _, pinv_fallback = safe_solve_inverse(A, Sigma_YX_reg, label="A")
    if rank != "full" and rank is not None:
        rank_val = int(rank)
        if rank_val < min(B_blp.shape):
            try:
                U, S, Vt = np.linalg.svd(B_blp, full_matrices=False)
                B_blp = U[:, :rank_val] @ np.diag(S[:rank_val]) @ Vt[:rank_val, :]
            except np.linalg.LinAlgError as e:
                logger.warning(f"SVD rank reduction failed: {e}")
    return (B_blp, Sigma_XX_reg, Sigma_YX_reg, Sigma_YY_reg, cond_num, pinv_fallback)


def compute_pca_prior(
    corr: np.ndarray,
    v0_static: np.ndarray | None,
    c_full: np.ndarray | None,
    *,
    k: int,
    lambda_lw: float,
    lambda_reg: float,
    lw_target: Any,
    n_j: int,
    n_u: int,
    min_raw_weight: float = 0.0,
) -> np.ndarray:
    """Compute PCA prior B_pca from eigen decomposition of regularized correlation."""
    B_pca = np.zeros((n_j, n_u))
    n = n_u + n_j
    if (
        v0_static is not None
        and c_full is not None
        and (corr.shape == (n, n))
        and (v0_static.ndim == 2)
        and (v0_static.shape[0] == n)
        and (c_full.shape == (n, n))
    ):
        c0_t = build_c0_from_v0(v0_static, c_full)
        c_t_reg = regularize_correlation(
            corr, c0_t, lambda_reg, lambda_lw, lw_target, min_raw_weight
        )
        eigvals, eigvecs = np.linalg.eigh(c_t_reg)
        sort_idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sort_idx]
        v_t_k = eigvecs[:, :k]
        v_u_t_k = v_t_k[:n_u, :]
        v_j_t_k = v_t_k[n_u:, :]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            B_pca = v_j_t_k @ v_u_t_k.T
    return B_pca


def solve_tikhonov(
    Sigma_XX_reg: np.ndarray,
    Sigma_YX_reg: np.ndarray,
    B_pca: np.ndarray,
    M_sector: np.ndarray,
    diag_mean: float,
    B_blp: np.ndarray | None = None,
    *,
    frobenius_scale_priors: bool,
    lambda_pca: float,
    lambda_sector: float,
    n_u: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-target Tikhonov regularization solve.

    When frobenius_scale_priors is True, B_pca and M_sector are scaled to
    match ||B_blp||_F before being added to the RHS, per the model spec.

    Returns (B_struct, inv_A_tikh).
    """
    l_pca = lambda_pca
    l_sec = lambda_sector
    lambda_sum = l_pca + l_sec
    if lambda_sum > 0.75:
        l_pca = lambda_pca / lambda_sum * 0.75
        l_sec = lambda_sector / lambda_sum * 0.75
    B_pca_used = B_pca
    M_sector_used = M_sector
    if frobenius_scale_priors and B_blp is not None:
        b_blp_norm = np.linalg.norm(B_blp, "fro")
        b_pca_norm = np.linalg.norm(B_pca, "fro")
        m_sector_norm = np.linalg.norm(M_sector, "fro")
        if b_blp_norm > 1e-12:
            if b_pca_norm > 1e-12:
                B_pca_used = B_pca * (b_blp_norm / b_pca_norm)
            if m_sector_norm > 1e-12:
                M_sector_used = M_sector * (b_blp_norm / m_sector_norm)
    lambda_tikh = rho * diag_mean + l_pca + l_sec
    A_tikh = Sigma_XX_reg + lambda_tikh * np.eye(n_u)
    rhs = Sigma_YX_reg + l_pca * B_pca_used + l_sec * M_sector_used
    B_struct, inv_A_tikh, _ = safe_solve_inverse(A_tikh, rhs, label="A_tikh")
    return (B_struct, inv_A_tikh)


def apply_confidence_weighting(
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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Sigma_Y_given_X = Sigma_YY_reg - Sigma_YX_reg @ inv_A_tikh @ Sigma_XY_reg
    pred_var = np.maximum(np.diag(Sigma_Y_given_X), 0.0)
    var_floor = 1e-08
    pred_var_floored = np.maximum(pred_var, var_floor)
    num_floored = int(np.sum(pred_var < var_floor))
    if beta_conf > 0.0:
        z_hat_j_t1 = z_hat_j_t1 / pred_var_floored**beta_conf
        z_hat_j_t1 = np.nan_to_num(z_hat_j_t1, nan=0.0, posinf=0.0, neginf=0.0)
        z_hat_j_t1 = np.clip(z_hat_j_t1, -5.0, 5.0)
    return (z_hat_j_t1, pred_var, num_floored)


def build_blp_diagnostics(
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
        diag.update(
            {
                "Sigma_XX": A,
                "Sigma_YX": Sigma_YX_reg,
                "Sigma_YY": Sigma_YY_reg,
                "inv_A": inv_A_tikh,
                "B_blp": B_blp,
                "B_pca_prior": B_pca,
                "B_sector_prior": M_sector,
                "B_struct": B_struct,
                "z_U_t": z_U_t,
                "pred_var_vec": pred_var,
                "sigma_X": sigma[: len(US_TICKERS)] if sigma is not None else None,
                "sigma_Y": sigma[len(US_TICKERS) :] if sigma is not None else None,
                "sigma_Y_denorm": sigma_j_t,
                "mu_X": mu[: len(US_TICKERS)] if mu is not None else None,
                "mu_Y": mu[len(US_TICKERS) :] if mu is not None else None,
            }
        )
    return diag


def solve_asymmetric_blp(
    C_YX_pos: np.ndarray,
    C_YX_neg: np.ndarray,
    C_XX: np.ndarray,
    C_YY: np.ndarray,
    B_pca: np.ndarray,
    M_sector: np.ndarray,
    B_blp: np.ndarray | None = None,
    *,
    alpha_xx: float,
    alpha_yx: float,
    alpha_yy: float,
    frobenius_scale_priors: bool,
    lambda_pca: float,
    lambda_sector: float,
    n_j: int,
    n_u: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve BLP coefficients separately for positive and negative regimes.

    Returns:
        B_pos_struct, B_neg_struct, inv_A_avg, Sigma_YX_reg_avg
    """
    Sigma_XX_reg = (1.0 - alpha_xx) * C_XX + alpha_xx * np.eye(n_u)
    Sigma_YX_reg_pos = (1.0 - alpha_yx) * C_YX_pos
    Sigma_YX_reg_neg = (1.0 - alpha_yx) * C_YX_neg
    (1.0 - alpha_yy) * C_YY + alpha_yy * np.eye(n_j)
    diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))
    B_pos_struct, inv_A_pos = solve_tikhonov(
        Sigma_XX_reg,
        Sigma_YX_reg_pos,
        B_pca,
        M_sector,
        diag_mean,
        B_blp,
        frobenius_scale_priors=frobenius_scale_priors,
        lambda_pca=lambda_pca,
        lambda_sector=lambda_sector,
        n_u=n_u,
        rho=rho,
    )
    B_neg_struct, inv_A_neg = solve_tikhonov(
        Sigma_XX_reg,
        Sigma_YX_reg_neg,
        B_pca,
        M_sector,
        diag_mean,
        B_blp,
        frobenius_scale_priors=frobenius_scale_priors,
        lambda_pca=lambda_pca,
        lambda_sector=lambda_sector,
        n_u=n_u,
        rho=rho,
    )
    inv_A_avg = 0.5 * (inv_A_pos + inv_A_neg)
    Sigma_YX_reg_avg = 0.5 * (Sigma_YX_reg_pos + Sigma_YX_reg_neg)
    return (B_pos_struct, B_neg_struct, inv_A_avg, Sigma_YX_reg_avg)
