"""Pure gap adjustment functions for V2 production.

These functions mirror the gap-adjustment logic in
``tools/research/compute_gap_adjusted_distribution.py`` but are decoupled
from file I/O so they can be used on-demand in production.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def compute_filtered_gap(
    gap_override: np.ndarray,
    betas_t: np.ndarray,
    topix_night_t: float,
    gap_open_coef: float,
    topix_beta_coef: float,
    denominator_floor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute filtered gap and denominator.

    Args:
        gap_override: raw JP opening gap returns (n_j,)
        betas_t: per-ticker TOPIX betas (n_j,)
        topix_night_t: TOPIX overnight return (scalar)
        gap_open_coef: idiosyncratic gap coefficient (c). Callers must select
            ``gap_open_coef_neg`` when the US market is negative if asymmetric
            gap correction is configured.
        topix_beta_coef: TOPIX systematic coefficient (b). Same neg override
            applies.
        denominator_floor: floor applied to 1 + gap_filt

    Returns:
        (gap_filt, denominator, denominator_floored)
    """
    gap_syst = betas_t * topix_night_t
    gap_idio = gap_override - gap_syst
    gap_filt = gap_open_coef * gap_idio + (gap_open_coef - topix_beta_coef) * gap_syst
    denominator = 1.0 + gap_filt
    denominator_floored = np.maximum(denominator, denominator_floor)
    return gap_filt, denominator, denominator_floored


def compute_gap_adjusted_distribution(
    mu_raw: np.ndarray,
    omega_raw: np.ndarray,
    gap_override: np.ndarray,
    betas_t: np.ndarray,
    topix_night_t: float,
    gap_open_coef: float = 0.70,
    topix_beta_coef: float = 0.60,
    denominator_floor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gap-adjusted predictive distribution (mu_gap, Omega_gap).

    Pure function: the inputs ``mu_raw`` and ``omega_raw`` must already be
    reconstructed from the BLPX signal at the signal date. The 9:10 inputs
    ``gap_override``, ``betas_t``, ``topix_night_t`` become available on
    trade_date.

    ``gap_open_coef`` and ``topix_beta_coef`` must reflect the US-direction
    sensitive selection used inside the BLPX model (``gap_open_coef_neg``
    and ``topix_beta_coef_neg`` when the US market is negative and the
    asymmetric configuration is set).
    """
    _, _, denom_floored = compute_filtered_gap(
        gap_override,
        betas_t,
        topix_night_t,
        gap_open_coef,
        topix_beta_coef,
        denominator_floor,
    )
    d = 1.0 / denom_floored
    D = np.diag(d)
    mu_gap = (1.0 + mu_raw) * d - 1.0
    omega_gap = D @ omega_raw @ D
    omega_gap = 0.5 * (omega_gap + omega_gap.T)
    return mu_gap, omega_gap


def _omega_from_blp_res(blpx_result: dict) -> np.ndarray:
    """Reconstruct the standardized JP return correlation structure.

    Mirrors ``tools/research/compute_gap_adjusted_distribution.py``:

        Omega_struct = Sigma_YY
                       - B_struct @ Sigma_XY
                       - Sigma_YX @ B_struct.T
                       + B_struct @ Sigma_XX @ B_struct.T
    """
    Sigma_XX = blpx_result["Sigma_XX"]
    Sigma_YX = blpx_result["Sigma_YX"]
    Sigma_YY = blpx_result["Sigma_YY"]
    B_struct = blpx_result["B_struct"]
    Sigma_XY = Sigma_YX.T

    Omega_struct = (
        Sigma_YY
        - B_struct @ Sigma_XY
        - Sigma_YX @ B_struct.T
        + B_struct @ Sigma_XX @ B_struct.T
    )
    Omega_struct = 0.5 * (Omega_struct + Omega_struct.T)
    return cast(np.ndarray, Omega_struct)


def build_raw_distribution(
    blpx_result: dict,
    vol_adjusted_target: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (mu_raw, Omega_raw) from BLPX signal output.

    ``vol_adjusted_target`` changes the de-normalization of ``z_hat_j_t1``:
    - True: ``mu_raw = z_hat_j_t1 * sigma_Y_denorm``
    - False: ``mu_raw = mu_Y + sigma_Y * z_hat_j_t1``
    """
    z = blpx_result["z_hat_j_t1"]
    sigma = blpx_result["sigma_Y_denorm"]
    mu_y = blpx_result["mu_Y"]
    sigma_Y = blpx_result.get("sigma_Y", sigma)

    if vol_adjusted_target:
        mu_raw = z * sigma
    else:
        mu_raw = mu_y + sigma_Y * z

    corr = _omega_from_blp_res(blpx_result)
    D = np.diag(sigma)
    omega_raw = D @ corr @ D
    omega_raw = 0.5 * (omega_raw + omega_raw.T)
    return mu_raw, omega_raw

def denormalize_signal(
    z_hat_j_t1: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    all_returns: np.ndarray,
    current_index: int,
    n_u: int,
    vol_adjusted_target: bool,
) -> np.ndarray:
    """Denormalize standardized JP return predictions to raw return space."""
    mu_jp = mu[n_u:]
    sigma_jp = sigma[n_u:]
    if vol_adjusted_target:
        if current_index >= 20:
            jp_returns_20 = all_returns[current_index - 20 : current_index, n_u:]
            jp_returns_20 = np.nan_to_num(jp_returns_20, nan=0.0, posinf=0.0, neginf=0.0)
            sigma_j_t = np.std(jp_returns_20, axis=0, ddof=1)
            sigma_j_t = np.maximum(sigma_j_t, 1e-8)
        else:
            sigma_j_t = sigma_jp
        r_hat_jp_cc = z_hat_j_t1 * sigma_j_t
    else:
        r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
    return cast(np.ndarray, np.nan_to_num(r_hat_jp_cc, nan=0.0, posinf=0.0, neginf=0.0))


def apply_gap_adjustment(
    r_hat_jp_cc: np.ndarray,
    z_hat_j_t1: np.ndarray,
    gap_override: np.ndarray | None,
    betas_t: np.ndarray | None,
    topix_night_t: float | None,
    gap_open_coef: float = 0.7,
    topix_beta_coef: float = 0.6,
) -> np.ndarray:
    """Apply gap override adjustment to the predicted signal."""
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
            assert topix_night_t is not None
            gap_syst = betas_vec * topix_night_t
            gap_idio = gap_vec - gap_syst
            gap_filt = gap_open_coef * gap_idio + (gap_open_coef - topix_beta_coef) * gap_syst
            denom = np.maximum(1.0 + gap_filt, 0.1)
            signal = (1.0 + r_hat_jp_cc) / denom - 1.0
        else:
            signal = r_hat_jp_cc - gap_open_coef * gap_vec
    else:
        signal = z_hat_j_t1
    return signal

