"""Pure one-day gap-distribution reconstruction.

The Step 2 research command and the V2 on-demand source historically carried
the same raw-distribution and gap-adjustment algebra in separate functions.
This module owns that algebra only.  File reads, model execution, bundle
publication, and diagnostics stay in their respective callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from leadlag.core.gap_adjustment import (
    build_raw_distribution,
    compute_filtered_gap,
    compute_gap_adjusted_distribution,
)


@dataclass(frozen=True)
class GapDistributionComputation:
    """Pure calculation result for one trade date and one horizon."""

    mu_raw: np.ndarray
    omega_raw: np.ndarray
    mu_gap: np.ndarray
    omega_gap: np.ndarray
    gap_filt: np.ndarray
    denominator: np.ndarray
    denominator_floored: np.ndarray


def select_gap_coefficients(model: Any, blpx_result: dict) -> tuple[float, float]:
    """Select the directional gap coefficients used by every caller.

    Production BLPX applies the negative-US regime override while building its
    signal.  The distribution reconstruction must use the same coefficients;
    otherwise cached Step 2 μ differs from the on-demand fallback even when
    their BLPX inputs are identical.
    """
    gap_open_coef = float(model.gap_open_coef)
    topix_beta_coef = float(model.topix_beta_coef)
    z_u = blpx_result.get("z_U_t")
    if z_u is None:
        return gap_open_coef, topix_beta_coef
    us_negative = float(np.nanmean(z_u)) < 0.0
    gap_open_neg = getattr(model, "gap_open_coef_neg", None)
    topix_beta_neg = getattr(model, "topix_beta_coef_neg", None)
    if us_negative and gap_open_neg is not None and topix_beta_neg is not None:
        try:
            gap_open_coef = float(gap_open_neg)
            topix_beta_coef = float(topix_beta_neg)
        except (TypeError, ValueError):
            # Compatibility test doubles and legacy models may expose the
            # attributes without numeric negative-regime coefficients.
            pass
    return gap_open_coef, topix_beta_coef


def compute_gap_distribution(
    blpx_result: dict,
    *,
    gap_override: np.ndarray,
    betas_t: np.ndarray,
    topix_night_t: float,
    vol_adjusted_target: bool,
    gap_open_coef: float,
    topix_beta_coef: float,
    denominator_floor: float = 0.1,
    omega_struct: np.ndarray | None = None,
) -> GapDistributionComputation:
    """Reconstruct raw and gap-adjusted parameters from one BLPX result.

    The function performs no I/O and does not mutate any input array.  It is
    intentionally shared by the production on-demand path and the Step 2
    research command.  ``blpx_result`` is the already-computed, horizon-specific
    BLPX output; callers retain responsibility for PIT and signal-date checks.
    """
    if omega_struct is not None:
        z = np.asarray(blpx_result["z_hat_j_t1"], dtype=float)
        sigma = np.asarray(blpx_result["sigma_Y_denorm"], dtype=float)
        mu_y = np.asarray(blpx_result["mu_Y"], dtype=float)
        sigma_y = np.asarray(blpx_result.get("sigma_Y", sigma), dtype=float)
        mu_raw = z * sigma if vol_adjusted_target else mu_y + sigma_y * z
        structure = np.asarray(omega_struct, dtype=float)
        omega_raw = np.diag(sigma) @ structure @ np.diag(sigma)
        omega_raw = 0.5 * (omega_raw + omega_raw.T)
    else:
        mu_raw, omega_raw = build_raw_distribution(
            blpx_result,
            vol_adjusted_target=vol_adjusted_target,
        )
    gap_filt, denominator, denominator_floored = compute_filtered_gap(
        np.asarray(gap_override, dtype=float),
        np.asarray(betas_t, dtype=float),
        float(topix_night_t),
        float(gap_open_coef),
        float(topix_beta_coef),
        denominator_floor=float(denominator_floor),
    )
    mu_gap, omega_gap = compute_gap_adjusted_distribution(
        mu_raw,
        omega_raw,
        np.asarray(gap_override, dtype=float),
        np.asarray(betas_t, dtype=float),
        float(topix_night_t),
        gap_open_coef=float(gap_open_coef),
        topix_beta_coef=float(topix_beta_coef),
        denominator_floor=float(denominator_floor),
    )
    return GapDistributionComputation(
        mu_raw=mu_raw,
        omega_raw=omega_raw,
        mu_gap=mu_gap,
        omega_gap=omega_gap,
        gap_filt=gap_filt,
        denominator=denominator,
        denominator_floored=denominator_floored,
    )


__all__ = ["GapDistributionComputation", "compute_gap_distribution", "select_gap_coefficients"]
