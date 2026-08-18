"""BLPX correlation and window helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from leadlag.core.correlation import (
    compute_correlation,
    compute_stress_weight,
)

if TYPE_CHECKING:
    from leadlag.models.blpx.model import ProductionBLPXModel

logger = logging.getLogger("leadlag.models.blpx")


def _prepare_window_returns(
    self: ProductionBLPXModel,
    all_returns: np.ndarray,
    current_index: int,
    rolling_std: np.ndarray | None,
) -> np.ndarray:
    """Slice window returns, apply vol-scaling and winsorization."""
    window_start = max(0, current_index - self.blp_window)
    window_returns = all_returns[window_start:current_index].copy()

    if self.exec_adjustment == "vol_scale" and rolling_std is not None:
        vol_factors = rolling_std[window_start:current_index]
        window_returns[:, self.n_u :] /= vol_factors

    window_returns = np.nan_to_num(window_returns, nan=0.0, posinf=0.0, neginf=0.0)

    if self.winsor_sigma is not None:
        mus = np.mean(window_returns, axis=0)
        stds = np.std(window_returns, axis=0)
        for c in range(window_returns.shape[1]):
            if stds[c] > 1e-8:
                window_returns[:, c] = np.clip(
                    window_returns[:, c],
                    mus[c] - self.winsor_sigma * stds[c],
                    mus[c] + self.winsor_sigma * stds[c],
                )
    return cast(np.ndarray, window_returns)


def _estimate_correlation(
    self: ProductionBLPXModel,
    window_returns: np.ndarray,
    current_index: int,
    is_residual: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate rolling mean, std, and correlation with caching.

    When copula_enabled is True, the Pearson correlation is blended with
    a t-copula correlation matrix. The blend weight is either fixed
    (copula_dynamic_blend=False) or dynamically increased during stress
    periods (copula_dynamic_blend=True).
    """
    cache_key = (
        current_index,
        self.blp_window,
        self.winsor_sigma,
        self.exec_adjustment,
        self.blp_ewma_halflife,
        is_residual,
        self.copula_enabled,
    )
    if cache_key in self._blp_corr_cache:
        return cast(tuple[np.ndarray, np.ndarray, np.ndarray], self._blp_corr_cache[cache_key])

    use_copula = False
    copula_weight = 0.0

    if self.copula_enabled and self.copula_blend_weight > 0.0:
        if self.copula_dynamic_blend:
            w_stress = compute_stress_weight(
                window_returns,
                threshold=self.copula_stress_threshold,
            )
            copula_weight = self.copula_blend_weight * w_stress
        else:
            copula_weight = self.copula_blend_weight

        if copula_weight > 0.05:
            use_copula = True

    mu, sigma, corr = compute_correlation(
        window_returns,
        self.blp_ewma_halflife,
        use_copula=use_copula,
        copula_blend_weight=copula_weight,
        copula_nu_init=self.copula_nu_init,
        use_cache=False,
    )
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(corr, 1.0)
    self._blp_corr_cache[cache_key] = (mu, sigma, corr)
    return mu, sigma, corr
