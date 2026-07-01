"""Shared experiment model variants for BLP Enhanced experiments."""

from __future__ import annotations

import numpy as np

from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)


class BlendSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Blend static M_sector with rolling cross-correlation."""

    def __init__(self, config, blend_alpha=0.5):
        super().__init__(config)
        self.blend_alpha = blend_alpha

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        C_YX = corr[self.n_u:, :self.n_u]
        if C_YX.shape != B_blp.shape:
            return np.zeros(B_blp.shape)
        static = self.M_sector
        if static.shape != C_YX.shape:
            return C_YX.copy()
        return (1.0 - self.blend_alpha) * static + self.blend_alpha * C_YX


class CorrOnlySectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Use rolling cross-correlation directly as sector prior."""

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        C_YX = corr[self.n_u:, :self.n_u]
        if C_YX.shape == B_blp.shape:
            return C_YX.copy()
        return np.zeros(B_blp.shape)


class RidgeDynamicSectorModel(SectorRelativeEnsembleBLPEnhancedModel):
    """Use rolling ridge regression coefficients as sector prior.

    At each time step, regress JP target returns on US returns over the
    BLP window with a small ridge penalty, producing a data-driven mapping.
    """

    def __init__(self, config, ridge_rho=0.05):
        super().__init__(config)
        self.ridge_rho = ridge_rho

    def _get_sector_prior(self, current_index, all_returns, corr, B_blp):
        window_start = max(0, current_index - self.blp_window)
        W = all_returns[window_start:current_index]
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
        X = W[:, :self.n_u]
        Y = W[:, self.n_u:]
        # Ridge: B = Y^T X (X^T X + rho*I)^{-1}
        XtX = X.T @ X
        ridge = self.ridge_rho * np.mean(np.diag(XtX)) * np.eye(self.n_u)
        try:
            A_inv = np.linalg.inv(XtX + ridge)
            B_ridge = Y.T @ X @ A_inv
        except Exception:
            B_ridge = np.zeros((self.n_j, self.n_u))
        if B_ridge.shape == B_blp.shape:
            return B_ridge
        return np.zeros(B_blp.shape)
