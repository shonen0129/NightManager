"""src/models/hinge_ridge_overlay.py — Sprint 3-A Ridge Hinge Overlay.

Implements Ridge regression overlay:
    hat_e_{j,t} = Ridge(H_{j,t})

with hyperparameter selection on validation window.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.models.hinge_overlay import BaseHingeOverlay

logger = logging.getLogger(__name__)

RIDGE_ALPHA_GRID_DEFAULT = [0.1, 1.0, 10.0, 100.0]


class HingeRidgeOverlay(BaseHingeOverlay):
    """Ridge regression hinge overlay model.

    Fits a Ridge regression on hinge features to predict the residual
    (y_intraday - mu_base), then blends back with alpha.

    Regularization is cross-validated on the validation window during
    walk-forward.

    Parameters
    ----------
    ridge_alpha:
        Ridge regularization strength.
    fit_intercept:
        Whether to fit an intercept in Ridge.
    alpha:
        Blend parameter: mu_final = mu_base + alpha * cap(hat_e).
    cap_overlay:
        Whether to apply the conservative overlay cap.
    max_overlay_ratio:
        Cap = max_ratio * |mu_base|.
    max_overlay_bps:
        Cap = max_bps / 10000 (absolute).
    """

    MODEL_NAME = "hinge_ridge_overlay"

    def __init__(
        self,
        ridge_alpha: float = 1.0,
        fit_intercept: bool = True,
        alpha: float = 0.5,
        cap_overlay: bool = True,
        max_overlay_ratio: float = 0.5,
        max_overlay_bps: float = 20.0,
    ) -> None:
        super().__init__(
            alpha=alpha,
            cap_overlay=cap_overlay,
            max_overlay_ratio=max_overlay_ratio,
            max_overlay_bps=max_overlay_bps,
        )
        self.ridge_alpha = ridge_alpha
        self.fit_intercept = fit_intercept

        self._scaler = StandardScaler()
        self._model: Ridge | None = None

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit Ridge on standardized features."""
        X_scaled = self._scaler.fit_transform(X_train)
        self._model = Ridge(alpha=self.ridge_alpha, fit_intercept=self.fit_intercept)
        self._model.fit(X_scaled, y_train)
        logger.debug(
            "HingeRidgeOverlay fitted: ridge_alpha=%.3f, n_features=%d, n_samples=%d",
            self.ridge_alpha, X_train.shape[1], X_train.shape[0],
        )

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Predict with Ridge (handle NaN rows gracefully)."""
        if self._model is None:
            return np.zeros(len(X))

        # Impute NaN in X with zero (after scaling would be mean)
        X_clean = np.where(np.isnan(X), 0.0, X)
        X_scaled = self._scaler.transform(X_clean)
        return self._model.predict(X_scaled)

    @classmethod
    def select_best_ridge_alpha(
        cls,
        X_train: np.ndarray,
        y_intraday_train: np.ndarray,
        mu_base_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mu_base_val: np.ndarray,
        ridge_alpha_grid: list[float] = RIDGE_ALPHA_GRID_DEFAULT,
        blend_alpha: float = 0.5,
        cap_overlay: bool = True,
        max_overlay_ratio: float = 0.5,
        max_overlay_bps: float = 20.0,
    ) -> "HingeRidgeOverlay":
        """Select best ridge_alpha using validation Rank IC.

        Fits one model per ridge_alpha on training data, evaluates Rank IC
        on validation data, and returns the best fitted model.

        Parameters
        ----------
        X_train, y_intraday_train, mu_base_train:
            Training data.
        X_val, y_val, mu_base_val:
            Validation data.
        ridge_alpha_grid:
            Ridge regularization candidates.
        blend_alpha:
            Blend factor to use during validation alpha selection.
        cap_overlay, max_overlay_ratio, max_overlay_bps:
            Cap parameters.

        Returns
        -------
        HingeRidgeOverlay
            Best fitted model with best ridge_alpha and best blend alpha.
        """
        from scipy.stats import spearmanr
        from experiments.models.hinge_overlay import select_best_alpha

        best_ic = -np.inf
        best_model = None

        for r_alpha in ridge_alpha_grid:
            model = cls(
                ridge_alpha=r_alpha,
                alpha=blend_alpha,
                cap_overlay=cap_overlay,
                max_overlay_ratio=max_overlay_ratio,
                max_overlay_bps=max_overlay_bps,
            )
            model.fit(X_train, y_intraday_train, mu_base_train)

            if not model._is_fitted:
                continue

            # Evaluate on validation
            mu_pred = model.predict(X_val, mu_base_val)
            valid = ~(np.isnan(mu_pred) | np.isnan(y_val))
            if valid.sum() < 5:
                continue

            rho, _ = spearmanr(mu_pred[valid], y_val[valid])
            if rho > best_ic:
                best_ic = rho
                best_model = model

        if best_model is None:
            logger.warning(
                "HingeRidgeOverlay: no model fitted successfully. Using ridge_alpha=1.0."
            )
            best_model = cls(ridge_alpha=1.0, alpha=0.0)
            best_model._is_fitted = False
        else:
            # Also select best blend alpha on validation
            from experiments.models.hinge_overlay import ALPHA_GRID_DEFAULT
            best_alpha = select_best_alpha(
                X_val, y_val, mu_base_val, best_model,
                alpha_grid=ALPHA_GRID_DEFAULT,
            )
            best_model.alpha = best_alpha

        logger.info(
            "HingeRidgeOverlay selection: ridge_alpha=%.3f, blend_alpha=%.2f, val_IC=%.4f",
            best_model.ridge_alpha, best_model.alpha, best_ic,
        )
        return best_model
