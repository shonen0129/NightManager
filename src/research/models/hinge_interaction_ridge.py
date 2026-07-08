"""src/models/hinge_interaction_ridge.py — Sprint 3-B Ridge Interaction Overlay.

Implements Ridge regression overlay for asset-specific hinge interaction features.
Each (date, ticker) is a sample, so features vary cross-sectionally.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research.models.hinge_overlay import BaseHingeOverlay
from research.models.hinge_interaction_overlay import impute_train_stats, apply_imputation

logger = logging.getLogger(__name__)

RIDGE_ALPHA_GRID_DEFAULT = [0.1, 1.0, 10.0, 100.0, 300.0]


class InteractionRidgeOverlay(BaseHingeOverlay):
    """Ridge regression overlay for asset-specific interaction features.

    Compared to Sprint 3-A HingeRidgeOverlay:
    - X is (n_date × n_ticker, n_features): asset-specific, not date-only
    - Imputation uses train-derived medians (not zero)
    - Stores train_medians_ and train_stds_ for consistent test-time imputation

    Parameters
    ----------
    ridge_alpha:
        Ridge regularization strength.
    fit_intercept:
        Whether to fit an intercept.
    model_name:
        Name identifier for this model (e.g., 'macro_hinge_x_asset_beta_ridge').
    """

    def __init__(
        self,
        ridge_alpha: float = 1.0,
        fit_intercept: bool = True,
        model_name: str = "interaction_ridge",
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
        self.MODEL_NAME = model_name

        self._scaler = StandardScaler()
        self._model: Ridge | None = None
        self.train_medians_: np.ndarray | None = None
        self.train_stds_: np.ndarray | None = None

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit Ridge on standardized, imputed features."""
        # Store imputation stats from train
        self.train_medians_, self.train_stds_ = impute_train_stats(X_train)
        X_imputed = apply_imputation(X_train, self.train_medians_, self.train_stds_)

        X_scaled = self._scaler.fit_transform(X_imputed)
        self._model = Ridge(alpha=self.ridge_alpha, fit_intercept=self.fit_intercept)
        self._model.fit(X_scaled, y_train)

        logger.debug(
            "%s fitted: ridge_alpha=%.3f, n_features=%d, n_samples=%d",
            self.MODEL_NAME, self.ridge_alpha, X_train.shape[1], X_train.shape[0],
        )

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Predict with Ridge using train-derived imputation."""
        if self._model is None:
            return np.zeros(len(X))

        # Apply train-derived imputation
        if self.train_medians_ is not None:
            X_imputed = apply_imputation(X, self.train_medians_, self.train_stds_)
        else:
            X_imputed = np.where(np.isnan(X), 0.0, X)

        X_scaled = self._scaler.transform(X_imputed)
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
        model_name: str = "interaction_ridge",
        alpha_grid: list[float] | None = None,
    ) -> "InteractionRidgeOverlay":
        """Select best ridge_alpha using validation Rank IC.

        Returns the best fitted model.
        """
        from scipy.stats import spearmanr
        from research.models.hinge_overlay import select_best_alpha, ALPHA_GRID_DEFAULT

        if len(X_train) == 0 or X_train.shape[1] == 0:
            best_model = cls(ridge_alpha=100.0, alpha=0.0, model_name=model_name)
            best_model._is_fitted = False
            return best_model

        # Pre-impute and scale once
        train_medians, train_stds = impute_train_stats(X_train)
        X_tr_imputed = apply_imputation(X_train, train_medians, train_stds)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imputed)

        if len(X_val) > 0 and X_val.shape[1] > 0:
            X_val_imputed = apply_imputation(X_val, train_medians, train_stds)
            X_val_scaled = scaler.transform(X_val_imputed)
        else:
            X_val_scaled = np.empty((0, X_train.shape[1]))

        best_ic = -np.inf
        best_model = None

        for r_alpha in ridge_alpha_grid:
            model = cls(
                ridge_alpha=r_alpha,
                alpha=blend_alpha,
                cap_overlay=cap_overlay,
                max_overlay_ratio=max_overlay_ratio,
                max_overlay_bps=max_overlay_bps,
                model_name=model_name,
            )
            model.train_medians_ = train_medians
            model.train_stds_ = train_stds
            model._scaler = scaler
            model._model = Ridge(alpha=r_alpha, fit_intercept=model.fit_intercept)
            model._model.fit(X_tr_scaled, y_intraday_train)
            model._is_fitted = True

            if len(X_val_scaled) == 0:
                continue

            mu_pred = model.predict(X_val, mu_base_val)
            valid = ~(np.isnan(mu_pred) | np.isnan(y_val))
            if valid.sum() < 5:
                continue

            rho, _ = spearmanr(mu_pred[valid], y_val[valid])
            if rho > best_ic:
                best_ic = rho
                best_model = model

        if best_model is None:
            logger.warning("%s: no model fitted. Using ridge_alpha=100.0, alpha=0.0.", model_name)
            best_model = cls(ridge_alpha=100.0, alpha=0.0, model_name=model_name)
            best_model._is_fitted = False
        else:
            # Select best blend alpha on validation
            grid = alpha_grid if alpha_grid is not None else ALPHA_GRID_DEFAULT
            best_blend = select_best_alpha(
                X_val, y_val, mu_base_val, best_model, alpha_grid=grid
            )
            best_model.alpha = best_blend

        logger.info(
            "%s: ridge_alpha=%.3f, blend_alpha=%.2f, val_IC=%.4f",
            model_name, best_model.ridge_alpha, best_model.alpha, best_ic,
        )
        return best_model
