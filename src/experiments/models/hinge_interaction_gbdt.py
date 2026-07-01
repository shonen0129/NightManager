"""src/models/hinge_interaction_gbdt.py — Phase 2C GBDT Interaction Overlay.

Extends Sprint 3B with Gradient Boosting Decision Tree overlay for
asset-specific hinge interaction features.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from models.hinge_overlay import BaseHingeOverlay, select_best_alpha, ALPHA_GRID_DEFAULT
from models.hinge_interaction_overlay import impute_train_stats, apply_imputation

logger = logging.getLogger(__name__)

GBDT_PARAM_GRID_DEFAULT = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.7},
]


class InteractionGBDTOverlay(BaseHingeOverlay):
    """GBDT regression overlay for asset-specific interaction features.

    Parameters
    ----------
    n_estimators, max_depth, learning_rate, subsample:
        GBDT hyperparameters.
    model_name:
        Name identifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        min_samples_leaf: int = 20,
        model_name: str = "interaction_gbdt",
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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.MODEL_NAME = model_name

        self._scaler = StandardScaler()
        self._model: GradientBoostingRegressor | None = None
        self.train_medians_: np.ndarray | None = None
        self.train_stds_: np.ndarray | None = None

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.train_medians_, self.train_stds_ = impute_train_stats(X_train)
        X_imputed = apply_imputation(X_train, self.train_medians_, self.train_stds_)
        X_scaled = self._scaler.fit_transform(X_imputed)

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self._model.fit(X_scaled, y_train)
        logger.debug(
            "%s fitted: n_est=%d, depth=%d, lr=%.3f, n_features=%d, n_samples=%d",
            self.MODEL_NAME, self.n_estimators, self.max_depth,
            self.learning_rate, X_train.shape[1], X_train.shape[0],
        )

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(X))
        if self.train_medians_ is not None:
            X_imputed = apply_imputation(X, self.train_medians_, self.train_stds_)
        else:
            X_imputed = np.where(np.isnan(X), 0.0, X)
        X_scaled = self._scaler.transform(X_imputed)
        return self._model.predict(X_scaled)

    @classmethod
    def select_best_hyperparams(
        cls,
        X_train: np.ndarray,
        y_intraday_train: np.ndarray,
        mu_base_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mu_base_val: np.ndarray,
        param_grid: list[dict] | None = None,
        blend_alpha: float = 0.5,
        cap_overlay: bool = True,
        max_overlay_ratio: float = 0.5,
        max_overlay_bps: float = 20.0,
        model_name: str = "interaction_gbdt",
        blend_alpha_grid: list[float] | None = None,
    ) -> "InteractionGBDTOverlay":
        from scipy.stats import spearmanr

        if param_grid is None:
            param_grid = GBDT_PARAM_GRID_DEFAULT

        if len(X_train) == 0 or X_train.shape[1] == 0:
            best_model = cls(n_estimators=100, alpha=0.0, model_name=model_name)
            best_model._is_fitted = False
            return best_model

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
        best_params = None

        for params in param_grid:
            try:
                model = cls(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 3),
                    learning_rate=params.get("learning_rate", 0.05),
                    subsample=params.get("subsample", 0.8),
                    min_samples_leaf=params.get("min_samples_leaf", 20),
                    alpha=blend_alpha,
                    cap_overlay=cap_overlay,
                    max_overlay_ratio=max_overlay_ratio,
                    max_overlay_bps=max_overlay_bps,
                    model_name=model_name,
                )
                model.train_medians_ = train_medians
                model.train_stds_ = train_stds
                model._scaler = scaler
                model._model = GradientBoostingRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 3),
                    learning_rate=params.get("learning_rate", 0.05),
                    subsample=params.get("subsample", 0.8),
                    min_samples_leaf=params.get("min_samples_leaf", 20),
                    random_state=42,
                )
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
                    best_params = params

            except Exception as e:
                logger.debug("%s grid search failed: %s — %s", model_name, params, e)
                continue

        if best_model is None:
            logger.warning("%s: no model fitted. Using defaults, alpha=0.0.", model_name)
            best_model = cls(n_estimators=100, alpha=0.0, model_name=model_name)
            best_model._is_fitted = False
        else:
            grid = blend_alpha_grid if blend_alpha_grid is not None else ALPHA_GRID_DEFAULT
            best_blend = select_best_alpha(
                X_val, y_val, mu_base_val, best_model, alpha_grid=grid
            )
            best_model.alpha = best_blend

        logger.info(
            "%s: params=%s, blend_alpha=%.2f, val_IC=%.4f",
            model_name, best_params, best_model.alpha, best_ic,
        )
        return best_model
