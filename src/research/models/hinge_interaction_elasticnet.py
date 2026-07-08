"""src/models/hinge_interaction_elasticnet.py — Sprint 3-B ElasticNet Interaction Overlay.

Implements ElasticNet regression overlay for asset-specific hinge interaction features.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from research.models.hinge_overlay import BaseHingeOverlay
from research.models.hinge_interaction_overlay import impute_train_stats, apply_imputation

logger = logging.getLogger(__name__)

ELASTICNET_ALPHA_GRID_DEFAULT = [0.0001, 0.001, 0.01, 0.1]
ELASTICNET_L1_RATIO_GRID_DEFAULT = [0.1, 0.3, 0.5, 0.7]


class InteractionElasticNetOverlay(BaseHingeOverlay):
    """ElasticNet regression overlay for asset-specific interaction features.

    Parameters
    ----------
    en_alpha:
        Overall regularization strength.
    l1_ratio:
        L1/L2 mix (0 = Ridge, 1 = Lasso).
    model_name:
        Name identifier.
    """

    def __init__(
        self,
        en_alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 2000,
        model_name: str = "interaction_elasticnet",
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
        self.en_alpha = en_alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.MODEL_NAME = model_name

        self._scaler = StandardScaler()
        self._model: ElasticNet | None = None
        self.train_medians_: np.ndarray | None = None
        self.train_stds_: np.ndarray | None = None

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit ElasticNet on standardized, imputed features."""
        self.train_medians_, self.train_stds_ = impute_train_stats(X_train)
        X_imputed = apply_imputation(X_train, self.train_medians_, self.train_stds_)

        X_scaled = self._scaler.fit_transform(X_imputed)
        self._model = ElasticNet(
            alpha=self.en_alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            fit_intercept=True,
            warm_start=False,
        )
        try:
            self._model.fit(X_scaled, y_train)
        except Exception as e:
            logger.warning("%s fit failed: %s", self.MODEL_NAME, e)
            self._model = None
            raise

        n_nonzero = int(np.sum(self._model.coef_ != 0))
        logger.debug(
            "%s fitted: en_alpha=%.4f, l1_ratio=%.2f, n_features=%d, n_nonzero=%d",
            self.MODEL_NAME, self.en_alpha, self.l1_ratio,
            X_train.shape[1], n_nonzero,
        )

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Predict with ElasticNet using train-derived imputation."""
        if self._model is None:
            return np.zeros(len(X))

        if self.train_medians_ is not None:
            X_imputed = apply_imputation(X, self.train_medians_, self.train_stds_)
        else:
            X_imputed = np.where(np.isnan(X), 0.0, X)

        X_scaled = self._scaler.transform(X_imputed)
        return self._model.predict(X_scaled)

    @property
    def nonzero_features_(self) -> int:
        if self._model is None:
            return 0
        return int(np.sum(self._model.coef_ != 0))

    @classmethod
    def select_best_hyperparams(
        cls,
        X_train: np.ndarray,
        y_intraday_train: np.ndarray,
        mu_base_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        mu_base_val: np.ndarray,
        alpha_grid: list[float] = ELASTICNET_ALPHA_GRID_DEFAULT,
        l1_ratio_grid: list[float] = ELASTICNET_L1_RATIO_GRID_DEFAULT,
        blend_alpha: float = 0.5,
        cap_overlay: bool = True,
        max_overlay_ratio: float = 0.5,
        max_overlay_bps: float = 20.0,
        max_iter: int = 2000,
        model_name: str = "interaction_elasticnet",
        blend_alpha_grid: list[float] | None = None,
    ) -> "InteractionElasticNetOverlay":
        """Grid search over ElasticNet hyperparameters using validation Rank IC."""
        from scipy.stats import spearmanr
        from research.models.hinge_overlay import select_best_alpha, ALPHA_GRID_DEFAULT

        if len(X_train) == 0 or X_train.shape[1] == 0:
            best_model = cls(en_alpha=0.01, l1_ratio=0.5, alpha=0.0, model_name=model_name)
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

        for en_a in alpha_grid:
            for l1_r in l1_ratio_grid:
                try:
                    model = cls(
                        en_alpha=en_a,
                        l1_ratio=l1_r,
                        max_iter=max_iter,
                        alpha=blend_alpha,
                        cap_overlay=cap_overlay,
                        max_overlay_ratio=max_overlay_ratio,
                        max_overlay_bps=max_overlay_bps,
                        model_name=model_name,
                    )
                    model.train_medians_ = train_medians
                    model.train_stds_ = train_stds
                    model._scaler = scaler
                    model._model = ElasticNet(
                        alpha=en_a,
                        l1_ratio=l1_r,
                        max_iter=max_iter,
                        fit_intercept=True,
                        warm_start=False,
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

                except Exception as e:
                    logger.debug(
                        "%s grid search: en_alpha=%.4f l1_ratio=%.2f failed: %s",
                        model_name, en_a, l1_r, e
                    )
                    continue

        if best_model is None:
            logger.warning(
                "%s: no model fitted. Using en_alpha=0.01, l1_ratio=0.5, alpha=0.0.",
                model_name
            )
            best_model = cls(en_alpha=0.01, l1_ratio=0.5, alpha=0.0, model_name=model_name)
            best_model._is_fitted = False
        else:
            grid = blend_alpha_grid if blend_alpha_grid is not None else ALPHA_GRID_DEFAULT
            best_blend = select_best_alpha(
                X_val, y_val, mu_base_val, best_model, alpha_grid=grid
            )
            best_model.alpha = best_blend

        logger.info(
            "%s: en_alpha=%.4f, l1_ratio=%.2f, blend_alpha=%.2f, val_IC=%.4f",
            model_name, best_model.en_alpha, best_model.l1_ratio, best_model.alpha, best_ic,
        )
        return best_model
