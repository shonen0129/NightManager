"""src/models/hinge_elasticnet_overlay.py — Sprint 3-A ElasticNet Hinge Overlay.

Implements ElasticNet regression overlay:
    hat_e_{j,t} = ElasticNet(H_{j,t})

ElasticNet combines L1 (Lasso) and L2 (Ridge) regularization, performing
implicit feature selection (sparsity) while maintaining ridge stability.

Hyperparameters (alpha, l1_ratio) are selected on the validation window.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from experiments.models.hinge_overlay import BaseHingeOverlay

logger = logging.getLogger(__name__)

ELASTICNET_ALPHA_GRID_DEFAULT = [0.0001, 0.001, 0.01, 0.1]
ELASTICNET_L1_RATIO_GRID_DEFAULT = [0.1, 0.3, 0.5, 0.7]


class HingeElasticNetOverlay(BaseHingeOverlay):
    """ElasticNet regression hinge overlay model.

    Provides sparse feature selection through L1 penalty while maintaining
    stability through L2 penalty.

    Parameters
    ----------
    en_alpha:
        Overall regularization strength (ElasticNet alpha).
    l1_ratio:
        Mixing ratio between L1 and L2 (0 = Ridge, 1 = Lasso).
    max_iter:
        Max iterations for ElasticNet solver.
    alpha:
        Blend parameter: mu_final = mu_base + alpha * cap(hat_e).
    cap_overlay:
        Whether to apply the conservative overlay cap.
    max_overlay_ratio:
        Cap = max_ratio * |mu_base|.
    max_overlay_bps:
        Cap = max_bps / 10000 (absolute).
    """

    MODEL_NAME = "hinge_elasticnet_overlay"

    def __init__(
        self,
        en_alpha: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
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

        self._scaler = StandardScaler()
        self._model: ElasticNet | None = None

    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit ElasticNet on standardized features."""
        X_scaled = self._scaler.fit_transform(X_train)
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
            logger.warning("ElasticNet fit failed: %s. Model not fitted.", e)
            self._model = None
            raise

        n_nonzero = int(np.sum(self._model.coef_ != 0))
        logger.debug(
            "HingeElasticNetOverlay fitted: en_alpha=%.4f, l1_ratio=%.2f, "
            "n_features=%d, n_nonzero=%d, n_samples=%d",
            self.en_alpha, self.l1_ratio, X_train.shape[1],
            n_nonzero, X_train.shape[0],
        )

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Predict with ElasticNet (handle NaN rows gracefully)."""
        if self._model is None:
            return np.zeros(len(X))

        # Impute NaN in X with zero (becomes mean after standardization)
        X_clean = np.where(np.isnan(X), 0.0, X)
        X_scaled = self._scaler.transform(X_clean)
        return self._model.predict(X_scaled)

    @property
    def nonzero_features_(self) -> int:
        """Number of non-zero coefficients in fitted model."""
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
        max_iter: int = 1000,
    ) -> "HingeElasticNetOverlay":
        """Grid search over ElasticNet hyperparameters using validation Rank IC.

        Fits one model per (en_alpha, l1_ratio) combination on training data,
        evaluates Rank IC on validation data, and returns the best model.

        Parameters
        ----------
        X_train, y_intraday_train, mu_base_train:
            Training data.
        X_val, y_val, mu_base_val:
            Validation data.
        alpha_grid:
            ElasticNet alpha candidates.
        l1_ratio_grid:
            ElasticNet l1_ratio candidates.
        blend_alpha:
            Blend alpha to use during grid search.
        cap_overlay, max_overlay_ratio, max_overlay_bps:
            Cap parameters.
        max_iter:
            Max solver iterations.

        Returns
        -------
        HingeElasticNetOverlay
            Best fitted model with selected hyperparameters.
        """
        from scipy.stats import spearmanr
        from experiments.models.hinge_overlay import select_best_alpha, ALPHA_GRID_DEFAULT

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
                    )
                    model.fit(X_train, y_intraday_train, mu_base_train)

                    if not model._is_fitted:
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
                        "ElasticNet grid search: en_alpha=%.4f l1_ratio=%.2f failed: %s",
                        en_a, l1_r, e,
                    )
                    continue

        if best_model is None:
            logger.warning(
                "HingeElasticNetOverlay: no model fitted successfully. "
                "Using en_alpha=0.01, l1_ratio=0.5, alpha=0.0."
            )
            best_model = cls(en_alpha=0.01, l1_ratio=0.5, alpha=0.0)
            best_model._is_fitted = False
        else:
            # Also select best blend alpha on validation
            best_blend = select_best_alpha(
                X_val, y_val, mu_base_val, best_model,
                alpha_grid=ALPHA_GRID_DEFAULT,
            )
            best_model.alpha = best_blend

        logger.info(
            "HingeElasticNetOverlay selection: en_alpha=%.4f, l1_ratio=%.2f, "
            "blend_alpha=%.2f, val_IC=%.4f, n_nonzero=%d",
            best_model.en_alpha, best_model.l1_ratio,
            best_model.alpha, best_ic, best_model.nonzero_features_,
        )
        return best_model
