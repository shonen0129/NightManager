"""src/models/hinge_overlay.py — Sprint 3-A Hinge Overlay Base Classes.

Implements the residual overlay framework:
    e_{j,t} = y^intraday_resid_{j,t} - mu_base_{j,t}   (residual target)
    hat_e_{j,t} = model(H_{j,t})                         (hinge feature model)
    mu_final_{j,t} = mu_base_{j,t} + alpha * hat_e_{j,t} (final prediction)

Conservative constraints (look-ahead free, over-fitting prevention):
    abs(delta_hinge) <= max_overlay_to_base_abs_ratio * abs(mu_base)
    abs(delta_hinge) <= max_overlay_bps / 10000
    Cap uses the STRICTER of the two constraints.

Walk-forward design:
    Train window  → fit model and rolling stats
    Validation    → select alpha_blend
    Test          → OOS prediction (no data from future)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

MAX_OVERLAY_RATIO_DEFAULT = 0.5     # abs(delta) <= ratio * abs(mu_base)
MAX_OVERLAY_BPS_DEFAULT = 20        # abs(delta) <= 20 bps
ALPHA_GRID_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Overlay cap helper
# ---------------------------------------------------------------------------


def cap_overlay_prediction(
    delta: np.ndarray | pd.Series,
    mu_base: np.ndarray | pd.Series,
    max_ratio: float = MAX_OVERLAY_RATIO_DEFAULT,
    max_bps: float = MAX_OVERLAY_BPS_DEFAULT,
) -> np.ndarray:
    """Apply conservative cap to overlay prediction.

    Cap rules:
        cap_ratio = max_ratio * abs(mu_base)
        cap_bps   = max_bps / 10000
        cap       = min(cap_ratio, cap_bps)   [element-wise]
        delta_capped = sign(delta) * min(abs(delta), cap)

    Parameters
    ----------
    delta:
        Raw overlay prediction (same units as mu_base, i.e., decimal returns).
    mu_base:
        Baseline model prediction.
    max_ratio:
        Maximum overlay as fraction of |mu_base|.
    max_bps:
        Maximum overlay in basis points (absolute).

    Returns
    -------
    np.ndarray
        Capped overlay prediction.
    """
    delta_arr = np.asarray(delta, dtype=float)
    mu_arr = np.asarray(mu_base, dtype=float)

    cap_ratio = max_ratio * np.abs(mu_arr)
    cap_bps = max_bps / 10000.0

    # Strict cap: minimum of the two caps
    cap = np.minimum(cap_ratio, cap_bps)
    cap = np.maximum(cap, 0.0)  # ensure non-negative cap

    delta_capped = np.sign(delta_arr) * np.minimum(np.abs(delta_arr), cap)
    return delta_capped


# ---------------------------------------------------------------------------
# Walk-forward window generator
# ---------------------------------------------------------------------------


def generate_walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_window_days: int = 252,
    validation_window_days: int = 63,
    test_window_days: int = 21,
    step_days: int = 21,
    purge_days: int = 1,
) -> list[dict]:
    """Generate walk-forward train/validation/test date windows.

    All splits are date-based (not index-based) to avoid intra-day leakage.

    Parameters
    ----------
    dates:
        Sorted DatetimeIndex of available trading dates.
    train_window_days, validation_window_days, test_window_days:
        Window sizes in trading days.
    step_days:
        Step between consecutive windows.
    purge_days:
        Gap between train end and validation start (removes border leakage).

    Returns
    -------
    list[dict]
        Each dict: window_id, train_start, train_end, val_start, val_end,
                   test_start, test_end, train_dates, val_dates, test_dates.
    """
    n = len(dates)
    windows = []
    required = train_window_days + purge_days + validation_window_days + test_window_days
    window_id = 0

    i = 0
    while i + required <= n:
        train_end_idx = i + train_window_days - 1
        val_start_idx = train_end_idx + 1 + purge_days
        val_end_idx = val_start_idx + validation_window_days - 1
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + test_window_days - 1

        if test_end_idx >= n:
            break

        train_dates = dates[i : train_end_idx + 1]
        val_dates = dates[val_start_idx : val_end_idx + 1]
        test_dates = dates[test_start_idx : test_end_idx + 1]

        windows.append({
            "window_id": window_id,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_start": val_dates[0],
            "val_end": val_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "train_dates": train_dates,
            "val_dates": val_dates,
            "test_dates": test_dates,
        })

        i += step_days
        window_id += 1

    logger.info("Generated %d walk-forward windows.", len(windows))
    return windows


# ---------------------------------------------------------------------------
# Base overlay model
# ---------------------------------------------------------------------------


class BaseHingeOverlay(ABC):
    """Abstract base class for hinge overlay models.

    Subclasses implement ``_fit_model`` and ``_predict_model``.

    The target for fitting is:
        residual = y_intraday - mu_base   (a.k.a. e_{j,t})

    The final prediction is:
        mu_final = mu_base + alpha * clip(hat_e)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        cap_overlay: bool = True,
        max_overlay_ratio: float = MAX_OVERLAY_RATIO_DEFAULT,
        max_overlay_bps: float = MAX_OVERLAY_BPS_DEFAULT,
    ) -> None:
        self.alpha = alpha
        self.cap_overlay = cap_overlay
        self.max_overlay_ratio = max_overlay_ratio
        self.max_overlay_bps = max_overlay_bps
        self._is_fitted = False

    @abstractmethod
    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Fit the overlay model on training features X_train and residual target y_train."""
        ...

    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Predict overlay delta (residual correction) for X."""
        ...

    def fit(
        self,
        X_train: np.ndarray,
        y_intraday_train: np.ndarray,
        mu_base_train: np.ndarray,
    ) -> "BaseHingeOverlay":
        """Fit overlay on training data.

        Computes residual target = y_intraday - mu_base and fits _fit_model.

        Parameters
        ----------
        X_train:
            Hinge features (n_samples, n_features).
        y_intraday_train:
            Actual intraday returns (n_samples,).
        mu_base_train:
            Baseline predictions (n_samples,).
        """
        y_resid = y_intraday_train - mu_base_train
        # Remove NaN rows
        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_resid) | np.isnan(mu_base_train))
        if valid.sum() < 10:
            logger.warning(
                "%s.fit: Only %d valid samples. Skipping fit.",
                self.__class__.__name__, valid.sum()
            )
            self._is_fitted = False
            return self

        self._fit_model(X_train[valid], y_resid[valid])
        self._is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        mu_base: np.ndarray,
    ) -> np.ndarray:
        """Predict final mu_final = mu_base + alpha * cap(hat_e).

        Parameters
        ----------
        X:
            Hinge features for prediction.
        mu_base:
            Baseline predictions.

        Returns
        -------
        np.ndarray
            Final predictions mu_final.
        """
        if not self._is_fitted:
            logger.warning(
                "%s.predict called before fitting. Returning mu_base.",
                self.__class__.__name__,
            )
            return np.asarray(mu_base, dtype=float)

        hat_e_raw = self._predict_model(X)

        if self.cap_overlay:
            hat_e = cap_overlay_prediction(
                hat_e_raw, mu_base,
                max_ratio=self.max_overlay_ratio,
                max_bps=self.max_overlay_bps,
            )
        else:
            hat_e = hat_e_raw

        mu_base_arr = np.asarray(mu_base, dtype=float)
        mu_final = mu_base_arr + self.alpha * hat_e
        return mu_final

    def predict_delta(
        self,
        X: np.ndarray,
        mu_base: np.ndarray,
    ) -> np.ndarray:
        """Return capped overlay delta only (alpha * hat_e, before adding mu_base)."""
        if not self._is_fitted:
            return np.zeros(len(mu_base))

        hat_e_raw = self._predict_model(X)

        if self.cap_overlay:
            hat_e = cap_overlay_prediction(
                hat_e_raw, mu_base,
                max_ratio=self.max_overlay_ratio,
                max_bps=self.max_overlay_bps,
            )
        else:
            hat_e = hat_e_raw

        return self.alpha * hat_e


# ---------------------------------------------------------------------------
# Validation alpha selection
# ---------------------------------------------------------------------------


def select_best_alpha(
    X_val: np.ndarray,
    y_val: np.ndarray,
    mu_base_val: np.ndarray,
    fitted_model: BaseHingeOverlay,
    alpha_grid: list[float] = ALPHA_GRID_DEFAULT,
) -> float:
    """Select best alpha blend on validation set using Rank IC criterion.

    Parameters
    ----------
    X_val:
        Hinge features for validation period.
    y_val:
        Actual intraday returns for validation period.
    mu_base_val:
        Baseline predictions for validation period.
    fitted_model:
        A fitted BaseHingeOverlay (will temporarily override alpha).
    alpha_grid:
        Alpha candidates to evaluate.

    Returns
    -------
    float
        Best alpha (highest mean Rank IC on validation).
    """
    from scipy.stats import spearmanr

    best_alpha = 0.0
    best_ic = -np.inf

    original_alpha = fitted_model.alpha

    for alpha in alpha_grid:
        fitted_model.alpha = alpha
        mu_pred = fitted_model.predict(X_val, mu_base_val)

        # Rank IC on the validation cross-sections
        # Use all validation observations together (pooled cross-section)
        valid = ~(np.isnan(mu_pred) | np.isnan(y_val))
        if valid.sum() < 5:
            continue

        rho, _ = spearmanr(mu_pred[valid], y_val[valid])
        if rho > best_ic:
            best_ic = rho
            best_alpha = alpha

    fitted_model.alpha = original_alpha
    logger.info(
        "Alpha selection on validation: best_alpha=%.2f (IC=%.4f).",
        best_alpha, best_ic,
    )
    return best_alpha
