"""Strictly historical rolling OLS beta estimation."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rolling_ols_betas(
    y_data: np.ndarray,
    x_data: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute rolling OLS regression coefficients.

    For each row i, the coefficients are estimated using historical data
    from row i - window to row i - 1 (inclusive).

    Args:
        y_data: Target variable array of shape (T, N_targets)
        x_data: Feature variable array of shape (T, N_features)
        window: Rolling window size

    Returns:
        Array of shape (T, N_targets, N_features) containing the rolling coefficients.
        Rows prior to the window size are filled with np.nan.
    """
    T, N_targets = y_data.shape
    N_features = x_data.shape[1]

    if N_features == 1:
        # Vectorized fast path for 1-dimensional features
        y_clean = np.where(np.isfinite(y_data), y_data, np.nan)
        x_clean = np.where(np.isfinite(x_data), x_data, np.nan)

        y_df = pd.DataFrame(y_clean)
        x_series = pd.Series(x_clean[:, 0])

        min_periods = max(window // 2, 5)

        # Calculate rolling covariance and variance
        cov_rolling = y_df.rolling(window, min_periods=min_periods).cov(x_series)
        var_rolling = x_series.rolling(window, min_periods=min_periods).var()

        # Safely divide covariance by variance
        var_mask = var_rolling > 1e-12
        betas_df = cov_rolling.divide(var_rolling.where(var_mask, np.nan), axis=0)

        # Shift by 1 because t-th beta is estimated on data from index t-window to t-1
        betas_shifted = betas_df.shift(1)

        # Convert back to numpy of shape (T, N_targets, 1)
        betas = cast(np.ndarray, betas_shifted.values[:, :, np.newaxis].copy())
        betas[:window] = np.nan
        return betas

    else:
        # Standard fallback for multi-dimensional features
        betas = np.zeros((T, N_targets, N_features))
        betas[:window] = np.nan

        for t in range(window, T):
            start_idx = t - window
            end_idx = t - 1

            X_train = x_data[start_idx : end_idx + 1]
            Y_train = y_data[start_idx : end_idx + 1]

            # Filter out rows containing non-finite values (NaNs, infs)
            valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
            X_train_clean = X_train[valid_mask]
            Y_train_clean = Y_train[valid_mask]

            if len(X_train_clean) < max(window // 2, 5):
                betas[t] = np.nan
                continue

            # Fit OLS regression: Y = X * B + Intercept
            # We append a column of ones for the intercept
            X_design = np.column_stack([np.ones(len(X_train_clean)), X_train_clean])
            try:
                # Solve normal equations: coef has shape (1 + N_features, N_targets)
                coef, _, _, _ = np.linalg.lstsq(X_design, Y_train_clean, rcond=None)
                # The first row is intercept, the rest are feature coefficients
                betas[t] = coef[1:].T  # shape (N_targets, N_features)
            except np.linalg.LinAlgError:
                betas[t] = np.nan

        return betas
