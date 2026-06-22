"""Residualized Supervised Low-Rank Lead-Lag Model.

This module implements the "Residualized Supervised Low-Rank Lead-Lag Model"
for predicting Japanese sector ETF idiosyncratic alphas using US sector ETF
idiosyncratic shocks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
        betas = betas_shifted.values[:, :, np.newaxis]
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


class ResidualizedSupervisedLowRankModel:
    """Residualized Supervised Low-Rank Lead-Lag Model."""

    def __init__(
        self,
        rank_k: int = 3,
        ridge_alpha: float = 0.1,
        train_window: int = 756,
        beta_window: int = 60,
        residualize_us_market: bool = True,
        residualize_us_macro: bool = False,
        residualize_jp_market: bool = True,
        residualize_jp_macro: bool = False,
    ):
        """Initialize the model.

        Args:
            rank_k: Rank constraint for coefficient matrix B (K <= min(11, 17))
            ridge_alpha: L2 regularization coefficient for Ridge regression
            train_window: Lookback window for fitting the Ridge-SVD model
            beta_window: Lookback window for estimating beta coefficients
            residualize_us_market: If True, residualize US returns on US market
            residualize_us_macro: If True, residualize US returns on macro variables
            residualize_jp_market: If True, residualize JP returns on TOPIX
            residualize_jp_macro: If True, residualize JP returns on macro variables (unused by default)
        """
        self.rank_k = rank_k
        self.ridge_alpha = ridge_alpha
        self.train_window = train_window
        self.beta_window = beta_window
        self.residualize_us_market = residualize_us_market
        self.residualize_us_macro = residualize_us_macro
        self.residualize_jp_market = residualize_jp_market
        self.residualize_jp_macro = residualize_jp_macro

    def fit_predict_step(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        x_predict: np.ndarray,
    ) -> np.ndarray:
        """Fit the Ridge + SVD low-rank model and predict the target for the next step.

        Args:
            X_train: In-sample feature matrix of shape (n_samples, n_features)
            Y_train: In-sample target matrix of shape (n_samples, n_targets)
            x_predict: Feature vector for the prediction step of shape (n_features,)

        Returns:
            Predicted target vector of shape (n_targets,)
        """
        # 1. Filter out non-finite training rows
        valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
        X_tr = X_train[valid_mask]
        Y_tr = Y_train[valid_mask]

        if len(X_tr) < 10:
            # Fallback if insufficient data
            return np.zeros(Y_train.shape[1])

        # 2. Standardize X and Y in-sample to avoid leaks
        mean_X = np.mean(X_tr, axis=0)
        std_X = np.std(X_tr, axis=0, ddof=1)
        std_X[std_X == 0.0] = 1e-8

        mean_Y = np.mean(Y_tr, axis=0)
        std_Y = np.std(Y_tr, axis=0, ddof=1)
        std_Y[std_Y == 0.0] = 1e-8

        X_tr_std = (X_tr - mean_X) / std_X
        Y_tr_std = (Y_tr - mean_Y) / std_Y

        # 3. Fit multi-output Ridge regression (without intercept since data is centered)
        # B_ridge has shape (n_features, n_targets)
        # Ridge solver in sklearn uses coef_ of shape (n_targets, n_features)
        # so we transpose it
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False, solver="svd")
        ridge.fit(X_tr_std, Y_tr_std)
        B_ridge = ridge.coef_.T  # shape (n_features, n_targets)

        # 4. SVD Decomposition of B_ridge
        # B_ridge = U * diag(S) * Vt
        # U: (n_features, n_features), S: (n_features,), Vt: (n_features, n_targets)
        U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)

        # 5. Apply low-rank constraint: keep only the top K singular values
        S_low = S.copy()
        k_capped = min(self.rank_k, len(S_low))
        if k_capped < len(S_low):
            S_low[k_capped:] = 0.0

        # Reconstruct low-rank B
        B_lowrank = U @ np.diag(S_low) @ Vt  # shape (n_features, n_targets)

        # 6. Predict for the next step (x_predict)
        x_pred_std = (x_predict - mean_X) / std_X
        y_pred_std = x_pred_std @ B_lowrank  # shape (n_targets,)

        # 7. Inverse transform standardized prediction
        y_pred = mean_Y + y_pred_std * std_Y

        return y_pred
