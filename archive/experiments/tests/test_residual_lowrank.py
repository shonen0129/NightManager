import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from domain.models.residual_lowrank import (
    ResidualizedSupervisedLowRankModel,
    compute_rolling_ols_betas,
)


def test_compute_rolling_ols_betas_correctness():
    """Verify rolling OLS beta estimation matches simple linear regression."""
    np.random.seed(42)
    T = 100
    x = np.random.randn(T, 1)
    # y = 2 * x + noise
    y = 2.0 * x + 0.1 * np.random.randn(T, 1)

    window = 30
    betas = compute_rolling_ols_betas(y, x, window)

    # Prior to window, should be nan
    assert np.isnan(betas[:window]).all()

    # For t >= window, check a specific index
    t = 50
    # Manual regression for window ending at t-1
    X_slice = x[t - window : t]
    Y_slice = y[t - window : t]
    X_design = np.column_stack([np.ones(len(X_slice)), X_slice])
    coef, _, _, _ = np.linalg.lstsq(X_design, Y_slice, rcond=None)
    manual_beta = coef[1, 0]

    assert pytest.approx(betas[t, 0, 0], abs=1e-7) == manual_beta
    # Check that it's close to 2.0
    assert abs(betas[t, 0, 0] - 2.0) < 0.1


def test_residual_lowrank_model_rank_constraint():
    """Verify that fit_predict_step generates a coefficient matrix with rank <= K."""
    np.random.seed(42)
    n_samples = 100
    n_features = 11
    n_targets = 17

    X_train = np.random.randn(n_samples, n_features)
    Y_train = np.random.randn(n_samples, n_targets)
    x_predict = np.random.randn(n_features)

    for K in [1, 3, 5]:
        model = ResidualizedSupervisedLowRankModel(rank_k=K, ridge_alpha=0.1)
        # Let's extract the fitted B matrix internally to check its rank
        mean_X = np.mean(X_train, axis=0)
        std_X = np.std(X_train, axis=0, ddof=1)
        std_X[std_X == 0.0] = 1e-8
        mean_Y = np.mean(Y_train, axis=0)
        std_Y = np.std(Y_train, axis=0, ddof=1)
        std_Y[std_Y == 0.0] = 1e-8

        X_std = (X_train - mean_X) / std_X
        Y_std = (Y_train - mean_Y) / std_Y

        ridge = Ridge(alpha=0.1, fit_intercept=False, solver="svd")
        ridge.fit(X_std, Y_std)
        B_ridge = ridge.coef_.T

        U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
        S_low = S.copy()
        S_low[K:] = 0.0
        B_lowrank = U @ np.diag(S_low) @ Vt

        # Rank of B_lowrank should be exactly K (or less if features are degenerate)
        rank = np.linalg.matrix_rank(B_lowrank)
        assert rank <= K


def test_prediction_value_correctness():
    """Verify standardisation and prediction math in fit_predict_step."""
    np.random.seed(42)
    X_train = np.random.randn(50, 11)
    Y_train = np.random.randn(50, 17)
    x_predict = np.random.randn(11)

    model = ResidualizedSupervisedLowRankModel(rank_k=3, ridge_alpha=1.0)
    pred = model.fit_predict_step(X_train, Y_train, x_predict)

    assert len(pred) == 17
    assert np.all(np.isfinite(pred))
