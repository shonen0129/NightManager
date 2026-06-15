"""Prior Residualized Low-Rank Gap Model.

This module implements the "Residualized Prior Subspace Low-Rank Gap Model"
for predicting Japanese sector ETF returns using US sector ETF returns
and prior subspace projections.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def project_to_subspace(
    data: np.ndarray,
    V0_sub: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Project standardized data onto a prior subspace V0.

    coord(z) = z * V0_sub * inv(V0_sub.T * V0_sub + eps * I)

    Args:
        data: Standardized return array of shape (N_samples, N_assets)
        V0_sub: Subspace matrix of shape (N_assets, K0)
        eps: Regularization coefficient for the projection

    Returns:
        Coordinate projection array of shape (N_samples, K0)
    """
    G = V0_sub.T @ V0_sub
    K0 = G.shape[0]
    inv_G = np.linalg.inv(G + eps * np.eye(K0))
    # data is (T, N_assets), V0_sub is (N_assets, K0), inv_G is (K0, K0)
    # result shape is (T, K0)
    return data @ V0_sub @ inv_G


def solve_factor_propagation(
    F: np.ndarray,
    G: np.ndarray,
    lambda_A: float,
    lambda_prior: float,
    A0: np.ndarray,
) -> np.ndarray:
    """Solve the Ridge factor propagation optimization problem.

    min_A ||G - F A||^2 + lambda_A ||A||_F^2 + lambda_prior ||A - A0||_F^2

    Closed-form solution:
    A = inv(F.T * F + (lambda_A + lambda_prior) * I) * (F.T * G + lambda_prior * A0)

    Args:
        F: In-sample US coordinate projections (n_samples, K0)
        G: In-sample JP coordinate projections (n_samples, K0)
        lambda_A: L2 regularization coefficient
        lambda_prior: Prior contraction coefficient
        A0: Prior propagation matrix (K0, K0)

    Returns:
        Fitted factor propagation matrix A of shape (K0, K0)
    """
    K0 = F.shape[1]
    FTF = F.T @ F
    FTG = F.T @ G
    reg_term = (lambda_A + lambda_prior) * np.eye(K0)
    lhs = FTF + reg_term
    rhs = FTG + lambda_prior * A0
    return np.linalg.solve(lhs, rhs)


class ResidualizedPriorSubspaceLowRankGapModel:
    """Prior Subspace Residualized Low-Rank Lead-Lag Model with Gap Correction."""

    def __init__(
        self,
        k_prior: int = 6,
        ridge_alpha_a: float = 1.0,
        lambda_prior_a: float = 1.0,
        train_window: int = 756,
        beta_window: int = 60,
        residualize_us_market: bool = True,
        residualize_jp_market: bool = True,
        gap_open_coef: float = 1.0,
        topix_beta_coef: float = 0.6,
        gap_signal_coef: float = 1.0,
        signal_mode: str = "prior_cc_to_oc_gap",
        target_mode: str = "cc_residual",
        gap_formula: str = "multiplicative",
    ):
        self.k_prior = k_prior
        self.ridge_alpha_a = ridge_alpha_a
        self.lambda_prior_a = lambda_prior_a
        self.train_window = train_window
        self.beta_window = beta_window
        self.residualize_us_market = residualize_us_market
        self.residualize_jp_market = residualize_jp_market
        self.gap_open_coef = gap_open_coef
        self.topix_beta_coef = topix_beta_coef
        self.gap_signal_coef = gap_signal_coef
        self.signal_mode = signal_mode
        self.target_mode = target_mode
        self.gap_formula = gap_formula

    def fit_predict_step(
        self,
        X_train: np.ndarray,  # US residualized/raw shocks in-sample (T_train, 11)
        Y_train: np.ndarray,  # JP residualized/raw targets in-sample (T_train, 17)
        x_predict: np.ndarray,  # US feature vector at current step (11,)
        V0_U: np.ndarray,  # US subspace (11, K0)
        V0_J: np.ndarray,  # JP subspace (17, K0)
        A0: np.ndarray,  # Prior matrix (K0, K0)
        Y_train_sigma20: np.ndarray | None = None,  # Vol-adjustment for Y_train (T_train, 17)
        y_predict_sigma20: np.ndarray | None = None,  # Vol-adjustment at prediction step (17,)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit factor propagation and predict.

        Returns:
            Tuple of: (predicted_return_vector (17,), effective_B_matrix (11, 17))
        """
        # 1. Filter out non-finite training rows
        valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
        if Y_train_sigma20 is not None:
            valid_mask = valid_mask & np.isfinite(Y_train_sigma20).all(axis=1)

        X_tr = X_train[valid_mask]
        Y_tr = Y_train[valid_mask]

        if Y_train_sigma20 is not None:
            Y_tr = Y_tr / Y_train_sigma20[valid_mask]

        n_features = X_train.shape[1]
        n_targets = Y_train.shape[1]

        if len(X_tr) < 10:
            return np.zeros(n_targets), np.zeros((n_features, n_targets))

        # 2. Standardize X and Y in-sample
        mean_X = np.mean(X_tr, axis=0)
        std_X = np.std(X_tr, axis=0, ddof=1)
        std_X[std_X == 0.0] = 1e-8

        mean_Y = np.mean(Y_tr, axis=0)
        std_Y = np.std(Y_tr, axis=0, ddof=1)
        std_Y[std_Y == 0.0] = 1e-8

        X_tr_std = (X_tr - mean_X) / std_X
        Y_tr_std = (Y_tr - mean_Y) / std_Y

        # 3. Project to coordinates
        F_tr = project_to_subspace(X_tr_std, V0_U)
        G_tr = project_to_subspace(Y_tr_std, V0_J)

        # 4. Solve propagation matrix A
        A = solve_factor_propagation(
            F_tr, G_tr, self.ridge_alpha_a, self.lambda_prior_a, A0
        )

        # 5. Build effective B matrix (standardized space)
        # B_eff_std = V0_U * inv(V0_U.T * V0_U) * A * V0_J.T
        G_U = V0_U.T @ V0_U
        inv_G_U = np.linalg.inv(G_U + 1e-8 * np.eye(self.k_prior))
        B_eff_std = V0_U @ inv_G_U @ A @ V0_J.T  # (11, 17)

        # 6. Predict for x_predict
        x_pred_std = (x_predict - mean_X) / std_X
        y_pred_std = x_pred_std @ B_eff_std  # (17,)

        # 7. Unstandardize prediction
        y_pred = mean_Y + y_pred_std * std_Y

        if y_predict_sigma20 is not None:
            y_pred = y_pred * y_predict_sigma20

        # 8. Unstandardize effective B for diagnostic save
        effective_std_Y = std_Y
        if y_predict_sigma20 is not None:
            effective_std_Y = std_Y * y_predict_sigma20

        B_eff_raw = (1.0 / std_X)[:, np.newaxis] * B_eff_std * effective_std_Y[np.newaxis, :]

        return y_pred, B_eff_raw
