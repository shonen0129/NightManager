"""BLPX meta-learning, macro confidence and asymmetric propagation helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from leadlag.core.correlation import compute_correlation
from leadlag.core.macro import (
    compute_factor_kappa_scale,
    compute_macro_surprise,
)

if TYPE_CHECKING:
    from leadlag.models.blpx.model import ProductionBLPXModel

logger = logging.getLogger("leadlag.models.blpx")


def combine_signals(
    self: ProductionBLPXModel,
    z0: np.ndarray,
    z3: np.ndarray,
    z_raw_blpx: np.ndarray,
    z_residual_blpx: np.ndarray,
) -> np.ndarray:
    """Combine component signals with ensemble weights."""
    return (
        self.raw_pca_weight * z0
        + self.residual_pca_weight * z3
        + self.raw_blpx_weight * z_raw_blpx
        + self.residual_blpx_weight * z_residual_blpx
    )


def _predict_meta_weight(
    self: ProductionBLPXModel,
    i: int,
    us_dispersions: list[float],
    cond_nums: list[float],
    vix_vals: list[float],
    ic_blpx_vals: list[float],
    ic_pca_vals: list[float],
) -> float:
    """Fit a low-capacity meta-model and predict tomorrow's ensemble weight w_t.

    Uses expanding/rolling window up to day i-1.
    """
    # We need at least some minimum number of samples to train, say 100 days
    min_samples = 100

    # Let's collect training features and targets
    X_train_list: list[list[float]] = []
    Y_train_list: list[float] = []

    start_train = max(self.corr_window + 10, i - self.meta_train_window)
    for j in range(start_train, i):
        if j - 10 < 0 or j >= len(ic_blpx_vals):
            continue
        rec_ic_blpx = np.nanmean(ic_blpx_vals[j - 10 : j])
        rec_ic_pca = np.nanmean(ic_pca_vals[j - 10 : j])

        if np.isnan(rec_ic_blpx) or np.isnan(rec_ic_pca):
            continue

        f_j = [
            us_dispersions[j],
            cond_nums[j],
            vix_vals[j],
            rec_ic_blpx,
            rec_ic_pca,
        ]

        target_j = ic_blpx_vals[j] - ic_pca_vals[j]
        if np.isnan(target_j):
            continue

        X_train_list.append(f_j)
        Y_train_list.append(target_j)

    if len(X_train_list) < min_samples:
        return 0.8  # Fallback to static weight

    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)

    # Current features at day i (to predict for tomorrow)
    rec_ic_blpx_i = np.nanmean(ic_blpx_vals[i - 10 : i])
    rec_ic_pca_i = np.nanmean(ic_pca_vals[i - 10 : i])
    F_i = np.array([[
        us_dispersions[i],
        cond_nums[i],
        vix_vals[i],
        rec_ic_blpx_i,
        rec_ic_pca_i,
    ]])

    if np.isnan(F_i).any():
        return 0.8

    try:
        if self.meta_model_type == "logistic_regression":
            # Binary target: 1 if blpx outperformed, 0 otherwise
            Y_train_bin = (Y_train > 0).astype(int)
            if len(np.unique(Y_train_bin)) < 2:
                model = Ridge(alpha=1.0)
                model.fit(X_train, Y_train)
                y_pred = float(model.predict(F_i)[0])
                w_t = np.clip(0.8 + y_pred, 0.6, 1.0)
            else:
                model = LogisticRegression(C=1.0, solver="liblinear")
                model.fit(X_train, Y_train_bin)
                prob = float(model.predict_proba(F_i)[0, 1])
                w_t = 0.6 + 0.4 * prob
        else:
            # Default: Ridge Regression
            model = Ridge(alpha=1.0)
            model.fit(X_train, Y_train)
            y_pred = float(model.predict(F_i)[0])
            w_t = np.clip(0.8 + y_pred, 0.6, 1.0)
    except (ValueError, TypeError, RuntimeError, IndexError) as e:
        logger.warning(f"Meta-model training failed at index {i}: {e}. Using static weight 0.8.")
        w_t = 0.8

    return float(w_t)


def _estimate_asymmetric_covariance(
    self: ProductionBLPXModel,
    window_returns: np.ndarray,
    corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate asymmetric covariance/correlation matrices based on US market factor sign.

    Returns:
        C_YX_pos, C_YX_neg, C_XX, C_YY
    """
    C_XX = corr[: self.n_u, : self.n_u]
    C_YY = corr[self.n_u :, self.n_u :]

    # US market factor: average of all US assets (first n_u columns)
    us_factor = np.mean(window_returns[:, : self.n_u], axis=1)
    pos_mask = us_factor >= 0.0
    neg_mask = us_factor < 0.0

    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos < 30 or n_neg < 30:
        C_YX = corr[self.n_u :, : self.n_u]
        return C_YX.copy(), C_YX.copy(), C_XX, C_YY

    try:
        _, _, corr_pos = compute_correlation(
            window_returns[pos_mask],
            self.blp_ewma_halflife,
            cache=self._blp_corr_cache,
        )
        C_YX_pos = corr_pos[self.n_u :, : self.n_u]
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        logger.warning(f"Failed to compute positive subset correlation: {e}. Falling back.")
        C_YX_pos = corr[self.n_u :, : self.n_u].copy()

    try:
        _, _, corr_neg = compute_correlation(
            window_returns[neg_mask],
            self.blp_ewma_halflife,
            cache=self._blp_corr_cache,
        )
        C_YX_neg = corr_neg[self.n_u :, : self.n_u]
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        logger.warning(f"Failed to compute negative subset correlation: {e}. Falling back.")
        C_YX_neg = corr[self.n_u :, : self.n_u].copy()

    return C_YX_pos, C_YX_neg, C_XX, C_YY


def _solve_asymmetric_blp(
    self: ProductionBLPXModel,
    C_YX_pos: np.ndarray,
    C_YX_neg: np.ndarray,
    C_XX: np.ndarray,
    C_YY: np.ndarray,
    B_pca: np.ndarray,
    M_sector: np.ndarray,
    B_blp: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve BLP coefficients separately for positive and negative regimes.

    Returns:
        B_pos_struct, B_neg_struct, inv_A_avg, Sigma_YX_reg_avg
    """
    Sigma_XX_reg = (1.0 - self.alpha_xx) * C_XX + self.alpha_xx * np.eye(self.n_u)
    Sigma_YX_reg_pos = (1.0 - self.alpha_yx) * C_YX_pos
    Sigma_YX_reg_neg = (1.0 - self.alpha_yx) * C_YX_neg
    (1.0 - self.alpha_yy) * C_YY + self.alpha_yy * np.eye(self.n_j)

    diag_mean = float(np.mean(np.diag(Sigma_XX_reg)))

    B_pos_struct, inv_A_pos = self._solve_tikhonov(
        Sigma_XX_reg, Sigma_YX_reg_pos, B_pca, M_sector, diag_mean, B_blp
    )
    B_neg_struct, inv_A_neg = self._solve_tikhonov(
        Sigma_XX_reg, Sigma_YX_reg_neg, B_pca, M_sector, diag_mean, B_blp
    )

    inv_A_avg = 0.5 * (inv_A_pos + inv_A_neg)
    Sigma_YX_reg_avg = 0.5 * (Sigma_YX_reg_pos + Sigma_YX_reg_neg)

    return B_pos_struct, B_neg_struct, inv_A_avg, Sigma_YX_reg_avg


def _load_vix_series(self: ProductionBLPXModel, df_exec: pd.DataFrame) -> pd.Series | None:
    """Load VIX series aligned to df_exec for meta-learning."""
    sim_dates = df_exec.index
    if not self.meta_enabled:
        return None

    vix_series = None
    from pathlib import Path

    macro_path = Path(__file__).resolve().parents[3] / "market_data" / "macro_data.pkl"
    if macro_path.exists():
        try:
            macro_df = pd.read_pickle(macro_path)
            macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None).normalize()
            vix_series = macro_df["^VIX"].reindex(sim_dates).ffill()
            vix_series = vix_series.bfill()
            logger.info("Successfully loaded VIX from macro_data.pkl for meta-learning.")
        except (OSError, EOFError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to load VIX from macro_data.pkl: {e}")
    if vix_series is None:
        vix_series = pd.Series(20.0, index=sim_dates)

    return vix_series


def _prepare_macro_confidence(self: ProductionBLPXModel, df_exec: pd.DataFrame) -> None:
    """Load macro returns and compute per-instance macro confidence scales."""
    if not (self.macro_confidence_enabled and self.macro_kappas is not None):
        return

    macro_returns = self._load_macro_returns(df_exec)
    if macro_returns is not None:
        surprise_raw = compute_macro_surprise(
            macro_returns,
            halflife_mean=self.macro_surprise_halflife_mean,
            halflife_vol=self.macro_surprise_halflife_vol,
        )
        self._macro_surprise_raw = surprise_raw
        self._macro_scales = compute_factor_kappa_scale(
            surprise_raw, self.macro_kappas, self._macro_sens_matrix
        )
        if self.macro_direction_enabled:
            from leadlag.core.macro import compute_macro_direction_adjustment

            self._macro_direction_adj = compute_macro_direction_adjustment(
                surprise_raw, self.macro_kappas, self._macro_sens_matrix
            )
        logger.info(
            "Macro confidence enabled: kappas=%s, halflife_mean=%.1f, halflife_vol=%.1f, "
            "direction=%s, sigma_yy_inflation=%s",
            self.macro_kappas.tolist(),
            self.macro_surprise_halflife_mean,
            self.macro_surprise_halflife_vol,
            self.macro_direction_enabled,
            self.macro_sigma_yy_inflation_enabled,
        )
    else:
        logger.warning("Macro confidence enabled but macro data unavailable; skipping.")
        self.macro_confidence_enabled = False


class BLPXMetaMixin:
    """Mixin providing meta-learning, macro confidence, and asymmetric propagation."""

    combine_signals = combine_signals
    _predict_meta_weight = _predict_meta_weight
    _estimate_asymmetric_covariance = _estimate_asymmetric_covariance
    _solve_asymmetric_blp = _solve_asymmetric_blp
    _load_vix_series = _load_vix_series
    _prepare_macro_confidence = _prepare_macro_confidence
