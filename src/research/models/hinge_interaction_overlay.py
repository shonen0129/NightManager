"""src/models/hinge_interaction_overlay.py — Sprint 3-B Interaction Overlay Base.

Extends BaseHingeOverlay for asset-specific interaction features.
The key difference from Sprint 3-A:
  - Features are now (date × ticker) long-format, making them asset-specific.
  - FDR selection is applied to asset-specific IC (features actually vary by ticker).
  - The within-date cross-sectional std is verified before model fitting.

Formula:
    delta_{j,t} = model(interaction_features_{j,t})
    mu_final_{j,t} = mu_base_{j,t} + alpha * cap(delta_{j,t})

Model groups (G1–G10):
    G1/G2: macro_hinge × asset_beta (Ridge/ElasticNet)
    G3/G4: sector_hinge × sector_exposure (Ridge/ElasticNet)
    G5/G6: regime_hinge × base_signal (Ridge/ElasticNet)
    G7/G8: gap_hinge asset-specific (Ridge/ElasticNet)
    G9/G10: combined all interactions (Ridge/ElasticNet)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research.models.hinge_overlay import (
    BaseHingeOverlay,
    cap_overlay_prediction,
    ALPHA_GRID_DEFAULT,
    MAX_OVERLAY_RATIO_DEFAULT,
    MAX_OVERLAY_BPS_DEFAULT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------


def build_flat_arrays_from_long(
    long_df: pd.DataFrame,
    feature_cols: list[str],
    signal_col: str,
    target_col: str,
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple]]:
    """Build flat (n_samples, n_features) arrays from long-format DataFrame.

    Each (date, ticker) pair is one sample.

    Parameters
    ----------
    long_df:
        Long-format DataFrame with columns [date, ticker, feature_cols...,
        signal_col, target_col].
    feature_cols:
        Interaction feature column names.
    signal_col:
        Column name of baseline signal/mu_base.
    target_col:
        Column name of target return.
    dates:
        Date subset to include.
    tickers:
        Ticker subset.

    Returns
    -------
    X, y_intraday, mu_base, index_pairs
        Arrays and list of (date, ticker) pairs.
    """
    sub = long_df[long_df["date"].isin(dates) & long_df["ticker"].isin(tickers)].copy()

    if sub.empty:
        return (
            np.empty((0, len(feature_cols))),
            np.empty(0),
            np.empty(0),
            [],
        )

    # Ensure columns exist; fill missing feature cols with NaN
    for col in feature_cols:
        if col not in sub.columns:
            sub[col] = np.nan

    X = sub[feature_cols].values.astype(float)
    y = sub[target_col].values.astype(float) if target_col in sub.columns else np.zeros(len(sub))
    mu = sub[signal_col].values.astype(float) if signal_col in sub.columns else np.zeros(len(sub))

    index_pairs = list(zip(sub["date"].tolist(), sub["ticker"].tolist()))

    return X, y, mu, index_pairs


def impute_train_stats(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature median and std for train-window imputation.

    Returns
    -------
    (medians, stds) arrays shaped (n_features,).
    """
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X_train, axis=0) if X_train.shape[0] > 0 else np.zeros(X_train.shape[1])
        stds = np.nanstd(X_train, axis=0) if X_train.shape[0] > 0 else np.ones(X_train.shape[1])
    stds = np.where(stds < 1e-10, 1.0, stds)
    return medians, stds


def apply_imputation(
    X: np.ndarray,
    medians: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Apply train-derived imputation (NaN → median) to X."""
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        nan_mask = np.isnan(X_out[:, j])
        X_out[nan_mask, j] = medians[j]
    return X_out


# ---------------------------------------------------------------------------
# Prediction assembler: collect OOS predictions into wide panel
# ---------------------------------------------------------------------------


def assemble_predictions_panel(
    predictions_by_model: dict[str, list[tuple]],
    baseline_predictions: list[tuple],
    tickers: list[str],
) -> pd.DataFrame:
    """Assemble OOS predictions from multiple models into a daily wide panel.

    Parameters
    ----------
    predictions_by_model:
        Dict: model_name → list of (date, ticker, mu_final, delta).
    baseline_predictions:
        List of (date, ticker, mu_base) tuples.
    tickers:
        List of tickers.

    Returns
    -------
    pd.DataFrame
        Wide panel indexed by date with columns for each model's prediction.
    """
    # Build baseline pivot
    if not baseline_predictions:
        return pd.DataFrame()

    base_records = pd.DataFrame(baseline_predictions, columns=["date", "ticker", "mu_base"])
    base_pivot = base_records.pivot_table(index="date", columns="ticker", values="mu_base")

    result_frames: dict[str, pd.DataFrame] = {"baseline_mu_base": base_pivot}

    for model_name, preds in predictions_by_model.items():
        if not preds:
            continue
        pred_df = pd.DataFrame(preds, columns=["date", "ticker", "mu_final", "delta"])
        mu_pivot = pred_df.pivot_table(index="date", columns="ticker", values="mu_final")
        delta_pivot = pred_df.pivot_table(index="date", columns="ticker", values="delta")
        result_frames[f"{model_name}_mu_final"] = mu_pivot
        result_frames[f"{model_name}_delta"] = delta_pivot

    # Combine into single DataFrame (wide by ticker×model)
    combined = pd.concat(result_frames, axis=1)
    return combined
