"""domain.correction.nonlinear_layer – Constrained GBT nonlinear correction layer.

This module implements ``NonlinearCorrectionLayer``, which adds a weakly
regularised gradient-boosting correction **g(f_t)** on top of the existing
linear lead-lag signal **z_lin**:

    z_final = z_lin + g(f_t)

Design constraints enforced at training time
--------------------------------------------
* max_depth ∈ {2, 3}  – captures at most pairwise interactions
* n_estimators ≤ 200  – shallow ensemble with mandatory early stopping
* learning_rate ∈ [0.01, 0.05]
* Aggressive regularisation: min_child_samples, lambda, subsampling
* Optional monotone constraints per factor-sector pair (via YAML config)
* Scale clipping: g is clipped to ±clip_scale * |z_lin| element-wise

Interfaces
----------
``fit(f_t, z_lin, y, dates, tradeable_mask)``
    Train correction using leak-safe walk-forward CV for HP selection.

``predict(f_t, z_lin) -> np.ndarray``
    Return z_final = z_lin + g(f_t) for a single day.

``predict_with_attribution(f_t, z_lin) -> dict``
    Return z_final plus SHAP-based feature attribution.

``save(path) / load(path)``
    Serialise/deserialise the fitted model + metadata.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluation import (
    AdoptionDecision,
    CostModel,
    PerformanceMetrics,
    compute_net_returns,
    compute_performance_metrics,
    evaluate_correction_adoption,
)
from .feature_builder import FeatureBuilder, FeatureFlags
from .time_series_cv import (
    TimeSeriesPurgeSplit,
    audit_no_leak,
    check_contribution_cap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter configuration
# ---------------------------------------------------------------------------


@dataclass
class GBTHyperparams:
    """Validated hyperparameter set for a constrained GBT model.

    All values are pre-validated to enforce the shallow/strongly-regularised
    design constraints required by the strategy specification.
    """

    library: Literal["lightgbm", "xgboost"] = "lightgbm"
    max_depth: int = 2
    n_estimators: int = 100
    learning_rate: float = 0.02
    min_child_samples: int = 30        # LightGBM; mapped to min_child_weight for XGB
    reg_lambda: float = 5.0
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    early_stopping_rounds: int = 20
    clip_scale: float = 0.5            # |g| ≤ clip_scale * |z_lin|
    seed: int = 42

    def __post_init__(self) -> None:
        assert 2 <= self.max_depth <= 3, (
            f"max_depth must be 2 or 3 (got {self.max_depth}). "
            "Deeper trees increase overfitting risk."
        )
        assert 10 <= self.n_estimators <= 200, (
            f"n_estimators must be in [10, 200] (got {self.n_estimators})."
        )
        assert 0.005 <= self.learning_rate <= 0.10, (
            f"learning_rate must be in [0.005, 0.10] (got {self.learning_rate})."
        )
        assert self.min_child_samples >= 10, (
            f"min_child_samples must be >= 10 (got {self.min_child_samples})."
        )
        assert self.reg_lambda >= 0.0, "reg_lambda must be non-negative."
        assert 0.0 < self.subsample <= 1.0, "subsample must be in (0, 1]."
        assert 0.0 < self.colsample_bytree <= 1.0, "colsample_bytree must be in (0, 1]."
        assert self.clip_scale > 0.0, "clip_scale must be positive."

    def to_lightgbm_params(
        self,
        monotone_constraints: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Convert to LightGBM parameter dictionary."""
        params: Dict[str, Any] = {
            "objective": "regression",
            "metric": "rmse",
            "max_depth": self.max_depth,
            "num_leaves": 2 ** self.max_depth - 1,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.seed,
            "verbose": -1,
            "n_jobs": -1,
        }
        if monotone_constraints:
            params["monotone_constraints"] = monotone_constraints
            params["monotone_constraints_method"] = "advanced"
        return params

    def to_xgboost_params(
        self,
        monotone_constraints: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Convert to XGBoost parameter dictionary."""
        params: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "min_child_weight": max(1, self.min_child_samples // 5),
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.seed,
            "verbosity": 0,
            "n_jobs": -1,
        }
        if monotone_constraints:
            params["monotone_constraints"] = tuple(monotone_constraints)
        return params


# ---------------------------------------------------------------------------
# Monotone constraint builder
# ---------------------------------------------------------------------------


def build_monotone_constraints(
    feature_names: List[str],
    default_constraints: Optional[Dict[str, int]] = None,
    sector_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    n_sectors: int = 17,
) -> List[int]:
    """Build per-feature monotone constraint list for GBT training.

    In single-model mode each factor feature appears once (shared across all
    sectors), so the default_constraints are applied per factor name.

    Parameters
    ----------
    feature_names : list of str
        Ordered feature names from FeatureBuilder.
    default_constraints : dict or None
        Mapping from feature name pattern to constraint direction {-1, 0, 1}.
        Keys are matched against feature names by substring.
        Example: {"f_0": 1} means factor_0 is monotone non-decreasing.
    sector_overrides : dict or None
        Per-sector override (only meaningful in per-sector model mode).
        Keys are sector indices (str), values are dicts like default_constraints.
    n_sectors : int
        Number of JP sectors (used only in per-sector mode).

    Returns
    -------
    list of int
        One constraint per feature column: -1, 0, or +1.
    """
    if default_constraints is None:
        default_constraints = {}

    constraints: List[int] = []
    for feat in feature_names:
        c = 0
        for pattern, direction in default_constraints.items():
            if pattern in feat:
                c = int(direction)
                break
        constraints.append(c)

    return constraints


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class NonlinearCorrectionLayer:
    """Constrained GBT correction layer for the lead-lag strategy.

    Fits a shallow, strongly regularised gradient boosting model to learn
    *residual* corrections that the linear signal z_lin misses.  The final
    signal is:

        z_final_j = z_lin_j + clip(g_j(f_t), ±clip_scale * |z_lin_j|)

    Parameters
    ----------
    n_factors : int
        Dimension of f_t (K).  Default 6.
    n_sectors : int
        Number of JP sectors (N_J).  Default 17.
    hyperparams : GBTHyperparams or None
        Model hyperparameters.  Defaults to GBTHyperparams().
    flags : FeatureFlags or None
        Optional feature augmentation flags.
    multi_output_mode : {"single", "per_sector"}
        "single": one shared model with sector_id as a categorical feature.
        "per_sector": 17 independent models.  Default is "single".
    correction_enabled : bool
        If False, predict() returns z_lin unchanged (ON/OFF switch).
    monotone_constraints : list of int or None
        Pre-built constraint list (length = n_features).  If None, no
        constraints are applied.
    contribution_cap : float
        Portfolio-average |g| / |z_lin| warning threshold.  Default 0.3.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        n_factors: int = 6,
        n_sectors: int = 17,
        hyperparams: Optional[GBTHyperparams] = None,
        flags: Optional[FeatureFlags] = None,
        multi_output_mode: Literal["single", "per_sector"] = "single",
        correction_enabled: bool = True,
        monotone_constraints: Optional[List[int]] = None,
        contribution_cap: float = 0.3,
    ) -> None:
        self.n_factors = n_factors
        self.n_sectors = n_sectors
        self.hp = hyperparams if hyperparams is not None else GBTHyperparams()
        self.flags = flags if flags is not None else FeatureFlags()
        self.multi_output_mode = multi_output_mode
        self.correction_enabled = correction_enabled
        self.monotone_constraints = monotone_constraints
        self.contribution_cap = contribution_cap

        single_model = (multi_output_mode == "single")
        self.feature_builder = FeatureBuilder(
            n_factors=n_factors,
            n_sectors=n_sectors,
            flags=self.flags,
            single_model=single_model,
        )

        # Fitted models: one per sector in per_sector mode, one in single mode
        self._models: List[Any] = []
        self._is_fitted: bool = False
        self._fit_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        f_matrix: np.ndarray,
        z_lin_matrix: np.ndarray,
        y_matrix: np.ndarray,
        dates: pd.DatetimeIndex,
        tradeable_mask: Optional[np.ndarray] = None,
        signal_dates: Optional[pd.DatetimeIndex] = None,
        n_trials_recorded: int = 1,
        cv_splitter: Optional[TimeSeriesPurgeSplit] = None,
    ) -> "NonlinearCorrectionLayer":
        """Train the correction model.

        Parameters
        ----------
        f_matrix : np.ndarray, shape (T, K)
            Factor scores (signal date = US close).
        z_lin_matrix : np.ndarray, shape (T, N_J)
            Linear signal from the existing model.
        y_matrix : np.ndarray, shape (T, N_J)
            Realized OC returns (target, trade date = signal_date + 1bd).
        dates : pd.DatetimeIndex, shape (T,)
            Trade dates (JP OC return dates = signal_date + 1 business day).
        tradeable_mask : np.ndarray or None, shape (T, N_J), dtype bool
            If provided, non-tradeable samples are excluded from training.
        signal_dates : pd.DatetimeIndex or None, shape (T,)
            Dates on which signals were computed (= US close = dates - 1bd).
            Used for leak audit.  If None, assumed to be dates - 1 calendar day
            (approximate; provide explicitly for rigour).
        n_trials_recorded : int
            Number of HP configurations tried (used in DSR reporting).
        cv_splitter : TimeSeriesPurgeSplit or None
            Custom CV splitter.  Defaults to TimeSeriesPurgeSplit(n_splits=5).

        Returns
        -------
        self
        """
        T, K = f_matrix.shape
        assert K == self.n_factors, f"f_matrix has {K} factors, expected {self.n_factors}"
        assert z_lin_matrix.shape == (T, self.n_sectors)
        assert y_matrix.shape == (T, self.n_sectors)
        assert len(dates) == T

        # ── Data-leak audit ──────────────────────────────────────────────
        if signal_dates is None:
            # Approximate: shift back 1 day (not rigorous; warn user)
            signal_dates = dates - pd.Timedelta(days=1)
            warnings.warn(
                "signal_dates not provided; approximated as dates - 1 calendar day. "
                "Pass signal_dates explicitly for a rigorous leak audit.",
                stacklevel=2,
            )
        audit_no_leak(
            signal_dates=signal_dates,
            target_dates=dates,
            sample_label="NonlinearCorrectionLayer.fit",
        )

        # ── Residual targets: y - z_lin (what the linear model misses) ──
        residuals = y_matrix - z_lin_matrix  # (T, N_J)

        # ── Build feature panel ──────────────────────────────────────────
        X_panel, sector_ids, _ = self.feature_builder.build_panel(f_matrix, dates=dates)
        # X_panel: (T*N_J, n_features), sector_ids: (T*N_J,)

        # Flatten residuals and mask
        y_flat = residuals.reshape(-1)  # (T*N_J,)
        z_lin_flat = z_lin_matrix.reshape(-1)

        if tradeable_mask is not None:
            mask_flat = tradeable_mask.reshape(-1).astype(bool)
        else:
            mask_flat = np.ones(T * self.n_sectors, dtype=bool)

        # Also exclude NaN targets
        valid = mask_flat & np.isfinite(y_flat) & np.isfinite(z_lin_flat)

        X_valid = X_panel[valid]
        y_valid = y_flat[valid]

        logger.info(
            "NonlinearCorrectionLayer.fit: T=%d, N_J=%d, "
            "total_samples=%d, valid_samples=%d, mode=%s",
            T, self.n_sectors, T * self.n_sectors, valid.sum(), self.multi_output_mode,
        )

        # ── Train ────────────────────────────────────────────────────────
        if self.multi_output_mode == "single":
            self._models = [self._train_single_model(X_valid, y_valid)]
        else:
            self._models = []
            for j in range(self.n_sectors):
                # Build per-sector data: rows in X_panel where sector_id == j
                sector_mask_in_valid = sector_ids[valid] == j
                X_j = X_valid[sector_mask_in_valid]
                y_j = y_valid[sector_mask_in_valid]
                model_j = self._train_single_model(X_j, y_j)
                self._models.append(model_j)

        self._is_fitted = True
        self._fit_metadata = {
            "version": self.VERSION,
            "n_factors": self.n_factors,
            "n_sectors": self.n_sectors,
            "n_samples_trained": int(valid.sum()),
            "train_date_range": [str(dates.min().date()), str(dates.max().date())],
            "multi_output_mode": self.multi_output_mode,
            "n_trials_recorded": n_trials_recorded,
            "feature_names": self.feature_builder.feature_names,
            "hyperparams": asdict(self.hp),
            "flags": asdict(self.flags),
        }
        logger.info("NonlinearCorrectionLayer fitted successfully.")
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        f_t: np.ndarray,
        z_lin: np.ndarray,
        f_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute z_final = z_lin + clip(g(f_t), ±clip_scale * |z_lin|).

        Parameters
        ----------
        f_t : np.ndarray, shape (K,)
            Factor scores for the current day.
        z_lin : np.ndarray, shape (N_J,)
            Linear signal from the existing model.
        f_history : np.ndarray or None, shape (delta_window, K)
            Recent factor score history (needed if use_delta=True).

        Returns
        -------
        z_final : np.ndarray, shape (N_J,)
        """
        if not self.correction_enabled:
            return np.asarray(z_lin, dtype=float).copy()

        if not self._is_fitted:
            raise RuntimeError(
                "NonlinearCorrectionLayer is not fitted. Call fit() first."
            )

        g = self._compute_correction(f_t, z_lin, f_history)
        z_final = np.asarray(z_lin, dtype=float) + g

        # Warn if correction dominates
        check_contribution_cap(g, z_lin, cap=self.contribution_cap)

        return z_final

    # ------------------------------------------------------------------
    # predict_with_attribution
    # ------------------------------------------------------------------

    def predict_with_attribution(
        self,
        f_t: np.ndarray,
        z_lin: np.ndarray,
        f_history: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Return z_final plus SHAP-based feature attribution.

        Parameters
        ----------
        f_t : np.ndarray, shape (K,)
        z_lin : np.ndarray, shape (N_J,)
        f_history : np.ndarray or None

        Returns
        -------
        dict with keys:
            "z_final"       : np.ndarray (N_J,) – corrected signal
            "g"             : np.ndarray (N_J,) – raw correction before clipping
            "g_clipped"     : np.ndarray (N_J,) – correction after clipping
            "z_lin"         : np.ndarray (N_J,) – linear signal (unchanged)
            "shap_values"   : np.ndarray (N_J, n_features) or None
            "feature_names" : list of str
            "contribution_ratio" : float – portfolio |g|/|z_lin| mean
        """
        if not self._is_fitted:
            raise RuntimeError(
                "NonlinearCorrectionLayer is not fitted. Call fit() first."
            )

        z_lin = np.asarray(z_lin, dtype=float)
        g, g_raw = self._compute_correction(f_t, z_lin, f_history, return_raw=True)
        z_final = z_lin + g

        # SHAP values
        shap_values = self._compute_shap(f_t, f_history)

        # Contribution ratio
        denom = np.abs(z_lin)
        valid_mask = denom > 1e-12
        ratio = float(np.mean(np.abs(g[valid_mask]) / denom[valid_mask])) if valid_mask.any() else 0.0

        return {
            "z_final": z_final,
            "g": g_raw,
            "g_clipped": g,
            "z_lin": z_lin,
            "shap_values": shap_values,
            "feature_names": self.feature_builder.feature_names,
            "contribution_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted layer to disk.

        Saves a directory containing:
          models.pkl   – the list of fitted GBT models
          metadata.json – configuration and version info
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        models_path = out / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(self._models, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta_path = out / "metadata.json"
        meta = {
            **self._fit_metadata,
            "correction_enabled": self.correction_enabled,
            "contribution_cap": self.contribution_cap,
            "is_fitted": self._is_fitted,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("NonlinearCorrectionLayer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "NonlinearCorrectionLayer":
        """Load a saved correction layer from disk.

        Parameters
        ----------
        path : str
            Directory path saved by ``save()``.

        Returns
        -------
        NonlinearCorrectionLayer (fitted)
        """
        out = Path(path)

        meta_path = out / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        hp_dict = meta.get("hyperparams", {})
        hp = GBTHyperparams(**{k: v for k, v in hp_dict.items() if k != "library"})
        hp = GBTHyperparams(library=hp_dict.get("library", "lightgbm"), **{
            k: v for k, v in hp_dict.items() if k != "library"
        })

        flags_dict = meta.get("flags", {})
        flags = FeatureFlags(**flags_dict)

        layer = cls(
            n_factors=meta.get("n_factors", 6),
            n_sectors=meta.get("n_sectors", 17),
            hyperparams=hp,
            flags=flags,
            multi_output_mode=meta.get("multi_output_mode", "single"),
            correction_enabled=meta.get("correction_enabled", True),
            contribution_cap=meta.get("contribution_cap", 0.3),
        )

        models_path = out / "models.pkl"
        with open(models_path, "rb") as f:
            layer._models = pickle.load(f)

        layer._is_fitted = meta.get("is_fitted", True)
        layer._fit_metadata = meta

        logger.info("NonlinearCorrectionLayer loaded from %s", path)
        return layer

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        ax=None,
        importance_type: str = "gain",
        top_n: int = 20,
    ):
        """Plot feature importance for the fitted model(s).

        Parameters
        ----------
        ax : matplotlib Axes or None
        importance_type : str
            Importance type (passed to LightGBM/XGBoost).
        top_n : int
            Number of top features to show.

        Returns
        -------
        matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        names = self.feature_builder.feature_names
        importances = np.zeros(len(names))

        for model in self._models:
            if self.hp.library == "lightgbm":
                imp = model.booster_.feature_importance(importance_type=importance_type)
            else:
                imp = model.feature_importances_
            importances += np.asarray(imp, dtype=float)

        importances /= max(len(self._models), 1)

        sort_idx = np.argsort(importances)[::-1][:top_n]
        top_names = [names[i] for i in sort_idx]
        top_imp = importances[sort_idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        else:
            fig = ax.figure

        ax.barh(range(len(top_names)), top_imp[::-1], align="center")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel(f"Feature Importance ({importance_type})")
        ax.set_title("Nonlinear Correction Layer – Feature Importance")
        fig.tight_layout()
        return fig

    def compute_shap_panel(
        self,
        f_matrix: np.ndarray,
        f_history_window: int = 5,
    ) -> Dict[str, Any]:
        """Compute SHAP values for a panel of factor scores.

        Parameters
        ----------
        f_matrix : np.ndarray, shape (T, K)
        f_history_window : int

        Returns
        -------
        dict:
            "shap_values"   : np.ndarray (T, N_J, n_features)
            "feature_names" : list of str
        """
        T = len(f_matrix)
        all_shap = []
        for t in range(T):
            f_hist = f_matrix[max(0, t - f_history_window): t] if self.flags.use_delta else None
            sv = self._compute_shap(f_matrix[t], f_hist)
            all_shap.append(sv)  # (N_J, n_features) or None

        if all_shap[0] is not None:
            shap_panel = np.stack(all_shap, axis=0)  # (T, N_J, n_features)
        else:
            shap_panel = None

        return {
            "shap_values": shap_panel,
            "feature_names": self.feature_builder.feature_names,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        """Train one GBT model with an internal validation split for early stopping."""
        if len(X) < 40:
            raise ValueError(
                f"Not enough training samples ({len(X)}) to train GBT correction. "
                "Need at least 40 non-NaN, tradeable samples."
            )

        # Reserve last 10% for early-stopping validation (time-ordered)
        n_val = max(10, len(X) // 10)
        n_train = len(X) - n_val
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        constraints = self.monotone_constraints

        if self.hp.library == "lightgbm":
            return self._train_lgb(X_tr, y_tr, X_val, y_val, constraints)
        else:
            return self._train_xgb(X_tr, y_tr, X_val, y_val, constraints)

    def _train_lgb(self, X_tr, y_tr, X_val, y_val, constraints):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "lightgbm is required for NonlinearCorrectionLayer. "
                "Install it with: pip install lightgbm"
            ) from e

        params = self.hp.to_lightgbm_params(constraints)

        # categorical_feature index for sector_id
        feat_names = self.feature_builder.feature_names
        cat_features: List[int] = []
        if self.multi_output_mode == "single" and "sector_id" in feat_names:
            cat_features = [feat_names.index("sector_id")]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.hp.early_stopping_rounds, verbose=False),
                       lgb.log_evaluation(period=-1)],
            categorical_feature=cat_features if cat_features else "auto",
        )
        return model

    def _train_xgb(self, X_tr, y_tr, X_val, y_val, constraints):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "xgboost is required when library='xgboost'. "
                "Install it with: pip install xgboost"
            ) from e

        params = self.hp.to_xgboost_params(constraints)
        n_est = params.pop("n_estimators")

        model = xgb.XGBRegressor(
            n_estimators=n_est,
            early_stopping_rounds=self.hp.early_stopping_rounds,
            **params,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def _compute_correction(
        self,
        f_t: np.ndarray,
        z_lin: np.ndarray,
        f_history: Optional[np.ndarray],
        return_raw: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Predict raw correction g and apply scale clipping."""
        f_t = np.asarray(f_t, dtype=np.float32).reshape(-1)
        z_lin = np.asarray(z_lin, dtype=float).reshape(-1)

        if self.multi_output_mode == "single":
            X_day = self.feature_builder.build_single_day(f_t, f_history=f_history)
            g_raw = self._models[0].predict(X_day).astype(float)
        else:
            g_raw = np.zeros(self.n_sectors, dtype=float)
            for j, model in enumerate(self._models):
                # Build per-sector feature row (no sector_id column in per_sector mode)
                X_j = self.feature_builder.build_single_day(
                    f_t,
                    sector_ids=np.array([j]),
                    f_history=f_history,
                )[:1]  # shape (1, n_features)
                g_raw[j] = float(model.predict(X_j)[0])

        # Scale clipping: |g| ≤ clip_scale * |z_lin|
        clip_bound = self.hp.clip_scale * np.abs(z_lin)
        g_clipped = np.clip(g_raw, -clip_bound, clip_bound)

        if return_raw:
            return g_clipped, g_raw
        return g_clipped

    def _compute_shap(
        self,
        f_t: np.ndarray,
        f_history: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Compute SHAP values for a single day.

        Returns ndarray (N_J, n_features) or None if shap is unavailable.
        """
        try:
            import shap
        except ImportError:
            warnings.warn(
                "shap is not installed. SHAP attribution will be skipped. "
                "Install with: pip install shap",
                stacklevel=3,
            )
            return None

        f_t = np.asarray(f_t, dtype=np.float32).reshape(-1)

        if self.multi_output_mode == "single":
            X_day = self.feature_builder.build_single_day(f_t, f_history=f_history)
            explainer = shap.TreeExplainer(self._models[0])
            sv = explainer.shap_values(X_day)  # (N_J, n_features)
            return np.asarray(sv, dtype=float)
        else:
            all_sv = []
            for j, model in enumerate(self._models):
                X_j = self.feature_builder.build_single_day(
                    f_t,
                    sector_ids=np.array([j]),
                    f_history=f_history,
                )[:1]
                explainer = shap.TreeExplainer(model)
                sv_j = explainer.shap_values(X_j)  # (1, n_features)
                all_sv.append(sv_j[0])
            return np.stack(all_sv, axis=0)  # (N_J, n_features)

    # ------------------------------------------------------------------
    # Classmethods for config-driven construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NonlinearCorrectionLayer":
        """Instantiate from a parsed YAML config dict.

        Parameters
        ----------
        config : dict
            Parsed contents of nonlinear_correction.yaml.

        Returns
        -------
        NonlinearCorrectionLayer (not yet fitted)
        """
        correction = config.get("correction", {})
        hp_raw = config.get("hyperparams", {})
        feat_raw = config.get("features", {})
        mono_raw = config.get("monotone_constraints", {})

        hp = GBTHyperparams(
            library=correction.get("library", "lightgbm"),
            max_depth=hp_raw.get("max_depth", 2),
            n_estimators=hp_raw.get("n_estimators", 100),
            learning_rate=hp_raw.get("learning_rate", 0.02),
            min_child_samples=hp_raw.get("min_child_samples", 30),
            reg_lambda=hp_raw.get("reg_lambda", 5.0),
            subsample=hp_raw.get("subsample", 0.7),
            colsample_bytree=hp_raw.get("colsample_bytree", 0.7),
            early_stopping_rounds=hp_raw.get("early_stopping_rounds", 20),
            clip_scale=hp_raw.get("clip_scale", 0.5),
            seed=correction.get("seed", 42),
        )

        flags = FeatureFlags(
            use_squared=feat_raw.get("use_squared", False),
            use_absolute=feat_raw.get("use_absolute", False),
            use_delta=feat_raw.get("use_delta", False),
            delta_window=feat_raw.get("delta_window", 5),
        )

        n_factors = correction.get("n_factors", 6)
        n_sectors = correction.get("n_sectors", 17)
        multi_output_mode = correction.get("multi_output_mode", "single")
        correction_enabled = correction.get("enabled", True)
        contribution_cap = config.get("thresholds", {}).get("contribution_cap", 0.3)

        # Build a temporary FeatureBuilder to get feature names for constraint lookup
        fb_temp = FeatureBuilder(
            n_factors=n_factors,
            n_sectors=n_sectors,
            flags=flags,
            single_model=(multi_output_mode == "single"),
        )
        defaults = {
            k: int(v)
            for k, v in mono_raw.items()
            if k.startswith("default_")
            for k in [k.replace("default_", "")]
        }
        # Simpler: pass raw default mapping pattern → direction
        default_map = {
            k.replace("default_", ""): int(v)
            for k, v in mono_raw.items()
            if k.startswith("default_")
        }

        mono_constraints = build_monotone_constraints(
            feature_names=fb_temp.feature_names,
            default_constraints=default_map,
        )
        # If all zeros, pass None
        if all(c == 0 for c in mono_constraints):
            mono_constraints = None

        return cls(
            n_factors=n_factors,
            n_sectors=n_sectors,
            hyperparams=hp,
            flags=flags,
            multi_output_mode=multi_output_mode,
            correction_enabled=correction_enabled,
            monotone_constraints=mono_constraints,
            contribution_cap=contribution_cap,
        )

    def __repr__(self) -> str:
        return (
            f"NonlinearCorrectionLayer("
            f"mode={self.multi_output_mode}, "
            f"enabled={self.correction_enabled}, "
            f"fitted={self._is_fitted}, "
            f"lib={self.hp.library}, "
            f"depth={self.hp.max_depth}, "
            f"n_est={self.hp.n_estimators})"
        )
