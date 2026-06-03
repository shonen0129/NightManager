"""domain.correction.feature_builder – Feature construction for the nonlinear correction layer.

All features are derived exclusively from f_t (the K-dim factor scores from the
linear model) and optionally z_lin (the linear signal itself as a covariate).
The sector_id categorical feature is appended when using a single shared model
across all 17 JP sectors (recommended default).

Feature catalogue
-----------------
Required (always included):
  f_0 .. f_{K-1}     : raw factor scores (K=6 by default)
  sector_id           : integer 0..N_J-1 (only in "single" model mode)

Optional (controlled by flags):
  f_0^2 .. f_{K-1}^2 : squared factor scores  (use_squared=True)
  |f_0| .. |f_{K-1}|  : absolute factor scores (use_absolute=True)
  Δf_0 .. Δf_{K-1}   : factor score differences vs. delta_window days ago
                        (use_delta=True, requires f_t history buffer)

Notes
-----
* z_lin is NOT included as a feature by default.  The correction is designed
  to capture what z_lin *misses*, so conditioning on z_lin itself can cause
  the tree to learn a trivial identity correction.
* All features are returned as a numpy float32 array for LightGBM/XGBoost
  compatibility.  The ``feature_names`` property lists the column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureFlags:
    """Flags controlling optional feature generation."""

    use_squared: bool = False
    """Whether to append squared factor scores (f_k^2)."""

    use_absolute: bool = False
    """Whether to append absolute factor scores (|f_k|)."""

    use_delta: bool = False
    """Whether to append rolling delta of factor scores (f_{t} - f_{t-window})."""

    delta_window: int = 5
    """Lookback days for delta features (relevant only when use_delta=True)."""


class FeatureBuilder:
    """Constructs the feature matrix used by the nonlinear correction model.

    Parameters
    ----------
    n_factors : int
        Dimensionality of f_t (K).  Typically 6.
    n_sectors : int
        Number of JP sectors (N_J).  Typically 17.
    flags : FeatureFlags
        Optional feature flags.
    single_model : bool
        If True, sector_id is added as a feature (shared model mode).
    """

    def __init__(
        self,
        n_factors: int = 6,
        n_sectors: int = 17,
        flags: Optional[FeatureFlags] = None,
        single_model: bool = True,
    ) -> None:
        self.n_factors = n_factors
        self.n_sectors = n_sectors
        self.flags = flags if flags is not None else FeatureFlags()
        self.single_model = single_model
        self._feature_names: List[str] = self._build_feature_names()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature column names."""
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        """Total number of features per sample."""
        return len(self._feature_names)

    def build_single_day(
        self,
        f_t: np.ndarray,
        sector_ids: Optional[np.ndarray] = None,
        f_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build feature matrix for a single day across all sectors.

        Parameters
        ----------
        f_t : np.ndarray, shape (K,)
            Factor scores for day t.
        sector_ids : np.ndarray or None, shape (N_J,)
            Integer sector identifiers 0..N_J-1.  Inferred if None.
        f_history : np.ndarray or None, shape (delta_window, K)
            Most-recent ``delta_window`` rows of f_t used for delta features.
            Required when ``flags.use_delta=True``.

        Returns
        -------
        X : np.ndarray, shape (N_J, n_features), dtype float32
        """
        f_t = np.asarray(f_t, dtype=np.float32).reshape(-1)
        if f_t.shape[0] != self.n_factors:
            raise ValueError(
                f"f_t must have {self.n_factors} elements, got {f_t.shape[0]}"
            )

        if sector_ids is None:
            sector_ids = np.arange(self.n_sectors, dtype=np.int32)
        sector_ids = np.asarray(sector_ids, dtype=np.int32).reshape(-1)

        cols: List[np.ndarray] = []

        # Base factor scores – broadcast to N_J rows
        base = np.tile(f_t, (self.n_sectors, 1))  # (N_J, K)
        cols.append(base)

        # Optional features
        if self.flags.use_squared:
            cols.append(base ** 2)

        if self.flags.use_absolute:
            cols.append(np.abs(base))

        if self.flags.use_delta:
            delta = self._compute_delta(f_t, f_history)  # (K,)
            cols.append(np.tile(delta, (self.n_sectors, 1)))

        # Sector ID column (float representation for compatibility)
        if self.single_model:
            cols.append(sector_ids.reshape(-1, 1).astype(np.float32))

        X = np.concatenate(cols, axis=1).astype(np.float32)
        assert X.shape == (self.n_sectors, self.n_features), (
            f"Expected shape ({self.n_sectors}, {self.n_features}), got {X.shape}"
        )
        return X

    def build_panel(
        self,
        f_matrix: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[pd.DatetimeIndex]]:
        """Build a flat (long) feature matrix for training from a time-series panel.

        Each day t is expanded into N_J rows (one per sector), giving a
        total of T * N_J rows.

        Parameters
        ----------
        f_matrix : np.ndarray, shape (T, K)
            Panel of factor scores.
        dates : pd.DatetimeIndex or None, shape (T,)
            Trade dates corresponding to each row of f_matrix.

        Returns
        -------
        X : np.ndarray, shape (T * N_J, n_features), dtype float32
        sector_ids : np.ndarray, shape (T * N_J,), int
        expanded_dates : pd.DatetimeIndex or None, shape (T * N_J,)
        """
        T, K = f_matrix.shape
        if K != self.n_factors:
            raise ValueError(
                f"f_matrix must have {self.n_factors} columns, got {K}"
            )

        rows: List[np.ndarray] = []
        sector_rows: List[np.ndarray] = []

        for t in range(T):
            f_t = f_matrix[t].astype(np.float32)
            f_hist = f_matrix[max(0, t - self.flags.delta_window): t] if self.flags.use_delta else None
            x_day = self.build_single_day(f_t, f_history=f_hist)  # (N_J, n_features)
            rows.append(x_day)
            sector_rows.append(np.arange(self.n_sectors, dtype=np.int32))

        X = np.concatenate(rows, axis=0)  # (T*N_J, n_features)
        s = np.concatenate(sector_rows, axis=0)  # (T*N_J,)

        exp_dates: Optional[pd.DatetimeIndex] = None
        if dates is not None:
            exp_dates = pd.DatetimeIndex(
                np.repeat(dates, self.n_sectors)
            )

        return X, s, exp_dates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        for k in range(self.n_factors):
            names.append(f"f_{k}")
        if self.flags.use_squared:
            for k in range(self.n_factors):
                names.append(f"f_{k}_sq")
        if self.flags.use_absolute:
            for k in range(self.n_factors):
                names.append(f"f_{k}_abs")
        if self.flags.use_delta:
            for k in range(self.n_factors):
                names.append(f"f_{k}_delta{self.flags.delta_window}")
        if self.single_model:
            names.append("sector_id")
        return names

    def _compute_delta(
        self,
        f_t: np.ndarray,
        f_history: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute f_t - f_{t-window}.  Returns zeros if history insufficient."""
        if f_history is None or len(f_history) == 0:
            return np.zeros(self.n_factors, dtype=np.float32)
        f_lag = np.asarray(f_history[0], dtype=np.float32).reshape(-1)
        return (f_t - f_lag).astype(np.float32)
