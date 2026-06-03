"""domain.correction.time_series_cv – Leak-safe time-series cross-validation.

Key design principles
---------------------
1. **Temporal ordering is sacred** – all splits respect chronological order.
   Random K-Fold is explicitly forbidden (enforced by assertion in TimeSeriesPurgeSplit).

2. **Purge** – training samples whose target overlaps with the validation period
   are excluded.  For a 1-day-ahead prediction model the purge is typically
   gap_purge=1 (the signal at t predicts t+1, so the last signal in the train
   fold overlaps with validation day 1 – remove it).

3. **Embargo** – a buffer of ``embargo`` days after each validation fold is
   excluded from the immediately following training window.  This prevents the
   model from learning patterns that are adjacent in time to the validation
   period (regime bleed-through).

4. **Walk-forward** – supports both *expanding-window* (train grows with each
   fold) and *rolling-window* (fixed-size train) modes.

5. **Leak audit** – ``audit_no_leak()`` asserts that every feature sample was
   computed strictly from information available at the signal date and that the
   corresponding target is a *future* value.

Usage example
-------------
>>> splitter = TimeSeriesPurgeSplit(n_splits=5, embargo=5)
>>> for train_idx, val_idx in splitter.split(dates):
...     X_tr, y_tr = X[train_idx], y[train_idx]
...     X_val, y_val = X[val_idx], y[val_idx]
...     model.fit(X_tr, y_tr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TimeSeriesPurgeSplit
# ---------------------------------------------------------------------------


@dataclass
class SplitInfo:
    """Metadata about a single CV split."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_val: int
    n_purged: int
    n_embargoed: int


class TimeSeriesPurgeSplit:
    """Time-series cross-validator with purge and embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    train_size : int or None
        If None, use expanding window.  If int, use rolling window of this
        many samples.
    gap_purge : int
        Number of samples to remove from the *end* of each training fold
        because they overlap with the validation period.
    embargo : int
        Number of samples to remove from the *start* of each training fold
        that immediately follow the previous validation fold (regime bleed
        prevention).
    min_train_size : int
        Minimum required training samples; folds smaller than this are skipped.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        gap_purge: int = 1,
        embargo: int = 5,
        min_train_size: int = 120,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if gap_purge < 0:
            raise ValueError("gap_purge must be >= 0")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_splits = n_splits
        self.train_size = train_size
        self.gap_purge = gap_purge
        self.embargo = embargo
        self.min_train_size = min_train_size

    def split(
        self,
        dates: pd.DatetimeIndex,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_indices, val_indices) for each fold.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Chronologically ordered dates corresponding to individual *day*
            observations (not expanded sector rows).

        Yields
        ------
        train_idx : np.ndarray of int
        val_idx   : np.ndarray of int
        """
        n = len(dates)
        assert isinstance(dates, pd.DatetimeIndex), (
            "dates must be pd.DatetimeIndex – random K-Fold is forbidden"
        )
        assert dates.is_monotonic_increasing, (
            "dates must be sorted in ascending chronological order"
        )

        # Compute fold boundaries in sample-index space
        # Reserve the first portion for training; split the rest into n_splits val folds
        test_size = n // (self.n_splits + 1)
        fold_size = max(test_size, 1)

        for fold in range(self.n_splits):
            val_start = n - (self.n_splits - fold) * fold_size
            val_end = val_start + fold_size

            if val_start <= 0:
                continue  # not enough data

            # Training range before purge
            if self.train_size is not None:
                raw_train_start = max(0, val_start - self.train_size)
            else:
                raw_train_start = 0

            # Apply embargo from previous val fold
            if fold > 0:
                prev_val_end = n - (self.n_splits - (fold - 1)) * fold_size
                train_start = prev_val_end + self.embargo
                train_start = max(train_start, raw_train_start)
            else:
                train_start = raw_train_start

            # Apply purge at the end of training
            train_end = val_start - self.gap_purge

            if train_end - train_start < self.min_train_size:
                continue  # not enough training data

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, min(val_end, n))

            if len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_split_info(self, dates: pd.DatetimeIndex) -> list[SplitInfo]:
        """Return metadata for each fold without yielding arrays."""
        infos: list[SplitInfo] = []
        for fold_i, (tr, val) in enumerate(self.split(dates)):
            purged = max(0, self.gap_purge)
            embargoed = self.embargo if fold_i > 0 else 0
            infos.append(
                SplitInfo(
                    fold_index=fold_i,
                    train_start=dates[tr[0]] if len(tr) > 0 else pd.NaT,
                    train_end=dates[tr[-1]] if len(tr) > 0 else pd.NaT,
                    val_start=dates[val[0]],
                    val_end=dates[val[-1]],
                    n_train=len(tr),
                    n_val=len(val),
                    n_purged=purged,
                    n_embargoed=embargoed,
                )
            )
        return infos


# ---------------------------------------------------------------------------
# Data-leak audit
# ---------------------------------------------------------------------------


def audit_no_leak(
    signal_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex,
    sample_label: str = "sample",
) -> None:
    """Assert that every signal date strictly precedes its target date.

    The lead-lag pipeline generates signal on date ``t`` (after US close)
    and the target is the JP Open-to-Close return on date ``t+1``.
    This function verifies ``signal_date < target_date`` for every row.

    Parameters
    ----------
    signal_dates : pd.DatetimeIndex
        Dates on which the signal (f_t, z_lin) is computed.
        Must be the *US close date* (= JP trade date - 1 business day).
    target_dates : pd.DatetimeIndex
        Dates for which the realized return y_{j,t+1} is measured.
    sample_label : str
        Label used in error messages.

    Raises
    ------
    AssertionError
        If any signal_date >= target_date (data leak detected).
    """
    assert len(signal_dates) == len(target_dates), (
        f"signal_dates ({len(signal_dates)}) and target_dates ({len(target_dates)}) "
        "must have equal length"
    )
    sig_arr = signal_dates.to_numpy(dtype="datetime64[D]")
    tgt_arr = target_dates.to_numpy(dtype="datetime64[D]")

    leaky_mask = sig_arr >= tgt_arr
    if leaky_mask.any():
        leaky_positions = np.where(leaky_mask)[0][:5]  # show first 5 offenders
        details = [
            f"  [{i}] signal={signal_dates[i].date()}, target={target_dates[i].date()}"
            for i in leaky_positions
        ]
        raise AssertionError(
            f"DATA LEAK DETECTED in {sample_label}!\n"
            f"signal_date >= target_date for {leaky_mask.sum()} samples:\n"
            + "\n".join(details)
        )


def audit_feature_dates_within_window(
    feature_dates: pd.DatetimeIndex,
    max_lookback_days: int,
    sample_label: str = "features",
) -> None:
    """Check that feature computations do not use data older than expected.

    This is a sanity check – it cannot catch all future-data leaks but
    ensures the rolling window assumption is not violated by accident.

    Parameters
    ----------
    feature_dates : pd.DatetimeIndex
        Sorted dates of each feature sample (= signal computation dates).
    max_lookback_days : int
        Maximum allowed lookback in calendar days (e.g. corr_window=60 ≈ 90 calendar days).
    sample_label : str
        Label used in error messages.
    """
    assert feature_dates.is_monotonic_increasing, (
        f"{sample_label}: dates must be sorted ascending"
    )
    if len(feature_dates) < 2:
        return

    gaps = np.diff(feature_dates.to_numpy(dtype="datetime64[D]")).astype(int)
    max_gap = gaps.max()
    # A single gap > max_lookback_days would imply a break in the rolling
    # window but is NOT itself a leak; just a warning heuristic.
    if max_gap > max_lookback_days:
        import warnings
        warnings.warn(
            f"{sample_label}: maximum consecutive date gap is {max_gap} calendar days "
            f"(> max_lookback_days={max_lookback_days}). "
            "Verify that no future data is inadvertently included.",
            stacklevel=2,
        )


def check_contribution_cap(
    g: np.ndarray,
    z_lin: np.ndarray,
    cap: float = 0.3,
    sample_label: str = "day",
) -> bool:
    """Warn if the nonlinear correction dominates the linear signal.

    Parameters
    ----------
    g : np.ndarray, shape (N_J,)
        Nonlinear correction values.
    z_lin : np.ndarray, shape (N_J,)
        Linear signal values.
    cap : float
        Portfolio-average |g| / |z_lin| threshold.
    sample_label : str
        Label for warning messages.

    Returns
    -------
    bool
        True if the cap is exceeded (a warning is emitted).
    """
    import warnings

    denom = np.abs(z_lin)
    valid = denom > 1e-12
    if not valid.any():
        return False

    ratio = np.mean(np.abs(g[valid]) / denom[valid])
    if ratio > cap:
        warnings.warn(
            f"Contribution cap exceeded ({sample_label}): "
            f"|g|/|z_lin| portfolio mean = {ratio:.3f} > cap = {cap:.3f}. "
            "Consider tightening clip_scale or regularization.",
            stacklevel=2,
        )
        return True
    return False
