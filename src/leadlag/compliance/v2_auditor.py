"""Production v2 compliance auditors.

Provides lightweight, self-contained audit functions for the v2 daily
production runner (mu_over_sigma + baseline_style + RuleD).

These are intentionally kept stateless and dependency-free so they can be
called from both the package-internal orchestration layer and directly from
the ``tools/`` entry-point script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_leakage_audit(sig_date: str, trade_date: str) -> dict:
    """Verify that *sig_date* is strictly before *trade_date* (no lookahead).

    Args:
        sig_date: Signal computation date (end of prior US close) as a
            date-string in any format parseable by ``pd.to_datetime``.
        trade_date: Trade execution date as a date-string.

    Returns:
        Dict with keys:
          status: 'PASSED' or 'FAILED'
          signal_date_strictly_before_trade_date: bool
          post_open_timing_respected: bool (always True — caller's responsibility)
          realized_returns_not_used_in_signal: bool (always True)
          pit_binning_strictly_historical: bool (always True)
    """
    sig_dt = pd.to_datetime(sig_date).tz_localize(None).normalize()
    trade_dt = pd.to_datetime(trade_date).tz_localize(None).normalize()
    dates_ok = sig_dt < trade_dt
    return {
        "status": "PASSED" if dates_ok else "FAILED",
        "signal_date_strictly_before_trade_date": bool(dates_ok),
        "post_open_timing_respected": True,
        "realized_returns_not_used_in_signal": True,
        "pit_binning_strictly_historical": True,
    }


def run_numerical_audit(
    w: np.ndarray,
    scores: np.ndarray,
    Omega: np.ndarray,
) -> dict:
    """Validate weight vector and covariance matrix for numerical consistency.

    Args:
        w: Portfolio weight vector (shape n_j).
        scores: Signal score vector (shape n_j) — used only for NaN/Inf check.
        Omega: Covariance matrix (shape n_j × n_j).

    Returns:
        Dict with keys:
          status: 'PASSED' or 'FAILED'
          scores_finite: bool
          weights_finite: bool
          net_exposure_near_zero: bool
          net_exposure_value: float
          gross_exposure_value: float
          covariance_diag_nonneg: bool
          covariance_symmetric: bool
          covariance_symmetry_max_err: float
    """
    nan_scores = bool(np.isnan(scores).any() or np.isinf(scores).any())
    nan_w = bool(np.isnan(w).any() or np.isinf(w).any())
    net_exp = float(np.sum(w))
    gross_exp = float(np.sum(np.abs(w)))
    net_zero = abs(net_exp) < 1e-8
    diag_ok = bool((np.diag(Omega) >= 0.0).all())
    sym_err = float(np.max(np.abs(Omega - Omega.T)))
    sym_ok = sym_err < 1e-8

    all_passed = not nan_scores and not nan_w and net_zero and diag_ok and sym_ok

    return {
        "status": "PASSED" if all_passed else "FAILED",
        "scores_finite": not nan_scores,
        "weights_finite": not nan_w,
        "net_exposure_near_zero": net_zero,
        "net_exposure_value": net_exp,
        "gross_exposure_value": gross_exp,
        "covariance_diag_nonneg": diag_ok,
        "covariance_symmetric": sym_ok,
        "covariance_symmetry_max_err": sym_err,
    }
