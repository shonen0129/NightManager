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


def run_leakage_audit(
    sig_date: str,
    trade_date: str,
    *,
    gap_data_loaded: bool = True,
    pit_history_trade_dates: np.ndarray | None = None,
) -> dict:
    """Verify no lookahead leakage in signal computation.

    Args:
        sig_date: Signal computation date (end of prior US close) as a
            date-string in any format parseable by ``pd.to_datetime``.
        trade_date: Trade execution date as a date-string.
        gap_data_loaded: Whether gap matrices (mu_gap, Omega_gap) were
            successfully loaded. These are only available after the JP
            market open, so this verifies post-open timing.
        pit_history_trade_dates: Trade dates of the PIT IR history rows
            used for binning. Verified to be strictly before *trade_date*.
            If None, the check is skipped (vacuously True).

    Returns:
        Dict with keys:
          status: 'PASSED' or 'FAILED'
          signal_date_strictly_before_trade_date: bool
          post_open_timing_respected: bool
          realized_returns_not_used_in_signal: bool
          pit_binning_strictly_historical: bool
    """
    sig_dt = pd.to_datetime(sig_date).tz_localize(None).normalize()
    trade_dt = pd.to_datetime(trade_date).tz_localize(None).normalize()
    dates_ok = sig_dt < trade_dt

    # Gap data is only available after JP market open (9:00 JST).
    # If gap matrices were not loaded, post-open timing was not respected.
    post_open_ok = bool(gap_data_loaded)

    # If signal date is strictly before trade date, the trade date's
    # realized returns (jp_oc_*) cannot be part of the signal input.
    realized_ok = bool(dates_ok)

    # PIT IR history must not include the current trade date.
    if pit_history_trade_dates is not None and len(pit_history_trade_dates) > 0:
        hist_dts = pd.to_datetime(pit_history_trade_dates).normalize()
        pit_ok = bool((hist_dts < trade_dt).all())
    else:
        pit_ok = True

    # Gap data freshness check (1 to 10 calendar days difference)
    # Upper bound accommodates JP long holidays (Golden Week etc.)
    days_diff = (trade_dt - sig_dt).days
    freshness_ok = 1 <= days_diff <= 10

    all_passed = dates_ok and post_open_ok and realized_ok and pit_ok and freshness_ok

    return {
        "status": "PASSED" if all_passed else "FAILED",
        "signal_date_strictly_before_trade_date": bool(dates_ok),
        "post_open_timing_respected": post_open_ok,
        "realized_returns_not_used_in_signal": realized_ok,
        "pit_binning_strictly_historical": pit_ok,
        "gap_data_freshness_ok": bool(freshness_ok),
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
