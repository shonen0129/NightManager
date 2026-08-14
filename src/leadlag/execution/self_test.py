"""Self-test diagnostics for the production CLI.

These checks were previously embedded in the legacy daily production script.
They are now shared by ``python3 -m leadlag.cli self-test`` and the test
suite.  Each check returns a non-zero exit code on failure instead of using
``assert``.
"""

from __future__ import annotations

import logging

import numpy as np

from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.core.portfolio import get_rolling_pit_bin, solve_baseline_style

logger = logging.getLogger(__name__)


def run_self_tests() -> int:
    """Run production self-tests.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    failed = False

    # T1: baseline_style sizing
    scores = np.array([1.5, 0.5, 0.2, -0.1, -0.5, -1.0, 0.0, 0.1, -0.3, 0.8])
    longs = np.array([0, 1, 2, 7, 9])
    shorts = np.array([3, 4, 5, 6, 8])
    w = solve_baseline_style(scores, longs, shorts, baseline_gross=2.0)

    if abs(np.sum(w)) >= 1e-12:
        logger.error("T1a: net must be 0; got %s", np.sum(w))
        failed = True
    if abs(np.sum(np.abs(w)) - 2.0) >= 1e-10:
        logger.error("T1b: gross must be 2.0; got %s", np.sum(np.abs(w)))
        failed = True
    if not (w[longs] >= 0.0).all():
        logger.error("T1c: longs must be non-negative")
        failed = True
    if not (w[shorts] <= 0.0).all():
        logger.error("T1d: shorts must be non-positive")
        failed = True
    if not failed:
        logger.info("[PASS] T1: baseline_style sizing")

    # T2: PIT binning
    hist = np.linspace(0.0, 3.0, 500)

    b, lo, hi, m = get_rolling_pit_bin(hist, 0.5, rolling_window=252)
    if b != "Low" or abs(m - 0.75) >= 1e-9:
        logger.error("T2a: expected Low/0.75 got %s/%s", b, m)
        failed = True
    else:
        logger.info("[PASS] T2a: PIT Low bin")

    b, _, _, m = get_rolling_pit_bin(hist, 2.2, rolling_window=252)
    if b != "Medium" or abs(m - 1.0) >= 1e-9:
        logger.error("T2b: expected Medium/1.0 got %s/%s", b, m)
        failed = True
    else:
        logger.info("[PASS] T2b: PIT Medium bin")

    b, _, _, m = get_rolling_pit_bin(hist, 3.5, rolling_window=252)
    if b != "High" or abs(m - 1.0) >= 1e-9:
        logger.error("T2c: expected High/1.0 got %s/%s", b, m)
        failed = True
    else:
        logger.info("[PASS] T2c: PIT High bin")

    # T3: insufficient history -> Medium/1.0 with NaN thresholds
    b, lo, _, m = get_rolling_pit_bin(hist, 2.0, rolling_window=600)
    if b != "Medium" or not np.isnan(lo) or abs(m - 1.0) >= 1e-9:
        logger.error("T3: expected Medium/NaN/1.0 got %s/%s/%s", b, lo, m)
        failed = True
    else:
        logger.info("[PASS] T3: insufficient history fallback")

    # T4: leakage audit
    res = run_leakage_audit("2026-06-15", "2026-06-16")
    if res["status"] != "PASSED":
        logger.error("T4a: valid dates should PASS; got %s", res)
        failed = True
    res_bad = run_leakage_audit("2026-06-16", "2026-06-16")
    if res_bad["status"] != "FAILED":
        logger.error("T4b: same date must FAIL; got %s", res_bad)
        failed = True
    if res["status"] == "PASSED" and res_bad["status"] == "FAILED":
        logger.info("[PASS] T4: leakage audit")

    # T5: numerical audit with valid inputs
    w_ok = np.array([0.2] * 5 + [-0.2] * 5)
    scores_ok = np.ones(10)
    Omega_ok = np.eye(10) * 0.01
    audit = run_numerical_audit(w_ok, scores_ok, Omega_ok)
    if audit["status"] != "PASSED":
        logger.error("T5a: valid weights should PASS; got %s", audit)
        failed = True
    else:
        logger.info("[PASS] T5: numerical audit")

    # T6: cost formula consistency
    gross_ex = float(np.sum(np.abs(w_ok)))
    cost_bps_per_gross = 10.0
    cost_bps = gross_ex * cost_bps_per_gross
    if abs(cost_bps - 20.0) >= 1e-9:
        logger.error("T6: expected 20 bps got %s", cost_bps)
        failed = True
    else:
        logger.info("[PASS] T6: cost formula")

    if failed:
        logger.error("=== Self-Tests FAILED ===")
        return 1
    logger.info("=== All Self-Tests PASSED ===")
    return 0
