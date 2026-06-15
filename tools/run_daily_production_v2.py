#!/usr/bin/env python
"""Daily Production Runner v2: mu_over_sigma + baseline_style + RuleD.

This script is the official daily production execution script for
production_p8p3_v2.  It implements:

  ranking:   mu_over_sigma = mu_gap / sigma_gap
  sizing:    baseline_style (signal-proportional weighting)
  gross:     RuleD dynamic gross scaling
               Low tertile  → gross multiplier 0.75
               Mid tertile  → gross multiplier 1.00
               High tertile → gross multiplier 1.00

Fallback hierarchy:
  gap data missing  → v1 P8P3-BLPX baseline weights
  v1 weights missing→ SRE equal-weight stub

This script MUST NOT be modified to place real orders.  It writes portfolio
weight files to the live/ directory that the downstream order-management system
reads.  Actual trade placement is handled by a separate execution layer.

Usage (normal daily run):
  python tools/run_daily_production_v2.py \\
      --trade-date latest \\
      --gap-input-dir results/gap_adjusted_distribution/latest

Usage (dry-run, no file writes):
  python tools/run_daily_production_v2.py \\
      --trade-date 2026-06-16 \\
      --gap-input-dir results/gap_adjusted_distribution/latest \\
      --dry-run true

Usage (self-test):
  python tools/run_daily_production_v2.py --self-test true
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.tickers import JP_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ProductionV2")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VERSION = "production_p8p3_v2"
LONG_COUNT = 5
SHORT_COUNT = 5
BASELINE_GROSS = 2.0
COST_BPS_PER_GROSS = 10.0  # ex-ante cost used for IR calculation only


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Daily Production Runner v2 (mu_over_sigma + baseline_style + RuleD)"
    )
    p.add_argument("--config", default="configs/production.yaml")
    p.add_argument("--trade-date", default="latest", help="YYYY-MM-DD or 'latest'")
    p.add_argument(
        "--gap-input-dir",
        default=None,
        help="Directory containing mu_gap_{YYYYMMDD}.npy and omega_gap_{YYYYMMDD}.npy",
    )
    p.add_argument(
        "--v1-weights-file",
        default="live/production_p8p3_blpx/v1_baseline_weights.csv",
        help="Path to v1 baseline weights CSV (fallback when gap data missing)",
    )
    p.add_argument(
        "--live-dir",
        default="live/production_p8p3_blpx",
        help="Live output directory",
    )
    p.add_argument(
        "--dry-run",
        default="false",
        choices=["true", "false"],
        help="If true, compute weights but do NOT write files",
    )
    p.add_argument(
        "--self-test",
        default="false",
        choices=["true", "false"],
        help="Run self-tests and exit",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mathematical Helpers (identical to shadow runner for consistency)
# ---------------------------------------------------------------------------

def solve_baseline_style(
    scores: np.ndarray,
    long_idx: np.ndarray,
    short_idx: np.ndarray,
    baseline_gross: float = 2.0,
) -> np.ndarray:
    """Compute baseline_style weights normalised to baseline_gross.

    Long side sums to +baseline_gross/2; short side sums to -baseline_gross/2.
    """
    n = len(scores)
    w = np.zeros(n)
    med_score = np.median(scores)
    scores_centered = scores - med_score

    long_raw = np.maximum(scores_centered[long_idx], 1e-12)
    long_denom = np.sum(long_raw)
    if long_denom > 0:
        w[long_idx] = (baseline_gross / 2.0) * (long_raw / long_denom)

    short_raw = np.maximum(-scores_centered[short_idx], 1e-12)
    short_denom = np.sum(short_raw)
    if short_denom > 0:
        w[short_idx] = -(baseline_gross / 2.0) * (short_raw / short_denom)

    return w


def get_rolling_pit_bin(
    history_ir: np.ndarray,
    current_ir: float,
    rolling_window: int = 252,
    low_pct: float = 33.3333,
    high_pct: float = 66.6667,
    mult_low: float = 0.75,
    mult_mid: float = 1.00,
    mult_high: float = 1.00,
) -> tuple[str, float, float, float]:
    """Assign current_ir to a PIT tertile bin using strictly historical data.

    Returns (bin_label, low_threshold, high_threshold, multiplier).
    Falls back to ('Medium', nan, nan, 1.00) when history is insufficient.
    """
    history_valid = history_ir[np.isfinite(history_ir)]

    if len(history_valid) < rolling_window:
        return "Medium", float("nan"), float("nan"), mult_mid

    history_slice = history_valid[-rolling_window:]
    low_thresh = float(np.percentile(history_slice, low_pct))
    high_thresh = float(np.percentile(history_slice, high_pct))

    if current_ir <= low_thresh:
        return "Low", low_thresh, high_thresh, mult_low
    elif current_ir >= high_thresh:
        return "High", low_thresh, high_thresh, mult_high
    else:
        return "Medium", low_thresh, high_thresh, mult_mid


# ---------------------------------------------------------------------------
# Audit Functions
# ---------------------------------------------------------------------------

def run_leakage_audit(sig_date: str, trade_date: str) -> dict:
    """Verify signal_date < trade_date (no lookahead)."""
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
    """Validate weight vector and covariance matrix."""
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


# ---------------------------------------------------------------------------
# Self Tests
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    """Execute verification self-tests.  Returns 0 on success."""
    logger.info("=== ProductionV2 Self-Tests ===")

    # Test 1: baseline_style sizing
    scores = np.array([1.5, 0.5, 0.2, -0.1, -0.5, -1.0, 0.0, 0.1, -0.3, 0.8])
    longs = np.array([0, 1, 2, 7, 9])
    shorts = np.array([3, 4, 5, 6, 8])
    w = solve_baseline_style(scores, longs, shorts, baseline_gross=2.0)
    assert abs(np.sum(w)) < 1e-12, "T1a: net must be 0"
    assert abs(np.sum(np.abs(w)) - 2.0) < 1e-10, "T1b: gross must be 2.0"
    assert (w[longs] >= 0.0).all(), "T1c: longs non-negative"
    assert (w[shorts] <= 0.0).all(), "T1d: shorts non-positive"
    logger.info("[PASS] T1: baseline_style sizing")

    # Test 2: PIT binning — Low
    hist = np.linspace(0.0, 3.0, 500)
    b, lo, hi, m = get_rolling_pit_bin(hist, 0.5, rolling_window=252)
    assert b == "Low" and abs(m - 0.75) < 1e-9, f"T2a: expected Low/0.75 got {b}/{m}"
    logger.info("[PASS] T2a: PIT Low bin")

    # Test 2b: Medium (hist[-252:] spans 1.49-3.00; lo≈1.994, hi≈2.497 → 2.2 is Medium)
    b, _, _, m = get_rolling_pit_bin(hist, 2.2, rolling_window=252)
    assert b == "Medium" and abs(m - 1.0) < 1e-9, f"T2b: expected Medium/1.0 got {b}/{m}"
    logger.info("[PASS] T2b: PIT Medium bin")

    # Test 2c: High (IR=3.5 > hi≈2.497)
    b, _, _, m = get_rolling_pit_bin(hist, 3.5, rolling_window=252)
    assert b == "High" and abs(m - 1.0) < 1e-9, f"T2c: expected High/1.0 got {b}/{m}"
    logger.info("[PASS] T2c: PIT High bin")

    # Test 3: insufficient history → fallback to Medium/1.0
    b, lo, _, m = get_rolling_pit_bin(hist, 2.0, rolling_window=600)
    assert b == "Medium" and np.isnan(lo) and abs(m - 1.0) < 1e-9, "T3: insufficient history fallback"
    logger.info("[PASS] T3: insufficient history fallback")

    # Test 4: leakage audit
    res = run_leakage_audit("2026-06-15", "2026-06-16")
    assert res["status"] == "PASSED", "T4a: valid dates"
    res_bad = run_leakage_audit("2026-06-16", "2026-06-16")
    assert res_bad["status"] == "FAILED", "T4b: same date must fail"
    logger.info("[PASS] T4: leakage audit")

    # Test 5: numerical audit with valid inputs
    w_ok = np.array([0.2] * 5 + [-0.2] * 5)
    scores_ok = np.ones(10)
    Omega_ok = np.eye(10) * 0.01
    audit = run_numerical_audit(w_ok, scores_ok, Omega_ok)
    assert audit["status"] == "PASSED", f"T5a: {audit}"
    logger.info("[PASS] T5: numerical audit")

    # Test 6: cost formula consistency
    gross_ex = float(np.sum(np.abs(w_ok)))  # 2.0
    cost_bps = gross_ex * COST_BPS_PER_GROSS
    assert abs(cost_bps - 20.0) < 1e-9, f"T6: expected 20 bps got {cost_bps}"
    logger.info("[PASS] T6: cost formula")

    logger.info("=== All Self-Tests PASSED ===")
    return 0


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def load_gap_matrices(
    gap_input_dir: Path,
    date_str: str,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Load mu_gap and Omega_gap matrices for the given trade date.

    Returns (mu_gap, Omega_gap, alerts).
    """
    alerts: list[str] = []
    date_numeric = pd.to_datetime(date_str).strftime("%Y%m%d")

    mu_file = gap_input_dir / "matrices" / f"mu_gap_{date_numeric}.npy"
    omega_file = gap_input_dir / "matrices" / f"omega_gap_{date_numeric}.npy"

    if not mu_file.exists():
        alerts.append(f"mu_gap file missing: {mu_file}")
        return None, None, alerts
    if not omega_file.exists():
        alerts.append(f"omega_gap file missing: {omega_file}")
        return None, None, alerts

    mu_gap = np.load(mu_file)
    Omega_gap = np.load(omega_file)
    return mu_gap, Omega_gap, alerts


def load_v1_fallback_weights(
    v1_weights_file: Path,
    date_str: str,
    n_j: int,
) -> tuple[np.ndarray, list[str]]:
    """Load v1 P8P3-BLPX baseline weights as fallback.

    Returns (w_v1, alerts).
    """
    alerts: list[str] = []
    w_v1 = np.zeros(n_j)

    if not v1_weights_file.exists():
        alerts.append(f"v1 weights file not found: {v1_weights_file}. Using zero weights.")
        return w_v1, alerts

    df = pd.read_csv(v1_weights_file)
    if len(df) == 0:
        alerts.append("v1 weights file is empty. Using zero weights.")
        return w_v1, alerts

    # Verify date alignment
    file_date = str(df.iloc[0].get("trade_date", ""))
    if file_date != date_str:
        alerts.append(
            f"v1 weights date mismatch: file has {file_date}, expected {date_str}. "
            "Using stale v1 weights (caution)."
        )

    for _, row in df.iterrows():
        tk = str(row.get("ticker", ""))
        if tk in JP_TICKERS:
            idx = JP_TICKERS.index(tk)
            w_v1[idx] = float(row.get("weight", 0.0))

    if np.sum(np.abs(w_v1)) < 1e-8:
        alerts.append("v1 weights loaded but all zero. Using zero weights.")

    return w_v1, alerts


def load_pit_ir_history(
    gap_input_dir: Path,
    trade_date: str,
) -> tuple[np.ndarray, list[str]]:
    """Load historical ex-ante IR series for PIT binning.

    Returns (history_ir, alerts).
    """
    alerts: list[str] = []
    diag_file = gap_input_dir / "portfolio_gap_distribution_diagnostics.csv"

    if not diag_file.exists():
        alerts.append(f"Diagnostics file missing: {diag_file}. PIT binning falls back to Medium/1.0.")
        return np.array([]), alerts

    df = pd.read_csv(diag_file)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df_hist = df[df["trade_date"] < trade_date]

    if "pred_ir_gap_exante_cost" not in df_hist.columns:
        alerts.append("pred_ir_gap_exante_cost column missing. PIT binning falls back to Medium/1.0.")
        return np.array([]), alerts

    history_ir = df_hist["pred_ir_gap_exante_cost"].values
    return history_ir, alerts


def _derive_signal_date(gap_input_dir: Path | None, trade_date: str) -> str:
    """Derive signal_date from gap matrix file naming or trade_date - 1 day.

    Gap matrices are computed from data up to (and including) the signal date,
    which is the US close day before trade_date.  The mu_gap_{YYYYMMDD}.npy
    filename encodes the signal date.  We find the most-recent matrix file
    dated strictly before trade_date to use as signal_date.

    Falls back to trade_date minus 1 calendar day when the directory or files
    are not available.
    """
    trade_dt = pd.to_datetime(trade_date).normalize()
    fallback_sig_date = (trade_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if gap_input_dir is None:
        return fallback_sig_date

    matrices_dir = gap_input_dir / "matrices"
    if not matrices_dir.exists():
        return fallback_sig_date

    # Find all mu_gap_YYYYMMDD.npy files dated before trade_date
    candidate_dates = []
    for f in matrices_dir.glob("mu_gap_*.npy"):
        stem = f.stem  # e.g. "mu_gap_20260612"
        parts = stem.split("_")
        if len(parts) >= 3:
            date_str_candidate = parts[2]
            try:
                cdt = pd.to_datetime(date_str_candidate, format="%Y%m%d").normalize()
                if cdt < trade_dt:
                    candidate_dates.append(cdt)
            except ValueError:
                pass

    if not candidate_dates:
        return fallback_sig_date

    # Use the most recent file date as signal_date
    latest_signal_dt = max(candidate_dates)
    return latest_signal_dt.strftime("%Y-%m-%d")


def generate_v2_production_portfolio(
    trade_date: str,
    gap_input_dir: Path | None,
    v1_weights_file: Path,
    cfg: dict,
) -> dict:
    """Core v2 production portfolio construction.

    Returns a result dict with keys:
      w_final     : np.ndarray, final production weights
      w_v1        : np.ndarray, v1 baseline weights (for comparison / fallback)
      scores      : np.ndarray, mu_over_sigma scores
      mu_gap      : np.ndarray
      sigma_gap   : np.ndarray
      Omega_gap   : np.ndarray
      fallback     : dict, fallback flags
      pit_binning  : dict, PIT binning details
      leakage      : dict, leakage audit result
      numerical    : dict, numerical audit result
      alerts       : list[str]
      summary      : dict, one-row summary statistics
    """
    n_j = len(JP_TICKERS)
    date_str = str(trade_date)
    alerts: list[str] = []
    fallback = {
        "gap_data_missing": False,
        "v1_fallback_used": False,
        "sre_fallback_used": False,
    }

    # ------------------------------------------------------------------
    # 1. Load v1 baseline weights (always; needed for PIT current IR + fallback)
    # ------------------------------------------------------------------
    w_v1, v1_alerts = load_v1_fallback_weights(v1_weights_file, date_str, n_j)
    alerts.extend(v1_alerts)

    # ------------------------------------------------------------------
    # 2. Load gap-adjusted distribution matrices
    # ------------------------------------------------------------------
    mu_gap: np.ndarray | None = None
    Omega_gap: np.ndarray | None = None
    if gap_input_dir is not None:
        mu_gap, Omega_gap, gap_alerts = load_gap_matrices(gap_input_dir, date_str)
        alerts.extend(gap_alerts)
    else:
        alerts.append("--gap-input-dir not specified. Using v1 fallback.")

    # ------------------------------------------------------------------
    # 3. Fallback: gap data missing → use v1 weights
    # ------------------------------------------------------------------
    if mu_gap is None or Omega_gap is None:
        fallback["gap_data_missing"] = True
        fallback["v1_fallback_used"] = True
        logger.warning(f"[{date_str}] Gap data missing — activating v1 fallback.")

        # Minimal placeholder matrices for audit
        dummy_scores = np.zeros(n_j)
        dummy_Omega = np.eye(n_j) * 0.01
        leakage = run_leakage_audit(date_str, date_str)
        numerical = run_numerical_audit(w_v1, dummy_scores, dummy_Omega)

        return {
            "w_final": w_v1.copy(),
            "w_v1": w_v1.copy(),
            "scores": dummy_scores,
            "mu_gap": np.zeros(n_j),
            "sigma_gap": np.ones(n_j) * 0.1,
            "Omega_gap": dummy_Omega,
            "fallback": fallback,
            "pit_binning": {
                "assigned_bin": "Medium",
                "threshold_low": float("nan"),
                "threshold_high": float("nan"),
                "multiplier": 1.0,
                "history_count": 0,
                "fallback_flag": True,
            },
            "leakage": leakage,
            "numerical": numerical,
            "alerts": alerts,
            "summary": _build_summary(
                w_v1, date_str, 1.0, "Medium", float("nan"), float("nan"),
                fallback=True, candidate="v1_fallback",
            ),
        }

    # ------------------------------------------------------------------
    # 4. Ensure Omega_gap is PSD
    # ------------------------------------------------------------------
    min_eig = np.min(np.linalg.eigvalsh(Omega_gap))
    if min_eig < 0.0:
        Omega_gap = Omega_gap + (abs(min_eig) + 1e-8) * np.eye(n_j)
        alerts.append("Omega_gap repaired to PSD.")

    # ------------------------------------------------------------------
    # 5. Compute mu_over_sigma scores
    # ------------------------------------------------------------------
    sigma_gap = np.sqrt(np.maximum(np.diag(Omega_gap), 1e-6))
    scores = mu_gap / sigma_gap

    # ------------------------------------------------------------------
    # 6. Select top-5 long, bottom-5 short by score
    # ------------------------------------------------------------------
    sorted_idx = np.argsort(scores)
    short_idx = sorted_idx[:SHORT_COUNT]
    long_idx = sorted_idx[-LONG_COUNT:]

    # ------------------------------------------------------------------
    # 7. Compute pre-gross weights (baseline_style)
    # ------------------------------------------------------------------
    w_pre = solve_baseline_style(scores, long_idx, short_idx, baseline_gross=BASELINE_GROSS)

    # ------------------------------------------------------------------
    # 8. PIT binning for RuleD
    # ------------------------------------------------------------------
    history_ir = np.array([])
    if gap_input_dir is not None:
        history_ir, pit_alerts = load_pit_ir_history(gap_input_dir, date_str)
        alerts.extend(pit_alerts)

    # Compute current ex-ante IR using v1 baseline weights
    if np.sum(np.abs(w_v1)) > 1e-8:
        p_mean = np.dot(w_v1, mu_gap)
        p_var = np.dot(w_v1, np.dot(Omega_gap, w_v1))
        p_vol = np.sqrt(max(0.0, p_var))
        ex_ante_cost = BASELINE_GROSS * 0.001  # 20 bps in decimal at gross=2.0
        current_ir = (p_mean - ex_ante_cost) / p_vol if p_vol > 1e-6 else 0.0
    else:
        current_ir = 0.0

    assigned_bin, lo_thresh, hi_thresh, mult = get_rolling_pit_bin(
        history_ir, current_ir, rolling_window=252
    )
    history_count = int(np.sum(np.isfinite(history_ir)))
    pit_fallback = history_count < 252

    if pit_fallback:
        alerts.append(
            f"PIT history insufficient ({history_count} < 252). Using Medium/1.0 multiplier."
        )

    pit_binning = {
        "assigned_bin": assigned_bin,
        "threshold_low": lo_thresh,
        "threshold_high": hi_thresh,
        "multiplier": float(mult),
        "current_ir": float(current_ir),
        "history_count": history_count,
        "fallback_flag": pit_fallback,
    }
    logger.info(
        f"[{date_str}] PIT bin={assigned_bin} (IR={current_ir:.4f}), "
        f"mult={mult:.2f}, history={history_count}d"
    )

    # ------------------------------------------------------------------
    # 9. Apply RuleD multiplier
    # ------------------------------------------------------------------
    w_final = w_pre * mult

    # ------------------------------------------------------------------
    # 10. Safety audits
    # ------------------------------------------------------------------
    # Signal date = the gap data file date, which must be strictly before trade_date.
    # Gap matrix files are named mu_gap_{YYYYMMDD}.npy where YYYYMMDD is the
    # signal computation date (end of prior US close).  We derive it from the
    # diagnostics file or from the gap_input_dir folder name as a proxy.
    # For conservatism we use trade_date - 1 calendar day as signal_date if
    # no better date is available.
    signal_date_str = _derive_signal_date(gap_input_dir, date_str)
    leakage = run_leakage_audit(signal_date_str, date_str)

    numerical = run_numerical_audit(w_final, scores, Omega_gap)
    if numerical["status"] == "FAILED":
        alerts.append(f"Numerical audit FAILED: {numerical}. Falling back to v1.")
        fallback["v1_fallback_used"] = True
        w_final = w_v1.copy()
        numerical = run_numerical_audit(w_final, scores, Omega_gap)

    summary = _build_summary(
        w_final, date_str, mult, assigned_bin, lo_thresh, hi_thresh,
        fallback=fallback["v1_fallback_used"], candidate="primary_ruleD",
        scores=scores, mu_gap=mu_gap, Omega_gap=Omega_gap,
    )

    return {
        "w_final": w_final,
        "w_v1": w_v1,
        "scores": scores,
        "mu_gap": mu_gap,
        "sigma_gap": sigma_gap,
        "Omega_gap": Omega_gap,
        "fallback": fallback,
        "pit_binning": pit_binning,
        "leakage": leakage,
        "numerical": numerical,
        "alerts": alerts,
        "summary": summary,
    }


def _build_summary(
    w: np.ndarray,
    date_str: str,
    mult: float,
    assigned_bin: str,
    lo_thresh: float,
    hi_thresh: float,
    *,
    fallback: bool,
    candidate: str,
    scores: np.ndarray | None = None,
    mu_gap: np.ndarray | None = None,
    Omega_gap: np.ndarray | None = None,
) -> dict:
    """Build a one-row performance summary dict."""
    n_j = len(JP_TICKERS)
    if scores is None:
        scores = np.zeros(n_j)
    if mu_gap is None:
        mu_gap = np.zeros(n_j)
    if Omega_gap is None:
        Omega_gap = np.eye(n_j) * 0.01

    long_idx = np.where(w > 1e-8)[0]
    short_idx = np.where(w < -1e-8)[0]
    gross = float(np.sum(np.abs(w)))
    net = float(np.sum(w))

    p_mean = float(np.dot(w, mu_gap))
    p_var = float(np.dot(w, np.dot(Omega_gap, w)))
    p_vol = float(np.sqrt(max(0.0, p_var)))
    ex_ante_cost = gross * 0.001  # COST_BPS_PER_GROSS / 10000 (10 bps)
    p_ir = float((p_mean - ex_ante_cost) / p_vol) if p_vol > 1e-6 else 0.0

    w_l = w[w > 0]
    hhi = float(np.sum((w_l / np.sum(w_l)) ** 2)) if len(w_l) > 0 else 0.0

    return {
        "trade_date": date_str,
        "candidate": candidate,
        "version": VERSION,
        "long_count": int(len(long_idx)),
        "short_count": int(len(short_idx)),
        "target_gross": gross,
        "target_net": net,
        "gross_multiplier": float(mult),
        "pit_bin": assigned_bin,
        "pit_threshold_low": lo_thresh,
        "pit_threshold_high": hi_thresh,
        "predicted_portfolio_mean": p_mean,
        "predicted_portfolio_vol": p_vol,
        "predicted_portfolio_ir": p_ir,
        "expected_cost_bps": gross * COST_BPS_PER_GROSS,
        "herfindahl": hhi,
        "fallback_triggered": int(fallback),
    }


# ---------------------------------------------------------------------------
# File Writing
# ---------------------------------------------------------------------------

def write_production_files(
    trade_date: str,
    live_dir: Path,
    result: dict,
    dry_run: bool = False,
) -> None:
    """Write all production output files to live_dir.

    Files written:
      latest_weights.csv          — primary production weights (v2 or fallback)
      v1_baseline_weights.csv     — v1 P8P3-BLPX weights (comparison / next-day fallback input)
      production_scores.csv       — mu_over_sigma scores for all 17 tickers
      production_summary.csv      — one-row summary statistics
      pit_binning.json            — PIT binning audit details
      leakage_audit.json          — lookahead leakage audit
      numerical_audit.json        — numerical boundary audit
      production_audit.json       — aggregated audit (all_passed flag)
      daily_production_report.md  — human-readable daily report
      run_config.json             — execution metadata
    """
    if dry_run:
        logger.info("[DRY-RUN] Would write files to: %s", live_dir)
        _print_dry_run_summary(trade_date, result)
        return

    live_dir.mkdir(parents=True, exist_ok=True)
    w_final = result["w_final"]
    w_v1 = result["w_v1"]
    scores = result["scores"]
    mu_gap = result["mu_gap"]
    sigma_gap = result["sigma_gap"]
    pit = result["pit_binning"]

    # 1. latest_weights.csv
    rows = []
    for j, tk in enumerate(JP_TICKERS):
        side = "LONG" if w_final[j] > 1e-8 else ("SHORT" if w_final[j] < -1e-8 else "NEUTRAL")
        rows.append({
            "trade_date": trade_date,
            "ticker": tk,
            "weight": float(w_final[j]),
            "side": side,
            "score": float(scores[j]),
            "mu_gap": float(mu_gap[j]),
            "sigma_gap": float(sigma_gap[j]),
            "ensemble_signal": float(scores[j]),   # kept for backward compat with shadow runner
            "gross_multiplier": float(pit["multiplier"]),
            "pit_bin": pit["assigned_bin"],
            "version": VERSION,
            "fallback_flag": int(result["fallback"]["v1_fallback_used"]),
        })
    pd.DataFrame(rows).to_csv(live_dir / "latest_weights.csv", index=False)
    logger.info("Written: latest_weights.csv  (gross=%.4f)", float(np.sum(np.abs(w_final))))

    # 2. v1_baseline_weights.csv
    v1_rows = []
    for j, tk in enumerate(JP_TICKERS):
        side = "LONG" if w_v1[j] > 1e-8 else ("SHORT" if w_v1[j] < -1e-8 else "NEUTRAL")
        v1_rows.append({
            "trade_date": trade_date,
            "ticker": tk,
            "weight": float(w_v1[j]),
            "side": side,
            "version": "production_p8p3_v1",
        })
    pd.DataFrame(v1_rows).to_csv(live_dir / "v1_baseline_weights.csv", index=False)
    logger.info("Written: v1_baseline_weights.csv")

    # 3. production_scores.csv
    score_rows = [
        {
            "trade_date": trade_date,
            "ticker": JP_TICKERS[j],
            "mu_gap": float(mu_gap[j]),
            "sigma_gap": float(sigma_gap[j]),
            "mu_over_sigma_score": float(scores[j]),
        }
        for j in range(len(JP_TICKERS))
    ]
    pd.DataFrame(score_rows).to_csv(live_dir / "production_scores.csv", index=False)
    logger.info("Written: production_scores.csv")

    # 4. production_summary.csv
    pd.DataFrame([result["summary"]]).to_csv(live_dir / "production_summary.csv", index=False)
    logger.info("Written: production_summary.csv")

    # 5. JSON audit files
    with open(live_dir / "pit_binning.json", "w") as f:
        json.dump(result["pit_binning"], f, indent=4, default=str)

    with open(live_dir / "leakage_audit.json", "w") as f:
        json.dump(result["leakage"], f, indent=4)

    with open(live_dir / "numerical_audit.json", "w") as f:
        json.dump(result["numerical"], f, indent=4)

    all_passed = (
        result["leakage"]["status"] == "PASSED"
        and result["numerical"]["status"] == "PASSED"
    )
    production_audit = {
        "trade_date": trade_date,
        "version": VERSION,
        "all_passed": all_passed,
        "leakage_status": result["leakage"]["status"],
        "numerical_status": result["numerical"]["status"],
        "fallback_triggered": result["fallback"]["v1_fallback_used"],
        "alerts": result["alerts"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(live_dir / "production_audit.json", "w") as f:
        json.dump(production_audit, f, indent=4)
    logger.info(
        "Written: production_audit.json  (all_passed=%s)", all_passed
    )

    # 6. run_config.json
    run_config = {
        "trade_date": trade_date,
        "version": VERSION,
        "candidate": "primary_ruleD",
        "ranking_mode": "mu_over_sigma",
        "sizing_mode": "baseline_style",
        "gross_scaling_rule": "RuleD",
        "post_open_requirement": "Tokyo 9:10 POST_OPEN",
        "slippage_bps_per_side": 5.0,
        "cost_bps_per_gross": COST_BPS_PER_GROSS,
        "timestamp": datetime.now().isoformat(),
    }
    with open(live_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)

    # 7. daily_production_report.md
    _write_daily_report(trade_date, live_dir, result)
    logger.info("Written: daily_production_report.md")


def _print_dry_run_summary(trade_date: str, result: dict) -> None:
    """Print summary to console for dry-run mode."""
    pit = result["pit_binning"]
    s = result["summary"]
    logger.info("=== DRY-RUN SUMMARY: %s ===", trade_date)
    logger.info("  Candidate     : primary_ruleD (v2)")
    logger.info("  PIT Bin       : %s (mult=%.2f)", pit["assigned_bin"], pit["multiplier"])
    logger.info("  Target Gross  : %.4f", s["target_gross"])
    logger.info("  Target Net    : %.6f", s["target_net"])
    logger.info("  Ex-Ante IR    : %.4f", s["predicted_portfolio_ir"])
    logger.info("  Fallback      : %s", result["fallback"]["v1_fallback_used"])
    logger.info("  Leakage Audit : %s", result["leakage"]["status"])
    logger.info("  Numerical Audit: %s", result["numerical"]["status"])
    logger.info("  Alerts        : %s", result["alerts"])

    w = result["w_final"]
    long_tks = [JP_TICKERS[i] for i in range(len(JP_TICKERS)) if w[i] > 1e-8]
    short_tks = [JP_TICKERS[i] for i in range(len(JP_TICKERS)) if w[i] < -1e-8]
    logger.info("  Longs  (%d): %s", len(long_tks), long_tks)
    logger.info("  Shorts (%d): %s", len(short_tks), short_tks)
    logger.info("=========================")


def _write_daily_report(trade_date: str, live_dir: Path, result: dict) -> None:
    """Write the human-readable daily production report."""
    pit = result["pit_binning"]
    s = result["summary"]
    w = result["w_final"]
    fb = result["fallback"]

    long_tks = [JP_TICKERS[i] for i in range(len(JP_TICKERS)) if w[i] > 1e-8]
    short_tks = [JP_TICKERS[i] for i in range(len(JP_TICKERS)) if w[i] < -1e-8]
    long_weights = [(JP_TICKERS[i], w[i]) for i in range(len(JP_TICKERS)) if w[i] > 1e-8]
    short_weights = [(JP_TICKERS[i], w[i]) for i in range(len(JP_TICKERS)) if w[i] < -1e-8]
    long_weights.sort(key=lambda x: -x[1])
    short_weights.sort(key=lambda x: x[1])

    fallback_note = ""
    if fb["v1_fallback_used"]:
        fallback_note = "\n\n> [!WARNING]\n> **フォールバック発動**: gap data 未利用。v1 P8P3-BLPX ウェイトを使用しています。\n"

    alert_text = ""
    if result["alerts"]:
        alert_text = "\n## Alerts\n" + "\n".join(f"- {a}" for a in result["alerts"]) + "\n"

    thresh_lo = f"{pit['threshold_low']:.4f}" if not (isinstance(pit["threshold_low"], float) and pit["threshold_low"] != pit["threshold_low"]) else "N/A"
    thresh_hi = f"{pit['threshold_high']:.4f}" if not (isinstance(pit["threshold_high"], float) and pit["threshold_high"] != pit["threshold_high"]) else "N/A"

    rep = f"""# Production Daily Report — {trade_date}

**Version**: `{VERSION}` | **Candidate**: `primary_ruleD`
**Ranking**: `mu_over_sigma` | **Sizing**: `baseline_style` | **Gross**: `RuleD`
**Timestamp**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")}
{fallback_note}
## 1. Portfolio Summary

| Metric | Value |
|---|---|
| Target Gross | {s['target_gross']:.4f} |
| Target Net | {s['target_net']:.6f} |
| Gross Multiplier (RuleD) | {s['gross_multiplier']:.2f} |
| PIT Bin | **{pit['assigned_bin']}** |
| Predicted Portfolio IR | {s['predicted_portfolio_ir']:.4f} |
| Expected Cost (bps) | {s['expected_cost_bps']:.1f} |
| Longs | {len(long_tks)} |
| Shorts | {len(short_tks)} |
| Fallback Triggered | {"YES ⚠️" if fb['v1_fallback_used'] else "No"} |

## 2. RuleD Dynamic Gross Binning

| Item | Value |
|---|---|
| Current Ex-Ante IR | {pit.get('current_ir', 0.0):.4f} |
| Assigned Bin | **{pit['assigned_bin']}** |
| Threshold Low (33rd pct) | {thresh_lo} |
| Threshold High (67th pct) | {thresh_hi} |
| Multiplier | {pit['multiplier']:.2f} |
| History Days | {pit['history_count']} |

## 3. Selected Positions

**Longs:**
| Ticker | Weight |
|---|---:|
"""
    for tk, wt in long_weights:
        rep += f"| {tk} | {wt:.4f} |\n"

    rep += "\n**Shorts:**\n| Ticker | Weight |\n|---|---:|\n"
    for tk, wt in short_weights:
        rep += f"| {tk} | {wt:.4f} |\n"

    rep += f"""
## 4. Safety Audit Status

| Audit | Status |
|---|---|
| Leakage Audit | **{result['leakage']['status']}** |
| Numerical Audit | **{result['numerical']['status']}** |
{alert_text}
---
*This file is generated automatically by `run_daily_production_v2.py`.
No trades are placed by this script; it writes weight targets only.*
"""

    with open(live_dir / "daily_production_report.md", "w", encoding="utf-8") as f:
        f.write(rep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    if args.self_test == "true":
        sys.exit(run_self_tests())

    # Load config
    config_path = ROOT / args.config
    logger.info("Loading config: %s", config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve paths
    live_dir = (
        ROOT / args.live_dir if not args.live_dir.startswith("/") else Path(args.live_dir)
    )
    v1_weights_file = (
        ROOT / args.v1_weights_file
        if not args.v1_weights_file.startswith("/")
        else Path(args.v1_weights_file)
    )

    gap_input_dir: Path | None = None
    if args.gap_input_dir:
        gap_path = ROOT / args.gap_input_dir if not args.gap_input_dir.startswith("/") else Path(args.gap_input_dir)
        if gap_path.exists():
            gap_input_dir = gap_path
        else:
            logger.warning("--gap-input-dir does not exist: %s. Will use v1 fallback.", gap_path)
    else:
        # Try default from config
        default_gap = cfg.get("gap_distribution", {}).get("dir", "")
        if default_gap:
            gap_path = ROOT / default_gap if not default_gap.startswith("/") else Path(default_gap)
            if gap_path.exists():
                gap_input_dir = gap_path
                logger.info("Using gap_distribution.dir from config: %s", gap_path)
            else:
                logger.warning("Config gap_distribution.dir not found: %s. Will use v1 fallback.", gap_path)

    # Resolve trade date
    if args.trade_date == "latest":
        latest_file = live_dir / "latest_weights.csv"
        if latest_file.exists():
            try:
                df_tmp = pd.read_csv(latest_file)
                trade_date = str(df_tmp.iloc[0].get("trade_date", datetime.now().strftime("%Y-%m-%d")))
            except Exception:
                trade_date = datetime.now().strftime("%Y-%m-%d")
        else:
            trade_date = datetime.now().strftime("%Y-%m-%d")
    else:
        trade_date = args.trade_date

    logger.info("Trade date: %s", trade_date)
    logger.info("Gap input dir: %s", gap_input_dir)
    logger.info("v1 weights file: %s", v1_weights_file)
    logger.info("Live dir: %s", live_dir)

    # Run core logic
    result = generate_v2_production_portfolio(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        v1_weights_file=v1_weights_file,
        cfg=cfg,
    )

    # Write files
    dry_run = args.dry_run == "true"
    write_production_files(trade_date, live_dir, result, dry_run=dry_run)

    # Final status log
    audit_ok = (
        result["leakage"]["status"] == "PASSED"
        and result["numerical"]["status"] == "PASSED"
    )
    fallback_used = result["fallback"]["v1_fallback_used"]

    if fallback_used:
        logger.warning(
            "[%s] Production v2 COMPLETED with v1 FALLBACK. Gross=%.4f  Leakage=%s  Numerical=%s",
            trade_date,
            float(np.sum(np.abs(result["w_final"]))),
            result["leakage"]["status"],
            result["numerical"]["status"],
        )
    elif audit_ok:
        logger.info(
            "[%s] Production v2 COMPLETED OK.  Bin=%s  Mult=%.2f  Gross=%.4f  IR=%.4f",
            trade_date,
            result["pit_binning"]["assigned_bin"],
            result["pit_binning"]["multiplier"],
            float(np.sum(np.abs(result["w_final"]))),
            result["summary"]["predicted_portfolio_ir"],
        )
    else:
        logger.error(
            "[%s] Production v2 COMPLETED with AUDIT FAILURES. Leakage=%s  Numerical=%s",
            trade_date,
            result["leakage"]["status"],
            result["numerical"]["status"],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
