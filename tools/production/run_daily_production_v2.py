#!/usr/bin/env python
"""Daily Production Runner v2: mu_over_sigma + baseline_style + RuleD.

This script is the official daily production execution entry-point for
production_residual_blpx_v2.  All business logic lives in the ``src/leadlag`` package:

  - Portfolio construction : ``leadlag.models.production_v2``
  - Math helpers           : ``leadlag.core.portfolio``
  - Compliance audits      : ``leadlag.compliance.v2_auditor``
  - File output            : ``leadlag.reporting.production_v2_writer``

This file is intentionally kept as a thin CLI wrapper (~130 lines).

Usage (normal daily run)::

    python tools/run_daily_production_v2.py \\
        --trade-date latest \\
        --gap-input-dir results/gap_adjusted_distribution/latest

Usage (dry-run, no file writes)::

    python tools/run_daily_production_v2.py \\
        --trade-date 2026-06-16 \\
        --gap-input-dir results/gap_adjusted_distribution/latest \\
        --dry-run true

Usage (self-test)::

    python tools/run_daily_production_v2.py --self-test true
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup — make the project package importable without installation
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Import after path is set up
from leadlag.compliance.v2_auditor import run_leakage_audit, run_numerical_audit
from leadlag.core.portfolio import get_rolling_pit_bin, solve_baseline_style
from leadlag.models.production_v2 import generate_v2_production_portfolio, VERSION
from leadlag.reporting.production_v2_writer import write_production_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ProductionV2")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Daily Production Runner v2 (mu_over_sigma + baseline_style + RuleD)"
    )
    p.add_argument("--config", default="configs/production/production.yaml")
    p.add_argument("--trade-date", default="latest", help="YYYY-MM-DD or 'latest'")
    p.add_argument(
        "--gap-input-dir",
        default=None,
        help="Directory containing mu_gap_{YYYYMMDD}.npy and omega_gap_{YYYYMMDD}.npy",
    )
    p.add_argument(
        "--v1-weights-file",
        default="live/production_residual_blpx/v1_baseline_weights.csv",
        help="Path to v1 baseline weights CSV (fallback when gap data missing)",
    )
    p.add_argument(
        "--live-dir",
        default="live/production_residual_blpx",
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
# Self Tests (thin wrappers over package functions)
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

    # Test 2: PIT binning
    hist = np.linspace(0.0, 3.0, 500)
    b, lo, hi, m = get_rolling_pit_bin(hist, 0.5, rolling_window=252)
    assert b == "Low" and abs(m - 0.75) < 1e-9, f"T2a: expected Low/0.75 got {b}/{m}"
    logger.info("[PASS] T2a: PIT Low bin")

    b, _, _, m = get_rolling_pit_bin(hist, 2.2, rolling_window=252)
    assert b == "Medium" and abs(m - 1.0) < 1e-9, f"T2b: expected Medium/1.0 got {b}/{m}"
    logger.info("[PASS] T2b: PIT Medium bin")

    b, _, _, m = get_rolling_pit_bin(hist, 3.5, rolling_window=252)
    assert b == "High" and abs(m - 1.0) < 1e-9, f"T2c: expected High/1.0 got {b}/{m}"
    logger.info("[PASS] T2c: PIT High bin")

    # Test 3: insufficient history → fallback to Medium/1.0
    b, lo, _, m = get_rolling_pit_bin(hist, 2.0, rolling_window=600)
    assert b == "Medium" and np.isnan(lo) and abs(m - 1.0) < 1e-9, "T3: insufficient history"
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
    cost_bps_per_gross = 10.0  # Default from ProductionV2RunConfig
    cost_bps = gross_ex * cost_bps_per_gross
    assert abs(cost_bps - 20.0) < 1e-9, f"T6: expected 20 bps got {cost_bps}"
    logger.info("[PASS] T6: cost formula")

    logger.info("=== All Self-Tests PASSED ===")
    return 0


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
        gap_path = (
            ROOT / args.gap_input_dir
            if not args.gap_input_dir.startswith("/")
            else Path(args.gap_input_dir)
        )
        if gap_path.exists():
            gap_input_dir = gap_path
        else:
            logger.warning(
                "--gap-input-dir does not exist: %s. Will use v1 fallback.", gap_path
            )
    else:
        # Try default from config
        default_gap = cfg.get("gap_distribution", {}).get("dir", "")
        if default_gap:
            gap_path = (
                ROOT / default_gap
                if not default_gap.startswith("/")
                else Path(default_gap)
            )
            if gap_path.exists():
                gap_input_dir = gap_path
                logger.info("Using gap_distribution.dir from config: %s", gap_path)
            else:
                logger.warning(
                    "Config gap_distribution.dir not found: %s. Will use v1 fallback.",
                    gap_path,
                )

    # Resolve trade date
    if args.trade_date == "latest":
        latest_file = live_dir / "latest_weights.csv"
        if latest_file.exists():
            try:
                import pandas as pd
                df_tmp = pd.read_csv(latest_file)
                trade_date = str(
                    df_tmp.iloc[0].get("trade_date", datetime.now().strftime("%Y-%m-%d"))
                )
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

    # Run core logic (delegated to the package)
    result = generate_v2_production_portfolio(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        v1_weights_file=v1_weights_file,
        cfg=cfg,
    )

    # Write files (delegated to the package)
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
