#!/usr/bin/env python
"""Production V2 decision entry point.

Thin wrapper around `leadlag.execution.v2_bridge.run_v2_decision` so the
launcher shell script can call a script file instead of an inline `python -c`
expression (the latter is a known source of CLI hangs).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Path setup — make the project package importable without installation
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.v2_bridge import run_v2_decision  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("v2_decision")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Run V2 production decision and submit orders via broker API."
    )
    p.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to V2 production YAML config.",
    )
    p.add_argument(
        "--gap-input-dir",
        default="live/pipeline_data/gap_adjusted_distribution/latest",
        help="Directory containing mu_gap/omega_gap .npy files.",
    )
    p.add_argument(
        "--live-dir",
        default="live/production_residual_blpx",
        help="Live output directory for V2 artifacts.",
    )
    p.add_argument(
        "--trade-date",
        default=None,
        help="Trade date string (YYYY-MM-DD). Defaults to today.",
    )
    p.add_argument(
        "--api-enable",
        action="store_true",
        help="Submit orders to broker API.",
    )
    p.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Simulate order submission.",
    )
    p.add_argument(
        "--capital-from-wallet",
        action="store_true",
        help="Use wallet balance as max capital.",
    )
    p.add_argument(
        "--text-output",
        action="store_true",
        help="Print text order summary.",
    )
    p.add_argument(
        "--output-root",
        default="results",
        help="Root directory for decision output.",
    )
    p.add_argument(
        "--jp-opens-csv",
        default=None,
        help="Path to JP opens CSV (fallback if API unavailable).",
    )
    p.add_argument(
        "--google-opens",
        action="store_true",
        help="Use Google Finance for JP opens.",
    )
    return p.parse_args()


def main() -> int:
    """Run V2 decision with command-line arguments."""
    args = parse_arguments()
    try:
        run_v2_decision(
            config_path=args.config,
            gap_input_dir=args.gap_input_dir,
            live_dir=args.live_dir,
            trade_date=args.trade_date,
            api_enable=args.api_enable,
            api_dry_run=args.api_dry_run,
            capital_from_wallet=args.capital_from_wallet,
            text_output=args.text_output,
            output_root=args.output_root,
            jp_opens_csv=args.jp_opens_csv,
            google_opens=args.google_opens,
        )
        return 0
    except Exception:
        logger.exception("V2 decision failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
