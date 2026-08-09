"""leadlag/cli.py — Command-line interface for the lead-lag trading package.

Supports subparsers:
  - decision: Run one-day V2 trade decision pipeline
  - backtest: Run full V2 historical simulation
  - close: Run end-of-day position closing logic
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import pandas as pd

from leadlag.reporting.results_format import get_default_results_root

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool for the lead-lag market-neutral trading strategy."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # --- DECISION SUBCOMMAND ---
    decision_parser = subparsers.add_parser("decision", help="Run one-day V2 trade decision pipeline")
    decision_parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to V2 production YAML config (default: configs/production/production.yaml).",
    )
    decision_parser.add_argument(
        "--gap-dir",
        default=None,
        help="Directory containing mu_gap/omega_gap .npy files. "
             "Defaults to gap_distribution.dir in the YAML config.",
    )
    decision_parser.add_argument(
        "--live-dir",
        default="live/production_residual_blpx",
        help="Live output directory for V2 artifacts (default: live/production_residual_blpx).",
    )
    decision_parser.add_argument(
        "--output-root",
        default=get_default_results_root(),
        help="Directory root where outputs are written.",
    )
    decision_parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag. If omitted, a timestamp is used.",
    )
    decision_parser.add_argument(
        "--trade-date",
        default=None,
        help="Trade date in YYYY-MM-DD for decision mode (default: today).",
    )
    decision_parser.add_argument(
        "--jp-opens-csv",
        default=None,
        help="CSV file with TOPIX-17 opens (columns: ticker, open_price).",
    )
    decision_parser.add_argument(
        "--capital",
        type=float,
        default=1000000.0,
        help="Equity capital in JPY for position sizing (default: 1000000).",
    )
    decision_parser.add_argument(
        "--capital-from-wallet",
        action="store_true",
        help="Use cash account wallet balance from kabu API for sizing (requires --api-enable).",
    )
    decision_parser.add_argument(
        "--api-enable",
        action="store_true",
        help="Enable kabuステーション API for live order submission.",
    )
    decision_parser.add_argument(
        "--api-url",
        default=None,
        help="kabuステーション API URL. Defaults to KABU_API_URL environment variable.",
    )
    decision_parser.add_argument(
        "--api-token",
        default=None,
        help="kabuステーション API token. Defaults to KABU_API_TOKEN environment variable.",
    )
    decision_parser.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Simulate API calls without actually submitting orders.",
    )
    decision_parser.add_argument(
        "--auto-close",
        action="store_true",
        help="(Deprecated) Use the separate 'close' subcommand instead.",
    )
    decision_parser.add_argument(
        "--auto-close-time",
        default="14:50",
        help="(Deprecated) Time to auto-close positions (HH:MM format, default: 14:50).",
    )
    decision_parser.add_argument(
        "--close-position-order",
        type=int,
        default=0,
        help="(Deprecated) Close position order priority (0-7).",
    )
    decision_parser.add_argument(
        "--google-opens",
        action="store_true",
        help="Fetch JP open prices from Google Finance.",
    )
    decision_parser.add_argument(
        "--text-output",
        action="store_true",
        help="Output trade orders in text format to the console.",
    )

    # --- BACKTEST SUBCOMMAND ---
    backtest_parser = subparsers.add_parser("backtest", help="Run full V2 historical simulation")
    backtest_parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to V2 production YAML config (default: configs/production/production.yaml).",
    )
    backtest_parser.add_argument(
        "--gap-dir",
        default=None,
        help="Directory containing mu_gap/omega_gap .npy files. "
             "Defaults to gap_distribution.dir in the YAML config.",
    )
    backtest_parser.add_argument(
        "--start-date",
        default="2015-01-05",
        help="Simulation start date (default: 2015-01-05).",
    )
    backtest_parser.add_argument(
        "--output-root",
        default=get_default_results_root(),
        help="Directory root where outputs are written.",
    )
    backtest_parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag. If omitted, a timestamp is used.",
    )
    backtest_parser.add_argument(
        "--skip-chart",
        action="store_true",
        help="Skip cumulative return and drawdown chart generation.",
    )
    backtest_parser.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help="Slippage cost per side in bps. If omitted, YAML default is used.",
    )
    backtest_parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers for signal computation (1=sequential, -1=all cores).",
    )

    # --- CLOSE SUBCOMMAND ---
    close_parser = subparsers.add_parser("close", help="Run end-of-day position closing logic")
    close_parser.add_argument(
        "--output-root",
        default=get_default_results_root(),
        help="Directory root where outputs are written.",
    )
    close_parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag. If omitted, a timestamp is used.",
    )
    close_parser.add_argument(
        "--api-url",
        default=None,
        help="kabuステーション API URL.",
    )
    close_parser.add_argument(
        "--api-token",
        default=None,
        help="kabuステーション API token.",
    )
    close_parser.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Simulate API calls without actually submitting orders.",
    )
    close_parser.add_argument(
        "--close-position-order",
        type=int,
        default=0,
        help="Close position order priority (0-7).",
    )

    return parser


def _handle_decision(args: argparse.Namespace) -> int:
    """Run the one-day trade decision pipeline."""
    if args.auto_close:
        logger.warning(
            "--auto-close is deprecated: the decision process will block until %s. "
            "Use the separate 'close' subcommand via launchd/cron (com.leadlag.close) instead. "
            "Remove --auto-close from batch scripts to avoid indefinite hangs.",
            getattr(args, "auto_close_time", "14:50"),
        )

    if args.capital_from_wallet and not args.api_enable:
        raise ValueError("--capital-from-wallet requires --api-enable")

    # --- V2 decision ---
    from leadlag.execution.v2_bridge import run_v2_decision

    result_path = run_v2_decision(
        config_path=args.config,
        gap_input_dir=args.gap_dir,
        live_dir=args.live_dir,
        trade_date=args.trade_date,
        api_enable=args.api_enable,
        api_dry_run=args.api_dry_run,
        capital_from_wallet=args.capital_from_wallet,
        text_output=args.text_output,
        output_root=args.output_root,
        jp_opens_csv=args.jp_opens_csv,
        google_opens=args.google_opens,
        max_capital=args.capital,
        api_url=args.api_url,
        api_token=args.api_token,
        run_tag=args.run_tag,
    )
    logger.info("V2 decision completed. Output: %s", result_path)

    return 0


def _handle_backtest(args: argparse.Namespace) -> int:
    """Run full V2 historical backtest simulation."""
    from leadlag.execution.backtest import run_production

    backtest_kwargs: dict = {}
    if args.slippage_bps is not None:
        backtest_kwargs["slippage_bps"] = args.slippage_bps
    if hasattr(args, "n_jobs") and args.n_jobs != 1:
        backtest_kwargs["n_jobs"] = args.n_jobs
    if args.config is not None:
        backtest_kwargs["config_path"] = args.config
    if args.gap_dir is not None:
        backtest_kwargs["gap_input_dir"] = args.gap_dir
    run_production(
        start_date=args.start_date,
        output_root=args.output_root,
        run_tag=args.run_tag,
        skip_chart=args.skip_chart,
        **backtest_kwargs,
    )
    return 0


def _handle_close(args: argparse.Namespace) -> int:
    """Run end-of-day position closing logic."""
    from leadlag.execution.close import run_close_positions_mode

    run_close_positions_mode(
        output_root=args.output_root,
        run_tag=args.run_tag,
        api_url=args.api_url,
        api_token=args.api_token,
        api_dry_run=args.api_dry_run,
        close_position_order=args.close_position_order,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = setup_parser()

    # If no arguments are passed, show help
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)

    # Skip execution on non-trading days (weekends & Japanese holidays)
    if args.command in ("decision", "close"):
        from leadlag.core.market_calendar import is_market_closed

        today = pd.Timestamp.now().date()
        if is_market_closed(today):
            holiday_name = None
            try:
                from leadlag.core.market_calendar import get_holiday_name

                holiday_name = get_holiday_name(today)
            except Exception:
                pass
            label = holiday_name or "non-trading day"
            logger.info("Market closed today (%s: %s). Skipping %s.", today, label, args.command)
            return 0

    if args.command == "decision":
        return _handle_decision(args)
    if args.command == "backtest":
        return _handle_backtest(args)
    if args.command == "close":
        return _handle_close(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
