"""leadlag/cli.py — Command-line interface for the lead-lag trading package.

Supports subparsers:
  - decision: Run one-day trade decision pipeline (with optional --fast-mode)
  - backtest: Run full historical simulation
  - close: Run end-of-day position closing logic
"""

from __future__ import annotations

import argparse
import logging
import os
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
    decision_parser = subparsers.add_parser("decision", help="Run one-day trade decision pipeline")
    decision_parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Backtest start date (default: 2015-01-01).",
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
        "--fast-mode",
        action="store_true",
        help="Use precomputed cache for faster decision-making (skips heavy decomposition).",
    )
    decision_parser.add_argument(
        "--auto-close",
        action="store_true",
        help="Automatically close all positions at end-of-day.",
    )
    decision_parser.add_argument(
        "--auto-close-time",
        default="14:50",
        help="Time to auto-close positions (HH:MM format, default: 14:50).",
    )
    decision_parser.add_argument(
        "--close-position-order",
        type=int,
        default=0,
        help="Close position order priority (0-7).",
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
    backtest_parser = subparsers.add_parser("backtest", help="Run full historical simulation")
    backtest_parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Simulation start date (default: 2015-01-01).",
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
        if args.auto_close:
            logger.warning(
                "--auto-close is deprecated: the decision process will block until %s. "
                "Use the separate 'close' subcommand via launchd/cron (com.leadlag.close) instead. "
                "Remove --auto-close from batch scripts to avoid indefinite hangs.",
                getattr(args, "auto_close_time", "14:50"),
            )

        if args.capital_from_wallet and not args.api_enable:
            raise ValueError("--capital-from-wallet requires --api-enable")

        if args.fast_mode:
            logger.info("=== FAST MODE ENABLED (No yfinance) ===")
            if not args.api_enable:
                raise ValueError(
                    "FAST MODE requires --api-enable to fetch US returns and JP opens "
                    "from kabuステーション API (no yfinance dependency)."
                )

            # Lazy imports
            from leadlag.data.cache import (
                exclusive_lock as _exclusive_lock,
            )
            from leadlag.data.cache import (
                is_strategy_cache_valid as _is_cache_valid,
            )
            from leadlag.data.cache import (
                load_df_exec_from_local_cache as _load_df_exec_from_local_cache,
            )
            from leadlag.data.cache import load_jp_close_from_cache
            from leadlag.data.market_data import (
                compute_gap_from_jp_close as _compute_gap_from_jp_close,
            )
            from leadlag.data.market_data import (
                compute_topix_night_override as _compute_topix_night_override,
            )
            from leadlag.data.market_data import (
                normalize_to_tokyo_date as _normalize_to_tokyo_date,
            )
            from leadlag.execution.config import load_config_from_yaml
            from leadlag.execution.fast import (
                build_precomputed_cache,
                fetch_jp_opens_for_fast_mode,
                fetch_us_returns_from_api,
                run_decision_fast,
            )
            from leadlag.execution.helpers import (
                build_api_client,
                build_output_dir,
                fetch_current_positions,
                resolve_wallet_capital,
            )

            # Load config from YAML
            config = load_config_from_yaml()

            output_dir = build_output_dir(
                args.output_root,
                args.run_tag,
                run_name="production_decision_fast",
            )
            cache_path = os.path.join(args.output_root, ".cache", "strategy_cache.npz")

            api_client = None
            try:
                api_client = build_api_client(args.api_url, args.api_token, args.api_dry_run)

                t_trade = (
                    pd.to_datetime(args.trade_date).normalize()
                    if args.trade_date is not None
                    else pd.Timestamp.now().normalize()
                )

                logger.info("[1/3] Fetching US ETF returns from kabu API...")
                us_returns_today = fetch_us_returns_from_api(api_client, args.output_root)

                logger.info("[2/3] Fetching JP opens...")
                manual_opens, topix_open = fetch_jp_opens_for_fast_mode(
                    api_client=api_client,
                    config=config.strategy,
                    jp_opens_csv=args.jp_opens_csv,
                    google_opens=args.google_opens,
                )

                # Build or validate precomputed strategy cache
                with _exclusive_lock(cache_path + ".lock"):
                    if not _is_cache_valid(cache_path, config=config.strategy):
                        logger.info("[FAST MODE] Building precomputed cache from local cache...")
                        df_exec = _load_df_exec_from_local_cache()
                        build_precomputed_cache(config.strategy, df_exec, cache_path)
                        logger.info("[FAST MODE] Cache built: %s", cache_path)
                    else:
                        logger.info("[FAST MODE] Using existing cache: %s", cache_path)

                # Gap override from local jp_close cache
                jp_close = load_jp_close_from_cache()
                jp_close.index = _normalize_to_tokyo_date(jp_close.index)
                gap_override = _compute_gap_from_jp_close(jp_close, t_trade, manual_opens)
                topix_night_override = None
                if topix_open is not None:
                    topix_night_override = _compute_topix_night_override(
                        jp_close, t_trade, topix_open
                    )

                logger.info("[3/3] Generating trade decision (FAST path)...")
                max_capital = args.capital
                if args.capital_from_wallet:
                    max_capital = resolve_wallet_capital(api_client)

                # Fetch existing positions for delta-based order submission
                current_positions = None
                try:
                    current_positions = fetch_current_positions(api_client)
                except Exception as e:
                    logger.warning("Failed to fetch current positions: %s. Will submit full target.", e)

                result_path = run_decision_fast(
                    config=config.strategy,
                    cache_path=cache_path,
                    trade_date=t_trade,
                    manual_opens=manual_opens,
                    gap_override=gap_override,
                    topix_night_override=topix_night_override,
                    us_returns_today=us_returns_today,
                    max_capital=max_capital,
                    output_dir=output_dir,
                    output_root=args.output_root,
                    api_client=api_client,
                    api_dry_run=args.api_dry_run,
                    text_output=args.text_output,
                    current_positions=current_positions,
                )
                logger.info("Fast decision completed. Output: %s", result_path)

                if args.auto_close:
                    from leadlag.execution.close import wait_and_auto_close

                    wait_and_auto_close(
                        api_client=api_client,
                        output_dir=output_dir,
                        auto_close_time=args.auto_close_time,
                        dry_run=args.api_dry_run,
                        close_position_order=args.close_position_order,
                    )
            finally:
                if api_client is not None:
                    api_client.close()

        else:
            # ---- Standard mode ----
            from leadlag.execution.decision import run_decision

            run_decision(
                start_date=args.start_date,
                output_root=args.output_root,
                run_tag=args.run_tag,
                trade_date=args.trade_date,
                opens_csv=args.jp_opens_csv,
                max_capital=args.capital,
                api_enable=args.api_enable,
                api_url=args.api_url,
                api_token=args.api_token,
                api_dry_run=args.api_dry_run,
                use_google_opens=args.google_opens,
                text_output=args.text_output,
                use_wallet_capital=args.capital_from_wallet,
            )

            if args.auto_close:
                from leadlag.execution.close import wait_and_auto_close
                from leadlag.execution.helpers import build_api_client, build_output_dir

                api_client = None
                output_dir = build_output_dir(
                    args.output_root, args.run_tag, run_name="production_decision"
                )
                try:
                    if args.api_enable:
                        api_client = build_api_client(
                            args.api_url, args.api_token, args.api_dry_run
                        )
                        wait_and_auto_close(
                            api_client=api_client,
                            output_dir=output_dir,
                            auto_close_time=args.auto_close_time,
                            dry_run=args.api_dry_run,
                            close_position_order=args.close_position_order,
                        )
                finally:
                    if api_client is not None:
                        api_client.close()

    elif args.command == "backtest":
        from leadlag.execution.backtest import run_production

        backtest_kwargs: dict = {}
        if args.slippage_bps is not None:
            backtest_kwargs["slippage_bps"] = args.slippage_bps
        run_production(
            start_date=args.start_date,
            output_root=args.output_root,
            run_tag=args.run_tag,
            skip_chart=args.skip_chart,
            **backtest_kwargs,
        )

    elif args.command == "close":
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


if __name__ == "__main__":
    sys.exit(main())
