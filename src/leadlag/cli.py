"""leadlag/cli.py — Command-line interface for the lead-lag trading package.

Supports subparsers:
  - decision: Run one-day V2 trade decision pipeline
  - backtest: Run full V2 historical simulation
  - daily: Run decision before the morning cutoff and close at/after it
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


def _add_decision_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the decision (and daily) subcommand."""
    parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to V2 production YAML config (default: configs/production/production.yaml).",
    )
    parser.add_argument(
        "--gap-dir",
        default=None,
        help="Directory containing mu_gap/omega_gap .npy files. "
             "Defaults to gap_distribution.dir in the YAML config.",
    )
    parser.add_argument(
        "--live-dir",
        default="var/live/production_residual_blpx",
        help="Live output directory for V2 artifacts (default: var/live/production_residual_blpx).",
    )
    parser.add_argument(
        "--output-root",
        default=get_default_results_root(),
        help="Directory root where outputs are written.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag. If omitted, a timestamp is used.",
    )
    parser.add_argument(
        "--trade-date",
        default=None,
        help="Trade date in YYYY-MM-DD or 'latest' for decision mode (default: today).",
    )
    parser.add_argument(
        "--jp-opens-csv",
        default=None,
        help="CSV file with TOPIX-17 opens (columns: ticker, open_price).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=300000.0,
        help="Equity capital in JPY for position sizing (default: 300000).",
    )
    parser.add_argument(
        "--capital-from-wallet",
        action="store_true",
        help="Use cash account wallet balance from kabu API for sizing (requires --api-enable).",
    )
    parser.add_argument(
        "--api-enable",
        action="store_true",
        help="Enable kabuステーション API for live order submission.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="kabuステーション API URL. Defaults to KABU_API_URL environment variable.",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="kabuステーション API token. Defaults to KABU_API_TOKEN environment variable.",
    )
    parser.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Simulate API calls without actually submitting orders.",
    )
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help="(Deprecated) Use the separate 'close' subcommand instead.",
    )
    parser.add_argument(
        "--auto-close-time",
        default="14:50",
        help="(Deprecated) Time to auto-close positions (HH:MM format, default: 14:50).",
    )
    parser.add_argument(
        "--close-position-order",
        type=int,
        default=0,
        help="(Deprecated) Close position order priority (0-7).",
    )
    parser.add_argument(
        "--google-opens",
        action="store_true",
        help="Fetch JP open prices from Google Finance.",
    )
    parser.add_argument(
        "--text-output",
        action="store_true",
        help="Output trade orders in text format to the console.",
    )
    parser.add_argument(
        "--engine",
        choices=["nextgen", "v2"],
        default="nextgen",
        help="Execution engine: 'nextgen' (default, Unified Convex + Async FSM) or 'v2' (legacy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate weights but do not write files or submit orders.",
    )


def _add_close_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the close (and daily) subcommand."""
    parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to V2 production YAML config (used by the Next-Gen path; ignored by the V2 legacy path).",
    )
    parser.add_argument(
        "--engine",
        choices=["nextgen", "v2"],
        default="nextgen",
        help="Execution engine: 'nextgen' (default, Async FSM) or 'v2' (legacy).",
    )
    parser.add_argument(
        "--output-root",
        default=get_default_results_root(),
        help="Directory root where outputs are written.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag. If omitted, a timestamp is used.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="kabuステーション API URL.",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="kabuステーション API token.",
    )
    parser.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Simulate API calls without actually submitting orders.",
    )
    parser.add_argument(
        "--close-position-order",
        type=int,
        default=0,
        help="Close position order priority (0-7).",
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI tool for the lead-lag market-neutral trading strategy."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # --- DECISION SUBCOMMAND ---
    decision_parser = subparsers.add_parser("decision", help="Run one-day V2 trade decision pipeline")
    _add_decision_args(decision_parser)

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
        "--gap-store",
        default=None,
        help="Optional path to a .sqlite GapStore. If provided it overrides "
             "--gap-dir for matrix loading while keeping --gap-dir as a .npy fallback.",
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
    backtest_parser.add_argument(
        "--data-source",
        choices=["download", "cache"],
        default="cache",
        help="How to obtain df_exec: cache (default) or download.",
    )
    backtest_parser.add_argument(
        "--end-date",
        default="latest",
        help="Backtest end date ('latest' for last available, default: latest).",
    )
    backtest_parser.add_argument(
        "--side-leverage",
        type=float,
        default=None,
        help="Notional side leverage (overrides config if set).",
    )
    backtest_parser.add_argument(
        "--output-level",
        choices=["minimal", "detailed"],
        default="detailed",
        help="Output level: detailed (default) writes full daily CSVs; minimal writes summary only.",
    )
    backtest_parser.add_argument(
        "--overlay-model-dir",
        default=None,
        help="Path to an ML order-overlay model directory (optional).",
    )

    # --- DAILY SUBCOMMAND ---
    daily_parser = subparsers.add_parser(
        "daily", help="Run the appropriate daily operation (decision in morning, close in afternoon)"
    )
    _add_decision_args(daily_parser)
    # Decision args already include output_root, run_tag, api_url, api_token,
    # api_dry_run and close_position_order (latter is deprecated for decision).
    # The daily dispatcher re-uses those values for the close phase as well.
    daily_parser.add_argument(
        "--decision-cutoff",
        default="09:15",
        help="HH:MM cutoff. Before this time 'daily' runs 'decision'; after it runs 'close' (default: 09:15).",
    )

    # --- CLOSE SUBCOMMAND ---
    close_parser = subparsers.add_parser("close", help="Run end-of-day position closing logic")
    _add_close_args(close_parser)

    # --- NEXTGEN SHADOW SUBCOMMAND ---
    shadow_parser = subparsers.add_parser(
        "nextgen-shadow", help="Run Next-Gen Convex Pipeline shadow comparison against Production V2"
    )
    shadow_parser.add_argument(
        "--trade-date",
        default="latest",
        help="Trade date in YYYY-MM-DD or 'latest' (default: latest).",
    )
    shadow_parser.add_argument(
        "--config",
        default="configs/production/production.yaml",
        help="Path to V2 production YAML config (default: configs/production/production.yaml).",
    )
    shadow_parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Equity capital in JPY (default: 1000000).",
    )

    # --- SELF-TEST SUBCOMMAND ---
    subparsers.add_parser("self-test", help="Run CLI production self-tests and exit")

    return parser


def _handle_decision(args: argparse.Namespace) -> int:
    """Run the one-day trade decision pipeline."""
    if args.auto_close:
        logger.warning(
            "--auto-close is deprecated and ignored inside the 'decision' subcommand. "
            "Use the separate 'close' subcommand via launchd/cron (com.leadlag.close) instead. "
            "Remove --auto-close from batch scripts to avoid this warning."
        )

    if args.capital_from_wallet and not args.api_enable:
        raise ValueError("--capital-from-wallet requires --api-enable")

    if getattr(args, "engine", "nextgen") == "v2":
        # --- V2 Legacy Decision ---
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
            dry_run=args.dry_run,
        )
        logger.info("V2 legacy decision completed. Output: %s", result_path)
        return 0

    # --- Next-Gen Unified Convex + Async FSM Decision ---
    import asyncio
    from pathlib import Path

    from leadlag.broker.async_base import (
        AsyncBrokerClient,
        AsyncDryRunBrokerClient,
        AsyncThreadedBrokerClient,
    )
    from leadlag.config.paths import project_root
    from leadlag.core.market_calendar import is_market_closed, previous_trading_day
    from leadlag.data.market_data_cache import load_df_exec_from_local_cache
    from leadlag.data.pit_lake import PITDataLake
    from leadlag.execution.broker_ops import build_api_client
    from leadlag.execution.config import load_config_from_yaml
    from leadlag.execution.nextgen_pipeline import NextGenDecisionResult, NextGenPipeline

    df_exec = load_df_exec_from_local_cache()
    if df_exec is None:
        raise RuntimeError("df_exec cache not found.")

    lake = PITDataLake(df_exec)

    # Resolve trade date consistently with V2 (latest -> today/previous trading day)
    trade_date = args.trade_date or "latest"
    today = pd.Timestamp.now().tz_localize(None).normalize()
    if trade_date == "latest":
        resolved = today if not is_market_closed(today) else previous_trading_day(today)
        trade_date = resolved.strftime("%Y-%m-%d")
        if pd.to_datetime(trade_date) not in lake.df_exec.index:
            trade_date = lake.end_date.strftime("%Y-%m-%d")
            logger.warning("Resolved latest trade date not in PITDataLake; using %s", trade_date)
    else:
        if pd.to_datetime(trade_date).normalize() > today:
            raise ValueError(f"trade_date {trade_date} is in the future (today: {today.date()})")
        if pd.to_datetime(trade_date) not in lake.df_exec.index:
            raise ValueError(
                f"trade_date {trade_date} is not a trade date in the local cache. "
                "Use 'latest' or a valid historical trading day."
            )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    app_config = load_config_from_yaml(str(config_path))

    # Override gap input dir if --gap-dir provided
    if args.gap_dir:
        gap_dir = Path(args.gap_dir)
        if not gap_dir.is_absolute():
            gap_dir = project_root() / gap_dir
        gap_dir_str = str(gap_dir)
        v2_cfg = app_config.v2.model_copy(update={"gap_input_dir": gap_dir_str})
        app_config = app_config.model_copy(
            update={"gap_distribution_dir": gap_dir_str, "v2": v2_cfg}
        )

    # Resolve live output dir
    live_dir = None
    if args.live_dir:
        live_dir = Path(args.live_dir)
        if not live_dir.is_absolute():
            live_dir = project_root() / live_dir
        app_config = app_config.model_copy(update={"output_live_dir": str(live_dir)})
    else:
        live_dir = Path(app_config.output_live_dir)
        if not live_dir.is_absolute():
            live_dir = project_root() / live_dir

    # Resolve output root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root() / output_root

    nextgen = NextGenPipeline(app_config)

    async def _run_nextgen() -> NextGenDecisionResult:
        # Build broker client: live API or dry-run simulator
        broker: AsyncBrokerClient
        if args.api_enable and not args.dry_run and not args.api_dry_run:
            sync_client = build_api_client(
                api_url=args.api_url,
                api_token=args.api_token,
                api_dry_run=False,
            )
            broker = AsyncThreadedBrokerClient(
                sync_client,
                broker_request_timeout=app_config.nextgen.broker_request_timeout_seconds,
            )
        elif args.api_enable and args.api_dry_run:
            sync_client = build_api_client(
                api_url=args.api_url,
                api_token=args.api_token,
                api_dry_run=True,
            )
            broker = AsyncThreadedBrokerClient(
                sync_client,
                broker_request_timeout=app_config.nextgen.broker_request_timeout_seconds,
            )
        else:
            broker = AsyncDryRunBrokerClient(simulated_latency_ms=10.0)

        async with broker:
            capital = args.capital
            if args.capital_from_wallet:
                wallet = await broker.get_wallet()
                capital = wallet.cash_available if wallet.cash_available > 0 else args.capital

            res = await nextgen.run_daily_decision(
                trade_date=trade_date,
                lake=lake,
                broker=broker,
                capital=capital,
                submit_orders=not args.dry_run and args.api_enable,
            )

            nextgen.save_decision_artifacts(
                res,
                output_root=output_root,
                run_tag=args.run_tag,
                live_dir=live_dir,
                dry_run=args.dry_run,
                text_output=args.text_output,
            )
            return res

    res = asyncio.run(_run_nextgen())
    logger.info(
        "Next-Gen decision completed. Date: %s, Success: %s, IR: %.4f, Gross: %.2f",
        res.trade_date,
        res.success,
        res.opt_result.ex_ante_ir,
        res.opt_result.gross_exposure,
    )
    return 0 if res.success else 1


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
    if args.gap_store is not None:
        backtest_kwargs["gap_store_path"] = args.gap_store
    if args.data_source is not None:
        backtest_kwargs["data_source"] = args.data_source
    if args.end_date is not None:
        backtest_kwargs["end_date"] = args.end_date
    if args.side_leverage is not None:
        backtest_kwargs["side_leverage"] = args.side_leverage
    if args.output_level is not None:
        backtest_kwargs["output_level"] = args.output_level
    if args.overlay_model_dir is not None:
        backtest_kwargs["overlay_model_dir"] = args.overlay_model_dir
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
    if getattr(args, "engine", "nextgen") == "v2":
        # --- V2 Legacy Close ---
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

    # --- Next-Gen Async Close ---
    import asyncio
    import json
    from pathlib import Path

    from leadlag.broker.async_base import (
        AsyncBrokerClient,
        AsyncDryRunBrokerClient,
        AsyncThreadedBrokerClient,
    )
    from leadlag.config.paths import project_root
    from leadlag.execution.async_fsm import AsyncExecutionEngine, ExecutionJournal
    from leadlag.execution.broker_ops import build_api_client
    from leadlag.execution.config import load_config_from_yaml
    from leadlag.execution.output_ops import build_output_dir

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    app_config = load_config_from_yaml(str(config_path))

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root() / output_root
    output_dir = Path(
        build_output_dir(
            str(output_root),
            args.run_tag,
            "production_close_positions",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = AsyncExecutionEngine()

    async def _run_close() -> ExecutionJournal:
        broker: AsyncBrokerClient
        if args.api_enable and not args.api_dry_run:
            sync_client = build_api_client(
                api_url=args.api_url,
                api_token=args.api_token,
                api_dry_run=False,
            )
            broker = AsyncThreadedBrokerClient(
                sync_client,
                broker_request_timeout=app_config.nextgen.broker_request_timeout_seconds,
            )
        elif args.api_enable and args.api_dry_run:
            sync_client = build_api_client(
                api_url=args.api_url,
                api_token=args.api_token,
                api_dry_run=True,
            )
            broker = AsyncThreadedBrokerClient(
                sync_client,
                broker_request_timeout=app_config.nextgen.broker_request_timeout_seconds,
            )
        else:
            broker = AsyncDryRunBrokerClient(simulated_latency_ms=10.0)

        async with broker:
            trade_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            journal = await engine.close_all_positions(
                broker,
                trade_date=trade_date,
                close_position_order=args.close_position_order,
            )
            logger.info(
                "Next-Gen async close completed. Filled: %d, Failed: %d, Success: %s",
                journal.filled_orders,
                journal.failed_orders,
                journal.success,
            )

            journal_data = {
                "trade_date": journal.trade_date,
                "total_orders": journal.total_orders,
                "filled_orders": journal.filled_orders,
                "failed_orders": journal.failed_orders,
                "close_orders_count": journal.close_orders_count,
                "new_orders_count": journal.new_orders_count,
                "elapsed_seconds": journal.elapsed_seconds,
                "success": journal.success,
            }
            journal_file = output_dir / "execution_journal.json"
            with journal_file.open("w") as f:
                json.dump(journal_data, f, indent=2, ensure_ascii=False)
            logger.info("Close journal saved: %s", journal_file)
            return journal

    journal = asyncio.run(_run_close())
    return 0 if journal.success else 1


def _handle_nextgen_shadow(args: argparse.Namespace) -> int:
    """Run Next-Gen Convex Pipeline shadow comparison against Production V2."""
    import importlib.util
    import sys
    from pathlib import Path

    # tools/validation is not a package; load it as a module without mutating
    # sys.path so it cannot collide with installed packages.
    script_path = Path(__file__).resolve().parents[2] / "tools" / "validation" / "run_nextgen_shadow_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "run_nextgen_shadow_pipeline",
        script_path,
        submodule_search_locations=[str(script_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_nextgen_shadow_pipeline"] = module
    spec.loader.exec_module(module)

    module.run_shadow_comparison(
        trade_date=args.trade_date,
        config_path=args.config,
        capital=args.capital,
    )
    return 0


def _handle_self_test(args: argparse.Namespace) -> int:
    """Run production self-tests."""
    from leadlag.execution.self_test import run_self_tests

    return run_self_tests()


def _handle_daily(args: argparse.Namespace) -> int:
    """Run the daily operation: decision before cutoff, close after.

    This lets a single cron/launchd entry call ``leadlag.cli daily`` at any
    time of day and have the correct phase execute.
    """
    from datetime import datetime

    now = datetime.now().time()
    try:
        hour, minute = args.decision_cutoff.split(":")
        cutoff = datetime.strptime(f"{hour}:{minute}", "%H:%M").time()
    except Exception as e:
        raise ValueError(f"--decision-cutoff must be HH:MM, got {args.decision_cutoff}") from e

    if now < cutoff:
        logger.info("Current time %s is before daily cutoff %s. Running 'decision'.", now, cutoff)
        return _handle_decision(args)

    logger.info("Current time %s is at/after daily cutoff %s. Running 'close'.", now, cutoff)
    return _handle_close(args)


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

    # Skip execution on non-trading days (weekends & Japanese holidays).
    # An explicit --trade-date (e.g. a backfill or ``latest`` resolution)
    # bypasses this check so users can rerun for a specific date.
    if args.command in ("close", "daily") or (
        args.command == "decision" and args.trade_date is None
    ) or (
        args.command == "nextgen-shadow" and args.trade_date != "latest"
    ):
        from leadlag.core.market_calendar import is_market_closed

        check_date = pd.Timestamp.now().date()
        if args.command == "nextgen-shadow" and args.trade_date != "latest":
            check_date = pd.to_datetime(args.trade_date).date()
        if is_market_closed(check_date):
            holiday_name = None
            try:
                from leadlag.core.market_calendar import get_holiday_name

                holiday_name = get_holiday_name(check_date)
            except Exception:
                pass
            label = holiday_name or "non-trading day"
            logger.info("Market closed on %s (%s). Skipping %s.", check_date, label, args.command)
            return 0

    if args.command == "decision":
        return _handle_decision(args)
    if args.command == "backtest":
        return _handle_backtest(args)
    if args.command == "close":
        return _handle_close(args)
    if args.command == "daily":
        return _handle_daily(args)
    if args.command == "nextgen-shadow":
        return _handle_nextgen_shadow(args)
    if args.command == "self-test":
        return _handle_self_test(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
