"""production.py — entry point and CLI argument parser.

This module is now a thin facade. All run-mode implementations have been
extracted to dedicated modules under ``runner/``:

    - runner.decision   → run_decision()        (--mode decision, standard)
    - runner.fast       → run_decision_fast()   (--mode decision --fast-mode)
    - runner.close      → close_all_positions() (--mode close-positions)
    - runner.backtest   → run_production()      (--mode backtest)
    - runner.helpers    → shared utilities
    - runner.config     → ProductionConfig

External code that imports from this module directly continues to work
via the re-exports at the bottom of this file.
"""

import argparse
import logging
import os
from typing import Optional

import pandas as pd

from config import KABU_API_CONFIG
from data_loader import JP_TICKERS, TOPIX_TICKER, load_jp_close_from_cache
from results_format import get_default_results_root
from runner.backtest import run_production
from runner.close import close_all_positions, wait_and_auto_close
from runner.config import ProductionConfig
from runner.decision import run_decision
from runner.fast import (
    build_precomputed_cache,
    fetch_jp_opens_for_fast_mode,
    fetch_us_returns_from_api,
    run_decision_fast,
)
from runner.helpers import (
    build_api_client,
    build_output_dir,
    resolve_wallet_capital,
)
from services.cache_service import (
    exclusive_lock as _exclusive_lock,
    is_strategy_cache_valid as _is_cache_valid,
    load_df_exec_from_local_cache as _load_df_exec_from_local_cache,
)
from services.market_data import (
    compute_gap_from_jp_close as _compute_gap_from_jp_close,
    compute_topix_night_override as _compute_topix_night_override,
    normalize_to_tokyo_date as _normalize_to_tokyo_date,
    validate_topix_open as _validate_topix_open,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production runner for the lead-lag market-neutral strategy."
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "decision", "close-positions"],
        default="decision",
        help=(
            "Run mode: 'decision' for one-day trade decision, "
            "'backtest' for full historical run, "
            "'close-positions' for end-of-day position closing (引け時反対売買)."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=ProductionConfig.start_date,
        help="Backtest start date in YYYY-MM-DD format (default: 2015-01-01).",
    )
    parser.add_argument(
        "--output-root",
        default=get_default_results_root(),
        help="Directory root where production outputs are written.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag. If omitted, a timestamp is used.",
    )
    parser.add_argument(
        "--skip-chart",
        action="store_true",
        help="Skip cumulative return and drawdown chart generation.",
    )
    parser.add_argument(
        "--trade-date",
        default=None,
        help="Trade date in YYYY-MM-DD for decision mode (default: today).",
    )
    parser.add_argument(
        "--jp-opens-csv",
        default=None,
        help="CSV file with TOPIX-17 opens (columns: ticker, open_price).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000000.0,
        help=(
            "Equity capital in JPY for position sizing (default: 1000000). "
            "Market-neutral sizing assumes margin: long and short notionals can "
            "be allocated simultaneously (default gross cap: 3.0)."
        ),
    )
    parser.add_argument(
        "--capital-from-wallet",
        action="store_true",
        help=(
            "Use cash account wallet balance from kabu API for position sizing "
            "(requires --api-enable)."
        ),
    )
    parser.add_argument(
        "--api-enable",
        action="store_true",
        help="Enable kabuステーション API for live order submission (decision mode only).",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=(
            "kabuステーション API URL (e.g., http://localhost:18080). "
            "If not provided, uses KABU_API_URL environment variable."
        ),
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help=(
            "kabuステーション API token. "
            "If not provided, uses KABU_API_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Simulate API calls without actually submitting orders (test mode).",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help=(
            "Use precomputed cache for faster decision-making. "
            "Skips heavy correlation/eigen-decomposition computation."
        ),
    )
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help="Automatically close all positions at end-of-day (引け時反対売買を自動実行).",
    )
    parser.add_argument(
        "--auto-close-time",
        default="14:50",
        help="Time to auto-close positions (HH:MM format, default: 14:50).",
    )
    parser.add_argument(
        "--close-position-order",
        type=int,
        default=0,
        help=(
            "Close position order priority for credit repayment (ClosePositionOrder: 0-7). "
            "Used for close-positions/auto-close."
        ),
    )
    parser.add_argument(
        "--google-opens",
        action="store_true",
        help="Fetch JP open prices from Google Finance (used when API is not enabled).",
    )
    parser.add_argument(
        "--text-output",
        action="store_true",
        help="Output trade orders in text format to the console.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help=(
            "スリッページコスト（片道、basis points）。"
            "内部で往復コスト = 2 × slippage_bps × gross_exposure として適用。"
            "未指定時はデフォルト値 (5.0 bps = 0.05％片道) を使用。"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    if args.mode == "decision":
        if args.capital_from_wallet and not args.api_enable:
            raise ValueError("--capital-from-wallet requires --api-enable")

        if args.fast_mode:
            # ---- Fast mode: precomputed cache + broker API (no yfinance) ----
            logger.info("=== FAST MODE ENABLED (No yfinance) ===")
            if not args.api_enable:
                raise ValueError(
                    "FAST MODE requires --api-enable to fetch US returns and JP opens "
                    "from kabuステーション API (no yfinance dependency)."
                )

            config = ProductionConfig(start_date=args.start_date)
            output_dir = build_output_dir(
                args.output_root,
                args.run_tag,
                run_name="production_decision_fast",
            )
            cache_path = os.path.join(
                args.output_root, ".cache", "strategy_cache.npz"
            )

            api_client = None
            try:
                api_client = build_api_client(
                    args.api_url, args.api_token, args.api_dry_run
                )

                t_trade = (
                    pd.to_datetime(args.trade_date).normalize()
                    if args.trade_date is not None
                    else pd.Timestamp.now().normalize()
                )

                logger.info("[1/3] Fetching US ETF returns from kabu API...")
                us_returns_today = fetch_us_returns_from_api(
                    api_client, args.output_root
                )

                logger.info("[2/3] Fetching JP opens...")
                manual_opens, topix_open = fetch_jp_opens_for_fast_mode(
                    api_client=api_client,
                    config=config,
                    jp_opens_csv=args.jp_opens_csv,
                    google_opens=args.google_opens,
                )

                # Build or validate precomputed strategy cache
                with _exclusive_lock(cache_path + ".lock"):
                    if not _is_cache_valid(cache_path, config=config):
                        logger.info(
                            "[FAST MODE] Building precomputed cache from local cache..."
                        )
                        df_exec = _load_df_exec_from_local_cache()
                        build_precomputed_cache(config, df_exec, cache_path)
                        logger.info("[FAST MODE] Cache built: %s", cache_path)
                    else:
                        logger.info("[FAST MODE] Using existing cache: %s", cache_path)

                # Gap override from local jp_close cache
                jp_close = load_jp_close_from_cache()
                jp_close.index = _normalize_to_tokyo_date(jp_close.index)
                gap_override = _compute_gap_from_jp_close(
                    jp_close, t_trade, manual_opens
                )
                topix_night_override = None
                if topix_open is not None:
                    topix_night_override = _compute_topix_night_override(
                        jp_close, t_trade, topix_open
                    )

                logger.info("[3/3] Generating trade decision (FAST path)...")
                max_capital = args.capital
                if args.capital_from_wallet:
                    max_capital = resolve_wallet_capital(api_client)

                result_path = run_decision_fast(
                    config=config,
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
                )
                logger.info("Fast decision completed. Output: %s", result_path)

                if args.auto_close:
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

    elif args.mode == "close-positions":
        # ---- Close-positions mode ----
        logger.info("=== CLOSE-POSITIONS MODE ===")
        if not args.api_enable:
            raise ValueError(
                "CLOSE-POSITIONS mode requires --api-enable to fetch positions "
                "and submit close orders."
            )
        output_dir = build_output_dir(
            args.output_root,
            args.run_tag,
            run_name="production_close_positions",
        )
        api_client = None
        try:
            api_client = build_api_client(
                args.api_url, args.api_token, args.api_dry_run
            )
            close_summary = close_all_positions(
                api_client=api_client,
                output_dir=output_dir,
                dry_run=args.api_dry_run,
                margin_trade_type=KABU_API_CONFIG.get("margin_trade_type", 3),
                account_type=KABU_API_CONFIG.get("account_type", 4),
                close_position_order=args.close_position_order,
            )
            logger.info(
                "Close-positions completed. Positions closed: %d",
                close_summary.get("close_orders_count", 0),
            )
        finally:
            if api_client is not None:
                api_client.close()

    else:
        # ---- Backtest mode ----
        # slippage_bpsオプション指定時は ProductionConfig を上書きして渡す
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Backward-compatible re-exports (for external code importing production.py)
# ---------------------------------------------------------------------------
from runner.config import ProductionConfig  # noqa: F811 (intentional re-export)
from runner.decision import run_decision  # noqa: F811
from runner.backtest import run_production  # noqa: F811
from runner.close import close_all_positions, wait_and_auto_close  # noqa: F811
