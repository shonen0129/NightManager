"""CLI runner: entry point for the production system."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from config import PRODUCTION_DEFAULTS, KABU_API_CONFIG, get_validated_kabu_config
from data_loader import JP_TICKERS, TOPIX_TICKER, download_data
from domain.models.types import (
    RiskConfig,
    StrategyConfig,
)
from results_format import create_results_output_dir, get_default_results_root
from app.workflow import TradeWorkflow
from services.market_data import (
    compute_topix_night_override as _compute_topix_night_override,
    validate_topix_open as _validate_topix_open,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production runner for the lead-lag market-neutral strategy.",
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "decision", "close-positions"],
        default="decision",
        help="Run mode: 'decision' for one-day trade decision, 'backtest' for full historical run, "
        "'close-positions' for end-of-day position closing.",
    )
    parser.add_argument(
        "--start-date",
        default=PRODUCTION_DEFAULTS["start_date"],
        help="Backtest start date in YYYY-MM-DD format.",
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
        help="Available capital in JPY for position sizing.",
    )
    parser.add_argument(
        "--api-enable",
        action="store_true",
        help="Enable kabu API for live order submission.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="kabu API URL.",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="kabu API token.",
    )
    parser.add_argument(
        "--api-dry-run",
        action="store_true",
        help="Simulate API calls without submitting orders.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use precomputed cache for faster decision-making.",
    )
    return parser.parse_args()


def _build_output_dir(
    output_root: str,
    run_tag: Optional[str],
    run_name: str,
) -> str:
    return create_results_output_dir(
        run_name=run_name,
        output_root=output_root,
        run_tag=run_tag,
        manifest_extra={"entry_point": "app/runner.py"},
    )


def _load_opens_from_csv(csv_path: str) -> dict:
    """Load TOPIX-17 opens from CSV."""
    df = pd.read_csv(csv_path, dtype={"ticker": str, "open_price": float})
    if len(df) < len(JP_TICKERS):
        raise ValueError(
            f"CSV must have at least {len(JP_TICKERS)} rows, got {len(df)}"
        )
    parsed = {}
    for _, row in df.iterrows():
        tk = row["ticker"].strip()
        if tk not in JP_TICKERS and tk != TOPIX_TICKER:
            raise ValueError(f"Unknown ticker in CSV: {tk}")
        parsed[tk] = float(row["open_price"])

    missing = [tk for tk in JP_TICKERS if tk not in parsed]
    if missing:
        raise ValueError(f"Missing opens in CSV for: {', '.join(missing)}")
    return parsed


def _compute_gap_override(
    data: dict, trade_date: pd.Timestamp, manual_opens: dict
) -> np.ndarray:
    """Compute gap override from manual opens."""
    jp_close = data["jp_close"].copy()
    jp_close.index = pd.to_datetime(jp_close.index).tz_localize(None).normalize()

    gaps = []
    for tk in JP_TICKERS:
        series = jp_close[tk].dropna()
        prev = series[series.index < trade_date]
        if len(prev) == 0:
            raise ValueError(
                f"Previous close not found for {tk} before {trade_date.date()}"
            )
        prev_close = float(prev.iloc[-1])
        open_price = float(manual_opens[tk])
        if prev_close <= 0:
            raise ValueError(f"Invalid previous close for {tk}: {prev_close}")
        gaps.append(open_price / prev_close - 1.0)

    return np.array(gaps, dtype=float)


def _build_strategy_config(args: argparse.Namespace) -> StrategyConfig:
    """Build StrategyConfig from CLI args."""
    return StrategyConfig(
        k=PRODUCTION_DEFAULTS["k"],
        lambda_reg=PRODUCTION_DEFAULTS["lambda_reg"],
        q=PRODUCTION_DEFAULTS["q"],
        weight_mode=PRODUCTION_DEFAULTS["weight_mode"],
        dispersion_filter=PRODUCTION_DEFAULTS["dispersion_filter"],
        dispersion_metric=PRODUCTION_DEFAULTS["dispersion_metric"],
        v3_mode=PRODUCTION_DEFAULTS["v3_mode"],
        ewma_half_life=PRODUCTION_DEFAULTS["ewma_half_life"],
        lambda_lw=PRODUCTION_DEFAULTS["lambda_lw"],
        lw_target=PRODUCTION_DEFAULTS["lw_target"],
        corr_window=PRODUCTION_DEFAULTS["corr_window"],
        include_v4_prior=PRODUCTION_DEFAULTS["include_v4_prior"],
        signal_mode=PRODUCTION_DEFAULTS["signal_mode"],
        gap_open_coef=PRODUCTION_DEFAULTS["gap_open_coef"],
        topix_beta_coef=PRODUCTION_DEFAULTS["topix_beta_coef"],
        beta_window=PRODUCTION_DEFAULTS["beta_window"],
        gamma=PRODUCTION_DEFAULTS.get("gamma", 0.5),
    )


def _build_risk_config() -> RiskConfig:
    """Build RiskConfig with defaults."""
    return RiskConfig()


def run_decision_mode(args: argparse.Namespace) -> str:
    """Run in decision mode."""
    config = _build_strategy_config(args)
    risk_config = _build_risk_config()
    output_dir = _build_output_dir(
        args.output_root,
        args.run_tag,
        run_name="app_runner_decision",
    )

    # Build API client if enabled
    api_client = None
    if args.api_enable:
        from kabu_client import KabuClient, KabuConfig, issue_api_token

        final_api_url = args.api_url if args.api_url else KABU_API_CONFIG.get("api_url")
        final_api_token = (
            args.api_token if args.api_token else KABU_API_CONFIG.get("api_token")
        )
        request_timeout = KABU_API_CONFIG.get("request_timeout", 10)

        if not final_api_url:
            try:
                validated = get_validated_kabu_config()
                final_api_url = validated["api_url"]
            except ValueError as e:
                logger.error(f"Configuration validation failed: {e}")
                raise

        # If token is not provided, issue one via /token using KABU_API_PASSWORD.
        if not final_api_token:
            api_password = os.environ.get("KABU_API_PASSWORD", "")
            if api_password:
                logger.info(
                    "[API] KABU_API_TOKEN is empty. Issuing token via /token..."
                )
                try:
                    final_api_token = issue_api_token(
                        final_api_url,
                        api_password,
                        request_timeout=request_timeout,
                    )
                    os.environ["KABU_API_TOKEN"] = final_api_token
                    KABU_API_CONFIG["api_token"] = final_api_token
                    logger.info("[API] Token issued and assigned to KABU_API_TOKEN")
                except Exception as e:
                    logger.error(f"Failed to issue API token: {e}")
                    raise RuntimeError(
                        "Failed to issue API token via /token. "
                        "Verify KABU_API_URL and KABU_API_PASSWORD."
                    ) from e

        if not final_api_token:
            try:
                get_validated_kabu_config()
            except ValueError as e:
                logger.error(f"Configuration validation failed: {e}")
                raise
            raise ValueError(
                "KABU_API_TOKEN is required. "
                "Set KABU_API_TOKEN directly, or set KABU_API_PASSWORD "
                "to auto-issue token via API."
            )

        kabu_config = KabuConfig(
            api_url=final_api_url,
            api_token=final_api_token,
            request_timeout=request_timeout,
        )
        api_client = KabuClient(kabu_config)

        logger.info("[API] Checking API connectivity...")
        if not api_client.health_check():
            if args.api_dry_run:
                logger.warning(
                    "[API] Health check failed, but continuing in dry-run mode..."
                )
            else:
                raise RuntimeError("Failed to connect to kabu API.")
        else:
            logger.info("[API] Connection successful")

    # Determine trade date
    t_trade = (
        pd.to_datetime(args.trade_date).normalize()
        if args.trade_date is not None
        else pd.Timestamp.now().normalize()
    )

    # Load data
    logger.info("[1/5] Loading market data...")
    data = download_data(beta_window=config.beta_window)

    # Obtain JP open prices
    if args.jp_opens_csv is not None:
        logger.info("[2/5] Loading JP opens from CSV...")
        manual_opens = _load_opens_from_csv(args.jp_opens_csv)
    elif api_client is not None:
        logger.info("[2/5] Fetching JP opens from kabu API...")
        manual_opens = api_client.fetch_jp_open_prices(
            JP_TICKERS + [TOPIX_TICKER]
        )
    else:
        raise ValueError("--jp-opens-csv is required when API is not enabled.")

    # Compute gap override
    gap_override = _compute_gap_override(data, t_trade, manual_opens)
    topix_night_override = None
    if config.signal_mode == "gap_residual":
        topix_open = _validate_topix_open(manual_opens)
        topix_night_override = _compute_topix_night_override(
            data["jp_close"],
            t_trade,
            topix_open,
        )

    # Run workflow
    workflow = TradeWorkflow(
        strategy_config=config,
        risk_config=risk_config,
        output_dir=output_dir,
    )

    result = workflow.run_decision(
        trade_date=t_trade,
        open_prices=manual_opens,
        max_capital=args.capital,
        api_client=api_client,
        dry_run=args.api_dry_run,
        gap_override=gap_override,
        topix_night_override=topix_night_override,
    )

    # Save decision output
    decision_df = result["decision_df"]
    out_path = os.path.join(output_dir, f"decision_{t_trade.strftime('%Y%m%d')}.csv")
    decision_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Decision saved: {out_path}")

    # Log summary
    buy_count = (decision_df["action"] == "BUY").sum()
    sell_count = (decision_df["action"] == "SELL").sum()
    hold_count = (decision_df["action"] == "HOLD").sum()
    logger.info(f"Positions: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
    logger.info(f"Available capital: {args.capital:,.0f} JPY")

    risk_report = result["risk_report"]
    logger.info("=== Risk Check ===")
    logger.info(f"Target net exposure: {risk_report.target_net_exposure:.4f}")
    logger.info(f"Target gross exposure: {risk_report.target_gross_exposure:.4f}")

    if risk_report.var_es.available:
        logger.info(
            f"VaR/ES(99%,250d): VaR={risk_report.var_es.var_loss:.4%}, "
            f"ES={risk_report.var_es.es_loss:.4%}"
        )

    for msg in risk_report.warning_breaches:
        logger.warning(f"[RISK-WARNING] {msg}")
    for msg in risk_report.stop_breaches:
        logger.error(f"[RISK-STOP] {msg}")

    return output_dir


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    if args.mode == "decision":
        output_dir = run_decision_mode(args)
        print(f"Output directory: {output_dir}")
    elif args.mode == "backtest":
        # Delegate to backtest runner
        from backtest.runner import main as backtest_main

        backtest_main()
    else:
        print(f"Mode '{args.mode}' not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
