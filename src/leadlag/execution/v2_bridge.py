"""V2 execution bridge: connect the unified ProductionRunner to broker order submission.

This module bridges the V2 production pipeline (Residual-BLPX-RA v2) with the
existing broker execution layer (``execute_post_decision_flow``).  It builds the
unified ``df_exec`` once and runs ``ProductionRunner`` to obtain weights,
then converts those weights into a decision dict compatible with the broker
submission flow.

Flow:
  1. Load V2 config YAML
  2. Build ``df_exec`` (historical data + today's placeholders)
  3. Fetch JP 09:10 current prices via broker API / date-scoped cache
  4. Run ``ProductionRunner`` to obtain w_final, scores, etc.
  5. Write V2 production files
  6. Convert V2 weights into a decision dict
  7. Execute order submission via the existing infrastructure
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.broker.tachibana.session_cache import load_current_prices_cache
from leadlag.config.paths import execution_state_path, project_root
from leadlag.config.paths import live as live_path
from leadlag.config.paths import results as results_path
from leadlag.config.schemas import AppConfig
from leadlag.data import adr_features as adr_data
from leadlag.data import macro as macro_data
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.quote_snapshot import FrozenQuoteSnapshot, load_frozen_quote_snapshot
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.domain.inputs import DecisionInputs, HistoricalInputs
from leadlag.execution.account_risk import (
    AccountRiskPreflight,
    AccountRiskSnapshot,
    AccountRiskSnapshotError,
)
from leadlag.execution.backtest import _load_df_exec
from leadlag.execution.broker_ops import (
    build_api_client,
    fetch_current_positions,
    resolve_wallet_capital,
)
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.job_guard import execution_lease
from leadlag.execution.output_ops import build_output_dir
from leadlag.execution.post_decision import execute_post_decision_flow
from leadlag.execution.runtime_manifest import (
    build_decision_manifest,
    update_decision_manifest,
)
from leadlag.execution.state_store import ExecutionStateStore
from leadlag.execution.var_history import get_hist_returns_for_risk as _get_hist_returns_for_risk
from leadlag.models.v2.pit import load_pit_ir_history
from leadlag.reporting.ml_overlay_shadow import append_ml_overlay_shadow
from leadlag.reporting.production_v2_writer import write_production_files
from leadlag.runner.production import ProductionRunner
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurrentPricePreflight:
    """Selected, run-owned price input before it is paired with prior closes."""

    prices: Mapping[str, float]
    source: Literal["frozen_quote", "broker_current", "previous_close_placeholder"]
    frozen_snapshot: FrozenQuoteSnapshot | None = None

    def __post_init__(self) -> None:
        if (self.source == "frozen_quote") != (self.frozen_snapshot is not None):
            raise ValueError("frozen quote source and snapshot must be supplied together")
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))


@dataclass(frozen=True)
class QuotePreflight:
    """Validated price and market snapshot contract for one decision run."""

    current_prices: Mapping[str, float]
    data_lake: PITDataLake
    market_snapshot: MarketSnapshot
    decision_as_of: pd.Timestamp
    price_input: CurrentPricePreflight

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_prices", MappingProxyType(dict(self.current_prices)))

    @property
    def requires_actual_account_risk(self) -> bool:
        return self.price_input.frozen_snapshot is not None


def _decision_as_of(trade_date: pd.Timestamp | str) -> pd.Timestamp:
    """Return the live bridge's 09:10 JST decision cutoff for a trade date."""
    return normalize_jst_date(trade_date) + pd.Timedelta(hours=9, minutes=10)


def _resolve_gap_dir(
    gap_input_dir: str | Path | None,
    app_config: AppConfig,
    root: Path,
) -> Path | None:
    """Resolve the V2 gap input directory from CLI or config."""
    gap_dir: Path | None = None
    if gap_input_dir is not None:
        gap_dir = Path(gap_input_dir)
        if not gap_dir.is_absolute():
            gap_dir = root / gap_dir
    else:
        default_gap = app_config.gap_distribution_dir or app_config.v2.gap_input_dir
        if default_gap:
            gap_dir = Path(default_gap)
            if not gap_dir.is_absolute():
                gap_dir = root / gap_dir

    if gap_dir is not None and not gap_dir.exists():
        logger.warning("Gap input dir not found: %s. Will use flat position.", gap_dir)
        gap_dir = None
    return gap_dir


def _resolve_trade_date(
    trade_date: str | None,
    live_dir: Path,
) -> str:
    """Resolve the trade date, including the ``latest`` shortcut.

    - ``latest`` reads ``live_dir/latest_weights.csv`` and uses its ``trade_date``.
    - If the CSV is missing, ``latest`` falls back to today (or the previous
      trading day if today is a market holiday).
    - ``None`` defaults to today.
    - Future dates are rejected to prevent accidentally running for a date
      that has not yet occurred.
    """
    today = normalize_jst_date(pd.Timestamp.now(tz="Asia/Tokyo"))

    def _assert_not_future(date_value: object) -> str:
        parsed = normalize_jst_date(date_value)
        if parsed > today:
            raise ValueError(
                f"trade_date {date_value} is in the future (today: {today.date()})"
            )
        return cast(str, parsed.strftime("%Y-%m-%d"))

    if trade_date == "latest":
        latest_file = live_dir / "latest_weights.csv"
        if latest_file.exists():
            try:
                df_tmp = pd.read_csv(latest_file)
                raw_date = df_tmp.iloc[0]["trade_date"]
                if pd.isna(raw_date):
                    raise ValueError("trade_date is NaN")
                resolved = _assert_not_future(raw_date)
                logger.info("Resolved latest trade date from %s: %s", latest_file, resolved)
                return resolved
            except Exception as e:
                logger.error("Failed to parse latest trade_date from %s: %s", latest_file, e)
                raise
        else:
            from leadlag.core.market_calendar import is_market_closed, previous_trading_day

            if is_market_closed(today):
                resolved = str(previous_trading_day(today).strftime("%Y-%m-%d"))
                logger.warning(
                    "latest_weights.csv not found and today is a non-trading day; using %s",
                    resolved,
                )
            else:
                resolved = str(today.strftime("%Y-%m-%d"))
                logger.warning("latest_weights.csv not found; using today: %s", resolved)
            return _assert_not_future(resolved)

    if trade_date is None:
        from leadlag.core.market_calendar import is_market_closed, previous_trading_day

        if is_market_closed(today):
            resolved = str(previous_trading_day(today).strftime("%Y-%m-%d"))
            logger.info(
                "No trade date provided and today is a non-trading day; using %s",
                resolved,
            )
        else:
            resolved = str(today.strftime("%Y-%m-%d"))
            logger.info("No trade date provided; using today: %s", resolved)
        return _assert_not_future(resolved)

    return _assert_not_future(trade_date)


def _resolve_current_prices(
    app_config: AppConfig,
    api_client: BrokerClient,
    jp_opens_csv: str | None,
    google_opens: bool,
) -> dict[str, float]:
    """Return prices observed at the decision time (not daily opens)."""
    if api_client is None:
        raise ValueError("A broker API client is required for 09:10 current prices")
    tickers = JP_TICKERS + [TOPIX_TICKER]
    prices = api_client.fetch_current_prices(tickers, allow_missing=True)
    current_prices: dict[str, float] = {}
    for tk, value in prices.items():
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float) and value_float > 0.0:
            current_prices[tk] = value_float
    missing = [tk for tk in JP_TICKERS if tk not in current_prices]
    if missing:
        raise ValueError("Missing 09:10 current prices: " + ", ".join(missing))
    return current_prices


def _has_complete_current_prices(prices: dict[str, float]) -> bool:
    """Return whether all JP decision prices are finite and strictly positive."""
    for tk in JP_TICKERS:
        if tk not in prices:
            return False
        try:
            value = float(prices[tk])
        except (TypeError, ValueError):
            return False
        if not np.isfinite(value) or value <= 0.0:
            return False
    return True


def _resolve_current_price_preflight(
    app_config: AppConfig,
    api_client: BrokerClient | None,
    trade_date: pd.Timestamp,
    live_quote_snapshot: FrozenQuoteSnapshot | None,
    *,
    api_enable: bool,
    jp_opens_csv: str | None,
    google_opens: bool,
) -> CurrentPricePreflight:
    """Choose one current-price source without reinterpreting it downstream."""
    if live_quote_snapshot is not None:
        logger.info(
            "[2/5] Using frozen quote-mid prices from %s.",
            live_quote_snapshot.snapshot_id,
        )
        return CurrentPricePreflight(
            prices={
                ticker: float(live_quote_snapshot.prices[ticker])
                for ticker in JP_TICKERS
            },
            source="frozen_quote",
            frozen_snapshot=live_quote_snapshot,
        )

    if not api_enable:
        return CurrentPricePreflight(prices={}, source="previous_close_placeholder")
    if api_client is None:
        raise ValueError("An API client is required for broker current prices")

    trade_date_str = trade_date.strftime("%Y%m%d")
    cached_prices = (
        load_current_prices_cache(trade_date_str)
        if app_config.broker_provider == "tachibana"
        else None
    )
    current_prices: dict[str, float] = {}
    if cached_prices is not None:
        cached_jp, topix_current = cached_prices
        current_prices = dict(cached_jp)
        if topix_current is not None:
            current_prices[TOPIX_TICKER] = float(topix_current)
        if _has_complete_current_prices(current_prices):
            logger.info("[2/5] Using cached 09:10 current prices.")
        else:
            logger.info(
                "[2/5] Current-price cache incomplete or invalid; fetching from API..."
            )
            current_prices = _resolve_current_prices(
                app_config, api_client, jp_opens_csv, google_opens
            )
    else:
        logger.info("[2/5] Fetching 09:10 current prices...")
        current_prices = _resolve_current_prices(
            app_config, api_client, jp_opens_csv, google_opens
        )
    return CurrentPricePreflight(prices=current_prices, source="broker_current")


def _build_quote_preflight(
    df_exec: pd.DataFrame,
    trade_date: pd.Timestamp,
    trade_date_text: str,
    decision_as_of: pd.Timestamp,
    price_input: CurrentPricePreflight,
    *,
    api_enable: bool,
    api_client: BrokerClient | None,
) -> QuotePreflight:
    """Pair selected prices with the same-day PIT close and validate once."""
    lake = PITDataLake(df_exec)
    if trade_date not in lake.history_frame().index:
        # A previous row is never a valid substitute for today's trade.  It
        # would combine yesterday's signal/gap with today's prices and could
        # replay an old order plan.  Fail closed for both dry-run and live
        # paths; callers can explicitly request a historical date instead.
        raise RuntimeError(
            f"Requested trade_date {trade_date_text} is not available in df_exec; "
            "refusing stale PIT fallback."
        )

    lake_snapshot = lake.get_snapshot(decision_as_of)
    snapshot_prev_closes = lake_snapshot.prev_closes
    current_prices = dict(price_input.prices)
    if not api_enable:
        # A no-API decision uses prior closes only as zero-gap placeholders.
        current_prices = {
            ticker: float(snapshot_prev_closes[ticker])
            for ticker in JP_TICKERS
            if ticker in snapshot_prev_closes
            and np.isfinite(snapshot_prev_closes[ticker])
            and snapshot_prev_closes[ticker] > 0.0
        }

    api_current_prices: dict[str, float] = {
        ticker: float(current_prices[ticker])
        for ticker in JP_TICKERS
        if ticker in current_prices
        and np.isfinite(current_prices[ticker])
        and current_prices[ticker] > 0.0
    }
    jp_gap_api = np.zeros(len(JP_TICKERS), dtype=float)
    for index, ticker in enumerate(JP_TICKERS):
        price = api_current_prices.get(ticker)
        prior_close = snapshot_prev_closes.get(ticker)
        if price is not None and prior_close is not None and prior_close > 0.0:
            jp_gap_api[index] = price / prior_close - 1.0

    topix_night_return = lake_snapshot.topix_night_return
    if price_input.frozen_snapshot is not None:
        topix_prev_close = snapshot_prev_closes.get(TOPIX_TICKER)
        topix_price = price_input.frozen_snapshot.prices[TOPIX_TICKER]
        if (
            topix_prev_close is None
            or not np.isfinite(topix_prev_close)
            or topix_prev_close <= 0.0
        ):
            raise RuntimeError(
                "Frozen quote snapshot cannot be paired with a valid prior TOPIX close"
            )
        topix_night_return = float(topix_price) / float(topix_prev_close) - 1.0

    frozen_snapshot = price_input.frozen_snapshot
    snapshot = MarketSnapshot(
        as_of=lake_snapshot.as_of,
        trade_date=lake_snapshot.trade_date,
        us_returns=lake_snapshot.us_returns,
        jp_gap_returns=jp_gap_api,
        jp_betas=lake_snapshot.jp_betas,
        topix_night_return=topix_night_return,
        current_prices=api_current_prices,
        prev_closes=snapshot_prev_closes,
        price_sources={
            ticker: (
                "tachibana:CLMMfdsGetMarketPrice:quote_mid"
                if frozen_snapshot is not None
                else "broker_current" if api_enable else "previous_close_placeholder"
            )
            for ticker in api_current_prices
        },
        price_observed_at=(
            {ticker: frozen_snapshot.observed_at[ticker] for ticker in JP_TICKERS}
            if frozen_snapshot is not None
            else {}
        ),
        quote_snapshot_id=(
            frozen_snapshot.snapshot_id if frozen_snapshot is not None else None
        ),
    )
    is_valid, snapshot_errors = snapshot.validate()
    if not is_valid:
        logger.error("MarketSnapshot validation failed: %s", snapshot_errors)
        if api_client is not None:
            try:
                api_client.close()
            except Exception:
                pass
        raise RuntimeError(
            f"MarketSnapshot validation failed: {snapshot_errors}. "
            "Aborting to avoid trading on bogus data."
        )

    return QuotePreflight(
        current_prices=current_prices,
        data_lake=lake,
        market_snapshot=snapshot,
        decision_as_of=decision_as_of,
        price_input=price_input,
    )


def _preflight_account_risk(
    project_root_path: Path,
    trade_date: pd.Timestamp,
    account_key: str,
    quote_preflight: QuotePreflight,
) -> AccountRiskPreflight:
    """Load the required live account evidence as one explicit typed result."""
    if not quote_preflight.requires_actual_account_risk:
        return AccountRiskPreflight.not_required()

    configured_path = os.environ.get("LEADLAG_ACCOUNT_RISK_SNAPSHOT")
    snapshot_path = (
        Path(configured_path)
        if configured_path
        else project_root_path / "var/live/pipeline_data/account_risk/latest.json"
    )
    if not snapshot_path.is_absolute():
        snapshot_path = project_root_path / snapshot_path
    try:
        snapshot = AccountRiskSnapshot.load(
            snapshot_path,
            trade_date=trade_date.strftime("%Y-%m-%d"),
            decision_as_of=quote_preflight.decision_as_of,
            account_key=account_key,
        )
    except AccountRiskSnapshotError as exc:
        logger.error("Actual-account loss evidence unavailable: %s", exc)
        return AccountRiskPreflight.unavailable(exc)
    return AccountRiskPreflight.verified(snapshot)


def _build_run_owned_decision_inputs(
    app_config: AppConfig,
    df_exec: pd.DataFrame,
    trade_date: pd.Timestamp,
    trade_date_text: str,
    gap_dir: Path | None,
    quote_preflight: QuotePreflight,
) -> DecisionInputs:
    """Load date-bounded history and bind it to the validated quote snapshot."""
    effective_trade_date = trade_date_text
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    macro_prices = None
    if app_config.v2.macro_kappa_enabled or app_config.v2.macro_direction_enabled:
        try:
            macro_prices = macro_data.load_macro_prices(
                start=df_exec.index.min().strftime("%Y-%m-%d"),
                end=(df_exec.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                period="max",
            )
            # Keep provider rows bounded by the run's decision date.
            macro_prices = macro_prices.loc[macro_prices.index <= effective_trade_date].copy()
        except Exception as exc:
            logger.warning("Failed to load run-owned macro prices: %s", exc)

    adr_features = None
    if app_config.v2.ml_overlay_enabled:
        try:
            # Keep the complete run-owned artifact; the overlay applies its
            # own date and staleness validation symmetrically with backtests.
            adr_features = adr_data.load_adr_features()
            if adr_features is not None:
                adr_features = adr_features.loc[
                    adr_features.index <= effective_trade_date
                ].copy()
        except Exception as exc:
            logger.warning("Failed to load run-owned ADR features: %s", exc)

    pit_ir_history = None
    pit_history_trade_dates = None
    if gap_dir is not None:
        pit_ir_history, _pit_alerts, pit_history_trade_dates = load_pit_ir_history(
            gap_dir, trade_date_text
        )
    rank_reversal_signals = None
    if app_config.v2.cs_overlay_enabled:
        rank_reversal_signals = load_rank_reversal_frame(
            gap_dir,
            [trade_date],
            file_pattern=app_config.v2.cs_rank_reversal_file_pattern,
        )

    historical_observed_at_by_date = {
        normalize_jst_date(date).strftime("%Y-%m-%d"): {
            "open_910_returns": f"{normalize_jst_date(date).date()} 09:10",
            "macro_prices": f"{normalize_jst_date(date).date()} 09:00",
            "adr_features": f"{normalize_jst_date(date).date()} 09:00",
            "rank_reversal_signals": f"{normalize_jst_date(date).date()} 09:00",
            "pit_ir_history": f"{normalize_jst_date(date).date()} 09:10",
        }
        for date in df_exec.index
    }
    historical_inputs = HistoricalInputs(
        df_exec,
        source="v2_bridge_live",
        open_910_returns=open_910_returns,
        macro_prices=macro_prices,
        adr_features_frame=adr_features,
        pit_ir_history=pit_ir_history,
        pit_history_trade_dates=pit_history_trade_dates,
        rank_reversal_signals=rank_reversal_signals,
        observed_at={
            "open_910_returns": f"{effective_trade_date} 09:10",
            "macro_prices": f"{effective_trade_date} 09:00",
            "adr_features": f"{effective_trade_date} 09:00",
            "rank_reversal_signals": f"{effective_trade_date} 09:00",
            "pit_ir_history": f"{effective_trade_date} 09:10",
        },
        observed_at_by_date=historical_observed_at_by_date,
    )
    snapshot = quote_preflight.market_snapshot
    return quote_preflight.data_lake.build_decision_inputs(
        quote_preflight.decision_as_of,
        snapshot=snapshot,
        gap_input_dir=gap_dir,
        use_file_cache=True,
        source="v2_bridge_live",
        historical=historical_inputs,
        observed_at={
            "us_returns": f"{effective_trade_date} 09:00",
            "jp_gap_returns": f"{effective_trade_date} 09:10",
            "jp_betas": f"{effective_trade_date} 09:10",
            "topix_night_return": f"{effective_trade_date} 09:10",
            "current_prices": f"{effective_trade_date} 09:10",
            "prev_closes": f"{effective_trade_date} 09:10",
        },
    )


def run_v2_decision(
    config_path: str | Path,
    gap_input_dir: str | Path | None = None,
    live_dir: str | Path = live_path("production_residual_blpx"),
    trade_date: str | None = None,
    api_enable: bool = False,
    api_dry_run: bool = False,
    capital_from_wallet: bool = False,
    text_output: bool = False,
    output_root: str = str(results_path()),
    jp_opens_csv: str | None = None,
    google_opens: bool = False,
    max_capital: float | None = None,
    api_url: str | None = None,
    api_token: str | None = None,
    run_tag: str | None = None,
    dry_run: bool = False,
    ml_overlay_shadow_dir: str | Path | None = None,
    shadow_only: bool = False,
) -> str:
    """Run V2 production decision and optionally submit orders via broker API.

    Args:
        config_path: Path to V2 production YAML config.
        gap_input_dir: Directory containing mu_gap/omega_gap .npy files.
        live_dir: Live output directory for V2 artifacts.
        trade_date: Trade date string (YYYY-MM-DD or ``latest``). Defaults to today.
        api_enable: If True, submit orders to broker API.
        api_dry_run: If True, simulate order submission.
        capital_from_wallet: If True, use wallet balance as max capital.
        text_output: If True, print text order summary.
        output_root: Root directory for decision output.
        jp_opens_csv: Path to JP opens CSV (fallback if API unavailable).
        google_opens: If True, use Google Sheets for JP opens.
        max_capital: Default max capital in JPY. Defaults to 300,000.0.
        api_url: Optional broker API URL override.
        api_token: Optional broker API token override.
        run_tag: Optional run tag for output directory.
        dry_run: If True, calculate weights but do not write files or submit orders.
            Also forces ``api_dry_run=True`` when ``api_enable=True`` to avoid
            live API calls.
        ml_overlay_shadow_dir: Optional append-only paired ML-on/off shadow output.
            Recorded only when live 09:10 prices are enabled and not dry-run.
        shadow_only: Return after recording paired decisions, before production
            file writes, position queries, or order submission.

    Returns:
        Path to the decision output CSV, or a dry-run summary path.
    """
    ROOT = project_root()

    # Resolve config
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    logger.info("Loading V2 config: %s", config_path)
    app_config = load_config_from_yaml(str(config_path))

    # Resolve live dir before trade date, because ``latest`` reads from it.
    live_path = Path(live_dir)
    if not live_path.is_absolute():
        live_path = ROOT / live_dir

    # Resolve trade date
    trade_date = _resolve_trade_date(trade_date, live_path)
    t_trade = normalize_jst_date(trade_date)
    # Keep the trade-date key at midnight for date/index lookups, but build the
    # market snapshot at the actual decision cutoff.  KnownMarketInputs checks
    # every observed_at timestamp against this value.
    decision_as_of = _decision_as_of(t_trade)
    logger.info("Trade date: %s", trade_date)

    # Resolve gap input dir
    gap_dir = _resolve_gap_dir(gap_input_dir, app_config, ROOT)
    if gap_dir is not None:
        logger.info("Using gap input dir: %s", gap_dir)

    if max_capital is None:
        max_capital = 300000.0  # default fallback

    # Dry-run should never hit a live API: force api_dry_run when live orders
    # are already suppressed, but keep api_enable so price/wallet simulation
    # can still run through a dry-run broker client if requested.
    if dry_run and api_enable and not api_dry_run:
        logger.warning("[DRY-RUN] Forcing api_dry_run=True to avoid live API calls.")
        api_dry_run = True

    live_quote_snapshot: FrozenQuoteSnapshot | None = None
    if api_enable and not api_dry_run and not dry_run:
        quote_dir_value = os.environ.get("LEADLAG_CAPTURE_OUTPUT_DIR")
        quote_dir = Path(quote_dir_value) if quote_dir_value else (
            ROOT / "var/shadow_runs/ml_overlay_value/microstructure"
        )
        if not quote_dir.is_absolute():
            quote_dir = ROOT / quote_dir
        try:
            live_quote_snapshot = load_frozen_quote_snapshot(
                quote_dir,
                trade_date=t_trade.strftime("%Y-%m-%d"),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Live decision requires today's complete, frozen 09:10 bid/ask snapshot; "
                f"no last-price or previous-close substitution is allowed ({exc})."
            ) from exc
        decision_as_of = live_quote_snapshot.as_of.tz_localize(None)
        logger.info(
            "Using frozen quote snapshot %s received at %s",
            live_quote_snapshot.snapshot_id,
            live_quote_snapshot.as_of.isoformat(),
        )

    # Without a real price source the 1000 JPY dummy would create bogus gaps
    # and corrupt output files / ML overlay features. Force dry-run and use
    # previous closes as placeholders (gap = 0) for non-API execution.
    if not api_enable and not dry_run:
        logger.warning(
            "[API] --api-enable not set. Forcing dry_run=True to avoid writing "
            "files with dummy open prices."
        )
        dry_run = True

    if shadow_only:
        if ml_overlay_shadow_dir is None:
            raise ValueError("--shadow-only requires --ml-overlay-shadow-dir")
        if not api_enable or api_dry_run or dry_run:
            raise ValueError("--shadow-only requires live read-only market data; dry-run is not allowed")
        if not app_config.v2.ml_overlay_enabled:
            raise ValueError("--shadow-only requires the production ML overlay to be enabled")

    # --- Step 1: Build df_exec (historical data + today's placeholders) ---
    logger.info("[1/5] Loading/building df_exec...")
    df_exec = _load_df_exec(app_config, data_source="cache")

    # --- Step 2: Select price evidence and build the typed quote preflight ---
    api_client: BrokerClient | None = None
    try:
        if api_enable:
            api_client = build_api_client(
                api_url=api_url,
                api_token=api_token,
                api_dry_run=api_dry_run,
            )
        else:
            logger.info(
                "[2/5] API disabled. Will use previous closes as placeholder open prices."
            )

        price_input = _resolve_current_price_preflight(
            app_config,
            api_client,
            t_trade,
            live_quote_snapshot,
            api_enable=api_enable,
            jp_opens_csv=jp_opens_csv,
            google_opens=google_opens,
        )
        if capital_from_wallet and api_client is not None:
            max_capital = resolve_wallet_capital(api_client)
            logger.info("[CAPITAL] Using wallet balance: %s JPY", f"{max_capital:,.0f}")
    except Exception as e:
        logger.error("[2/5] Failed to fetch opens: %s", e)
        if api_client is not None:
            api_client.close()
        raise

    logger.info("[3/5] Building PIT data lake and as-of market snapshot...")
    quote_preflight = _build_quote_preflight(
        df_exec,
        t_trade,
        trade_date,
        decision_as_of,
        price_input,
        api_enable=api_enable,
        api_client=api_client,
    )
    effective_trade_date = trade_date
    t_effective = t_trade

    # --- Step 4: Generate V2 portfolio ---
    logger.info("[4/5] Generating V2 production portfolio...")
    decision_inputs = _build_run_owned_decision_inputs(
        app_config,
        df_exec,
        t_trade,
        effective_trade_date,
        gap_dir,
        quote_preflight,
    )
    runner = ProductionRunner(app_config)
    result = runner.run(decision_inputs)
    shadow_path: Path | None = None

    if ml_overlay_shadow_dir is not None:
        if api_enable and not api_dry_run and not dry_run:
            overlay_model = getattr(runner.model, "_overlay_model", None)
            overlay_metadata = getattr(overlay_model, "metadata", None)
            try:
                shadow_path = append_ml_overlay_shadow(
                    app_config=app_config,
                    decision_inputs=decision_inputs,
                    ml_enabled_result=result,
                    output_dir=ml_overlay_shadow_dir,
                    overlay_metadata=overlay_metadata,
                    capital_jpy=max_capital,
                    raise_on_baseline_failure=shadow_only,
                )
                logger.info("Paired ML overlay shadow recorded: %s", shadow_path)
            except Exception:
                if shadow_only and api_client is not None:
                    try:
                        api_client.close()
                    finally:
                        api_client = None
                logger.exception(
                    "Paired ML overlay shadow failed; production decision is unchanged."
                )
                if shadow_only:
                    raise
        else:
            logger.warning(
                "ML overlay shadow skipped because the run has no verified live "
                "09:10 price observation (API disabled or dry-run)."
            )

    if shadow_only:
        if shadow_path is None:
            if api_client is not None:
                api_client.close()
                api_client = None
            raise RuntimeError("Shadow-only run completed without writing a paired shadow record")
        if api_client is not None:
            api_client.close()
            api_client = None
        logger.info(
            "Shadow-only decision complete; no production portfolio outputs or orders were created."
        )
        return str(shadow_path)

    write_production_files(effective_trade_date, live_path, result, dry_run=dry_run)

    fallback_used = (
        result.fallback.get("gap_data_missing", False)
        or result.fallback.get("audit_failure", False)
    )
    if fallback_used:
        reason = (
            "gap data missing" if result.fallback.get("gap_data_missing")
            else "audit failure"
        )
        logger.warning("[V2] %s. Flat position (w_final=0) returned.", reason)
    else:
        logger.info(
            "[V2] Portfolio OK. Bin=%s, Mult=%.2f, Gross=%.4f, IR=%.4f",
            result.pit_binning["assigned_bin"],
            result.pit_binning["multiplier"],
            float(np.sum(np.abs(result.w_final))),
            result.summary["predicted_portfolio_ir"],
        )

    out_path: str
    if not dry_run:
        # --- Step 5: Build decision dict for execute_post_decision_flow ---
        logger.info("[5/6] Building decision dict for execution...")
        w_final = result.w_final
        scores = result.scores

        action = np.where(
            w_final > 1e-8, "LONG",
            np.where(w_final < -1e-8, "SHORT", "HOLD"),
        )

        decision = {
            "trade_date": t_effective,
            "input_version_digest": decision_inputs.version.digest,
            "quote_snapshot_id": decision_inputs.known.quote_snapshot_id,
            "tickers": JP_TICKERS,
            "signal": scores,
            "weight": w_final,
            "raw_weight": w_final,
            "scale": 1.0,
            "action": action,
            "sigma_s": 0.0,
            "dispersion_indicator": 0.0,
            "gross_before": float(np.sum(np.abs(w_final))),
            "gross_after": float(np.sum(np.abs(w_final))),
            "gross_adjusted": False,
            "gross_adjustment_factor": 1.0,
        }

        # --- Step 6: Execute post-decision flow ---
        logger.info("[6/6] Executing post-decision flow (risk checks, order submission)...")

        output_dir = build_output_dir(
            output_root,
            run_tag=run_tag,
            run_name="production_decision_v2",
        )
        update_decision_manifest(
            output_dir,
            build_decision_manifest(
                app_config=app_config,
                inputs=decision_inputs,
                result=result,
                config_path=config_path,
                gap_input_dir=gap_dir,
                model=runner.model,
            ),
        )

        current_positions = None
        if api_client is not None:
            try:
                current_positions = fetch_current_positions(api_client)
            except Exception as e:
                # Fail-closed: without current positions we cannot compute a safe
                # delta, and submitting the full target may double the position
                # if positions already exist or the same decision is re-run.
                logger.error("Failed to fetch current positions: %s", e)
                if api_client is not None:
                    try:
                        api_client.close()
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Failed to fetch current positions: {e}. "
                    "Aborting to avoid double-ordering."
                ) from e

        # Keep decision and close jobs mutually exclusive for the account and
        # strategy even when callers bypass the batch guard.  The lease is
        # released after reconciliation (or an exception) and never authorizes
        # a retry of an unresolved broker outcome.
        state_store = ExecutionStateStore(execution_state_path())
        account_key = f"{app_config.broker_provider}:default" if not api_dry_run else f"simulation:{output_dir}"
        account_risk_preflight = _preflight_account_risk(
            ROOT, t_effective, account_key, quote_preflight
        )
        # Batch wrappers hold the same scope for the complete process group.
        # Avoid a nested self-conflict while retaining the in-process lease for
        # direct CLI invocations.
        lease_context = execution_lease(
            state_store, metadata={"trade_date": str(t_effective), "job_type": "decision"},
        )
        with lease_context:
            hist_returns = _get_hist_returns_for_risk(
                    config=app_config.strategy,
                output_root=output_root,
                trade_date=t_effective,
                config_path=config_path,
                gap_input_dir=gap_dir,
                overlay_model=getattr(runner.model, "_overlay_model", None),
            )

            out_path = execute_post_decision_flow(
                decision=decision,
                config=app_config.strategy,
                risk_config=app_config.risk,
                manual_opens=dict(quote_preflight.current_prices),
                max_capital=max_capital,
                hist_returns=hist_returns,
                output_dir=output_dir,
                api_client=api_client,
                text_output=text_output,
                current_positions=current_positions,
                state_store=state_store,
                account_key=account_key,
                strategy_key="production_v2",
                account_risk_preflight=account_risk_preflight,
            )

        logger.info("V2 decision completed. Output: %s", out_path)
    else:
        logger.info("[DRY-RUN] Skipping order submission and file writes.")
        out_path = str(live_path)

    if api_client is not None:
        api_client.close()

    return out_path
