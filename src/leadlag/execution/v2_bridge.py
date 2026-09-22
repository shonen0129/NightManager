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

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from leadlag.broker.base import BrokerClient
from leadlag.broker.tachibana.session_cache import load_current_prices_cache
from leadlag.config.paths import execution_state_path
from leadlag.config.paths import live as live_path
from leadlag.config.paths import results as results_path
from leadlag.config.schemas import AppConfig
from leadlag.data import adr_features as adr_data
from leadlag.data import macro as macro_data
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.rank_reversal import load_rank_reversal_frame
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.domain.inputs import HistoricalInputs
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
from leadlag.execution.state_store import ExecutionStateStore
from leadlag.execution.var_history import get_hist_returns_for_risk as _get_hist_returns_for_risk
from leadlag.models.v2.pit import load_pit_ir_history
from leadlag.reporting.production_v2_writer import write_production_files
from leadlag.runner.production import ProductionRunner
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger(__name__)


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

    Returns:
        Path to the decision output CSV, or a dry-run summary path.
    """
    ROOT = Path(__file__).resolve().parents[3]

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

    # Without a real price source the 1000 JPY dummy would create bogus gaps
    # and corrupt output files / ML overlay features. Force dry-run and use
    # previous closes as placeholders (gap = 0) for non-API execution.
    if not api_enable and not dry_run:
        logger.warning(
            "[API] --api-enable not set. Forcing dry_run=True to avoid writing "
            "files with dummy open prices."
        )
        dry_run = True

    # --- Step 1: Build df_exec (historical data + today's placeholders) ---
    logger.info("[1/5] Loading/building df_exec...")
    df_exec = _load_df_exec(app_config, data_source="cache")

    # --- Step 2: Fetch JP open prices ---
    api_client = None
    current_prices: dict[str, float] = {}
    try:
        if api_enable:
            api_client = build_api_client(
                api_url=api_url,
                api_token=api_token,
                api_dry_run=api_dry_run,
            )

            trade_date_str = t_trade.strftime("%Y%m%d")
            cached_prices = (
                load_current_prices_cache(trade_date_str)
                if app_config.broker_provider == "tachibana"
                else None
            )
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

            if capital_from_wallet:
                max_capital = resolve_wallet_capital(api_client)
                logger.info("[CAPITAL] Using wallet balance: %s JPY", f"{max_capital:,.0f}")
        else:
            logger.info("[2/5] API disabled. Will use previous closes as placeholder open prices.")
    except Exception as e:
        logger.error("[2/5] Failed to fetch opens: %s", e)
        if api_client is not None:
            api_client.close()
        raise

    # --- Step 3: Build PIT data lake and the as-of market snapshot ---
    logger.info("[3/5] Building PIT data lake and as-of market snapshot...")
    lake = PITDataLake(df_exec)
    if t_trade not in lake.df_exec.index:
        # A previous row is never a valid substitute for today's trade.  It
        # would combine yesterday's signal/gap with today's prices and could
        # replay an old order plan.  Fail closed for both dry-run and live
        # paths; callers can explicitly request a historical date instead.
        raise RuntimeError(
            f"Requested trade_date {trade_date} is not available in df_exec; "
            "refusing stale PIT fallback."
        )

    lake_snapshot = lake.get_snapshot(decision_as_of)
    effective_trade_date = trade_date
    t_effective = t_trade
    snapshot_prev_closes = lake_snapshot.prev_closes

    # If the API was disabled, fall back to previous closes as placeholders.
    # This produces zero gap and avoids the dangerous 1000 JPY dummy that would
    # otherwise leak into the ML overlay and on-demand BLPX recomputation.
    if not api_enable:
        current_prices = {
            tk: float(snapshot_prev_closes[tk])
            for tk in JP_TICKERS
            if tk in snapshot_prev_closes
            and np.isfinite(snapshot_prev_closes[tk])
            and snapshot_prev_closes[tk] > 0.0
        }

    # Recompute the 9:10 gap returns using the live/cached current prices so
    # the MarketSnapshot is the single source of truth for this trade date.
    api_current_prices: dict[str, float] = {
        tk: float(current_prices[tk])
        for tk in JP_TICKERS
        if tk in current_prices and np.isfinite(current_prices[tk]) and current_prices[tk] > 0.0
    }
    jp_gap_api = np.zeros(len(JP_TICKERS), dtype=float)
    for j, tk in enumerate(JP_TICKERS):
        p = api_current_prices.get(tk)
        pc = snapshot_prev_closes.get(tk)
        if p is not None and pc is not None and pc > 0.0:
            jp_gap_api[j] = p / pc - 1.0

    snapshot = MarketSnapshot(
        as_of=lake_snapshot.as_of,
        trade_date=lake_snapshot.trade_date,
        us_returns=lake_snapshot.us_returns,
        jp_gap_returns=jp_gap_api,
        jp_betas=lake_snapshot.jp_betas,
        topix_night_return=lake_snapshot.topix_night_return,
        current_prices=api_current_prices,
        prev_closes=snapshot_prev_closes,
        price_sources={
            ticker: "broker_current" if api_enable else "previous_close_placeholder"
            for ticker in api_current_prices
        },
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

    # --- Step 4: Generate V2 portfolio ---
    logger.info("[4/5] Generating V2 production portfolio...")
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    macro_prices = None
    if app_config.v2.macro_kappa_enabled or app_config.v2.macro_direction_enabled:
        try:
            macro_prices = macro_data.load_macro_prices(
                start=df_exec.index.min().strftime("%Y-%m-%d"),
                end=(df_exec.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                period="max",
            )
            # The live snapshot is bounded by the decision date.  Keeping a
            # later provider row in the owned frame would make provenance
            # impossible to audit after a cache refresh.
            macro_prices = macro_prices.loc[macro_prices.index <= effective_trade_date].copy()
        except Exception as exc:
            logger.warning("Failed to load run-owned macro prices: %s", exc)

    adr_features = None
    if app_config.v2.ml_overlay_enabled:
        try:
            # Keep the complete run-owned artifact.  The overlay applies the
            # adapter's date/staleness validation symmetrically with backtests.
            adr_features = adr_data.load_adr_features()
            if adr_features is not None:
                adr_features = adr_features.loc[adr_features.index <= effective_trade_date].copy()
        except Exception as exc:
            logger.warning("Failed to load run-owned ADR features: %s", exc)

    pit_ir_history = None
    pit_history_trade_dates = None
    if gap_dir is not None:
        pit_ir_history, _pit_alerts, pit_history_trade_dates = load_pit_ir_history(
            gap_dir, trade_date
        )
    rank_reversal_signals = None
    if app_config.v2.cs_overlay_enabled:
        rank_reversal_signals = load_rank_reversal_frame(
            gap_dir,
            [t_trade],
            file_pattern=app_config.v2.cs_rank_reversal_file_pattern,
        )
    historical_observed_at_by_date = {
        normalize_jst_date(dt).strftime("%Y-%m-%d"): {
            "open_910_returns": f"{normalize_jst_date(dt).date()} 09:10",
            "macro_prices": f"{normalize_jst_date(dt).date()} 09:00",
            "adr_features": f"{normalize_jst_date(dt).date()} 09:00",
            "rank_reversal_signals": f"{normalize_jst_date(dt).date()} 09:00",
            "pit_ir_history": f"{normalize_jst_date(dt).date()} 09:10",
        }
        for dt in df_exec.index
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
    decision_inputs = lake.build_decision_inputs(
        decision_as_of,
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
    runner = ProductionRunner(app_config)
    result = runner.run(decision_inputs)

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
        # Batch wrappers hold the same scope for the complete process group.
        # Avoid a nested self-conflict while retaining the in-process lease for
        # direct CLI invocations.
        lease_context = execution_lease(
            state_store, metadata={"trade_date": str(t_effective), "job_type": "decision"},
        )
        with lease_context:
            hist_returns = _get_hist_returns_for_risk(
                strategy=None,
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
                manual_opens=current_prices,
                max_capital=max_capital,
                hist_returns=hist_returns,
                output_dir=output_dir,
                api_client=api_client,
                text_output=text_output,
                current_positions=current_positions,
                state_store=state_store,
                account_key=account_key,
                strategy_key="production_v2",
            )

        logger.info("V2 decision completed. Output: %s", out_path)
    else:
        logger.info("[DRY-RUN] Skipping order submission and file writes.")
        out_path = str(live_path)

    if api_client is not None:
        api_client.close()

    return out_path
