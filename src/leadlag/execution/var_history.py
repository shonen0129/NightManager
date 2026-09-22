"""Historical daily return cache for VaR/ES risk checks.

This module was split from ``leadlag.data.market_data_cache`` to remove the
reverse dependency of the data layer on the execution layer. It runs the V2
backtest only when a cached return series is not already available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from leadlag.core.market_calendar import count_tse_bdays, previous_trading_day
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution import var_inputs, var_worker
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.var_cache import DeadlineBudget, build_var_cache_key
from leadlag.runner.model_factory import resolve_overlay_settings
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from leadlag.utils.threading import run_with_timeout

logger = logging.getLogger(__name__)


def _build_var_historical_inputs(
    df_exec: pd.DataFrame,
    app_config: Any,
    gap_dir: Path | None,
    sim_dates: pd.DatetimeIndex,
    overlay_enabled: bool,
) -> Any:
    """Build the same run-owned historical snapshot used by VaR backtests."""
    from leadlag.data import adr_features as adr_data
    from leadlag.data import macro as macro_data
    from leadlag.data.intraday_inputs import build_open_910_returns
    from leadlag.data.rank_reversal import load_rank_reversal_frame
    from leadlag.data.tickers import JP_TICKERS
    from leadlag.domain.inputs import HistoricalInputs
    from leadlag.models.v2.pit import load_pit_ir_history

    run_config = app_config.v2
    # Keep the cache identity pure for narrow unit/diagnostic frames that do
    # not represent an execution dataset.  Production df_exec always has
    # these columns and therefore receives the complete adapter snapshot.
    required_columns = {
        f"jp_oc_{JP_TICKERS[0]}",
        f"jp_open_trade_{JP_TICKERS[0]}",
    }
    if not required_columns.issubset(df_exec.columns):
        return HistoricalInputs(df_exec, source="var_history_incomplete_frame")
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    macro_prices = None
    if run_config.macro_kappa_enabled or run_config.macro_direction_enabled:
        try:
            macro_prices = macro_data.load_macro_prices(
                start=df_exec.index.min().strftime("%Y-%m-%d"),
                end=(df_exec.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                period="max",
            )
        except Exception as exc:  # pragma: no cover - provider-dependent
            logger.warning("Failed to load VaR run-owned macro prices: %s", exc)

    adr_features = None
    if overlay_enabled:
        try:
            adr_features = adr_data.load_adr_features()
        except Exception as exc:  # pragma: no cover - artifact-dependent
            logger.warning("Failed to load VaR run-owned ADR features: %s", exc)

    pit_ir_history: dict[str, Any] | None = None
    pit_history_trade_dates: dict[str, Any] | None = None
    if gap_dir is not None:
        pit_ir_history = {}
        pit_history_trade_dates = {}
        for dt in sim_dates:
            date_str = dt.strftime("%Y-%m-%d")
            history_ir, _alerts, history_dates = load_pit_ir_history(gap_dir, date_str)
            pit_ir_history[date_str] = history_ir
            pit_history_trade_dates[date_str] = history_dates

    rank_reversal_signals = None
    if run_config.cs_overlay_enabled:
        rank_reversal_signals = load_rank_reversal_frame(
            gap_dir,
            sim_dates,
            file_pattern=run_config.cs_rank_reversal_file_pattern,
        )

    historical_observed_at_by_date = {
        dt.strftime("%Y-%m-%d"): {
            "open_910_returns": f"{dt.date()} 09:10",
            "macro_prices": f"{dt.date()} 09:00",
            "adr_features": f"{dt.date()} 09:00",
            "rank_reversal_signals": f"{dt.date()} 09:00",
            "pit_ir_history": f"{dt.date()} 09:10",
        }
        for dt in sim_dates
    }

    return HistoricalInputs(
        df_exec,
        source="var_history",
        open_910_returns=open_910_returns,
        macro_prices=macro_prices,
        adr_features_frame=adr_features,
        pit_ir_history=pit_ir_history,
        pit_history_trade_dates=pit_history_trade_dates,
        rank_reversal_signals=rank_reversal_signals,
        observed_at_by_date=historical_observed_at_by_date,
    )


def get_hist_returns_for_risk(
    strategy: Any,
    config: Any,
    output_root: str,
    trade_date: pd.Timestamp,
    config_path: str | Path | None = None,
    gap_input_dir: str | Path | None = None,
    overlay_model: Any | None = None,
) -> pd.Series:
    """Efficiently get historical daily returns for VaR/ES risk checks.

    Uses an SQLite cache if available, otherwise runs the V2 full backtest and
    caches the result. The ``strategy`` argument is kept for backward
    compatibility but is no longer used.
    """
    cache_dir = Path(output_root) / ".cache"
    # A single deadline covers snapshot acquisition and the backtest.  The
    # snapshot used to happen before the backtest timeout was even resolved,
    # so a SQLite lock could block the caller for the connection timeout.
    if isinstance(config, dict):
        configured_timeout = config.get("var_history_timeout", 300)
    else:
        configured_timeout = getattr(config, "var_history_timeout", 300)
    try:
        timeout = max(0.01, float(configured_timeout))
    except (TypeError, ValueError):
        timeout = 300.0
    deadline_budget = DeadlineBudget.from_timeout(timeout)

    def remaining_timeout() -> float:
        return deadline_budget.remaining(label="VaR/ES risk-history")

    def timed_call(function: Any, label: str) -> Any:
        """Bound synchronous preparation/cache operations by the same deadline."""
        return run_with_timeout(function, timeout=remaining_timeout(), label=label)

    try:
        returns_store = timed_call(
            lambda: SqliteCacheStore(
                cache_dir / "daily_returns.sqlite",
                timeout=remaining_timeout(),
            ),
            "VaR/ES return-cache initialization",
        )
    except TimeoutError as exc:
        logger.error("VaR/ES return-cache initialization timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)

    # Load and freshness-check the source data before consulting the cache.
    # A cache hit must never hide a corrected/stale df_exec bundle.
    try:
        df_exec = timed_call(
            lambda: load_df_exec_from_local_cache(max_stale_bdays=3),
            "VaR/ES df_exec load",
        )
    except TimeoutError as exc:
        logger.error("VaR/ES df_exec load timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)
    except RuntimeError as e:
        logger.error("VaR/ES cannot use stale df_exec: %s", e)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)

    # Resolve the canonical inherited config before consulting the cache.  A
    # VaR series is only reusable for the same effective config and cost
    # override; the old fixed key silently reused returns from another run.
    project_root = Path(__file__).resolve().parents[3]
    if config_path is None:
        resolved_cfg_path = project_root / "configs" / "production" / "production.yaml"
    else:
        resolved_cfg_path = Path(config_path)
        if not resolved_cfg_path.is_absolute():
            resolved_cfg_path = project_root / resolved_cfg_path
    start_date = config.get("start_date", "2015-01-05") if isinstance(config, dict) else getattr(config, "start_date", "2015-01-05")
    slippage_bps = config.get("slippage_bps") if isinstance(config, dict) else getattr(config, "slippage_bps", None)
    try:
        app_config = timed_call(
            lambda: load_config_from_yaml(resolved_cfg_path, strict=True),
            "VaR/ES effective config load",
        )
    except TimeoutError as exc:
        logger.error("VaR/ES effective config load timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)
    if slippage_bps is not None:
        costs = app_config.v2.costs.model_copy(update={"slippage_bps_per_side": float(slippage_bps)})
        app_config = app_config.model_copy(update={"v2": app_config.v2.model_copy(update={"costs": costs})})
    # Hash the validated, effective calculation settings used by this very run.
    effective_config = {"v2": app_config.v2.model_dump(mode="json"),
                        "strategy": app_config.strategy.model_dump(mode="json")}
    configured_gap = app_config.gap_distribution_dir
    gap_arg = gap_input_dir if gap_input_dir is not None else configured_gap
    gap_path = Path(gap_arg) if gap_arg else None
    if gap_path is not None and not gap_path.is_absolute():
        gap_path = project_root / gap_path
    # Freeze the exact gap input before hashing the cache key.  The same
    # snapshot path is passed to the V2 backtest below, so a concurrent Step 2
    # writer cannot change the distribution between key construction and use.
    try:
        gap_snapshot_path, gap_fingerprint, _gap_snapshot_temp = timed_call(
            lambda: var_inputs._snapshot_gap_input(
                gap_path,
                timeout=remaining_timeout(),
            ),
            "VaR/ES gap-input snapshot",
        )
    except (RuntimeError, TimeoutError) as exc:
        logger.error("VaR/ES cannot obtain a stable gap-input snapshot: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)
    snapshot_owner = _gap_snapshot_temp

    def _cleanup_snapshot() -> None:
        """Release a pre-worker snapshot on every non-worker exit path."""
        nonlocal snapshot_owner
        if snapshot_owner is not None:
            snapshot_owner.cleanup()
            snapshot_owner = None

    try:
        configured_overlay, overlay_path = resolve_overlay_settings(app_config)

        # Select the overlay exactly once.  A caller such as the live V2 bridge
        # passes the object already held by ProductionRunner.  Standalone VaR
        # calls load the configured version once before computing the cache key;
        # the same object is then passed to BacktestEngine below.  Never resolve
        # CURRENT again after the key has been calculated.
        selected_overlay_model = overlay_model
        if selected_overlay_model is None:
            if configured_overlay and overlay_path is not None:
                from leadlag.models.ml_order_overlay import load_overlay_model

                selected_overlay_model = timed_call(
                    lambda: load_overlay_model(overlay_path),
                    "VaR/ES overlay load",
                )
        overlay_identity = timed_call(
            lambda: var_inputs._overlay_model_fingerprint(selected_overlay_model),
            "VaR/ES overlay fingerprint",
        ) if selected_overlay_model is not None else "none"
        sim_dates_for_inputs = pd.DatetimeIndex(
            df_exec.index[df_exec.index >= pd.Timestamp(start_date)]
        )
        historical_input_snapshot = timed_call(
            lambda: _build_var_historical_inputs(
                df_exec,
                app_config,
                gap_snapshot_path,
                sim_dates_for_inputs,
                bool(configured_overlay),
            ),
            "VaR/ES run-input snapshot",
        )
        input_snapshot_hash = timed_call(
            lambda: historical_input_snapshot.fingerprint,
            "VaR/ES run-input fingerprint",
        )
        df_exec_hash = timed_call(
            lambda: dataframe_fingerprint(df_exec),
            "VaR/ES df_exec fingerprint",
        )
        code_hash = timed_call(
            lambda: var_inputs._file_manifest_fingerprint(
                project_root / "src" / "leadlag",
                timeout=remaining_timeout(),
            ),
            "VaR/ES code fingerprint",
        )
        cache_key = build_var_cache_key(
            effective_config=effective_config,
            start_date=start_date,
            slippage_bps=slippage_bps,
            df_exec_hash=df_exec_hash,
            code_hash=code_hash,
            overlay_identity=overlay_identity,
            gap_input_hash=gap_fingerprint,
            input_snapshot_hash=input_snapshot_hash,
        )

        # VaR/ES should use returns up to the previous TSE trading day.
        # If the cache does not include the most recent completed trading day,
        # we recompute to avoid stale risk thresholds.
        _RETURNS_MAX_STALE_BDAY = 0

        try:
            cached = timed_call(
                lambda: returns_store.get(cache_key),
                "VaR/ES return-cache read",
            )
        except TimeoutError as exc:
            logger.error("VaR/ES return-cache read timed out: %s", exc)
            _cleanup_snapshot()
            logger.warning("Returning empty historical return series so risk check blocks.")
            return pd.Series(dtype=float)
        if cached is not None:
            hist_results = cached
            if not hist_results.empty:
                cached_last = pd.to_datetime(hist_results.index.max()).normalize()
                if not pd.isna(cached_last):
                    required_last = pd.Timestamp(
                        previous_trading_day(trade_date.to_pydatetime())
                    )
                    stale_bdays = count_tse_bdays(cached_last, required_last)
                    if stale_bdays <= _RETURNS_MAX_STALE_BDAY:
                        hist_returns = hist_results["daily_return"]
                        hist_returns = hist_returns[hist_returns.index < trade_date]
                        logger.info(
                            "Loaded %d cached daily returns for VaR/ES (last=%s)",
                            len(hist_returns),
                            cached_last.date(),
                        )
                        remaining_timeout()
                        _cleanup_snapshot()
                        return hist_returns
                    logger.warning(
                        "Cached daily returns are stale: last=%s, trade_date=%s, "
                        "required_last=%s, %d TSE trading days old (max=%d); recomputing.",
                        cached_last.date(),
                        trade_date.date(),
                        required_last.date(),
                        stale_bdays,
                        _RETURNS_MAX_STALE_BDAY,
                    )
                else:
                    logger.warning("Cached daily returns have no valid index; recomputing.")
            else:
                logger.warning("Cached daily returns are empty; recomputing.")

        logger.info("No return cache found; running V2 full backtest for VaR/ES...")
        if gap_input_dir is None:
            gap_input_dir = app_config.gap_distribution_dir
        gap_dir = gap_snapshot_path
        if gap_input_dir and gap_dir is None:
            logger.warning(
                "Gap input dir not found or could not be snapshotted: %s. "
                "V2 VaR/ES history will fall back to flat positions.",
                gap_input_dir,
            )

        # Local import to avoid a module-level cycle; bound it as part of the
        # same deadline because a cold import can be expensive on startup.
        def _load_backtest_engine() -> Any:
            from leadlag.execution.backtester import BacktestEngine

            return BacktestEngine

        backtest_engine = timed_call(
            _load_backtest_engine,
            "VaR/ES backtest engine import",
        )

        # Resolve the remaining budget before transferring ownership.  If the
        # overall deadline has already elapsed, the outer cleanup path still
        # owns the snapshot and can remove it deterministically.
        worker_timeout = remaining_timeout()

        # Transfer ownership to the worker before starting it.  The caller
        # must not clean this directory after a timeout while the daemon thread
        # is still reading from the snapshot.
        worker_snapshot_owner = snapshot_owner
        snapshot_owner = None

        def _run_backtest_for_risk() -> dict[str, Any]:
            return cast(dict[str, Any], backtest_engine.run_v2_backtest(
                cfg=app_config, gap_input_dir=gap_dir, df_exec=df_exec,
                start_date=start_date, n_jobs=1, overlay_model=selected_overlay_model,
                historical_inputs=historical_input_snapshot,
            ))

        try:
            out_res = var_worker.run_snapshot_worker(
                _run_backtest_for_risk, worker_snapshot_owner, timeout=worker_timeout,
            )
        except TimeoutError:
            logger.warning(
                "V2 backtest for VaR/ES timed out after %d seconds; "
                "returning empty series so risk check blocks.",
                timeout,
            )
            return pd.Series(dtype=float)
    except TimeoutError as exc:
        _cleanup_snapshot()
        logger.error("VaR/ES preparation exceeded its deadline: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)
    except BaseException:
        _cleanup_snapshot()
        raise
    hist_results = pd.DataFrame(
        {"daily_return": out_res["daily_returns"]},
        index=out_res["daily_returns"].index,
    )

    try:
        var_worker._set_cache_with_deadline(
            returns_store.path,
            cache_key,
            hist_results,
            timeout=remaining_timeout(),
        )
        remaining_timeout()
    except TimeoutError as exc:
        logger.error("VaR/ES return-cache write timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)

    remaining_timeout()
    return pd.Series(
        hist_results.loc[
            hist_results.index < trade_date,
            "daily_return",
        ]
    )
