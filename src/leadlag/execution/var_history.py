"""Historical daily return cache for VaR/ES risk checks.

This module was split from ``leadlag.data.market_data_cache`` to remove the
reverse dependency of the data layer on the execution layer. It runs the V2
backtest only when a cached return series is not already available.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.config.paths import project_root as resolve_project_root
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


def _load_overlay_version(artifact_root: Path, version: str) -> Any:
    """Load and validate one immutable overlay version without changing CURRENT."""
    from leadlag.models.ml_overlay_artifact import load_overlay_model

    with tempfile.TemporaryDirectory(prefix="leadlag-overlay-history-") as temporary:
        temp_root = Path(temporary)
        (temp_root / "versions").symlink_to(artifact_root / "versions", target_is_directory=True)
        (temp_root / "CURRENT").write_text(version + "\n", encoding="utf-8")
        return load_overlay_model(temp_root)


def _resolve_overlay_history_chunks(
    sim_dates: pd.DatetimeIndex,
    selected_overlay_model: Any | None,
    artifact_root: Path | None,
) -> tuple[list[tuple[pd.DatetimeIndex, Any | None]], str]:
    """Resolve explicit date-to-artifact history while enforcing train cutoffs."""
    if selected_overlay_model is None:
        return [(sim_dates, None)], "none"

    current_meta = getattr(selected_overlay_model, "metadata", {}) or {}
    current_version = current_meta.get("artifact_version")
    if artifact_root is None:
        artifact_root = None
    history_path = artifact_root / "HISTORY.json" if artifact_root is not None else None
    manifest: dict[str, Any] | None = None
    if history_path is not None and history_path.is_file():
        manifest = json.loads(history_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("segments"), list):
            raise ValueError(f"Invalid overlay history manifest: {history_path}")

    model_by_version: dict[str, Any] = {}
    if current_version:
        model_by_version[str(current_version)] = selected_overlay_model

    assigned: list[tuple[pd.Timestamp, str, Any]] = []
    if manifest is None:
        if "train_end" in current_meta:
            train_end = pd.Timestamp(current_meta["train_end"]).normalize()
            invalid_dates = [date for date in sim_dates if date.normalize() <= train_end]
            if invalid_dates:
                raise ValueError(
                    f"Overlay {current_version or '<unknown>'} is in-sample through "
                    f"{train_end.date()}; a date-versioned HISTORY.json is required "
                    f"for {invalid_dates[0].date()}"
                )
        assigned = [(date, str(current_version or "selected"), selected_overlay_model) for date in sim_dates]
    else:
        assert history_path is not None and artifact_root is not None
        normalized_segments: list[tuple[pd.Timestamp, pd.Timestamp | None, str]] = []
        for item in manifest["segments"]:
            if not isinstance(item, dict) or not item.get("version") or not item.get("start_date"):
                raise ValueError(f"Invalid segment in overlay history manifest: {item!r}")
            start = pd.Timestamp(item["start_date"]).normalize()
            end = None if item.get("end_date") is None else pd.Timestamp(item["end_date"]).normalize()
            if end is not None and end < start:
                raise ValueError(f"Invalid overlay history interval: {item!r}")
            normalized_segments.append((start, end, str(item["version"])))
        normalized_segments.sort(key=lambda item: item[0])
        for left, right in zip(normalized_segments, normalized_segments[1:]):
            if left[1] is None or left[1] >= right[0]:
                raise ValueError("Overlay history segments overlap or have an open-ended non-final range")

        for date in sim_dates:
            normalized_date = date.normalize()
            matches = [
                version
                for start, end, version in normalized_segments
                if start <= normalized_date and (end is None or normalized_date <= end)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Overlay history does not map {normalized_date.date()} to exactly one artifact"
                )
            version = matches[0]
            model = model_by_version.get(version)
            if model is None:
                model = _load_overlay_version(artifact_root, version)
                model_by_version[version] = model
            metadata = getattr(model, "metadata", {}) or {}
            train_end_value = metadata.get("train_end")
            if train_end_value is not None and normalized_date <= pd.Timestamp(train_end_value).normalize():
                raise ValueError(
                    f"Overlay {version} trained through {train_end_value} cannot be applied "
                    f"to {normalized_date.date()}"
                )
            assigned.append((normalized_date, version, model))

    grouped: list[tuple[pd.DatetimeIndex, Any | None]] = []
    group_dates: list[pd.Timestamp] = []
    group_version: str | None = None
    group_model: Any | None = None
    for date, version, model in assigned:
        if group_version is not None and version != group_version:
            grouped.append((pd.DatetimeIndex(group_dates), group_model))
            group_dates = []
        group_dates.append(date)
        group_version, group_model = version, model
    if group_dates:
        grouped.append((pd.DatetimeIndex(group_dates), group_model))

    identity_parts = []
    for dates, model in grouped:
        version = str((getattr(model, "metadata", {}) or {}).get("artifact_version", "selected"))
        model_identity = var_inputs._overlay_model_fingerprint(model)
        identity_parts.append(
            {
                "version": version,
                "start_date": str(dates.min().date()),
                "end_date": str(dates.max().date()),
                "identity": model_identity,
            }
        )
    composite_identity = json.dumps(identity_parts, sort_keys=True, separators=(",", ":"))
    return grouped, composite_identity


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
    from leadlag.models.ml_overlay_features import _precompute_market_vol
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
        market_vol_frame=_precompute_market_vol(df_exec) if overlay_enabled else None,
        pit_ir_history=pit_ir_history,
        pit_history_trade_dates=pit_history_trade_dates,
        rank_reversal_signals=rank_reversal_signals,
        observed_at_by_date=historical_observed_at_by_date,
    )


def get_hist_returns_for_risk(
    config: Any,
    output_root: str,
    trade_date: pd.Timestamp,
    config_path: str | Path | None = None,
    gap_input_dir: str | Path | None = None,
    overlay_model: Any | None = None,
) -> pd.Series:
    """Efficiently get historical daily returns for VaR/ES risk checks.

    Uses an SQLite cache if available, otherwise runs the V2 full backtest and
    caches the result.
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
    project_root = resolve_project_root()
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
    required_last = pd.Timestamp(previous_trading_day(trade_date.to_pydatetime())).normalize()
    start_boundary = pd.Timestamp(start_date).normalize()
    risk_window = int(app_config.risk.var_window)
    history_dates = pd.DatetimeIndex(
        df_exec.index[
            (df_exec.index >= start_boundary)
            & (df_exec.index <= required_last)
            & (df_exec.index < pd.Timestamp(trade_date).normalize())
        ]
    )
    if len(history_dates) > risk_window + 19:
        history_dates = history_dates[-(risk_window + 19):]
    if history_dates.empty:
        logger.error("No completed TSE rows available for the VaR/ES history window.")
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)
    history_start = str(history_dates.min().date())
    # Keep the point-in-time frame through the last completed session only.
    # Current/provisional rows cannot affect the signal, cache key or VaR series.
    df_exec = df_exec.loc[df_exec.index <= required_last].copy()
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
                max_trade_date=required_last,
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
                from leadlag.models.ml_overlay_artifact import load_overlay_model

                selected_overlay_model = timed_call(
                    lambda: load_overlay_model(overlay_path),
                    "VaR/ES overlay load",
                )
        overlay_chunks, overlay_identity = timed_call(
            lambda: _resolve_overlay_history_chunks(
                history_dates,
                selected_overlay_model,
                overlay_path,
            ),
            "VaR/ES overlay history resolution",
        )
        historical_input_snapshot = timed_call(
            lambda: _build_var_historical_inputs(
                df_exec,
                app_config,
                gap_snapshot_path,
                history_dates,
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
                Path(__file__).resolve().parents[1],
                timeout=remaining_timeout(),
            ),
            "VaR/ES code fingerprint",
        )
        cache_key = build_var_cache_key(
            effective_config=effective_config,
            start_date=history_start,
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
                if "daily_fallback" in hist_results and hist_results["daily_fallback"].astype(bool).any():
                    logger.warning("Cached VaR returns contain fallback dates; recomputing.")
                    hist_results = pd.DataFrame()
                if hist_results.empty:
                    cached = None
            if cached is not None and not hist_results.empty:
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

        logger.info(
            "No return cache found; replaying %d completed sessions for VaR/ES...",
            len(history_dates),
        )
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
            chunks: list[dict[str, Any]] = []
            audit_rows: list[dict[str, Any]] = []
            for chunk_dates, chunk_overlay in overlay_chunks:
                chunk_result = cast(dict[str, Any], backtest_engine.run_v2_backtest(
                    cfg=app_config,
                    gap_input_dir=gap_dir,
                    df_exec=df_exec,
                    start_date=str(chunk_dates.min().date()),
                    end_date=str(chunk_dates.max().date()),
                    n_jobs=4,
                    overlay_model=chunk_overlay,
                    historical_inputs=historical_input_snapshot,
                ))
                chunks.append(chunk_result)
                summaries = chunk_result.get("v2_summaries")
                if summaries is not None:
                    if len(summaries) != len(chunk_dates):
                        raise ValueError("VaR replay returned an incomplete decision summary set")
                    for date, summary, is_fallback in zip(
                        chunk_dates,
                        summaries,
                        chunk_result["daily_fallback"].to_numpy(dtype=bool),
                    ):
                        audit = summary.get("audit_status", {})
                        audit_rows.append(
                            {
                                "trade_date": str(date.date()),
                                "numerical": audit.get("numerical"),
                                "leakage": audit.get("leakage"),
                                "fallback": bool(is_fallback or audit.get("fallback", False)),
                            }
                        )
            combined = pd.concat([chunk["daily_returns"] for chunk in chunks]).sort_index()
            if not combined.index.equals(history_dates):
                raise ValueError("Versioned VaR replay returned a different date set than requested")
            if combined.isna().any() or not np.isfinite(combined.to_numpy(dtype=float)).all():
                raise ValueError("Versioned VaR replay returned non-finite daily returns")
            fallbacks = pd.concat(
                [
                    chunk.get(
                        "daily_fallback",
                        pd.Series(False, index=chunk["daily_returns"].index),
                    )
                    for chunk in chunks
                ]
            ).sort_index().astype(bool)
            if fallbacks.any():
                raise ValueError(
                    f"VaR replay produced {int(fallbacks.sum())} fallback days; refusing to cache"
                )
            if audit_rows and len(audit_rows) != len(combined):
                raise ValueError("VaR replay did not expose audit status for every decision date")
            if any(
                row["numerical"] != "PASSED" or row["leakage"] != "PASSED" or row["fallback"]
                for row in audit_rows
            ):
                failed = [
                    row for row in audit_rows
                    if row["numerical"] != "PASSED" or row["leakage"] != "PASSED" or row["fallback"]
                ]
                raise ValueError(f"VaR replay contains an audit failure: {failed[:5]}")
            return {"daily_returns": combined, "daily_fallback": fallbacks}

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
        {
            "daily_return": out_res["daily_returns"],
            "daily_fallback": out_res.get(
                "daily_fallback",
                pd.Series(False, index=out_res["daily_returns"].index),
            ),
        },
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
