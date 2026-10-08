"""Historical daily return cache for VaR/ES risk checks.

This module was split from ``leadlag.data.market_data_cache`` to remove the
reverse dependency of the data layer on the execution layer. It runs the V2
backtest only when a cached return series is not already available.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass
class VaRHistorySource:
    """Run-owned, deadline-bound source snapshot prepared before cache lookup."""

    returns_store: SqliteCacheStore
    deadline_budget: DeadlineBudget
    app_config: Any
    df_exec: pd.DataFrame
    trade_date: pd.Timestamp
    required_last: pd.Timestamp
    history_dates: pd.DatetimeIndex
    history_start: str
    slippage_bps: float | None
    effective_config: dict[str, Any]
    gap_input_path: Path | None
    gap_snapshot_path: Path | None
    gap_fingerprint: str
    snapshot_owner: tempfile.TemporaryDirectory[str] | None

    def remaining_timeout(self) -> float:
        return self.deadline_budget.remaining(label="VaR/ES risk-history")

    def cleanup_snapshot(self) -> None:
        """Release a snapshot only while this run still owns it."""
        owner = self.snapshot_owner
        self.snapshot_owner = None
        if owner is not None:
            owner.cleanup()

    def transfer_snapshot(self) -> tempfile.TemporaryDirectory[str] | None:
        """Transfer snapshot lifetime to the worker before it starts."""
        owner = self.snapshot_owner
        self.snapshot_owner = None
        return owner


@dataclass(frozen=True)
class VaRHistoryReplayPlan:
    """Exact inputs shared by the cache identity and the V2 replay worker."""

    source: VaRHistorySource
    overlay_chunks: list[tuple[pd.DatetimeIndex, Any | None]]
    historical_inputs: Any
    cache_key: str


def _configured_timeout(config: Any) -> float:
    if isinstance(config, dict):
        configured = config.get("var_history_timeout", 300)
    else:
        configured = getattr(config, "var_history_timeout", 300)
    try:
        return max(0.01, float(configured))
    except (TypeError, ValueError):
        return 300.0


def _timed_call(
    deadline_budget: DeadlineBudget,
    function: Callable[[], Any],
    label: str,
) -> Any:
    """Bound synchronous preparation with the run's single absolute deadline."""
    return run_with_timeout(
        function,
        timeout=deadline_budget.remaining(label="VaR/ES risk-history"),
        label=label,
    )


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


def _prepare_var_history_source(
    config: Any,
    output_root: str,
    trade_date: pd.Timestamp,
    config_path: str | Path | None,
    gap_input_dir: str | Path | None,
    deadline_budget: DeadlineBudget,
) -> VaRHistorySource | None:
    """Resolve effective configuration and freeze the completed source rows."""
    cache_dir = Path(output_root) / ".cache"
    try:
        returns_store = _timed_call(
            deadline_budget,
            lambda: SqliteCacheStore(
                cache_dir / "daily_returns.sqlite",
                timeout=deadline_budget.remaining(label="VaR/ES risk-history"),
            ),
            "VaR/ES return-cache initialization",
        )
    except TimeoutError as exc:
        logger.error("VaR/ES return-cache initialization timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return None

    try:
        df_exec = _timed_call(
            deadline_budget,
            lambda: load_df_exec_from_local_cache(max_stale_bdays=3),
            "VaR/ES df_exec load",
        )
    except TimeoutError as exc:
        logger.error("VaR/ES df_exec load timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return None
    except RuntimeError as exc:
        logger.error("VaR/ES cannot use stale df_exec: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return None

    project_root = resolve_project_root()
    if config_path is None:
        resolved_config_path = project_root / "configs" / "production" / "production.yaml"
    else:
        resolved_config_path = Path(config_path)
        if not resolved_config_path.is_absolute():
            resolved_config_path = project_root / resolved_config_path
    start_date = (
        config.get("start_date", "2015-01-05")
        if isinstance(config, dict)
        else getattr(config, "start_date", "2015-01-05")
    )
    slippage_bps = (
        config.get("slippage_bps")
        if isinstance(config, dict)
        else getattr(config, "slippage_bps", None)
    )
    try:
        app_config = _timed_call(
            deadline_budget,
            lambda: load_config_from_yaml(resolved_config_path, strict=True),
            "VaR/ES effective config load",
        )
    except TimeoutError as exc:
        logger.error("VaR/ES effective config load timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return None
    if slippage_bps is not None:
        costs = app_config.v2.costs.model_copy(
            update={"slippage_bps_per_side": float(slippage_bps)}
        )
        app_config = app_config.model_copy(
            update={"v2": app_config.v2.model_copy(update={"costs": costs})}
        )

    required_last = pd.Timestamp(
        previous_trading_day(trade_date.to_pydatetime())
    ).normalize()
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
        return None

    history_start = str(history_dates.min().date())
    df_exec = df_exec.loc[df_exec.index <= required_last].copy()
    effective_config = {
        "v2": app_config.v2.model_dump(mode="json"),
        "strategy": app_config.strategy.model_dump(mode="json"),
    }
    configured_gap = app_config.gap_distribution_dir
    gap_arg = gap_input_dir if gap_input_dir is not None else configured_gap
    gap_path = Path(gap_arg) if gap_arg else None
    if gap_path is not None and not gap_path.is_absolute():
        gap_path = project_root / gap_path

    try:
        gap_snapshot_path, gap_fingerprint, snapshot_owner = _timed_call(
            deadline_budget,
            lambda: var_inputs._snapshot_gap_input(
                gap_path,
                timeout=deadline_budget.remaining(label="VaR/ES risk-history"),
                max_trade_date=required_last,
            ),
            "VaR/ES gap-input snapshot",
        )
    except (RuntimeError, TimeoutError) as exc:
        logger.error("VaR/ES cannot obtain a stable gap-input snapshot: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return None

    return VaRHistorySource(
        returns_store=returns_store,
        deadline_budget=deadline_budget,
        app_config=app_config,
        df_exec=df_exec,
        trade_date=trade_date,
        required_last=required_last,
        history_dates=history_dates,
        history_start=history_start,
        slippage_bps=slippage_bps,
        effective_config=effective_config,
        gap_input_path=gap_path,
        gap_snapshot_path=gap_snapshot_path,
        gap_fingerprint=gap_fingerprint,
        snapshot_owner=snapshot_owner,
    )


def _build_var_history_replay_plan(
    source: VaRHistorySource,
    overlay_model: Any | None,
) -> VaRHistoryReplayPlan | None:
    """Bind the cache key to the exact overlay and historical inputs to replay."""
    deadline_budget = source.deadline_budget
    try:
        configured_overlay, overlay_path = resolve_overlay_settings(source.app_config)
        selected_overlay_model = overlay_model
        if selected_overlay_model is None and configured_overlay and overlay_path is not None:
            from leadlag.models.ml_overlay_artifact import load_overlay_model

            selected_overlay_model = _timed_call(
                deadline_budget,
                lambda: load_overlay_model(overlay_path),
                "VaR/ES overlay load",
            )
        overlay_chunks, overlay_identity = _timed_call(
            deadline_budget,
            lambda: _resolve_overlay_history_chunks(
                source.history_dates,
                selected_overlay_model,
                overlay_path,
            ),
            "VaR/ES overlay history resolution",
        )
        historical_inputs = _timed_call(
            deadline_budget,
            lambda: _build_var_historical_inputs(
                source.df_exec,
                source.app_config,
                source.gap_snapshot_path,
                source.history_dates,
                bool(configured_overlay),
            ),
            "VaR/ES run-input snapshot",
        )
        input_snapshot_hash = _timed_call(
            deadline_budget,
            lambda: historical_inputs.fingerprint,
            "VaR/ES run-input fingerprint",
        )
        df_exec_hash = _timed_call(
            deadline_budget,
            lambda: dataframe_fingerprint(source.df_exec),
            "VaR/ES df_exec fingerprint",
        )
        code_hash = _timed_call(
            deadline_budget,
            lambda: var_inputs._file_manifest_fingerprint(
                Path(__file__).resolve().parents[1],
                timeout=source.remaining_timeout(),
            ),
            "VaR/ES code fingerprint",
        )
        cache_key = build_var_cache_key(
            effective_config=source.effective_config,
            start_date=source.history_start,
            slippage_bps=source.slippage_bps,
            df_exec_hash=df_exec_hash,
            code_hash=code_hash,
            overlay_identity=overlay_identity,
            gap_input_hash=source.gap_fingerprint,
            input_snapshot_hash=input_snapshot_hash,
        )
        return VaRHistoryReplayPlan(
            source=source,
            overlay_chunks=overlay_chunks,
            historical_inputs=historical_inputs,
            cache_key=cache_key,
        )
    except TimeoutError as exc:
        source.cleanup_snapshot()
        logger.error("VaR/ES preparation exceeded its deadline: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return None
    except BaseException:
        source.cleanup_snapshot()
        raise


def _load_cached_var_returns(plan: VaRHistoryReplayPlan) -> pd.Series | None:
    """Return fresh cached returns, None on a miss, or an empty series on timeout."""
    source = plan.source
    try:
        cached = _timed_call(
            source.deadline_budget,
            lambda: source.returns_store.get(plan.cache_key),
            "VaR/ES return-cache read",
        )
    except TimeoutError as exc:
        logger.error("VaR/ES return-cache read timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)

    if cached is None:
        return None
    hist_results = cached
    if not hist_results.empty:
        if "daily_fallback" in hist_results and hist_results["daily_fallback"].astype(bool).any():
            logger.warning("Cached VaR returns contain fallback dates; recomputing.")
            hist_results = pd.DataFrame()
            cached = None
    if cached is not None and not hist_results.empty:
        cached_last = pd.to_datetime(hist_results.index.max()).normalize()
        if not pd.isna(cached_last):
            required_last = pd.Timestamp(
                previous_trading_day(source.trade_date.to_pydatetime())
            )
            stale_bdays = count_tse_bdays(cached_last, required_last)
            if stale_bdays <= 0:
                hist_returns = hist_results["daily_return"]
                hist_returns = hist_returns[hist_returns.index < source.trade_date]
                logger.info(
                    "Loaded %d cached daily returns for VaR/ES (last=%s)",
                    len(hist_returns),
                    cached_last.date(),
                )
                source.remaining_timeout()
                return hist_returns
            logger.warning(
                "Cached daily returns are stale: last=%s, trade_date=%s, "
                "required_last=%s, %d TSE trading days old (max=0); recomputing.",
                cached_last.date(),
                source.trade_date.date(),
                required_last.date(),
                stale_bdays,
            )
        else:
            logger.warning("Cached daily returns have no valid index; recomputing.")
    else:
        logger.warning("Cached daily returns are empty; recomputing.")
    return None


def _run_var_history_backtest(
    plan: VaRHistoryReplayPlan,
    backtest_engine: Any,
) -> dict[str, Any]:
    """Replay each date-scoped overlay chunk and validate all audit contracts."""
    source = plan.source
    gap_dir = source.gap_snapshot_path
    if source.gap_input_path is not None and gap_dir is None:
        logger.warning(
            "Gap input dir not found or could not be snapshotted: %s. "
            "V2 VaR/ES history will fall back to flat positions.",
            source.gap_input_path,
        )

    chunks: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for chunk_dates, chunk_overlay in plan.overlay_chunks:
        chunk_result = cast(
            dict[str, Any],
            backtest_engine.run_v2_backtest(
                cfg=source.app_config,
                gap_input_dir=gap_dir,
                df_exec=source.df_exec,
                start_date=str(chunk_dates.min().date()),
                end_date=str(chunk_dates.max().date()),
                n_jobs=4,
                overlay_model=chunk_overlay,
                historical_inputs=plan.historical_inputs,
            ),
        )
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
    if not combined.index.equals(source.history_dates):
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
    failed_audits = [
        row
        for row in audit_rows
        if row["numerical"] != "PASSED" or row["leakage"] != "PASSED" or row["fallback"]
    ]
    if failed_audits:
        raise ValueError(f"VaR replay contains an audit failure: {failed_audits[:5]}")
    return {"daily_returns": combined, "daily_fallback": fallbacks}


def get_hist_returns_for_risk(
    config: Any,
    output_root: str,
    trade_date: pd.Timestamp,
    config_path: str | Path | None = None,
    gap_input_dir: str | Path | None = None,
    overlay_model: Any | None = None,
) -> pd.Series:
    """Load fresh cached VaR returns or replay the same run-owned history."""
    timeout = _configured_timeout(config)
    deadline_budget = DeadlineBudget.from_timeout(timeout)
    source = _prepare_var_history_source(
        config,
        output_root,
        trade_date,
        config_path,
        gap_input_dir,
        deadline_budget,
    )
    if source is None:
        return pd.Series(dtype=float)

    plan = _build_var_history_replay_plan(source, overlay_model)
    if plan is None:
        return pd.Series(dtype=float)

    try:
        cached_returns = _load_cached_var_returns(plan)
        if cached_returns is not None:
            source.cleanup_snapshot()
            return cached_returns

        logger.info(
            "No return cache found; replaying %d completed sessions for VaR/ES...",
            len(source.history_dates),
        )

        # Keep the local import inside the same absolute deadline as snapshot
        # preparation and fingerprinting.
        def load_backtest_engine() -> Any:
            from leadlag.execution.backtester import BacktestEngine

            return BacktestEngine

        backtest_engine = _timed_call(
            deadline_budget,
            load_backtest_engine,
            "VaR/ES backtest engine import",
        )
        worker_timeout = source.remaining_timeout()
        worker_snapshot_owner = source.transfer_snapshot()
        try:
            out_res = var_worker.run_snapshot_worker(
                lambda: _run_var_history_backtest(plan, backtest_engine),
                worker_snapshot_owner,
                timeout=worker_timeout,
            )
        except TimeoutError:
            logger.warning(
                "V2 backtest for VaR/ES timed out after %d seconds; "
                "returning empty series so risk check blocks.",
                timeout,
            )
            return pd.Series(dtype=float)
    except TimeoutError as exc:
        source.cleanup_snapshot()
        logger.error("VaR/ES preparation exceeded its deadline: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)
    except BaseException:
        source.cleanup_snapshot()
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
            source.returns_store.path,
            plan.cache_key,
            hist_results,
            timeout=source.remaining_timeout(),
        )
        source.remaining_timeout()
    except TimeoutError as exc:
        logger.error("VaR/ES return-cache write timed out: %s", exc)
        logger.warning("Returning empty historical return series so risk check blocks.")
        return pd.Series(dtype=float)

    source.remaining_timeout()
    return pd.Series(
        hist_results.loc[
            hist_results.index < trade_date,
            "daily_return",
        ]
    )
