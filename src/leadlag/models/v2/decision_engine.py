"""V2 decision orchestration and final weight construction."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.market_calendar import previous_trading_day
from leadlag.core.portfolio import solve_baseline_style
from leadlag.core.signal import build_weights_minvar
from leadlag.data.gap_store import GapStore, is_gap_store_path
from leadlag.data.pit_lake import MarketSnapshot
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.distribution import (
    DistributionReason,
    DistributionResolutionError,
    DistributionStatus,
)
from leadlag.domain.inputs import DecisionInputs
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.v2.audit_comparator import _run_safety_audits
from leadlag.models.v2.fallback import _apply_pit_ruleD, _repair_and_adjust
from leadlag.models.v2.fallback_policy import FallbackPolicy
from leadlag.models.v2.overlay_applier import (
    _apply_overlay,
    _apply_rank_reversal_overlay,
    _multi_horizon_scores_with_metadata,
)
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger(__name__)


def _derive_signal_date(
    gap_input_dir: Path | None,
    trade_date: str,
    distribution_metadata: dict[str, Any] | None = None,
) -> str:
    """Derive signal_date as the previous TSE trading day of trade_date.

    Gap matrices for *trade_date* are computed from the US close on the prior
    JP business day and the JP opening gap on *trade_date*.  The actual signal
    inputs therefore stop at the previous business day, so the signal_date for
    the leakage audit is that prior business day.  Japanese holidays are
    taken into account so that a holiday does not become the signal date.
    """
    metadata = distribution_metadata
    if metadata is None and gap_input_dir is not None and is_gap_store_path(gap_input_dir):
        try:
            _mu, _omega, metadata = GapStore(gap_input_dir).load_horizon(trade_date)
        except Exception as exc:  # pragma: no cover - defensive cache fallback
            logger.warning("[%s] Could not read gap bundle metadata: %s", trade_date, exc)
    if metadata:
        raw_signal_date = metadata.get("sig_date", metadata.get("signal_date"))
        if raw_signal_date is None and isinstance(metadata.get("horizons"), dict):
            signal_dates = [
                item.get("metadata", {}).get("sig_date")
                for item in metadata["horizons"].values()
                if isinstance(item, dict)
            ]
            signal_dates = [value for value in signal_dates if value is not None]
            if signal_dates:
                # The full list is audited separately; this value is retained
                # for the legacy single-date result field.
                raw_signal_date = min(signal_dates)
        if raw_signal_date is not None:
            return str(normalize_jst_date(raw_signal_date).strftime("%Y-%m-%d"))

    trade_dt = cast(pd.Timestamp, normalize_jst_date(trade_date))
    prev_day = previous_trading_day(trade_dt.to_pydatetime())
    return prev_day.strftime("%Y-%m-%d")


def _distribution_signal_dates(
    distribution_metadata: dict[str, Any] | None,
    fallback_signal_date: str,
) -> list[str]:
    """Return every signal date that contributed to a distribution."""
    if not distribution_metadata:
        return [fallback_signal_date]
    dates: list[str] = []
    horizons = distribution_metadata.get("horizons")
    if isinstance(horizons, dict):
        for item in horizons.values():
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                value = metadata.get("sig_date", metadata.get("signal_date"))
                if value is not None:
                    dates.append(str(normalize_jst_date(value).strftime("%Y-%m-%d")))
    if not dates:
        value = distribution_metadata.get("sig_date", distribution_metadata.get("signal_date"))
        if value is not None:
            dates.append(str(normalize_jst_date(value).strftime("%Y-%m-%d")))
    return dates or [fallback_signal_date]


def generate_v2_production_portfolio_from_distribution(
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    trade_date: str,
    run_config: ProductionV2RunConfig,
    df_exec: pd.DataFrame | None,
    gap_input_dir: Path | None,
    scores: np.ndarray | None = None,
    cache: dict | None = None,
    distribution_metadata: dict[str, Any] | None = None,
    macro_prices: pd.DataFrame | None = None,
    pit_ir_history: np.ndarray | None = None,
    pit_history_trade_dates: np.ndarray | None = None,
    allow_implicit_io: bool = True,
    rank_reversal_signal: np.ndarray | None = None,
) -> PortfolioDecision:
    """Build a V2 portfolio from a pre-computed (mu_gap, Omega_gap) distribution.

    This is the rank/RuleD/weight stage of the pipeline.  It does **not** load
    gap matrix files and it does **not** run the on-demand BLPX computation.
    Multi-horizon blending (if any) must already be reflected in *mu_gap* / *Omega_gap*
    or in the optional *scores* argument.

    Returns the standard V2 result including ``w_final``, ``scores``,
    ``pit_binning``, ``summary`` and audits.
    """
    from leadlag.models.v2 import VERSION

    n_j = len(JP_TICKERS)
    date_str = str(normalize_jst_date(trade_date).strftime("%Y-%m-%d"))
    alerts: list[str] = []

    # Ensure PSD and optionally apply macro adjustments.
    mu_gap, omega_gap, alerts = _repair_and_adjust(
        mu_gap,
        omega_gap,
        run_config,
        date_str,
        n_j,
        alerts,
        cache=cache,
        macro_prices=macro_prices,
        allow_implicit_io=allow_implicit_io,
    )

    # Scores (mu_over_sigma).  If the caller already blended multiple horizons,
    # use the supplied scores; otherwise derive from mu_gap / sigma_gap.
    sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), run_config.sigma_floor))
    if scores is None:
        scores = mu_gap / sigma_gap

    # Cross-sectional rank-reversal overlay (file-based pre-computed signal).
    scores, alerts = _apply_rank_reversal_overlay(
        scores, gap_input_dir, date_str, run_config, alerts,
        rank_reversal_signal=rank_reversal_signal,
        allow_implicit_io=allow_implicit_io,
    )

    # Long/short selection.
    sorted_idx = np.argsort(scores)
    short_idx = sorted_idx[:run_config.short_count]
    long_idx = sorted_idx[-run_config.long_count:]

    # Pre-gross weights.
    if run_config.minvar_enabled:
        w_minvar = build_weights_minvar(
            signal=scores,
            q=float(run_config.long_count) / n_j,
            n_j=n_j,
            Sigma_YY=omega_gap,
            alpha=run_config.minvar_alpha,
            enforce_sign=False,
        )
        w_pre = w_minvar * (run_config.baseline_gross / 2.0)
        logger.info("[%s] MinVar weights applied: alpha=%.2f, gross=%.4f", date_str, run_config.minvar_alpha, float(np.sum(np.abs(w_pre))))
    else:
        w_pre = solve_baseline_style(scores, long_idx, short_idx, baseline_gross=run_config.baseline_gross)

    # PIT binning and RuleD.
    w_final, pit_binning, alerts, pit_history_trade_dates = _apply_pit_ruleD(
        w_pre,
        mu_gap,
        omega_gap,
        gap_input_dir,
        date_str,
        run_config,
        alerts,
        pit_ir_history=pit_ir_history,
        pit_history_trade_dates=pit_history_trade_dates,
        allow_implicit_io=allow_implicit_io,
    )

    # Safety audits and final assembly.
    signal_date = _derive_signal_date(gap_input_dir, date_str, distribution_metadata)
    signal_dates = _distribution_signal_dates(distribution_metadata, signal_date)
    return _run_safety_audits(
        w_final=w_final,
        scores=scores,
        mu_gap=mu_gap,
        Omega_gap=omega_gap,
        sigma_gap=sigma_gap,
        gap_input_dir=gap_input_dir,
        date_str=date_str,
        signal_date=signal_date,
        signal_dates=signal_dates,
        run_cfg=run_config,
        fallback={"gap_data_missing": False},
        pit_binning=pit_binning,
        alerts=alerts,
        pit_history_trade_dates=pit_history_trade_dates,
        candidate="primary_ruleD",
        version=VERSION,
        diagnostics={"distribution_provenance": distribution_metadata}
        if distribution_metadata is not None else None,
    )


def _file_cache_or_flat(
    model: Any,
    trade_date: str,
    gap_input_dir: Path | None,
) -> PortfolioDecision:
    """Load pre-computed gap matrices or return a flat-position result.

    This is the file-cache decision path; it does not use the on-demand
    BLPX model.  It is now a thin wrapper around the ``FallbackPolicy``
    chain for backward compatibility.
    """
    policy = FallbackPolicy.default(model, use_file_cache=True, gap_input_dir=gap_input_dir)
    result = policy.resolve(trade_date)
    if result.is_flat:
        return cast(PortfolioDecision, result.flat_decision)
    assert result.mu_gap is not None and result.Omega_gap is not None
    return generate_v2_production_portfolio_from_distribution(
        mu_gap=result.mu_gap,
        omega_gap=result.Omega_gap,
        trade_date=trade_date,
        run_config=model.run_config,
        df_exec=None,
        gap_input_dir=gap_input_dir,
        scores=None,
        cache=model._macro_price_cache,
        distribution_metadata=result.metadata,
    )


def _decide(
    model: Any,
    trade_date: str,
    gap_input_dir: Path | None = None,
    overlay_enabled: bool = True,
    use_file_cache: bool = True,
    inputs: DecisionInputs | None = None,
) -> PortfolioDecision:
    """Resolve one typed decision, or a cache-only request without market data.

    Public legacy arguments are normalized by ProductionV2Model before this
    internal boundary. There is no precedence among competing data sources.
    """
    df_exec = None
    current_prices = None
    snapshot = None
    open_910_returns = None
    macro_prices = None
    pit_ir_history = None
    pit_history_trade_dates = None
    rank_reversal_signal = None
    # Public legacy arguments are normalized by ProductionV2Model through the
    # adapter source below; preserve their historical adapter-owned PIT/macro
    # reads.  Direct runner/backtest contracts must provide run-owned frames.
    allow_implicit_io = inputs is None or (
        inputs is not None
        and inputs.known.source in {"public_model_adapter", "pit_lake"}
    )
    if inputs is not None:
        input_date = inputs.trade_date.strftime("%Y-%m-%d")
        if trade_date is not None and normalize_jst_date(trade_date) != inputs.trade_date:
            raise ValueError("trade_date does not match DecisionInputs.known.trade_date")
        trade_date = input_date
        gap_input_dir = inputs.gap_input_dir
        # The model sees only rows and close-derived labels available at the
        # known timestamp.  The owned HistoricalInputs object remains intact
        # for reproducibility and later evaluation.
        df_exec = inputs.historical.calculation_frame(inputs.known.as_of)
        open_910_returns = inputs.historical.open_910_returns
        macro_prices = inputs.historical.macro_prices
        pit_ir_history = inputs.historical.pit_ir_history_for(inputs.known.as_of)
        pit_history_trade_dates = inputs.historical.pit_history_trade_dates_for(inputs.known.as_of)
        rank_frame = inputs.historical.rank_reversal_signals
        if rank_frame is not None and inputs.trade_date in rank_frame.index:
            rank_reversal_signal = rank_frame.loc[inputs.trade_date].to_numpy(dtype=float, copy=True)
        current_prices = dict(inputs.known.current_prices)
        known = inputs.known
        snapshot = MarketSnapshot(
            as_of=known.as_of,
            trade_date=input_date,
            us_returns=known.us_returns,
            jp_gap_returns=known.jp_gap_returns,
            jp_betas=known.jp_betas,
            topix_night_return=known.topix_night_return,
            current_prices=known.current_prices,
            price_sources=known.price_sources,
            prev_closes=known.prev_closes,
        )
    if trade_date is None:
        raise ValueError("trade_date is required when DecisionInputs is not supplied")
    if gap_input_dir is not None:
        gap_input_dir = Path(gap_input_dir)

    # Multi-horizon blend still uses its own file->on-demand loop; for single
    # horizon and for the no-blpx file-cache path we use the FallbackPolicy.
    if (
        model._blpx_model is not None
        and df_exec is not None
        and current_prices is not None
        and model.run_config.mh_blend_enabled
        and len(model.run_config.mh_horizons) > 1
    ):
        try:
            mu_gap, omega_gap, scores, distribution_metadata = _multi_horizon_scores_with_metadata(
                model,
                trade_date=trade_date,
                df_exec=df_exec,
                current_prices=current_prices,
                use_file_cache=use_file_cache,
                snapshot=snapshot,
                gap_input_dir=gap_input_dir,
                open_910_returns=open_910_returns,
                allow_implicit_io=allow_implicit_io,
            )
        except DistributionResolutionError as e:
            if not model.run_config.fallback_on_gap_data_missing:
                raise
            logger.error(
                "[%s] Multi-horizon computation failed: %s. Falling back to flat.",
                trade_date, e,
            )
            from leadlag.models.v2.distribution_source import FlatPositionSource

            error_text = str(e)
            provenance_failure = e.reason == DistributionReason.PROVENANCE_REJECTED or any(
                attempt.status == DistributionStatus.REJECTED for attempt in e.attempts
            )
            flat_result = FlatPositionSource(model).resolve_failure(
                trade_date,
                alerts=[f"Multi-horizon computation failed; flat fallback applied: {e}"],
                fallback={
                    "gap_data_missing": not provenance_failure,
                    "audit_failure": provenance_failure,
                },
                diagnostics={
                    "distribution_provenance": {
                        "status": "rejected",
                        "error": error_text,
                    },
                    "distribution_resolution": {
                        "status": DistributionStatus.FLAT.value,
                        "reason": e.reason.value,
                        "attempts": [attempt.to_dict() for attempt in e.attempts],
                    },
                },
                attempts=e.attempts,
            )
            if flat_result.flat_decision is None:
                raise RuntimeError("Flat fallback failed after multi-horizon error.")
            return flat_result.flat_decision
        except Exception as e:
            if not model.run_config.fallback_on_gap_data_missing:
                raise
            logger.error(
                "[%s] Multi-horizon computation failed: %s. Falling back to flat.",
                trade_date, e,
            )
            from leadlag.models.v2.distribution_source import FlatPositionSource

            flat_result = FlatPositionSource(model).resolve_failure(
                trade_date,
                alerts=[f"Multi-horizon computation failed; flat fallback applied: {e}"],
                fallback={"gap_data_missing": True, "audit_failure": False},
                diagnostics={
                    "distribution_resolution": {
                        "status": DistributionStatus.FLAT.value,
                        "reason": DistributionReason.UNKNOWN.value,
                    }
                },
            )
            if flat_result.flat_decision is None:
                raise RuntimeError("Flat fallback failed after multi-horizon error.")
            return flat_result.flat_decision

        result = generate_v2_production_portfolio_from_distribution(
            mu_gap=mu_gap,
            omega_gap=omega_gap,
            trade_date=trade_date,
            run_config=model.run_config,
            df_exec=df_exec,
            gap_input_dir=gap_input_dir,
            scores=scores,
            cache=model._macro_price_cache,
            distribution_metadata=distribution_metadata,
            macro_prices=macro_prices,
            pit_ir_history=pit_ir_history,
            pit_history_trade_dates=pit_history_trade_dates,
            allow_implicit_io=allow_implicit_io,
            rank_reversal_signal=rank_reversal_signal,
        )
    else:
        # Single-horizon (or no-blpx) path via the FallbackPolicy chain.
        if model._blpx_model is not None and df_exec is not None and current_prices is None:
            raise ValueError("current_prices is required for on-demand V2 decision.")

        policy = FallbackPolicy.default(model, use_file_cache=use_file_cache, gap_input_dir=gap_input_dir)
        dist = policy.resolve(
            trade_date,
            df_exec,
            current_prices,
            horizon=1,
            snapshot=snapshot,
            open_910_returns=open_910_returns,
            allow_implicit_io=allow_implicit_io,
        )

        # Every non-flat distribution must prove its signal date, including
        # legacy ``.npy`` bundles. Retry an invalid file-cache result through
        # the permitted on-demand source before applying a provenance-specific
        # flat fallback. A normal portfolio must never be generated from an
        # unprovenanced matrix.
        if not dist.is_flat:
            from leadlag.models.v2.distribution_source import (
                FlatPositionSource,
                _validate_distribution_metadata,
            )

            validated_metadata, provenance_alerts = _validate_distribution_metadata(
                dist.metadata, trade_date, 1
            )
            if provenance_alerts and dist.source == "file_cache" and use_file_cache:
                ondemand = FallbackPolicy.default(model, use_file_cache=False, gap_input_dir=gap_input_dir).resolve(
                    trade_date,
                    df_exec,
                    current_prices,
                    horizon=1,
                    snapshot=snapshot,
                    open_910_returns=open_910_returns,
                    allow_implicit_io=allow_implicit_io,
                )
                if not ondemand.is_flat:
                    validated_metadata, provenance_alerts = _validate_distribution_metadata(
                        ondemand.metadata, trade_date, 1
                    )
                    if not provenance_alerts:
                        dist = ondemand
                else:
                    provenance_alerts = provenance_alerts + list(ondemand.alerts or [])
            if provenance_alerts:
                failure_text = "; ".join(provenance_alerts)
                flat = FlatPositionSource(model).resolve_failure(
                    trade_date,
                    alerts=[f"Single-horizon provenance rejected; flat fallback applied: {failure_text}"],
                    fallback={"gap_data_missing": False, "audit_failure": True},
                    diagnostics={
                        "distribution_provenance": {
                            "status": "rejected",
                            "error": failure_text,
                        },
                        "distribution_resolution": {
                            "status": DistributionStatus.FLAT.value,
                            "reason": DistributionReason.PROVENANCE_REJECTED.value,
                            "attempts": [attempt.to_dict() for attempt in dist.attempts],
                        },
                    },
                    reason=DistributionReason.FLAT_FALLBACK,
                    attempts=dist.attempts,
                    horizon=1,
                )
                if flat.flat_decision is None:
                    raise RuntimeError("Flat fallback failed after single-horizon provenance error.")
                return flat.flat_decision
            if validated_metadata is not None:
                dist = replace(dist, metadata=validated_metadata)

        if dist.is_flat:
            result = cast(PortfolioDecision, dist.flat_decision)
        else:
            assert dist.mu_gap is not None and dist.Omega_gap is not None
            result = generate_v2_production_portfolio_from_distribution(
                mu_gap=dist.mu_gap,
                omega_gap=dist.Omega_gap,
                trade_date=trade_date,
                run_config=model.run_config,
                df_exec=df_exec,
                gap_input_dir=gap_input_dir,
                scores=None,
                cache=model._macro_price_cache,
                distribution_metadata=dist.metadata,
                macro_prices=macro_prices,
                pit_ir_history=pit_ir_history,
                pit_history_trade_dates=pit_history_trade_dates,
                allow_implicit_io=allow_implicit_io,
                rank_reversal_signal=rank_reversal_signal,
            )

    # Optional overlay.
    result = _apply_overlay(
        model,
        result,
        trade_date,
        df_exec,
        overlay_enabled,
        snapshot=snapshot,
        adr_features=inputs.historical.adr_features if inputs is not None else None,
        allow_implicit_io=allow_implicit_io,
    )
    if snapshot is not None and snapshot.price_sources:
        result = replace(result, summary={
            **result.summary,
            "execution_price_sources": dict(snapshot.price_sources),
        })
    return result
