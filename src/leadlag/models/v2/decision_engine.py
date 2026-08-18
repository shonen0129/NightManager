"""V2 decision orchestration and final weight construction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.core.market_calendar import previous_trading_day
from leadlag.core.portfolio import solve_baseline_style
from leadlag.core.signal import build_weights_minvar
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.models.v2.audit_comparator import _run_safety_audits
from leadlag.models.v2.distribution_resolver import (
    _apply_pit_ruleD,
    _repair_and_adjust,
)
from leadlag.models.v2.fallback_policy import FallbackPolicy
from leadlag.models.v2.overlay_applier import (
    _apply_overlay,
    _apply_rank_reversal_overlay,
    _multi_horizon_scores,
)

logger = logging.getLogger(__name__)


def _derive_signal_date(gap_input_dir: Path | None, trade_date: str) -> str:
    """Derive signal_date as the previous TSE trading day of trade_date.

    Gap matrices for *trade_date* are computed from the US close on the prior
    JP business day and the JP opening gap on *trade_date*.  The actual signal
    inputs therefore stop at the previous business day, so the signal_date for
    the leakage audit is that prior business day.  Japanese holidays are
    taken into account so that a holiday does not become the signal date.
    """
    trade_dt = cast(pd.Timestamp, pd.to_datetime(trade_date).normalize())
    prev_day = previous_trading_day(trade_dt.to_pydatetime())
    return prev_day.strftime("%Y-%m-%d")


def generate_v2_production_portfolio_from_distribution(
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    trade_date: str,
    run_config: ProductionV2RunConfig,
    df_exec: pd.DataFrame | None,
    gap_input_dir: Path | None,
    scores: np.ndarray | None = None,
    cache: dict | None = None,
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
    date_str = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
    alerts: list[str] = []

    # Ensure PSD and optionally apply macro adjustments.
    mu_gap, omega_gap, alerts = _repair_and_adjust(
        mu_gap, omega_gap, run_config, date_str, n_j, alerts, cache=cache
    )

    # Scores (mu_over_sigma).  If the caller already blended multiple horizons,
    # use the supplied scores; otherwise derive from mu_gap / sigma_gap.
    sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), run_config.sigma_floor))
    if scores is None:
        scores = mu_gap / sigma_gap

    # Cross-sectional rank-reversal overlay (file-based pre-computed signal).
    scores, alerts = _apply_rank_reversal_overlay(
        scores, gap_input_dir, date_str, run_config, alerts
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
        w_pre, mu_gap, omega_gap, gap_input_dir, date_str, run_config, alerts
    )

    # Safety audits and final assembly.
    signal_date = _derive_signal_date(gap_input_dir, date_str)
    return _run_safety_audits(
        w_final=w_final,
        scores=scores,
        mu_gap=mu_gap,
        Omega_gap=omega_gap,
        sigma_gap=sigma_gap,
        gap_input_dir=gap_input_dir,
        date_str=date_str,
        signal_date=signal_date,
        run_cfg=run_config,
        fallback={"gap_data_missing": False},
        pit_binning=pit_binning,
        alerts=alerts,
        pit_history_trade_dates=pit_history_trade_dates,
        candidate="primary_ruleD",
        version=VERSION,
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
    policy = FallbackPolicy.default(model, use_file_cache=True)
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
    )


def _decide(
    model: Any,
    trade_date: str,
    gap_input_dir: str | Path | None = None,
    df_exec: pd.DataFrame | None = None,
    current_prices: dict[str, float] | None = None,
    overlay_enabled: bool = True,
    use_file_cache: bool = True,
    lake: PITDataLake | None = None,
    snapshot: MarketSnapshot | None = None,
) -> PortfolioDecision:
    """Generate the v2 portfolio decision for *trade_date*.

    The decision is resolved by a ``FallbackPolicy`` distribution source chain:
      1. File cache (pre-computed Step 2 matrices, with optional shadow
         on-demand validation).
      2. On-demand BLPX computation from 9:10 prices.
      3. Flat position (``w_final=0``) as the terminal fallback.

    Args:
        trade_date: Execution date in ``YYYY-MM-DD`` format.
        gap_input_dir: Directory with pre-computed gap matrices, or None.
        df_exec: Execution DataFrame (required for on-demand path).
            Ignored when ``lake`` is provided.
        current_prices: 9:10 JP open prices by ticker (required for on-demand).
            Ignored when ``snapshot`` is provided.
        overlay_enabled: Whether to apply the ML overlay if configured.
        use_file_cache: Prefer the Step 2 file cache when available
            (production). Set to False to try on-demand first.
        lake: Optional PITDataLake holding the execution frame. If given,
            ``df_exec`` is taken from ``lake.df_exec``.
        snapshot: Optional MarketSnapshot for ``trade_date``. If given,
            ``current_prices`` are taken from ``snapshot.current_prices``.

    Returns:
        V2 portfolio decision.
    """
    if gap_input_dir is not None:
        gap_input_dir = Path(gap_input_dir)

    # PIT data lake is the preferred data source. If a snapshot is supplied,
    # use it for the per-date market state and the lake for the full history.
    if lake is not None:
        df_exec = lake.df_exec
        if snapshot is None:
            snapshot = lake.get_snapshot(trade_date)
    if snapshot is not None:
        # The MarketSnapshot is the single point-in-time input; its prices take
        # precedence over any explicit current_prices mapping.
        current_prices = dict(snapshot.current_prices)

    # Keep the directory available to distribution sources.
    model._current_gap_input_dir = gap_input_dir

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
            mu_gap, omega_gap, scores = _multi_horizon_scores(
                model,
                trade_date=trade_date,
                df_exec=df_exec,
                current_prices=current_prices,
                use_file_cache=use_file_cache,
                snapshot=snapshot,
            )
        except Exception as e:
            if not model.run_config.fallback_on_gap_data_missing:
                raise
            logger.error(
                "[%s] Multi-horizon computation failed: %s. Falling back to file cache / flat.",
                trade_date, e,
            )
            return _file_cache_or_flat(model, trade_date, gap_input_dir)

        result = generate_v2_production_portfolio_from_distribution(
            mu_gap=mu_gap,
            omega_gap=omega_gap,
            trade_date=trade_date,
            run_config=model.run_config,
            df_exec=df_exec,
            gap_input_dir=gap_input_dir,
            scores=scores,
            cache=model._macro_price_cache,
        )
    else:
        # Single-horizon (or no-blpx) path via the FallbackPolicy chain.
        if model._blpx_model is not None and df_exec is not None and current_prices is None:
            raise ValueError("current_prices is required for on-demand V2 decision.")

        policy = FallbackPolicy.default(model, use_file_cache=use_file_cache)
        dist = policy.resolve(trade_date, df_exec, current_prices, horizon=1, snapshot=snapshot)

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
            )

    # Optional overlay.
    return _apply_overlay(model, result, trade_date, df_exec, overlay_enabled, snapshot=snapshot)
