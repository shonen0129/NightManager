"""Operational publication of the production V2 gap distribution cache.

This module deliberately reuses the same on-demand and file-cache sources as
the decision path. It owns no research diagnostics and performs no fallback:
all configured horizons must compute successfully before publication starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.config.schemas import AppConfig
from leadlag.data.gap_store import GapStore
from leadlag.data.intraday_inputs import build_open_910_returns
from leadlag.data.pit_lake import MarketSnapshot, PITDataLake
from leadlag.data.quote_snapshot import FrozenQuoteSnapshot
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER
from leadlag.domain.distribution import DistributionStatus
from leadlag.domain.inputs import HistoricalInputs
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.v2.distribution_source import (
    FileCacheDistributionSource,
    OnDemandDistributionSource,
)
from leadlag.models.v2.gap_io import _extract_gap_inputs, _extract_horizon_snapshot_inputs
from leadlag.runner.model_factory import build_blpx_model
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity


@dataclass(frozen=True)
class GapPublicationResult:
    """Summary of one validated operational gap-cache publication."""

    trade_date: str
    quote_snapshot_id: str
    horizons: tuple[int, ...]
    gap_store: str
    rank_reversal_saved: bool
    verification: dict[int, dict[str, float | str]]


def build_market_snapshot(
    df_exec: pd.DataFrame,
    frozen: FrozenQuoteSnapshot,
    trade_date: str,
) -> tuple[MarketSnapshot, dict[str, float], pd.Timestamp]:
    """Pair an immutable 09:10 quote snapshot with the run-owned PIT frame."""
    quote_time = frozen.as_of.tz_convert("Asia/Tokyo")
    if frozen.trade_date != trade_date or quote_time.date().isoformat() != trade_date:
        raise ValueError("frozen quote date must match requested trade_date")
    decision_as_of = quote_time.tz_localize(None)
    lake = PITDataLake(df_exec)
    trade_ts = pd.Timestamp(trade_date).normalize()
    if trade_ts not in lake.history_frame().index:
        raise ValueError(f"trade_date {trade_date} is not present in df_exec")

    base = lake.get_snapshot(decision_as_of)
    current_prices = {ticker: float(frozen.prices[ticker]) for ticker in JP_TICKERS}
    gap = np.zeros(len(JP_TICKERS), dtype=float)
    for index, ticker in enumerate(JP_TICKERS):
        prior = base.prev_closes.get(ticker)
        if prior is None or not np.isfinite(prior) or float(prior) <= 0.0:
            raise ValueError(f"missing positive prior close for {ticker}")
        gap[index] = current_prices[ticker] / float(prior) - 1.0

    topix_prior = base.prev_closes.get(TOPIX_TICKER)
    topix_price = float(frozen.prices[TOPIX_TICKER])
    if topix_prior is None or not np.isfinite(topix_prior) or float(topix_prior) <= 0.0:
        raise ValueError("missing positive prior TOPIX close")
    topix_night = topix_price / float(topix_prior) - 1.0

    snapshot = MarketSnapshot(
        as_of=base.as_of,
        trade_date=base.trade_date,
        us_returns=base.us_returns,
        jp_gap_returns=gap,
        jp_betas=base.jp_betas,
        topix_night_return=topix_night,
        current_prices=current_prices,
        prev_closes=base.prev_closes,
        price_sources={
            ticker: "tachibana:CLMMfdsGetMarketPrice:quote_mid" for ticker in current_prices
        },
        price_observed_at={ticker: frozen.observed_at[ticker] for ticker in JP_TICKERS},
        quote_snapshot_id=frozen.snapshot_id,
    )
    valid, errors = snapshot.validate()
    if not valid:
        raise ValueError(f"market snapshot validation failed: {errors}")
    return snapshot, current_prices, decision_as_of


def compute_rank_reversal_signal(df_exec: pd.DataFrame, trade_date: str) -> np.ndarray:
    """Return the shifted cross-sectional rank-reversal signal for one date."""
    date = pd.Timestamp(trade_date).normalize()
    if date not in df_exec.index:
        raise ValueError(f"trade_date {trade_date} is not present in df_exec")
    columns = [f"jp_oc_{ticker}" for ticker in JP_TICKERS]
    missing = [column for column in columns if column not in df_exec.columns]
    if missing:
        raise ValueError(f"rank-reversal inputs missing: {missing}")

    frame = df_exec.loc[:date, columns].copy()
    frame.columns = JP_TICKERS
    ranks = frame.shift(1).rank(axis=1)
    return cast(np.ndarray, -ranks.diff().iloc[-1].to_numpy(dtype=float))


def _configured_horizons(config: AppConfig) -> tuple[int, ...]:
    horizons = [1]
    if config.v2.mh_blend_enabled:
        horizons.extend(int(value) for value in config.v2.mh_horizons)
    return tuple(dict.fromkeys(horizons))


def publish_gap_cache(
    *,
    app_config: AppConfig,
    df_exec: pd.DataFrame,
    frozen_snapshot: FrozenQuoteSnapshot,
    gap_store: Path,
    trade_date: str,
) -> GapPublicationResult:
    """Compute, atomically publish, and read-back validate one day's gap bundles."""
    if gap_store.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        raise ValueError("operational gap publication requires a SQLite GapStore")

    snapshot, current_prices, decision_as_of = build_market_snapshot(
        df_exec, frozen_snapshot, trade_date
    )
    open_910_returns = build_open_910_returns(df_exec, JP_TICKERS)
    historical = HistoricalInputs(
        df_exec,
        open_910_returns=open_910_returns,
        source="operational_gap_publisher",
    )
    calculation_frame = historical.calculation_frame(decision_as_of)

    model = ProductionV2Model(
        app_config.v2,
        blpx_model=build_blpx_model(app_config, clear_cache=True),
    )
    on_demand = OnDemandDistributionSource(model)
    horizons = _configured_horizons(app_config)

    computed: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for horizon in horizons:
        result = on_demand.resolve(
            trade_date,
            calculation_frame,
            current_prices,
            horizon=horizon,
            snapshot=snapshot,
            open_910_returns=open_910_returns,
            allow_implicit_io=False,
        )
        if result.status != DistributionStatus.READY or result.mu_gap is None or result.Omega_gap is None:
            details = "; ".join(result.alerts) if result.alerts else result.reason.value
            raise RuntimeError(f"h={horizon} on-demand gap computation failed: {details}")

        gap_inputs: tuple[np.ndarray, np.ndarray, float] | None
        if horizon == 1:
            gap_inputs = _extract_gap_inputs(
                calculation_frame,
                trade_date,
                current_prices,
                snapshot=snapshot,
            )
        else:
            gap_inputs = _extract_horizon_snapshot_inputs(
                calculation_frame,
                trade_date,
                horizon,
                snapshot,
            )
            if gap_inputs is None:
                raise RuntimeError(f"h={horizon} point-in-time gap inputs are unavailable")

        metadata = {
            **(result.metadata or {}),
            **bundle_identity(
                calculation_frame,
                trade_date,
                config=app_config.v2,
                model=model,
                open_910_returns=open_910_returns,
                gap_inputs=gap_inputs,
                horizon=horizon,
            ),
            "quote_snapshot_id": frozen_snapshot.snapshot_id,
            "publisher": "leadlag.pipeline.gap_publisher",
        }
        computed[horizon] = (
            np.asarray(result.mu_gap, dtype=float),
            np.asarray(result.Omega_gap, dtype=float),
            metadata,
        )

    rank_signal = None
    if app_config.v2.cs_overlay_enabled:
        rank_signal = compute_rank_reversal_signal(calculation_frame, trade_date)
        if not np.isfinite(rank_signal).all():
            raise RuntimeError("rank-reversal signal is incomplete")

    # Compute every configured horizon before the first write so an input/model
    # failure cannot publish a knowingly incomplete generation.
    for horizon, (mu_gap, omega_gap, metadata) in computed.items():
        if horizon == 1:
            ok = save_gap_matrices(
                gap_store,
                trade_date,
                mu_gap,
                omega_gap,
                metadata=metadata,
            )
        else:
            ok = save_gap_matrices(
                gap_store,
                trade_date,
                mu_gap,
                omega_gap,
                mu_pattern=app_config.v2.mh_mu_file_pattern_h,
                omega_pattern=app_config.v2.mh_omega_file_pattern_h,
                pattern_kwargs={"h": horizon},
                metadata=metadata,
            )
        if not ok:
            raise RuntimeError(f"h={horizon} gap bundle publication failed")

    rank_saved = False
    if rank_signal is not None:
        GapStore(gap_store).put(trade_date, "rank_reversal", rank_signal)
        rank_saved = True

    cache = FileCacheDistributionSource(model, gap_input_dir=gap_store)
    verification: dict[int, dict[str, float | str]] = {}
    for horizon, (expected_mu, expected_omega, _metadata) in computed.items():
        cached = cache.resolve(
            trade_date,
            calculation_frame,
            current_prices,
            horizon=horizon,
            snapshot=snapshot,
            open_910_returns=open_910_returns,
            allow_implicit_io=False,
        )
        if cached.status != DistributionStatus.READY or cached.mu_gap is None or cached.Omega_gap is None:
            details = "; ".join(cached.alerts) if cached.alerts else cached.reason.value
            raise RuntimeError(f"h={horizon} published cache failed validation: {details}")
        mu_delta = float(np.max(np.abs(np.asarray(cached.mu_gap) - expected_mu)))
        omega_delta = float(np.max(np.abs(np.asarray(cached.Omega_gap) - expected_omega)))
        if mu_delta != 0.0 or omega_delta != 0.0:
            raise RuntimeError(
                f"h={horizon} published cache differs from on-demand result "
                f"(mu={mu_delta}, omega={omega_delta})"
            )
        verification[horizon] = {
            "source": cached.source,
            "max_abs_mu_diff": mu_delta,
            "max_abs_omega_diff": omega_delta,
        }

    return GapPublicationResult(
        trade_date=trade_date,
        quote_snapshot_id=frozen_snapshot.snapshot_id,
        horizons=horizons,
        gap_store=str(gap_store),
        rank_reversal_saved=rank_saved,
        verification=verification,
    )


__all__ = [
    "GapPublicationResult",
    "build_market_snapshot",
    "compute_rank_reversal_signal",
    "publish_gap_cache",
]
