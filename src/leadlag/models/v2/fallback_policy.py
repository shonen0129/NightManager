"""V2 fallback policy that chains ``DistributionSource`` instances.

The default production chain is:

1. ``FileCacheDistributionSource`` — trusted pre-computed Step 2 matrices.
2. ``OnDemandDistributionSource`` — BLPX computation from 9:10 prices.
3. ``FlatPositionSource`` — terminal zero-weight decision.

The ``FallbackPolicy`` tries each source in order, logs the resolution path, and
returns the first successful ``DistributionResult``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from leadlag.data.pit_lake import MarketSnapshot
from leadlag.models.v2.distribution_source import (
    DistributionResult,
    DistributionSource,
    FileCacheDistributionSource,
    FlatPositionSource,
    OnDemandDistributionSource,
)

logger = logging.getLogger(__name__)


@dataclass
class FallbackPolicy:
    """Ordered chain of ``DistributionSource`` instances."""

    sources: list[DistributionSource] = field(default_factory=list)

    def add(self, source: DistributionSource) -> FallbackPolicy:
        """Append a source and return ``self`` for chaining."""
        self.sources.append(source)
        return self

    def resolve(
        self,
        trade_date: str,
        df_exec: Any | None = None,
        current_prices: dict[str, float] | None = None,
        *,
        horizon: int = 1,
        snapshot: MarketSnapshot | None = None,
    ) -> DistributionResult:
        """Run the chain and return the first successful result.

        If a source returns ``is_available=False`` the next source is tried.
        If no source can resolve, the terminal ``FlatPositionSource`` is used.
        """
        prior_alerts: list[str] = []
        for source in self.sources:
            result = source.resolve(
                trade_date, df_exec, current_prices, horizon=horizon, snapshot=snapshot
            )
            if result.is_flat:
                logger.info("[%s] FallbackPolicy resolved via %s (flat).", trade_date, result.source)
                return replace(result, alerts=(result.alerts or []) + prior_alerts)
            if result.is_available and result.mu_gap is not None and result.Omega_gap is not None:
                logger.info("[%s] FallbackPolicy resolved via %s.", trade_date, result.source)
                return replace(result, alerts=(result.alerts or []) + prior_alerts)
            prior_alerts.extend(result.alerts or [])
            logger.info("[%s] FallbackPolicy: %s unavailable, trying next source.", trade_date, result.source)

        # Terminal flat guard — should only be reached if no FlatPositionSource was added.
        logger.error("[%s] FallbackPolicy exhausted with no resolution; forcing flat.", trade_date)
        flat_result = FlatPositionSource(self._model_from_sources()).resolve(
            trade_date, df_exec, current_prices, horizon=horizon, snapshot=snapshot
        )
        return replace(flat_result, alerts=(flat_result.alerts or []) + prior_alerts)

    def _model_from_sources(self) -> Any:
        """Return the model attached to the first source (for terminal flat)."""
        if not self.sources:
            raise RuntimeError("FallbackPolicy has no sources and no explicit model for terminal flat.")
        return self.sources[0].model

    @classmethod
    def default(
        cls,
        model: Any,
        *,
        use_file_cache: bool = True,
    ) -> FallbackPolicy:
        """Build the standard production distribution source chain for *model*.

        ``use_file_cache=True`` prefers the Step 2 file cache; ``False``
        prefers on-demand but still falls back to the file cache and then
        to a flat position, matching the legacy ``compute_distribution``
        behavior used by shadow runs.
        """
        chain = cls()
        file_source = FileCacheDistributionSource(model)
        ondemand_source = OnDemandDistributionSource(model)
        if use_file_cache:
            chain.add(file_source)
            chain.add(ondemand_source)
        else:
            chain.add(ondemand_source)
            chain.add(file_source)
        chain.add(FlatPositionSource(model))
        return chain
