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
from pathlib import Path
from typing import Any

from leadlag.data.pit_lake import MarketSnapshot
from leadlag.domain.distribution import (
    DistributionAttempt,
    DistributionReason,
    DistributionStatus,
)
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
        open_910_returns: Any | None = None,
        allow_implicit_io: bool = True,
    ) -> DistributionResult:
        """Run the chain and return the first successful result.

        If a source returns ``is_available=False`` the next source is tried.
        If no source can resolve, the terminal ``FlatPositionSource`` is used.
        """
        prior_alerts: list[str] = []
        prior_attempts: list[DistributionAttempt] = []
        for source in self.sources:
            if isinstance(source, FlatPositionSource):
                # Preserve the reason for reaching the terminal source in the
                # decision's fallback flags.  In particular, a fatal bundle
                # integrity/provenance alert is an audit failure even when the
                # on-demand source was unavailable and the final position is
                # flat because of missing inputs.
                fallback = {"gap_data_missing": True}
                audit_failure = any(
                    attempt.reason == DistributionReason.PROVENANCE_REJECTED
                    or attempt.status == DistributionStatus.REJECTED
                    for attempt in prior_attempts
                )
                if audit_failure:
                    fallback["audit_failure"] = True
                terminal_attempt = DistributionAttempt(
                    source=source.name,
                    status=DistributionStatus.FLAT,
                    reason=DistributionReason.FLAT_FALLBACK,
                    detail="All distribution sources failed.",
                )
                diagnostics = None
                if prior_attempts:
                    diagnostics = {
                        "distribution_provenance": {
                            "status": "rejected" if audit_failure else "unavailable",
                            "error": "; ".join(prior_alerts),
                        },
                        "distribution_resolution": {
                            "status": DistributionStatus.FLAT.value,
                            "reason": DistributionReason.FLAT_FALLBACK.value,
                            "attempts": [
                                attempt.to_dict()
                                for attempt in (*prior_attempts, terminal_attempt)
                            ],
                        },
                    }
                result = source.resolve_failure(
                    trade_date,
                    alerts=prior_alerts,
                    fallback=fallback,
                    diagnostics=diagnostics,
                    attempts=tuple(prior_attempts) + (terminal_attempt,),
                    horizon=horizon,
                )
                logger.info("[%s] FallbackPolicy resolved via %s (flat).", trade_date, result.source)
                return result
            resolve_kwargs: dict[str, Any] = {"horizon": horizon, "snapshot": snapshot}
            # Keep third-party/test DistributionSource implementations that
            # predate the explicit snapshot fields source-compatible.  New
            # typed paths opt in by supplying a frame or strict I/O mode.
            if open_910_returns is not None:
                resolve_kwargs["open_910_returns"] = open_910_returns
            if not allow_implicit_io:
                resolve_kwargs["allow_implicit_io"] = False
            result = source.resolve(trade_date, df_exec, current_prices, **resolve_kwargs)
            prior_attempts.extend(result.attempts)
            if result.is_flat:
                logger.info("[%s] FallbackPolicy resolved via %s (flat).", trade_date, result.source)
                return replace(
                    result,
                    alerts=(result.alerts or []) + prior_alerts,
                    attempts=tuple(prior_attempts),
                )
            if result.is_available and result.mu_gap is not None and result.Omega_gap is not None:
                logger.info("[%s] FallbackPolicy resolved via %s.", trade_date, result.source)
                return replace(
                    result,
                    alerts=(result.alerts or []) + prior_alerts,
                    attempts=tuple(prior_attempts),
                )
            prior_alerts.extend(result.alerts or [])
            logger.info("[%s] FallbackPolicy: %s unavailable, trying next source.", trade_date, result.source)

        # Terminal flat guard — should only be reached if no FlatPositionSource was added.
        logger.error("[%s] FallbackPolicy exhausted with no resolution; forcing flat.", trade_date)
        resolve_kwargs = {"horizon": horizon, "snapshot": snapshot}
        if open_910_returns is not None:
            resolve_kwargs["open_910_returns"] = open_910_returns
        if not allow_implicit_io:
            resolve_kwargs["allow_implicit_io"] = False
        flat_result = FlatPositionSource(self._model_from_sources()).resolve(
            trade_date, df_exec, current_prices, **resolve_kwargs
        )
        return replace(
            flat_result,
            alerts=(flat_result.alerts or []) + prior_alerts,
            attempts=tuple(prior_attempts) + flat_result.attempts,
        )

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
        gap_input_dir: Path | None = None,
        mu_pattern: str | None = None,
        omega_pattern: str | None = None,
    ) -> FallbackPolicy:
        """Build the standard production distribution source chain for *model*.

        ``use_file_cache=True`` prefers the Step 2 file cache; ``False``
        prefers on-demand but still falls back to the file cache and then
        to a flat position, matching the legacy ``compute_distribution``
        behavior used by shadow runs.
        """
        chain = cls()
        file_source = FileCacheDistributionSource(model, gap_input_dir=gap_input_dir, mu_pattern=mu_pattern, omega_pattern=omega_pattern)
        ondemand_source = OnDemandDistributionSource(model, gap_input_dir=gap_input_dir)
        if use_file_cache:
            chain.add(file_source)
            chain.add(ondemand_source)
        else:
            chain.add(ondemand_source)
            chain.add(file_source)
        chain.add(FlatPositionSource(model, gap_input_dir=gap_input_dir))
        return chain
