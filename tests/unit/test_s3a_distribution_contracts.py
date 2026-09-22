"""S3a regression tests for typed distribution outcomes and fallback traces."""

from __future__ import annotations

import numpy as np

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.distribution import (
    DistributionAttempt,
    DistributionReason,
    DistributionResolutionError,
    DistributionResult,
    DistributionStatus,
)
from leadlag.models.v2.distribution_source import DistributionSource, FlatPositionSource
from leadlag.models.v2.fallback_policy import FallbackPolicy


class _MockModel:
    def __init__(self) -> None:
        self.run_config = ProductionV2RunConfig()
        self.n_j = len(JP_TICKERS)
        self._blpx_model = None


def _ready_result(source: str = "cache") -> DistributionResult:
    n_j = len(JP_TICKERS)
    return DistributionResult(
        mu_gap=np.ones(n_j) * 0.01,
        Omega_gap=np.eye(n_j) * 0.001,
        source=source,
        status=DistributionStatus.READY,
        reason=DistributionReason.RESOLVED,
        horizon=1,
    )


def test_distribution_result_keeps_legacy_fields_coherent() -> None:
    ready = _ready_result()
    unavailable = DistributionResult(
        source="cache",
        alerts=["cache missing"],
        status=DistributionStatus.UNAVAILABLE,
        reason=DistributionReason.CACHE_MISSING,
    )
    flat = DistributionResult(
        source="flat_position",
        status=DistributionStatus.FLAT,
        reason=DistributionReason.FLAT_FALLBACK,
    )

    assert ready.is_ready and ready.is_available and not ready.is_flat
    assert ready.horizon == 1
    assert ready.attempts[0].reason == DistributionReason.RESOLVED
    assert not unavailable.is_available and not unavailable.is_flat
    assert unavailable.reason == DistributionReason.CACHE_MISSING
    assert flat.is_flat and flat.is_available and not flat.is_ready


def test_fallback_policy_uses_typed_rejection_without_alert_text_heuristics() -> None:
    class _TypedFailure(DistributionSource):
        name = "typed_failure"

        def resolve(self, trade_date, df_exec, current_prices, *, horizon=1, snapshot=None):
            return DistributionResult(
                source=self.name,
                alerts=["input was rejected by validator"],
                status=DistributionStatus.REJECTED,
                reason=DistributionReason.PROVENANCE_REJECTED,
            )

    result = FallbackPolicy().add(_TypedFailure(_MockModel())).add(
        FlatPositionSource(_MockModel())
    ).resolve("2024-01-01")

    assert result.is_flat
    assert result.flat_decision.fallback["audit_failure"] is True
    assert result.attempts[0].reason == DistributionReason.PROVENANCE_REJECTED
    assert result.attempts[-1].reason == DistributionReason.FLAT_FALLBACK


def test_fallback_policy_does_not_promote_alert_word_to_audit_failure() -> None:
    class _TextOnlyFailure(DistributionSource):
        name = "text_only_failure"

        def resolve(self, trade_date, df_exec, current_prices, *, horizon=1, snapshot=None):
            return DistributionResult(
                source=self.name,
                alerts=["provenance-like text from an upstream log"],
                status=DistributionStatus.UNAVAILABLE,
                reason=DistributionReason.CACHE_INVALID,
            )

    model = _MockModel()
    result = FallbackPolicy().add(_TextOnlyFailure(model)).add(FlatPositionSource(model)).resolve(
        "2024-01-01"
    )

    assert result.is_flat
    assert result.flat_decision.fallback.get("audit_failure", False) is False


def test_resolution_error_carries_reason_and_attempts() -> None:
    attempts = (
        DistributionAttempt(
            source="file_cache",
            status=DistributionStatus.REJECTED,
            reason=DistributionReason.PROVENANCE_REJECTED,
        ),
    )
    error = DistributionResolutionError(
        DistributionReason.PROVENANCE_REJECTED,
        "signal date rejected",
        attempts=attempts,
    )

    assert error.reason == DistributionReason.PROVENANCE_REJECTED
    assert error.attempts == attempts
