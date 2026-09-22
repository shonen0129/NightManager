"""Typed outcomes for gap-distribution resolution.

The distribution resolver used to communicate availability and failure causes
through a pair of booleans and free-form alert strings.  These types make the
state transition explicit while keeping the legacy properties available at
the outer compatibility boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from leadlag.domain.portfolio import PortfolioDecision


class DistributionStatus(StrEnum):
    """Lifecycle state returned by a distribution source."""

    READY = "ready"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    FLAT = "flat"


class DistributionReason(StrEnum):
    """Stable reason code for a distribution resolution outcome."""

    RESOLVED = "resolved"
    CACHE_NOT_CONFIGURED = "cache_not_configured"
    CACHE_MISSING = "cache_missing"
    CACHE_INVALID = "cache_invalid"
    PROVENANCE_REJECTED = "provenance_rejected"
    ON_DEMAND_DISABLED = "on_demand_disabled"
    MODEL_UNAVAILABLE = "model_unavailable"
    INPUTS_MISSING = "inputs_missing"
    COMPUTATION_FAILED = "computation_failed"
    SHADOW_VALIDATION_FAILED = "shadow_validation_failed"
    FLAT_FALLBACK = "flat_fallback"
    POLICY_EXHAUSTED = "policy_exhausted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DistributionAttempt:
    """One source attempt retained in the final resolution trace."""

    source: str
    status: DistributionStatus
    reason: DistributionReason
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status.value,
            "reason": self.reason.value,
            "detail": self.detail,
        }


class DistributionResolutionError(RuntimeError):
    """Failure carrying a typed reason through multi-horizon orchestration."""

    def __init__(
        self,
        reason: DistributionReason,
        message: str,
        *,
        attempts: tuple[DistributionAttempt, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.attempts = attempts


@dataclass(frozen=True)
class DistributionResult:
    """Outcome of one distribution-source attempt or terminal fallback.

    ``status`` and ``reason`` are the canonical fields.  ``is_flat`` and
    ``is_available`` remain as compatibility properties represented as fields
    so existing callers and serialized summaries keep working during S3.
    """

    mu_gap: np.ndarray | None = None
    Omega_gap: np.ndarray | None = None
    is_flat: bool = False
    flat_decision: PortfolioDecision | None = None
    source: str = ""
    alerts: list[str] | None = None
    is_available: bool = False
    metadata: dict[str, Any] | None = None
    status: DistributionStatus | None = None
    reason: DistributionReason | None = None
    attempts: tuple[DistributionAttempt, ...] = ()
    horizon: int | None = None

    def __post_init__(self) -> None:
        status = self.status
        if status is None:
            if self.is_flat:
                status = DistributionStatus.FLAT
            elif self.is_available and self.mu_gap is not None and self.Omega_gap is not None:
                status = DistributionStatus.READY
            else:
                status = DistributionStatus.UNAVAILABLE
            object.__setattr__(self, "status", status)

        reason = self.reason
        if reason is None:
            reason = {
                DistributionStatus.READY: DistributionReason.RESOLVED,
                DistributionStatus.FLAT: DistributionReason.FLAT_FALLBACK,
            }.get(status, DistributionReason.UNKNOWN)
            object.__setattr__(self, "reason", reason)

        # Keep legacy booleans coherent with the typed status.
        object.__setattr__(self, "is_flat", status == DistributionStatus.FLAT)
        object.__setattr__(
            self,
            "is_available",
            status in (DistributionStatus.READY, DistributionStatus.FLAT),
        )
        if self.alerts is None:
            object.__setattr__(self, "alerts", [])

        if self.horizon is None and isinstance(self.metadata, dict):
            raw_horizon = self.metadata.get("horizon")
            if raw_horizon is not None:
                try:
                    object.__setattr__(self, "horizon", int(raw_horizon))
                except (TypeError, ValueError):
                    pass

        attempts = tuple(self.attempts)
        if not attempts and self.source:
            attempts = (
                DistributionAttempt(
                    source=self.source,
                    status=status,
                    reason=reason,
                    detail=(self.alerts[0] if self.alerts else None),
                ),
            )
        object.__setattr__(self, "attempts", attempts)

    @property
    def is_ready(self) -> bool:
        """Whether this result contains a usable distribution."""
        return self.status == DistributionStatus.READY

    @property
    def is_rejected(self) -> bool:
        """Whether the source produced data that failed a safety contract."""
        return self.status == DistributionStatus.REJECTED

    def resolution_trace(self) -> list[dict[str, Any]]:
        """Return a JSON-ready source-attempt trace."""
        return [attempt.to_dict() for attempt in self.attempts]
