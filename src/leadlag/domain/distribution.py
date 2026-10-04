"""Typed outcomes for gap-distribution resolution.

Status and reason are the source of truth for state transitions. Alerts carry
human-readable detail and never determine the fallback classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True, kw_only=True)
class DistributionResult:
    """Outcome of one distribution-source attempt or terminal fallback.

    Every source supplies an explicit ``status`` and ``reason``.
    """

    status: DistributionStatus
    reason: DistributionReason
    mu_gap: np.ndarray | None = None
    Omega_gap: np.ndarray | None = None
    flat_decision: PortfolioDecision | None = None
    source: str = ""
    alerts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    attempts: tuple[DistributionAttempt, ...] = ()
    horizon: int | None = None

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        if not attempts and self.source:
            attempts = (
                DistributionAttempt(
                    source=self.source,
                    status=self.status,
                    reason=self.reason,
                    detail=(self.alerts[0] if self.alerts else None),
                ),
            )
        object.__setattr__(self, "attempts", attempts)
