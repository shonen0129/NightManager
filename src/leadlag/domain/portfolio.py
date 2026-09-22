"""Domain types for portfolio construction and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RiskBudget:
    """Risk constraints passed to the portfolio builder."""

    max_gross: float
    max_net: float
    max_single_weight: float
    var_confidence: float = 0.99
    var_window: int = 250


@dataclass(frozen=True)
class CostBreakdown:
    """Per-day decision cost decomposition in decimal return units."""

    slippage: float = 0.0
    financing: float = 0.0
    borrow: float = 0.0
    reverse: float = 0.0

    @property
    def unit(self) -> str:
        """Unit label kept explicit when compared with bps/currency costs."""
        return "return_fraction"

    @property
    def total(self) -> float:
        return self.slippage + self.financing + self.borrow + self.reverse


@dataclass(frozen=True)
class PortfolioDecision:
    """Result of the production ``decide`` phase.

    Represents the V2 result shape produced by ``_run_safety_audits`` and
    returned by ``ProductionV2Model.decide``. Consumers use its typed
    attributes; serialization belongs at the reporting boundary.
    """

    w_final: np.ndarray
    scores: np.ndarray
    mu_gap: np.ndarray
    sigma_gap: np.ndarray
    Omega_gap: np.ndarray
    fallback: dict
    pit_binning: dict
    leakage: dict
    numerical: dict
    alerts: list[str]
    summary: dict
    # The concrete application config is owned by the config/execution
    # boundary.  Keeping this field opaque prevents the domain package from
    # depending on Pydantic or any upper layer while preserving the runtime
    # object for reporting and audit consumers.
    run_config: Any
    scores_overlay: np.ndarray | None = None
    costs: CostBreakdown | None = None
    diagnostics: dict | None = None
