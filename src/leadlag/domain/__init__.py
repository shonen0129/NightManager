"""Domain layer for lead-lag fund models.

Lightweight frozen dataclasses that describe the boundaries between
signal generation, distribution estimation, portfolio construction,
and execution. They are intentionally dependency-free of Pydantic so
research scripts and production can share a stable vocabulary.
"""

from __future__ import annotations

from leadlag.domain.market import AssetReturns, GapDistribution
from leadlag.domain.portfolio import CostBreakdown, PortfolioDecision, RiskBudget
from leadlag.domain.signal import SignalComponent, SignalPackage

__all__ = [
    "AssetReturns",
    "CostBreakdown",
    "GapDistribution",
    "PortfolioDecision",
    "RiskBudget",
    "SignalComponent",
    "SignalPackage",
]
