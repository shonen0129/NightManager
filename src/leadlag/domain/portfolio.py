"""Domain types for portfolio construction and execution."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from leadlag.config.schemas import ProductionV2RunConfig


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
    """Per-day cost decomposition in decimal return."""

    slippage: float = 0.0
    financing: float = 0.0
    borrow: float = 0.0
    reverse: float = 0.0

    @property
    def total(self) -> float:
        return self.slippage + self.financing + self.borrow + self.reverse


@dataclass(frozen=True)
class PortfolioDecision:
    """Result of the production ``decide`` phase.

    Represents the V2 result shape produced by ``_run_safety_audits`` and
    returned by ``ProductionV2Model.decide``.  It behaves like an immutable
    dict so existing call sites and tests that use ``result["w_final"]``,
    ``result.get(...)``, or key iteration continue to work.
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
    run_config: ProductionV2RunConfig
    scores_overlay: np.ndarray | None = None
    costs: CostBreakdown | None = None
    diagnostics: dict | None = None

    def _field_names(self) -> list[str]:
        return [f.name for f in fields(self)]

    def __getitem__(self, key: Any) -> Any:
        names = self._field_names()
        if key not in names:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Any:
        return iter(self._field_names())

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and key in self._field_names()

    def __len__(self) -> int:
        return len(self._field_names())

    def keys(self) -> list[str]:
        """Return the list of available result keys."""
        return self._field_names()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* if it is set and not ``None``.

        ``None`` values are treated as missing so that optional keys such as
        ``scores_overlay`` can use the standard ``get`` fallback pattern.
        """
        if key in self:
            value = getattr(self, key)
            if value is not None:
                return value
        return default

    def items(self) -> list[tuple[str, Any]]:
        """Return ``(key, value)`` pairs for all fields."""
        return [(k, self[k]) for k in self.keys()]

    def values(self) -> list[Any]:
        """Return the values for all fields."""
        return [self[k] for k in self.keys()]

    def to_dict(self) -> dict[str, Any]:
        """Convert the decision to a plain ``dict``."""
        return dict(self)

    @classmethod
    def from_dict(cls, d: PortfolioDecision | dict[str, Any]) -> PortfolioDecision:
        """Build a ``PortfolioDecision`` from a mapping.

        If *d* is already a ``PortfolioDecision`` it is returned unchanged.
        """
        if isinstance(d, cls):
            return d

        def _get(key: str, default: Any = None) -> Any:
            if isinstance(d, dict):
                return d.get(key, default)
            try:
                return d[key]
            except KeyError:
                return default

        return cls(
            w_final=d["w_final"],
            scores=d["scores"],
            mu_gap=d["mu_gap"],
            sigma_gap=d["sigma_gap"],
            Omega_gap=d["Omega_gap"],
            fallback=d["fallback"],
            pit_binning=d["pit_binning"],
            leakage=d["leakage"],
            numerical=d["numerical"],
            alerts=d["alerts"],
            summary=d["summary"],
            run_config=d["run_config"],
            scores_overlay=_get("scores_overlay", None),
            costs=_get("costs", None),
            diagnostics=_get("diagnostics", None),
        )
