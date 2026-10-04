"""Fitted model contract for the ML order overlay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MLOrderOverlayModel:
    """Fitted estimator and its feature, scaling, and provenance contract."""

    lgbm: Any
    cont_cols: list[str]
    target_std: float
    use_ticker: bool
    use_classification: bool
    per_ticker_interactions: bool
    p_trade_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
