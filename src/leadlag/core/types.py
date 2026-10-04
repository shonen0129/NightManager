"""Type-safe domain models for the lead-lag strategy.

This module defines trading types (dataclasses / Enums).
Application configuration is defined in ``leadlag.config.schemas``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class OrderSide(StrEnum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Order type enumeration."""

    MARKET = "MO"
    LIMIT = "LO"
    CLOSE = "CLO"


class OrderStatus(StrEnum):
    """Order status enumeration."""

    SUBMITTED = "SUBMITTED"
    SIMULATED = "SIMULATED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CapitalAllocation:
    """資金配分の結果."""

    quantities: np.ndarray
    allocated_amounts: np.ndarray
    buy_budget: float
    sell_budget: float
    gross_budget: float = 0.0


@dataclass(frozen=True)
class VarEsResult:
    """VaR/ES計算結果."""

    available: bool
    samples: int
    window: int
    var_loss: float
    es_loss: float
    var_quantile: float = 0.0
    tail_count: int = 0
    var_method: str = "historical"


@dataclass(frozen=True)
class RiskReport:
    """リスクチェック結果."""

    target_net_exposure: float
    target_gross_exposure: float
    allocated_net_ratio: float
    allocated_gross_ratio: float
    var_es: VarEsResult
    warning_breaches: list[str] = field(default_factory=list)
    stop_breaches: list[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        """取引停止かどうか."""
        return len(self.stop_breaches) > 0


@dataclass(frozen=True)
class GrossExposureAdjustment:
    """Gross露出自動調整の結果."""

    gross_before: float
    gross_after: float
    gross_limit: float
    adjustment_factor: float
    was_adjusted: bool


@dataclass(frozen=True)
class OrderRequest:
    """注文リクエスト."""

    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    margin_trade_type: int | None = None
    account_type: int | None = None
    is_close: bool = False
    close_position_order: int = 0


@dataclass
class OrderResult:
    """注文結果."""

    order_id: str
    status: OrderStatus
    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    margin_trade_type: int = 3
    message: str = ""
    eigyou_day: str = ""
