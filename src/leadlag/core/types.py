"""Type-safe domain models for the lead-lag strategy.

This module defines the core domain types (dataclasses / Enums).

NOTE: ``StrategyConfig`` and ``RiskConfig`` were previously defined here as
frozen dataclasses.  They have been migrated to ``leadlag.config.schemas``
(Pydantic BaseModel) to enable field-level validation and a single source of
truth.  The names are re-exported here for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np


class TradeAction(str, Enum):
    """Trade action enumeration."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "MO"
    LIMIT = "LO"
    CLOSE = "CLO"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    SUBMITTED = "SUBMITTED"
    SIMULATED = "SIMULATED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TickerPosition:
    """単一銘柄のポジション情報."""

    ticker: str
    open_price: float
    signal: float
    weight: float
    action: TradeAction
    allocated_amount: float
    quantity: int

    @property
    def notional(self) -> float:
        """取引金額."""
        return self.open_price * self.quantity


@dataclass(frozen=True)
class TradeDecision:
    """トレード意思決定の結果."""

    trade_date: datetime
    tickers: list[str]
    signals: np.ndarray
    raw_weights: np.ndarray
    scale: float
    weights: np.ndarray
    actions: list[TradeAction]
    sigma_s: float
    dispersion_indicator: float
    dispersion_metric: str

    def to_positions(
        self,
        open_prices: dict[str, float] | None = None,
        quantities: np.ndarray | None = None,
        allocated_amounts: np.ndarray | None = None,
    ) -> list[TickerPosition]:
        """TickerPositionのリストに変換."""
        positions = []
        for i, ticker in enumerate(self.tickers):
            price = open_prices.get(ticker, 0.0) if open_prices else 0.0
            qty = int(quantities[i]) if quantities is not None else 0
            alloc = float(allocated_amounts[i]) if allocated_amounts is not None else 0.0
            positions.append(
                TickerPosition(
                    ticker=ticker,
                    open_price=price,
                    signal=float(self.signals[i]),
                    weight=float(self.weights[i]),
                    action=self.actions[i],
                    allocated_amount=alloc,
                    quantity=qty,
                )
            )
        return positions


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


@dataclass(frozen=True)
class BacktestResult:
    """バックテスト結果."""

    trade_dates: list[datetime]
    daily_returns: np.ndarray
    long_returns: np.ndarray
    short_returns: np.ndarray
    sigma_s: np.ndarray
    dispersion_indicators: np.ndarray
    scales: np.ndarray


# ---------------------------------------------------------------------------
# Backward-compatible re-exports from config.schemas
# ---------------------------------------------------------------------------
# StrategyConfig and RiskConfig have been migrated to Pydantic BaseModel
# in leadlag.config.schemas for unified validation.  Import them from there
# in new code.  The aliases below keep existing imports working.
from leadlag.config.schemas import RiskConfig, StrategyConfig  # noqa: F401, E402
