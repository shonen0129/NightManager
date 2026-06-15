"""Type-safe domain models for the lead-lag strategy."""

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


@dataclass(frozen=True)
class StrategyConfig:
    """戦略設定."""

    k: int = 6
    lambda_reg: float = 0.75
    q: float = 0.3
    weight_mode: str = "signal"
    dispersion_filter: bool = True
    dispersion_metric: str = "long_short_mean_gap"
    v3_mode: str = "static"
    ewma_half_life: int | None = 45
    lambda_lw: float = 0.5
    lw_target: str = "equicorrelation"
    corr_window: int = 60
    include_v4_prior: bool = True
    signal_mode: str = "gap_residual"
    gap_open_coef: float = 0.70
    topix_beta_coef: float = 1.20
    beta_window: int = 60
    gamma: float = 0.5
    slippage_bps: float = (
        5.0  # 片道スリッページ (basis points)。往復コスト = 2 × slippage_bps × gross_exposure
    )
    vol_adjusted_target: bool = True


@dataclass(frozen=True)
class RiskConfig:
    """リスク管理設定."""

    var_confidence: float = 0.99
    var_window: int = 250
    var_warning: float = 0.02
    var_stop: float = 0.03
    es_warning: float = 0.025
    es_stop: float = 0.04
    daily_loss_warning: float = 0.015
    daily_loss_stop: float = 0.025
    monthly_loss_stop: float = 0.05
    max_net_exposure: float = 0.05
    max_gross_exposure: float = 2.0
