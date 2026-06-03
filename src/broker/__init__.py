"""Broker abstraction layer.

Provides a pluggable interface for order execution, position management,
and market data retrieval across different broker backends.
"""

from broker.base import (
    BrokerClient,
    BrokerConfig,
    WalletInfo,
    Position,
)
from broker.factory import create_broker

__all__ = [
    "BrokerClient",
    "BrokerConfig",
    "WalletInfo",
    "Position",
    "create_broker",
]
