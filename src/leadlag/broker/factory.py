"""Broker factory — create the right BrokerClient from configuration.

Usage::

    from leadlag.broker import create_broker, BrokerConfig

    config = BrokerConfig(provider="kabu", api_url="...", api_token="...")
    client = create_broker(config)

Supports:
    - ``"kabu"`` → KabuBrokerClient
    - ``"dry_run"`` → DryRunBrokerClient
"""

from __future__ import annotations

import logging

from leadlag.broker.base import BrokerClient, BrokerConfig

logger = logging.getLogger(__name__)


def create_broker(config: BrokerConfig) -> BrokerClient:
    """Instantiate the appropriate BrokerClient for the given config.

    Args:
        config: Broker configuration with ``provider`` field.

    Returns:
        BrokerClient implementation.

    Raises:
        ValueError: If ``config.provider`` is not recognized.
    """
    provider = config.provider.lower().strip()

    if provider == "kabu":
        from leadlag.broker.kabu.client import KabuBrokerClient

        logger.info("Creating KabuBrokerClient (url=%s)", config.api_url)
        return KabuBrokerClient(config)

    elif provider == "tachibana":
        from leadlag.broker.tachibana.client import TachibanaBrokerClient

        logger.info("Creating TachibanaBrokerClient (url=%s)", config.api_url)
        return TachibanaBrokerClient(config)

    elif provider == "dry_run":
        from leadlag.broker.dry_run import DryRunBrokerClient

        logger.info("Creating DryRunBrokerClient (simulated)")
        return DryRunBrokerClient(config)

    else:
        raise ValueError(
            f"Unknown broker provider: {config.provider!r}. Supported: 'kabu', 'tachibana', 'dry_run'"
        )


def create_broker_from_args(
    *,
    api_url: str,
    api_token: str | None = None,
    api_password: str | None = None,
    dry_run: bool = False,
    margin_trade_type: int = 3,
    account_type: int = 4,
    request_timeout: int = 10,
) -> BrokerClient:
    """Convenience factory matching production.py's patterns.

    Args:
        api_url: Broker API URL
        api_token: Pre-issued API token (optional if password is provided)
        api_password: Password for token auto-issuance
        dry_run: If True, create a DryRunBrokerClient
        margin_trade_type: Margin trade type
        account_type: Account type
        request_timeout: Request timeout in seconds

    Returns:
        BrokerClient ready for use
    """
    import os

    if dry_run:
        provider = "dry_run"
    else:
        provider = os.environ.get("BROKER_PROVIDER", "kabu").lower().strip()

    config = BrokerConfig(
        provider=provider,
        api_url=api_url,
        api_token=api_token or "",
        api_password=api_password or "",
        request_timeout=request_timeout,
        margin_trade_type=margin_trade_type,
        account_type=account_type,
    )
    return create_broker(config)
