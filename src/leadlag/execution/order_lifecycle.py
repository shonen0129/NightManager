"""Shared synchronous order-status polling for new and close orders."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, MutableSequence
from typing import Any

from leadlag.core.types import OrderStatus

logger = logging.getLogger(__name__)

PENDING_STATUSES = frozenset({OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED})


def poll_order_statuses(
    api_client: Any,
    results: MutableSequence[Any],
    *,
    order_id_getter: Callable[[Any], str],
    status_getter: Callable[[Any], Any],
    status_setter: Callable[[Any, OrderStatus], None],
    message_setter: Callable[[Any, str], None] | None = None,
    timeout_seconds: float,
    poll_interval: float,
    label: str,
) -> tuple[MutableSequence[Any], bool]:
    """Poll any mutable order-result representation until terminal or timeout.

    ``broker_ops`` uses ``OrderResult`` objects while ``close`` keeps legacy
    dictionaries for its JSON schema. The adapters supplied by each caller
    are the only representation-specific part; deadline, probe, pending-state,
    and exception handling live here.
    """
    if not results:
        return results, False
    get_status = getattr(api_client, "get_order_status", None)
    if not callable(get_status):
        logger.debug("Broker client does not support get_order_status; skip %s polling", label)
        return results, False

    def pending_items() -> list[Any]:
        return [
            item
            for item in results
            if _status_value(status_getter(item)) in PENDING_STATUSES
        ]

    pending = pending_items()
    if not pending:
        return results, False

    test_order_id = order_id_getter(pending[0])
    try:
        if test_order_id:
            get_status(test_order_id)
    except NotImplementedError:
        logger.debug("Broker get_order_status is not implemented; skip %s polling", label)
        return results, False
    except Exception:
        # A probe failure is non-fatal. The normal poll can still recover.
        pass

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    logger.info(
        "[%s POLL] Waiting for %d order(s) to reach a terminal state (timeout=%.1fs)...",
        label,
        len(pending),
        timeout_seconds,
    )
    polled = True
    while pending and time.monotonic() < deadline:
        still_pending: list[Any] = []
        for item in pending:
            order_id = order_id_getter(item)
            if not order_id:
                still_pending.append(item)
                continue
            try:
                status = _status_value(get_status(order_id))
                status_setter(item, status)
                if message_setter is not None:
                    suffix = (
                        f"Polled to {status.value}; awaiting terminal fill"
                        if status in PENDING_STATUSES
                        else f"Polled to {status.value}"
                    )
                    message_setter(item, suffix)
                logger.info("  [%s] %s: %s (order_id=%s)", label, order_id, status.value, order_id)
                if status in PENDING_STATUSES:
                    still_pending.append(item)
            except NotImplementedError:
                logger.debug("Broker get_order_status not implemented; stopping %s polling", label)
                return results, polled
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to poll %s order %s: %s", label, order_id, exc)
                still_pending.append(item)
        pending = still_pending
        if pending:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(max(0.0, float(poll_interval)), remaining))

    if pending:
        logger.warning(
            "%d %s order(s) still pending after %.1fs",
            len(pending),
            label,
            timeout_seconds,
        )
    return results, polled


def _status_value(value: Any) -> OrderStatus:
    if isinstance(value, OrderStatus):
        return value
    return OrderStatus(str(value))
