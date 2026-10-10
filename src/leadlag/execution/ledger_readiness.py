"""Read-only evidence readiness check for actual-account reconciliation.

This does not produce PnL, a reconciled ledger, or an account-risk snapshot.
An incomplete manifest must never permit new-risk trading.
"""

from __future__ import annotations

import re
from datetime import datetime
from collections.abc import Mapping
from typing import Any

REQUIRED_SOURCES = (
    "cash", "positions", "fills", "fees", "financing",
    "borrow", "reverse", "session_marks", "external_cash_flows",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def assess_ledger_readiness(manifest: Mapping[str, Any]) -> list[str]:
    """Return unresolved evidence requirements, without inferring missing values."""
    errors: list[str] = []
    if manifest.get("schema_version") != "account-ledger-evidence-v1":
        errors.append("unsupported evidence manifest schema")
    if not isinstance(manifest.get("account_key"), str) or not manifest["account_key"].strip():
        errors.append("missing account_key")
    if not isinstance(manifest.get("session_date"), str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", manifest["session_date"]
    ):
        errors.append("missing or invalid session_date")
    observed_at = manifest.get("observed_at")
    try:
        parsed_observed_at = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
    except ValueError:
        parsed_observed_at = None
    if parsed_observed_at is None or parsed_observed_at.tzinfo is None:
        errors.append("missing or invalid timezone-aware observed_at")
    elif (
        isinstance(manifest.get("session_date"), str)
        and parsed_observed_at.date().isoformat() != manifest["session_date"]
    ):
        errors.append("observed_at date does not match session_date")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        sources = {}
    for name in REQUIRED_SOURCES:
        item = sources.get(name)
        if not isinstance(item, Mapping):
            errors.append(f"missing source: {name}")
            continue
        if item.get("authority") != "broker_official":
            errors.append(f"{name}: broker official authority not established")
        if not isinstance(item.get("source_id"), str) or not item["source_id"].strip():
            errors.append(f"{name}: missing source_id")
        if not isinstance(item.get("sha256"), str) or not _SHA256.fullmatch(item["sha256"]):
            errors.append(f"{name}: invalid sha256")
        if not isinstance(item.get("reconciled_by"), str) or not item["reconciled_by"].strip():
            errors.append(f"{name}: missing reconciliation owner")
        if item.get("verified") is not True:
            errors.append(f"{name}: not verified")
    crosswalk = manifest.get("execution_crosswalk")
    if not isinstance(crosswalk, list):
        errors.append("missing execution crosswalk")
    else:
        seen_broker: set[str] = set()
        seen_local: set[str] = set()
        for index, row in enumerate(crosswalk):
            if not isinstance(row, Mapping):
                errors.append(f"crosswalk[{index}]: invalid row")
                continue
            broker_id = row.get("broker_execution_id")
            local_id = row.get("local_fill_id")
            if not isinstance(broker_id, str) or not broker_id.strip():
                errors.append(f"crosswalk[{index}]: missing broker execution ID")
            elif broker_id in seen_broker:
                errors.append(f"crosswalk[{index}]: duplicate broker execution ID")
            else:
                seen_broker.add(broker_id)
            if not isinstance(local_id, str) or not local_id.strip():
                errors.append(f"crosswalk[{index}]: missing local fill ID")
            elif local_id in seen_local:
                errors.append(f"crosswalk[{index}]: duplicate local fill ID")
            else:
                seen_local.add(local_id)
            if row.get("verified") is not True:
                errors.append(f"crosswalk[{index}]: not verified")
    if manifest.get("all_broker_fills_accounted_for") is not True:
        errors.append("broker fill coverage not established")
    if manifest.get("all_local_fills_accounted_for") is not True:
        errors.append("local fill coverage not established")
    if manifest.get("cash_position_fee_balances_verified") is not True:
        errors.append("cash/position/fee balances not verified")
    if manifest.get("pending_fills") != 0 or isinstance(manifest.get("pending_fills"), bool):
        errors.append("pending fills not proven zero")
    if manifest.get("external_cash_flows_reconciled") is not True:
        errors.append("external cash flows not reconciled")
    if manifest.get("session_mark_policy_verified") is not True:
        errors.append("session end marks not verified")
    return errors


__all__ = ["REQUIRED_SOURCES", "assess_ledger_readiness"]
