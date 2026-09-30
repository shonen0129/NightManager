"""Validated actual-account loss observations used beside model replay risk."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ACCOUNT_RISK_SCHEMA = "account-risk-snapshot-v1"
REQUIRED_PNL_BASIS = "daily_realized_plus_unrealized_change_less_observed_fees"


class AccountRiskSnapshotError(ValueError):
    """Raised when actual-account risk evidence is missing or incomplete."""


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise AccountRiskSnapshotError(f"{name} must be a timezone-aware timestamp")
    return timestamp.tz_convert("Asia/Tokyo")


@dataclass(frozen=True)
class AccountRiskSnapshot:
    """One reconciled daily/monthly account return observation for live gating."""

    valid_for_trade_date: str
    observed_through: str
    as_of: pd.Timestamp
    account_key: str
    daily_return: float
    month_return: float
    source: str
    pnl_basis: str
    cash_reconciled: bool
    positions_reconciled: bool
    fees_complete: bool
    reconciliation_status: str

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        trade_date: str,
        decision_as_of: pd.Timestamp | str,
        account_key: str,
    ) -> AccountRiskSnapshot:
        if payload.get("schema_version") != ACCOUNT_RISK_SCHEMA:
            raise AccountRiskSnapshotError("unsupported account-risk snapshot schema")
        if payload.get("valid_for_trade_date") != trade_date:
            raise AccountRiskSnapshotError("account-risk snapshot is not valid for this trade_date")
        if payload.get("account_key") != account_key:
            raise AccountRiskSnapshotError("account-risk snapshot account_key mismatch")
        if payload.get("reconciliation_status") != "complete":
            raise AccountRiskSnapshotError("account-risk reconciliation is incomplete")
        if payload.get("source") != "reconciled_execution_ledger":
            raise AccountRiskSnapshotError("account-risk source is not the reconciled execution ledger")
        if payload.get("pnl_basis") != REQUIRED_PNL_BASIS:
            raise AccountRiskSnapshotError("account-risk PnL basis is unsupported")
        for field in ("cash_reconciled", "positions_reconciled", "fees_complete"):
            if payload.get(field) is not True:
                raise AccountRiskSnapshotError(f"account-risk snapshot requires {field}=true")

        observed_through = str(payload.get("observed_through") or "")
        try:
            observed_date = pd.Timestamp(observed_through).date()
            target_date = pd.Timestamp(trade_date).date()
        except (TypeError, ValueError) as exc:
            raise AccountRiskSnapshotError("observed_through and trade_date must be valid dates") from exc
        if observed_date >= target_date:
            raise AccountRiskSnapshotError("account-risk snapshot must use completed prior-session data")

        as_of = _timestamp(payload.get("as_of"), "as_of")
        cutoff = pd.Timestamp(decision_as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("Asia/Tokyo")
        else:
            cutoff = cutoff.tz_convert("Asia/Tokyo")
        if as_of > cutoff:
            raise AccountRiskSnapshotError("account-risk snapshot was recorded after the decision cutoff")
        if as_of.date().isoformat() != observed_through:
            raise AccountRiskSnapshotError(
                "account-risk as_of date must match observed_through"
            )

        returns: dict[str, float] = {}
        for field in ("daily_return", "month_return"):
            try:
                value = float(payload[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise AccountRiskSnapshotError(f"{field} must be a finite fraction") from exc
            if not math.isfinite(value) or value < -1.0:
                raise AccountRiskSnapshotError(f"{field} must be finite and >= -100%")
            returns[field] = value

        return cls(
            valid_for_trade_date=trade_date,
            observed_through=observed_through,
            as_of=as_of,
            account_key=account_key,
            daily_return=returns["daily_return"],
            month_return=returns["month_return"],
            source=str(payload["source"]),
            pnl_basis=str(payload["pnl_basis"]),
            cash_reconciled=True,
            positions_reconciled=True,
            fees_complete=True,
            reconciliation_status="complete",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        trade_date: str,
        decision_as_of: pd.Timestamp | str,
        account_key: str,
    ) -> AccountRiskSnapshot:
        source_path = Path(path)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AccountRiskSnapshotError(
                f"actual-account risk snapshot is missing: {source_path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise AccountRiskSnapshotError(
                f"actual-account risk snapshot cannot be read: {source_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise AccountRiskSnapshotError("account-risk snapshot must be a JSON object")
        return cls.from_payload(
            payload,
            trade_date=trade_date,
            decision_as_of=decision_as_of,
            account_key=account_key,
        )


def evaluate_account_loss(snapshot: AccountRiskSnapshot, config: Any) -> dict[str, Any]:
    """Compare verified account returns with the existing daily/monthly limits."""
    daily_loss = max(0.0, -snapshot.daily_return)
    monthly_loss = max(0.0, -snapshot.month_return)
    warnings: list[str] = []
    stops: list[str] = []
    if daily_loss >= float(config.daily_loss_stop):
        stops.append(
            f"ActualAccountDailyLoss={daily_loss:.4%} >= stop {float(config.daily_loss_stop):.2%}"
        )
    elif daily_loss >= float(config.daily_loss_warning):
        warnings.append(
            f"ActualAccountDailyLoss={daily_loss:.4%} >= warning {float(config.daily_loss_warning):.2%}"
        )
    if monthly_loss >= float(config.monthly_loss_stop):
        stops.append(
            f"ActualAccountMonthlyLoss={monthly_loss:.4%} >= stop {float(config.monthly_loss_stop):.2%}"
        )
    return {
        "available": True,
        "status": "complete",
        "valid_for_trade_date": snapshot.valid_for_trade_date,
        "observed_through": snapshot.observed_through,
        "as_of": snapshot.as_of.isoformat(),
        "account_key": snapshot.account_key,
        "daily_return": snapshot.daily_return,
        "month_return": snapshot.month_return,
        "daily_loss": daily_loss,
        "monthly_loss": monthly_loss,
        "source": snapshot.source,
        "pnl_basis": snapshot.pnl_basis,
        "warnings": warnings,
        "stop_breaches": stops,
        "is_blocked": bool(stops),
    }


__all__ = [
    "ACCOUNT_RISK_SCHEMA",
    "REQUIRED_PNL_BASIS",
    "AccountRiskSnapshot",
    "AccountRiskSnapshotError",
    "evaluate_account_loss",
]
