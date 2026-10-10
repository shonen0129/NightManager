"""Validated actual-account loss observations used beside model replay risk."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from leadlag.core.market_calendar import previous_trading_day

ACCOUNT_RISK_SCHEMA = "account-risk-snapshot-v1"
ACCOUNT_RISK_POINTER_SCHEMA = "account-risk-pointer-v1"
REQUIRED_PNL_BASIS = "daily_realized_plus_unrealized_change_less_observed_fees"


class AccountRiskSnapshotError(ValueError):
    """Raised when actual-account risk evidence is missing or incomplete."""


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise AccountRiskSnapshotError(f"{name} must be a timezone-aware timestamp")
    return timestamp.tz_convert("Asia/Tokyo")


def snapshot_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return the stable content hash used by immutable account-risk snapshots."""
    stable = dict(payload)
    stable.pop("snapshot_id", None)
    stable.pop("snapshot_sha256", None)
    encoded = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class AccountRiskSnapshot:
    """One reconciled daily/monthly account return observation for live gating."""

    valid_for_trade_date: str
    observed_through: str
    observed_at: pd.Timestamp
    as_of: pd.Timestamp
    account_key: str
    daily_return: float
    month_return: float
    source: str
    source_ids: tuple[str, ...]
    source_sha256s: tuple[str, ...]
    snapshot_id: str
    snapshot_sha256: str
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
        expected_prior = previous_trading_day(target_date)
        if observed_date != expected_prior:
            raise AccountRiskSnapshotError(
                "account-risk snapshot must use the immediately completed prior trading session"
            )

        observed_at = _timestamp(payload.get("observed_at"), "observed_at")
        as_of = _timestamp(payload.get("as_of"), "as_of")
        if observed_at != as_of:
            raise AccountRiskSnapshotError("account-risk observed_at and as_of must match")
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

        source_ids = payload.get("source_ids")
        source_hashes = payload.get("source_sha256s")
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or not all(isinstance(item, str) and item for item in source_ids)
        ):
            raise AccountRiskSnapshotError("account-risk snapshot requires source_ids")
        if (
            not isinstance(source_hashes, list)
            or len(source_hashes) != len(source_ids)
            or not all(
                isinstance(item, str)
                and len(item) == 64
                and all(char in "0123456789abcdef" for char in item)
                for item in source_hashes
            )
        ):
            raise AccountRiskSnapshotError("account-risk snapshot requires source_sha256s")

        digest = snapshot_payload_sha256(payload)
        if payload.get("snapshot_sha256") != digest:
            raise AccountRiskSnapshotError("account-risk snapshot content hash mismatch")
        if payload.get("snapshot_id") != f"sha256:{digest}":
            raise AccountRiskSnapshotError("account-risk snapshot identity mismatch")

        return cls(
            valid_for_trade_date=trade_date,
            observed_through=observed_through,
            observed_at=observed_at,
            as_of=as_of,
            account_key=account_key,
            daily_return=returns["daily_return"],
            month_return=returns["month_return"],
            source=str(payload["source"]),
            source_ids=tuple(source_ids),
            source_sha256s=tuple(source_hashes),
            snapshot_id=str(payload["snapshot_id"]),
            snapshot_sha256=digest,
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
            source_bytes = source_path.read_bytes()
            payload = json.loads(source_bytes.decode("utf-8"))
        except FileNotFoundError as exc:
            raise AccountRiskSnapshotError(
                f"actual-account risk snapshot is missing: {source_path}"
            ) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AccountRiskSnapshotError(
                f"actual-account risk snapshot cannot be read: {source_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise AccountRiskSnapshotError("account-risk snapshot must be a JSON object")

        if payload.get("schema_version") == ACCOUNT_RISK_POINTER_SCHEMA:
            snapshot_path_value = payload.get("snapshot_path")
            expected_file_sha = payload.get("file_sha256")
            if not isinstance(snapshot_path_value, str) or not snapshot_path_value:
                raise AccountRiskSnapshotError("account-risk pointer snapshot_path is missing")
            if not isinstance(expected_file_sha, str) or len(expected_file_sha) != 64:
                raise AccountRiskSnapshotError("account-risk pointer file_sha256 is invalid")
            snapshot_path = Path(snapshot_path_value)
            if not snapshot_path.is_absolute():
                snapshot_path = source_path.parent / snapshot_path
            try:
                snapshot_bytes = snapshot_path.read_bytes()
            except OSError as exc:
                raise AccountRiskSnapshotError(
                    f"account-risk pointer target cannot be read: {snapshot_path}"
                ) from exc
            if _sha256_bytes(snapshot_bytes) != expected_file_sha:
                raise AccountRiskSnapshotError("account-risk pointer target hash mismatch")
            try:
                target_payload = json.loads(snapshot_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AccountRiskSnapshotError("account-risk pointer target is invalid JSON") from exc
            if not isinstance(target_payload, Mapping):
                raise AccountRiskSnapshotError("account-risk pointer target must be a JSON object")
            if payload.get("snapshot_id") != target_payload.get("snapshot_id"):
                raise AccountRiskSnapshotError("account-risk pointer snapshot_id mismatch")
            if payload.get("snapshot_sha256") != target_payload.get("snapshot_sha256"):
                raise AccountRiskSnapshotError("account-risk pointer snapshot hash mismatch")
            payload = target_payload

        return cls.from_payload(
            payload,
            trade_date=trade_date,
            decision_as_of=decision_as_of,
            account_key=account_key,
        )


@dataclass(frozen=True)
class AccountRiskPreflight:
    """Run-owned account-risk evidence and whether the live gate requires it."""

    required: bool
    snapshot: AccountRiskSnapshot | None = None
    error: AccountRiskSnapshotError | None = None

    def __post_init__(self) -> None:
        if self.required and (self.snapshot is None) == (self.error is None):
            raise ValueError(
                "required account-risk preflight must contain exactly one of snapshot or error"
            )
        if not self.required and (self.snapshot is not None or self.error is not None):
            raise ValueError("optional account-risk preflight cannot contain evidence or an error")

    @classmethod
    def not_required(cls) -> AccountRiskPreflight:
        return cls(required=False)

    @classmethod
    def verified(cls, snapshot: AccountRiskSnapshot) -> AccountRiskPreflight:
        return cls(required=True, snapshot=snapshot)

    @classmethod
    def unavailable(cls, error: AccountRiskSnapshotError) -> AccountRiskPreflight:
        return cls(required=True, error=error)


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
        "observed_at": snapshot.observed_at.isoformat(),
        "as_of": snapshot.as_of.isoformat(),
        "account_key": snapshot.account_key,
        "daily_return": snapshot.daily_return,
        "month_return": snapshot.month_return,
        "daily_loss": daily_loss,
        "monthly_loss": monthly_loss,
        "source": snapshot.source,
        "source_ids": list(snapshot.source_ids),
        "source_sha256s": list(snapshot.source_sha256s),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "pnl_basis": snapshot.pnl_basis,
        "warnings": warnings,
        "stop_breaches": stops,
        "is_blocked": bool(stops),
    }


__all__ = [
    "ACCOUNT_RISK_POINTER_SCHEMA",
    "ACCOUNT_RISK_SCHEMA",
    "REQUIRED_PNL_BASIS",
    "AccountRiskPreflight",
    "AccountRiskSnapshot",
    "AccountRiskSnapshotError",
    "evaluate_account_loss",
    "snapshot_payload_sha256",
]
