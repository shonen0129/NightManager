"""Fail-closed producer for reconciled actual-account ledgers and risk snapshots.

The producer consumes reviewer-verified canonical evidence. It never queries a broker,
submits/cancels orders, estimates missing fees, or treats collateral as PnL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from leadlag.core.market_calendar import is_trading_day, previous_trading_day
from leadlag.execution.account_risk import (
    ACCOUNT_RISK_POINTER_SCHEMA,
    ACCOUNT_RISK_SCHEMA,
    REQUIRED_PNL_BASIS,
    snapshot_payload_sha256,
)
from leadlag.execution.ledger_readiness import REQUIRED_SOURCES, assess_ledger_readiness

RECONCILED_LEDGER_SCHEMA = "reconciled-execution-ledger-v1"


class LedgerReconciliationError(ValueError):
    """Raised when evidence cannot support a complete reconciled ledger."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _aware_timestamp(value: Any, name: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerReconciliationError(f"{name} must be a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise LedgerReconciliationError(f"{name} must be timezone-aware")
    return timestamp


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LedgerReconciliationError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LedgerReconciliationError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise LedgerReconciliationError(f"{name} must be a finite number")
    return number


def _nonnegative_number(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0.0:
        raise LedgerReconciliationError(f"{name} must be non-negative")
    return number


def _ledger_digest(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("ledger_id", None)
    stable.pop("ledger_sha256", None)
    return _payload_sha256(stable)


def build_reconciled_execution_ledger(
    evidence_manifest: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one completed-session ledger only from fully verified evidence."""
    blockers = assess_ledger_readiness(evidence_manifest)
    if blockers:
        raise LedgerReconciliationError(
            "ledger evidence is incomplete: " + "; ".join(blockers)
        )

    account_key = str(evidence_manifest["account_key"])
    trade_date = str(evidence_manifest["session_date"])
    if session.get("account_key") != account_key:
        raise LedgerReconciliationError("session account_key does not match evidence manifest")
    if session.get("trade_date") != trade_date:
        raise LedgerReconciliationError("session trade_date does not match evidence manifest")

    observed_at = _aware_timestamp(evidence_manifest["observed_at"], "observed_at")
    if observed_at.date().isoformat() != trade_date:
        raise LedgerReconciliationError("observed_at date does not match trade_date")

    opening_equity = _nonnegative_number(session.get("opening_equity_jpy"), "opening_equity_jpy")
    if opening_equity <= 0.0:
        raise LedgerReconciliationError("opening_equity_jpy must be > 0")
    closing_equity = _nonnegative_number(session.get("closing_equity_jpy"), "closing_equity_jpy")
    realized = _finite_number(session.get("realized_pnl_jpy"), "realized_pnl_jpy")
    unrealized_change = _finite_number(
        session.get("unrealized_pnl_change_jpy"), "unrealized_pnl_change_jpy"
    )
    commission = _nonnegative_number(session.get("commission_jpy"), "commission_jpy")
    taxes = _nonnegative_number(session.get("taxes_jpy"), "taxes_jpy")
    financing = _nonnegative_number(session.get("financing_jpy"), "financing_jpy")
    borrow = _nonnegative_number(session.get("borrow_jpy"), "borrow_jpy")
    reverse = _nonnegative_number(session.get("reverse_jpy"), "reverse_jpy")
    external_cash_flow = _finite_number(
        session.get("external_cash_flow_jpy"), "external_cash_flow_jpy"
    )

    total_costs = commission + taxes + financing + borrow + reverse
    net_pnl = realized + unrealized_change - total_costs
    expected_closing = opening_equity + net_pnl + external_cash_flow
    if not math.isclose(closing_equity, expected_closing, abs_tol=0.01, rel_tol=0.0):
        raise LedgerReconciliationError(
            "equity bridge mismatch: closing equity does not equal opening equity + "
            "net PnL + external cash flow"
        )
    daily_return = net_pnl / opening_equity
    if daily_return < -1.0:
        raise LedgerReconciliationError("daily return below -100% is unsupported")

    sources = evidence_manifest["sources"]
    source_refs = [
        {
            "source_class": name,
            "source_id": str(sources[name]["source_id"]),
            "sha256": str(sources[name]["sha256"]),
            "reconciled_by": str(sources[name]["reconciled_by"]),
        }
        for name in REQUIRED_SOURCES
    ]

    payload: dict[str, Any] = {
        "schema_version": RECONCILED_LEDGER_SCHEMA,
        "account_key": account_key,
        "trade_date": trade_date,
        "observed_at": observed_at.isoformat(),
        "reconciliation_status": "complete",
        "pnl_basis": REQUIRED_PNL_BASIS,
        "return_denominator_basis": "opening_equity_before_external_cash_flows",
        "opening_equity_jpy": opening_equity,
        "closing_equity_jpy": closing_equity,
        "realized_pnl_jpy": realized,
        "unrealized_pnl_change_jpy": unrealized_change,
        "commission_jpy": commission,
        "taxes_jpy": taxes,
        "financing_jpy": financing,
        "borrow_jpy": borrow,
        "reverse_jpy": reverse,
        "observed_costs_jpy": total_costs,
        "external_cash_flow_jpy": external_cash_flow,
        "net_pnl_jpy": net_pnl,
        "return_denominator_jpy": opening_equity,
        "daily_return": daily_return,
        "cash_reconciled": True,
        "positions_reconciled": True,
        "fills_reconciled": True,
        "fees_complete": True,
        "pending_fills": 0,
        "execution_crosswalk_count": len(evidence_manifest["execution_crosswalk"]),
        "source_manifest_sha256": _payload_sha256(evidence_manifest),
        "source_refs": source_refs,
    }
    digest = _ledger_digest(payload)
    payload["ledger_sha256"] = digest
    payload["ledger_id"] = f"sha256:{digest}"
    return payload


def validate_reconciled_execution_ledger(payload: Mapping[str, Any]) -> None:
    """Validate an immutable ledger before using it as risk evidence."""
    if payload.get("schema_version") != RECONCILED_LEDGER_SCHEMA:
        raise LedgerReconciliationError("unsupported reconciled ledger schema")
    if payload.get("reconciliation_status") != "complete":
        raise LedgerReconciliationError("reconciled ledger is incomplete")
    if payload.get("pnl_basis") != REQUIRED_PNL_BASIS:
        raise LedgerReconciliationError("reconciled ledger PnL basis is unsupported")
    if payload.get("cash_reconciled") is not True:
        raise LedgerReconciliationError("reconciled ledger cash is incomplete")
    if payload.get("positions_reconciled") is not True:
        raise LedgerReconciliationError("reconciled ledger positions are incomplete")
    if payload.get("fills_reconciled") is not True:
        raise LedgerReconciliationError("reconciled ledger fills are incomplete")
    if payload.get("fees_complete") is not True:
        raise LedgerReconciliationError("reconciled ledger fees are incomplete")
    if payload.get("pending_fills") != 0 or isinstance(payload.get("pending_fills"), bool):
        raise LedgerReconciliationError("reconciled ledger has pending fills")

    account_key = payload.get("account_key")
    if not isinstance(account_key, str) or not account_key:
        raise LedgerReconciliationError("reconciled ledger account_key is missing")
    try:
        trade_date = date.fromisoformat(str(payload.get("trade_date")))
    except ValueError as exc:
        raise LedgerReconciliationError("reconciled ledger trade_date is invalid") from exc
    observed_at = _aware_timestamp(payload.get("observed_at"), "ledger observed_at")
    if observed_at.date() != trade_date:
        raise LedgerReconciliationError("ledger observed_at date does not match trade_date")

    opening = _nonnegative_number(payload.get("opening_equity_jpy"), "opening_equity_jpy")
    closing = _nonnegative_number(payload.get("closing_equity_jpy"), "closing_equity_jpy")
    pnl = _finite_number(payload.get("net_pnl_jpy"), "net_pnl_jpy")
    external = _finite_number(payload.get("external_cash_flow_jpy"), "external_cash_flow_jpy")
    denominator = _finite_number(payload.get("return_denominator_jpy"), "return_denominator_jpy")
    daily_return = _finite_number(payload.get("daily_return"), "daily_return")
    if opening <= 0.0 or denominator <= 0.0:
        raise LedgerReconciliationError("reconciled ledger return denominator must be > 0")
    if not math.isclose(closing, opening + pnl + external, abs_tol=0.01, rel_tol=0.0):
        raise LedgerReconciliationError("reconciled ledger equity bridge is inconsistent")
    if not math.isclose(daily_return, pnl / denominator, abs_tol=1e-12, rel_tol=1e-12):
        raise LedgerReconciliationError("reconciled ledger daily return is inconsistent")
    if daily_return < -1.0:
        raise LedgerReconciliationError("reconciled ledger daily return is below -100%")

    source_refs = payload.get("source_refs")
    if not isinstance(source_refs, list) or len(source_refs) != len(REQUIRED_SOURCES):
        raise LedgerReconciliationError("reconciled ledger source references are incomplete")
    classes = {
        item.get("source_class")
        for item in source_refs
        if isinstance(item, Mapping)
    }
    if classes != set(REQUIRED_SOURCES):
        raise LedgerReconciliationError("reconciled ledger source classes are incomplete")

    digest = _ledger_digest(payload)
    if payload.get("ledger_sha256") != digest or payload.get("ledger_id") != f"sha256:{digest}":
        raise LedgerReconciliationError("reconciled ledger content hash mismatch")


def _month_trade_dates(observed_through: date) -> list[str]:
    current = observed_through.replace(day=1)
    dates: list[str] = []
    while current <= observed_through:
        if is_trading_day(current):
            dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def build_account_risk_snapshot(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    valid_for_trade_date: str,
    account_key: str,
) -> dict[str, Any]:
    """Derive prior-session daily/monthly returns from complete immutable ledgers."""
    if not ledgers:
        raise LedgerReconciliationError("no reconciled ledgers were supplied")
    try:
        target_date = date.fromisoformat(valid_for_trade_date)
    except ValueError as exc:
        raise LedgerReconciliationError("valid_for_trade_date is invalid") from exc
    expected_prior = previous_trading_day(target_date).isoformat()

    by_date: dict[str, Mapping[str, Any]] = {}
    for ledger in ledgers:
        validate_reconciled_execution_ledger(ledger)
        if ledger["account_key"] != account_key:
            raise LedgerReconciliationError("ledger account_key mismatch")
        trade_date = str(ledger["trade_date"])
        if trade_date in by_date:
            raise LedgerReconciliationError(f"duplicate reconciled ledger for {trade_date}")
        by_date[trade_date] = ledger

    if max(by_date) != expected_prior:
        raise LedgerReconciliationError(
            f"latest reconciled ledger must be prior trading session {expected_prior}"
        )
    observed_date = date.fromisoformat(expected_prior)
    expected_month_dates = _month_trade_dates(observed_date)
    actual_month_dates = sorted(
        trade_date
        for trade_date in by_date
        if trade_date[:7] == observed_date.strftime("%Y-%m")
        and trade_date <= expected_prior
    )
    if actual_month_dates != expected_month_dates:
        missing = sorted(set(expected_month_dates) - set(actual_month_dates))
        extra = sorted(set(actual_month_dates) - set(expected_month_dates))
        raise LedgerReconciliationError(
            f"month ledger coverage is incomplete; missing={missing}, extra={extra}"
        )

    latest = by_date[expected_prior]
    latest_observed_at = _aware_timestamp(latest["observed_at"], "latest ledger observed_at")
    monthly_growth = 1.0
    month_ledgers = [by_date[trade_date] for trade_date in expected_month_dates]
    for ledger in month_ledgers:
        monthly_growth *= 1.0 + float(ledger["daily_return"])
    month_return = monthly_growth - 1.0

    ledger_ids = [str(ledger["ledger_id"]) for ledger in month_ledgers]
    ledger_hashes = [str(ledger["ledger_sha256"]) for ledger in month_ledgers]
    payload: dict[str, Any] = {
        "schema_version": ACCOUNT_RISK_SCHEMA,
        "valid_for_trade_date": valid_for_trade_date,
        "observed_through": expected_prior,
        "observed_at": latest_observed_at.isoformat(),
        "as_of": latest_observed_at.isoformat(),
        "account_key": account_key,
        "daily_return": float(latest["daily_return"]),
        "month_return": month_return,
        "source": "reconciled_execution_ledger",
        "source_ids": ledger_ids,
        "source_sha256s": ledger_hashes,
        "pnl_basis": REQUIRED_PNL_BASIS,
        "cash_reconciled": True,
        "positions_reconciled": True,
        "fees_complete": True,
        "reconciliation_status": "complete",
    }
    snapshot_sha = snapshot_payload_sha256(payload)
    payload["snapshot_sha256"] = snapshot_sha
    payload["snapshot_id"] = f"sha256:{snapshot_sha}"
    return payload


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_content_addressed(
    directory: Path,
    *,
    stem: str,
    payload: Mapping[str, Any],
    digest: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_{digest[:16]}.json"
    data = _pretty_json_bytes(payload)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise LedgerReconciliationError(f"immutable artifact already exists with different bytes: {path}")
    return path


def write_reconciled_execution_ledger(
    directory: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    validate_reconciled_execution_ledger(payload)
    return _write_content_addressed(
        Path(directory),
        stem=str(payload["trade_date"]),
        payload=payload,
        digest=str(payload["ledger_sha256"]),
    )


def write_account_risk_snapshot(
    directory: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write immutable snapshot and atomically update the mutable latest pointer."""
    directory = Path(directory)
    snapshot_sha = str(payload.get("snapshot_sha256") or "")
    if payload.get("snapshot_id") != f"sha256:{snapshot_sha}":
        raise LedgerReconciliationError("account-risk snapshot identity is invalid")
    if snapshot_payload_sha256(payload) != snapshot_sha:
        raise LedgerReconciliationError("account-risk snapshot content hash mismatch")

    snapshot_path = _write_content_addressed(
        directory / "snapshots",
        stem=str(payload["valid_for_trade_date"]),
        payload=payload,
        digest=snapshot_sha,
    )
    file_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    pointer = {
        "schema_version": ACCOUNT_RISK_POINTER_SCHEMA,
        "snapshot_path": str(snapshot_path.relative_to(directory)),
        "file_sha256": file_sha,
        "snapshot_id": payload["snapshot_id"],
        "snapshot_sha256": snapshot_sha,
    }
    directory.mkdir(parents=True, exist_ok=True)
    pointer_path = directory / "latest.json"
    temporary = directory / "latest.json.tmp"
    temporary.write_bytes(_pretty_json_bytes(pointer))
    temporary.replace(pointer_path)
    return snapshot_path


def load_ledger_directory(directory: str | Path) -> list[dict[str, Any]]:
    ledgers: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LedgerReconciliationError(f"cannot read reconciled ledger: {path}") from exc
        if not isinstance(payload, dict):
            raise LedgerReconciliationError(f"reconciled ledger must be a JSON object: {path}")
        validate_reconciled_execution_ledger(payload)
        ledgers.append(payload)
    return ledgers


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerReconciliationError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise LedgerReconciliationError(f"JSON input must be an object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ledger_parser = subparsers.add_parser("ledger", help="build one immutable reconciled session ledger")
    ledger_parser.add_argument("--evidence-manifest", type=Path, required=True)
    ledger_parser.add_argument("--session", type=Path, required=True)
    ledger_parser.add_argument("--output-dir", type=Path, required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="derive immutable prior-session risk snapshot")
    snapshot_parser.add_argument("--ledger-dir", type=Path, required=True)
    snapshot_parser.add_argument("--valid-for-trade-date", required=True)
    snapshot_parser.add_argument("--account-key", required=True)
    snapshot_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "ledger":
        manifest = _read_json_object(args.evidence_manifest)
        session = _read_json_object(args.session)
        payload = build_reconciled_execution_ledger(manifest, session)
        path = write_reconciled_execution_ledger(args.output_dir, payload)
    else:
        ledgers = load_ledger_directory(args.ledger_dir)
        payload = build_account_risk_snapshot(
            ledgers,
            valid_for_trade_date=args.valid_for_trade_date,
            account_key=args.account_key,
        )
        path = write_account_risk_snapshot(args.output_dir, payload)
    print(json.dumps({"path": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LedgerReconciliationError",
    "RECONCILED_LEDGER_SCHEMA",
    "build_account_risk_snapshot",
    "build_reconciled_execution_ledger",
    "load_ledger_directory",
    "validate_reconciled_execution_ledger",
    "write_account_risk_snapshot",
    "write_reconciled_execution_ledger",
]
