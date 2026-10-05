#!/usr/bin/env python3
"""Build a conservative Stage-1 read-only/shadow acceptance report from saved artifacts.

The command is local and read-only with respect to broker/network state.  It
validates already-written artifacts and keeps market→shadow acceptance
separate from actual-account risk and later controlled-live reconciliation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from leadlag.config.paths import project_root
from leadlag.core.market_calendar import previous_trading_day
from leadlag.data.gap_store import GapStore
from leadlag.data.quote_snapshot import load_frozen_quote_snapshot
from leadlag.domain.gap_bundle import canonical_json_bytes, sha256_bytes
from leadlag.execution.account_risk import (
    AccountRiskSnapshot,
    AccountRiskSnapshotError,
    evaluate_account_loss,
)
from leadlag.execution.config import load_config_from_yaml

ROOT = project_root()
SCHEMA_VERSION = "readonly-shadow-acceptance-report-v1"
DEFAULT_CAPTURE_DIR = ROOT / "var/shadow_runs/ml_overlay_value/microstructure"
DEFAULT_GAP_STORE = ROOT / "var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite"
DEFAULT_SHADOW_DIR = ROOT / "var/shadow_runs/ml_overlay_value"
DEFAULT_RISK_PATH = ROOT / "var/live/pipeline_data/account_risk/latest.json"
DEFAULT_JOB_LOG_DIR = ROOT / "var/logs/job_guard"
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"
DEFERRED = "DEFERRED"


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve(strict=False) if value.is_absolute() else (ROOT / value).resolve(strict=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _check_preflight(
    path: Path,
    *,
    trade_date: str,
    capture_dir: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {"status": BLOCKED, "reason": "preflight_missing", "path": str(path)}
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": FAIL,
            "reason": "preflight_unreadable",
            "error": str(exc),
            "path": str(path),
        }

    checked_at = payload.get("checked_at")
    try:
        checked_date = datetime.fromisoformat(str(checked_at)).date().isoformat()
    except ValueError:
        checked_date = None
    recorded_capture = payload.get("capture_output_dir")
    capture_matches = (
        isinstance(recorded_capture, str)
        and _absolute(recorded_capture) == capture_dir.resolve(strict=False)
    )
    date_matches = checked_date == trade_date
    ready = payload.get("status") == "READY"
    if not ready:
        status = BLOCKED
        reason = "preflight_blocked"
    elif not date_matches:
        status = BLOCKED
        reason = "preflight_trade_date_mismatch"
    elif not capture_matches:
        status = BLOCKED
        reason = "preflight_capture_output_mismatch"
    else:
        status = PASS
        reason = None
    return {
        "status": status,
        "reason": reason,
        "path": str(path),
        "checked_at": checked_at,
        "trade_date_matches": date_matches,
        "git": payload.get("git"),
        "api": payload.get("api"),
        "capture_output_dir": recorded_capture,
        "capture_output_matches": capture_matches,
        "mode": payload.get("mode"),
        "blocking_checks": payload.get("blocking_checks", []),
    }


def _check_capture(capture_dir: Path, trade_date: str) -> dict[str, Any]:
    compact = trade_date.replace("-", "")
    path = capture_dir / f"capture_{compact}.json"
    if not path.exists():
        return {"status": NOT_RUN, "reason": "capture_artifact_missing", "path": str(path)}
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": FAIL, "reason": "capture_artifact_unreadable", "error": str(exc), "path": str(path)}
    terminal = str(payload.get("status") or "")
    if terminal == "CAPTURED":
        status = PASS
        reason = None
    elif terminal == "MARKET_CLOSED":
        status = NOT_RUN
        reason = "market_closed"
    else:
        status = FAIL
        reason = f"capture_terminal_status:{terminal or 'missing'}"
    attempts = payload.get("attempts")
    return {
        "status": status,
        "reason": reason,
        "path": str(path),
        "run_id": payload.get("run_id"),
        "terminal_status": terminal,
        "attempt_count": len(attempts) if isinstance(attempts, list) else None,
    }


def _check_frozen(capture_dir: Path, trade_date: str, *, capture_status: str) -> dict[str, Any]:
    path = capture_dir / f"frozen_{trade_date.replace('-', '')}.json"
    if not path.exists():
        return {
            "status": FAIL if capture_status == PASS else NOT_RUN,
            "reason": "frozen_snapshot_missing",
            "path": str(path),
        }
    try:
        snapshot = load_frozen_quote_snapshot(capture_dir, trade_date=trade_date)
        file_hash = _file_sha256(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": FAIL, "reason": "frozen_snapshot_invalid", "error": str(exc), "path": str(path)}
    return {
        "status": PASS,
        "path": str(path),
        "file_sha256": file_hash,
        "snapshot_id": snapshot.snapshot_id,
        "request_started_at": snapshot.request_started_at.isoformat(),
        "response_received_at": snapshot.as_of.isoformat(),
        "observed_count": len(snapshot.prices),
        "source": snapshot.source,
        "timestamp_source": snapshot.timestamp_source,
    }


def _load_gap_bundle_readonly(
    path: Path,
    trade_date: str,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None, Any]:
    """Read a WAL-backed GapStore through a SQLite backup without mutating the source."""
    with tempfile.TemporaryDirectory(prefix="nightmanager-gap-acceptance-") as tmp:
        backup_path = Path(tmp) / "gap_store.sqlite"
        source = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        destination = sqlite3.connect(str(backup_path))
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        return GapStore(backup_path).load_horizon_bundle(trade_date)


def _check_gap(path: Path, trade_date: str, snapshot_id: str | None) -> dict[str, Any]:
    if snapshot_id is None:
        return {"status": NOT_RUN, "reason": "frozen_snapshot_not_available", "path": str(path)}
    if not path.exists():
        return {"status": NOT_RUN, "reason": "gap_store_missing", "path": str(path)}
    try:
        mu, omega, metadata, manifest = _load_gap_bundle_readonly(path, trade_date)
    except Exception as exc:
        return {"status": FAIL, "reason": "gap_bundle_unreadable", "error": str(exc), "path": str(path)}
    if mu is None or omega is None or metadata is None:
        return {"status": NOT_RUN, "reason": "gap_bundle_missing", "path": str(path)}
    gap_snapshot_id = metadata.get("quote_snapshot_id")
    if gap_snapshot_id != snapshot_id:
        return {
            "status": FAIL,
            "reason": "gap_snapshot_id_mismatch",
            "path": str(path),
            "frozen_snapshot_id": snapshot_id,
            "gap_snapshot_id": gap_snapshot_id,
        }

    manifest_status = "absent"
    manifest_errors: list[str] = []
    if manifest is not None:
        manifest_errors = manifest.validate(
            trade_date=trade_date,
            horizon=None,
            storage_format="sqlite",
            mu_sha256=sha256_bytes(np.ascontiguousarray(mu).tobytes()),
            omega_sha256=sha256_bytes(np.ascontiguousarray(omega).tobytes()),
            metadata_sha256=sha256_bytes(canonical_json_bytes(metadata)),
        )
        manifest_status = "valid" if not manifest_errors else "invalid"
        if manifest_errors:
            return {
                "status": FAIL,
                "reason": "gap_bundle_manifest_invalid",
                "path": str(path),
                "errors": manifest_errors,
                "gap_snapshot_id": gap_snapshot_id,
            }

    return {
        "status": PASS,
        "path": str(path),
        "quote_snapshot_id": gap_snapshot_id,
        "input_version": metadata.get("input_version"),
        "model_version": metadata.get("model_version"),
        "config_version": metadata.get("config_version"),
        "gap_inputs_version": metadata.get("gap_inputs_version"),
        "source": metadata.get("source"),
        "manifest_status": manifest_status,
        "mu_shape": list(mu.shape),
        "omega_shape": list(omega.shape),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            records.append(item)
    return records


def _check_shadow(shadow_dir: Path, trade_date: str, snapshot_id: str | None) -> dict[str, Any]:
    path = shadow_dir / "daily.jsonl"
    if snapshot_id is None:
        return {"status": NOT_RUN, "reason": "frozen_snapshot_not_available", "path": str(path)}
    if not path.exists():
        return {"status": NOT_RUN, "reason": "shadow_log_missing", "path": str(path)}
    try:
        records = _read_jsonl(path)
    except (OSError, ValueError) as exc:
        return {"status": FAIL, "reason": "shadow_log_unreadable", "error": str(exc), "path": str(path)}
    same_day = [item for item in records if item.get("trade_date") == trade_date]
    matching = [item for item in same_day if item.get("quote_snapshot_id") == snapshot_id]
    if not matching:
        return {
            "status": FAIL if same_day else NOT_RUN,
            "reason": "shadow_snapshot_id_mismatch" if same_day else "shadow_not_recorded",
            "path": str(path),
            "same_day_record_count": len(same_day),
        }
    record = matching[-1]
    input_version = record.get("input_version")
    digest = input_version.get("digest") if isinstance(input_version, Mapping) else None
    complete = record.get("status") == "complete" and bool(digest)
    return {
        "status": PASS if complete else FAIL,
        "reason": None if complete else "shadow_incomplete",
        "path": str(path),
        "quote_snapshot_id": record.get("quote_snapshot_id"),
        "record_fingerprint": record.get("record_fingerprint"),
        "input_digest": digest,
        "shadow_status": record.get("status"),
        "as_of": record.get("as_of"),
        "attempt": record.get("attempt"),
    }


def _phase_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": BLOCKED, "reason": "log_missing", "path": str(path)}
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": FAIL, "reason": "log_unreadable", "error": str(exc), "path": str(path)}
    completed = payload.get("status") == "completed" and payload.get("return_code") == 0
    return {
        "status": PASS if completed else FAIL,
        "reason": None if completed else "phase_not_successful",
        "path": str(path),
        "runtime_status": payload.get("status"),
        "return_code": payload.get("return_code"),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "command": payload.get("command"),
    }


def _check_jobs(job_log_dir: Path, trade_date: str) -> dict[str, Any]:
    compact = trade_date.replace("-", "")
    capture_guard = _phase_record(job_log_dir / f"microstructure_0910_{compact}.json")
    decision_guard = _phase_record(job_log_dir / f"decision_{compact}.json")
    gap_phase = _phase_record(job_log_dir / f"decision_{compact}_gap_distribution.json")
    decision_phase = _phase_record(job_log_dir / f"decision_{compact}_decision.json")
    command = decision_phase.get("command")
    shadow_only = isinstance(command, list) and "--shadow-only" in command

    components = [capture_guard, decision_guard, gap_phase, decision_phase]
    statuses = [str(item["status"]) for item in components]
    if FAIL in statuses:
        status = FAIL
        reason = "job_or_phase_failed"
    elif BLOCKED in statuses:
        status = BLOCKED
        reason = "job_or_phase_evidence_missing"
    elif not shadow_only:
        status = FAIL
        reason = "decision_phase_not_shadow_only"
    else:
        status = PASS
        reason = None
    return {
        "status": status,
        "reason": reason,
        "shadow_only": shadow_only,
        "capture_guard": capture_guard,
        "decision_guard": decision_guard,
        "gap_phase": gap_phase,
        "decision_phase": decision_phase,
        "broker_side_effect_reconciliation": DEFERRED,
    }


def _check_risk(
    path: Path,
    trade_date: str,
    *,
    decision_as_of: str | None,
    account_key: str,
    risk_config: Any | None,
) -> dict[str, Any]:
    if decision_as_of is None:
        return {"status": BLOCKED, "reason": "decision_cutoff_unavailable", "path": str(path)}
    if not path.exists():
        return {"status": BLOCKED, "reason": "account_risk_snapshot_missing", "path": str(path)}
    try:
        snapshot = AccountRiskSnapshot.load(
            path,
            trade_date=trade_date,
            decision_as_of=decision_as_of,
            account_key=account_key,
        )
    except (AccountRiskSnapshotError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": BLOCKED, "reason": "account_risk_invalid", "error": str(exc), "path": str(path)}

    expected_previous = previous_trading_day(date.fromisoformat(trade_date)).isoformat()
    if snapshot.observed_through != expected_previous:
        return {
            "status": BLOCKED,
            "reason": "account_risk_not_previous_trading_session",
            "path": str(path),
            "observed_through": snapshot.observed_through,
            "expected_observed_through": expected_previous,
        }
    if risk_config is None:
        return {
            "status": BLOCKED,
            "reason": "risk_config_unavailable",
            "path": str(path),
            "observed_through": snapshot.observed_through,
        }
    gate = evaluate_account_loss(snapshot, risk_config)
    return {
        "status": PASS,
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "account_key": snapshot.account_key,
        "observed_through": snapshot.observed_through,
        "as_of": snapshot.as_of.isoformat(),
        "source": snapshot.source,
        "pnl_basis": snapshot.pnl_basis,
        "daily_return": snapshot.daily_return,
        "month_return": snapshot.month_return,
        "reconciliation_status": snapshot.reconciliation_status,
        "gate": {
            "daily_loss": gate["daily_loss"],
            "monthly_loss": gate["monthly_loss"],
            "warnings": gate["warnings"],
            "stop_breaches": gate["stop_breaches"],
            "is_blocked": gate["is_blocked"],
        },
    }


def _combine(statuses: Sequence[str]) -> str:
    if FAIL in statuses:
        return FAIL
    if BLOCKED in statuses:
        return BLOCKED
    if NOT_RUN in statuses:
        return NOT_RUN
    return PASS


def build_acceptance_report(
    *,
    trade_date: str,
    capture_dir: Path,
    gap_store: Path,
    shadow_dir: Path,
    risk_path: Path,
    preflight_path: Path,
    job_log_dir: Path,
    account_key: str = "tachibana:default",
    risk_config: Any | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    date.fromisoformat(trade_date)
    preflight = _check_preflight(
        preflight_path,
        trade_date=trade_date,
        capture_dir=capture_dir,
    )
    capture = _check_capture(capture_dir, trade_date)
    frozen = _check_frozen(capture_dir, trade_date, capture_status=str(capture["status"]))
    snapshot_id = frozen.get("snapshot_id") if frozen.get("status") == PASS else None
    gap = _check_gap(gap_store, trade_date, snapshot_id)
    shadow = _check_shadow(shadow_dir, trade_date, snapshot_id)
    jobs = _check_jobs(job_log_dir, trade_date)
    risk = _check_risk(
        risk_path,
        trade_date,
        decision_as_of=shadow.get("as_of") if shadow.get("status") == PASS else frozen.get("response_received_at"),
        account_key=account_key,
        risk_config=risk_config,
    )

    market_components = [preflight, capture, frozen, gap, shadow, jobs]
    market_status = _combine([str(item["status"]) for item in market_components])
    propagation_ids = {
        "frozen": frozen.get("snapshot_id"),
        "gap": gap.get("quote_snapshot_id"),
        "shadow": shadow.get("quote_snapshot_id"),
    }
    propagation_values = [value for value in propagation_ids.values() if value is not None]
    propagation_match = len(propagation_values) == 3 and len(set(propagation_values)) == 1
    if market_status == PASS and not propagation_match:
        market_status = FAIL

    if market_status == FAIL:
        stage1_status = FAIL
    elif market_status != PASS:
        stage1_status = market_status
    elif risk["status"] != PASS:
        stage1_status = BLOCKED
    else:
        stage1_status = PASS

    now = generated_at or datetime.now(UTC)
    snapshot_suffix = str(snapshot_id)[:12] if snapshot_id else "pending"
    acceptance_id = f"{trade_date}-readonly-shadow-{snapshot_suffix}"
    return {
        "schema_version": SCHEMA_VERSION,
        "acceptance_id": acceptance_id,
        "trade_date": trade_date,
        "generated_at": now.isoformat(),
        "stage": "stage1_readonly_shadow",
        "market_to_shadow_status": market_status,
        "risk_inclusive_stage1_status": stage1_status,
        "issue_27_overall_status": DEFERRED,
        "issue_27_deferred_reason": (
            "controlled-live reconciliation and a normal scheduler cycle remain later acceptance stages"
        ),
        "propagation": {
            "snapshot_ids": propagation_ids,
            "all_three_match": propagation_match,
        },
        "checks": {
            "preflight": preflight,
            "capture": capture,
            "frozen_snapshot": frozen,
            "gap": gap,
            "shadow": shadow,
            "jobs": jobs,
            "account_risk": risk,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    checks = report["checks"]
    propagation = report["propagation"]
    rows = [
        ("preflight", checks["preflight"]["status"]),
        ("capture", checks["capture"]["status"]),
        ("frozen snapshot", checks["frozen_snapshot"]["status"]),
        ("gap", checks["gap"]["status"]),
        ("paired shadow", checks["shadow"]["status"]),
        ("job/phase + shadow-only", checks["jobs"]["status"]),
        ("actual-account risk", checks["account_risk"]["status"]),
    ]
    lines = [
        f"# NightManager read-only / shadow acceptance — {report['trade_date']}",
        "",
        f"- acceptance ID: `{report['acceptance_id']}`",
        f"- market → shadow: **{report['market_to_shadow_status']}**",
        f"- risk込みStage 1: **{report['risk_inclusive_stage1_status']}**",
        f"- Issue #27全体: **{report['issue_27_overall_status']}**",
        f"- frozen/gap/shadow snapshot ID一致: **{propagation['all_three_match']}**",
        "",
        "| check | status |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {status} |" for name, status in rows)
    lines.extend(
        [
            "",
            "## Snapshot propagation",
            "",
            f"- frozen: `{propagation['snapshot_ids'].get('frozen')}`",
            f"- gap: `{propagation['snapshot_ids'].get('gap')}`",
            f"- shadow: `{propagation['snapshot_ids'].get('shadow')}`",
            "",
            "## Boundaries",
            "",
            "- actual-account riskがBLOCKEDの場合、市場→shadowがPASSでもrisk込みStage 1をPASSにしない。",
            "- broker side-effect reconciliationはこのレポートではDEFERRED。controlled liveの受入へ持ち越す。",
            "- Issue #27全体はcontrolled-live reconciliationと通常scheduler一巡が完了するまでDEFERRED。",
            "",
            "再現用の詳細値・path・hashは同ディレクトリの `report.json` を参照。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True, help="YYYY-MM-DD")
    capture_default = os.environ.get("LEADLAG_CAPTURE_OUTPUT_DIR", str(DEFAULT_CAPTURE_DIR))
    risk_default = os.environ.get("LEADLAG_ACCOUNT_RISK_SNAPSHOT", str(DEFAULT_RISK_PATH))
    parser.add_argument("--capture-dir", type=Path, default=Path(capture_default))
    parser.add_argument("--gap-store", type=Path, default=DEFAULT_GAP_STORE)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--risk-snapshot", type=Path, default=Path(risk_default))
    parser.add_argument("--preflight", type=Path, default=None)
    parser.add_argument("--job-log-dir", type=Path, default=DEFAULT_JOB_LOG_DIR)
    parser.add_argument("--account-key", default="tachibana:default")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/production/production.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    capture_dir = _absolute(args.capture_dir)
    preflight_path = _absolute(args.preflight) if args.preflight is not None else capture_dir / "preflight.json"
    output_dir = (
        _absolute(args.output_dir)
        if args.output_dir is not None
        else ROOT / "reports" / f"{args.trade_date.replace('-', '')}_readonly_shadow_acceptance"
    )
    app_config = load_config_from_yaml(_absolute(args.config), strict=True)
    report = build_acceptance_report(
        trade_date=args.trade_date,
        capture_dir=capture_dir,
        gap_store=_absolute(args.gap_store),
        shadow_dir=_absolute(args.shadow_dir),
        risk_path=_absolute(args.risk_snapshot),
        preflight_path=preflight_path,
        job_log_dir=_absolute(args.job_log_dir),
        account_key=args.account_key,
        risk_config=app_config.risk,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if report["market_to_shadow_status"] == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
