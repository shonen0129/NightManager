#!/usr/bin/env python3
"""Safely orchestrate Issue #27 Stage-1 read-only/shadow acceptance.

The default action only prints the execution plan. ``--preflight-only`` performs
network-free host readiness checks. ``--execute`` coordinates the registered
read-only 09:10 LaunchAgent capture with the V2 ``--shadow-only`` path. It
never starts a competing capture and never requests controlled-live execution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from leadlag.config.paths import project_root
from leadlag.core.market_calendar import is_trading_day
from leadlag.data.quote_snapshot import load_frozen_quote_snapshot

ROOT = project_root()
JST = ZoneInfo("Asia/Tokyo")
DEFAULT_CAPTURE_DIR = ROOT / "var/shadow_runs/ml_overlay_value/microstructure"
DEFAULT_REPORT_ROOT = ROOT / "reports"
CAPTURE_START_HOUR = 9
CAPTURE_START_MINUTE = 10
CAPTURE_WINDOW_SECONDS = 30
CAPTURE_RESULT_DEADLINE_SECONDS = 70
CAPTURE_POLL_SECONDS = 0.5
MAX_EARLY_WAIT = timedelta(minutes=15)


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve(strict=False)
    return (ROOT / value).resolve(strict=False)


def _jst_now() -> datetime:
    return datetime.now(JST)


def _python_bin() -> str:
    venv_python = ROOT / ".venv/bin/python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def stage1_environment(
    capture_dir: Path,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that cannot silently leave Stage-1 shadow mode."""
    env = dict(os.environ if source is None else source)
    env["LEADLAG_SHADOW_ONLY"] = "1"
    env["LEADLAG_CAPTURE_0910"] = "1"
    env["LEADLAG_CAPTURE_OUTPUT_DIR"] = str(capture_dir.resolve(strict=False))
    return env


def build_execution_plan(
    *,
    capture_dir: Path,
    report_dir: Path,
    trade_date: str = "YYYY-MM-DD",
) -> dict[str, Any]:
    """Describe the guarded sequence without executing anything."""
    python_bin = _python_bin()
    preflight_path = capture_dir / "preflight.json"
    return {
        "read_only": True,
        "controlled_live": False,
        "trade_date": trade_date,
        "capture_dir": str(capture_dir),
        "report_dir": str(report_dir),
        "steps": [
            {
                "name": "preflight",
                "command": [
                    python_bin,
                    "tools/validation/check_readonly_acceptance_preflight.py",
                    "--output",
                    str(preflight_path),
                ],
                "env": {
                    "LEADLAG_SHADOW_ONLY": "1",
                    "LEADLAG_CAPTURE_0910": "1",
                },
            },
            {
                "name": "capture",
                "source": "registered com.leadlag.microstructure-0910 LaunchAgent",
                "action": "wait_for_and_verify_artifact",
                "constraint": "09:10:00-09:10:30 Asia/Tokyo on a trading day",
                "note": "do not start a competing manual capture",
            },
            {
                "name": "gap_and_v2_shadow",
                "command": ["bash", "scripts/batch/run_decision_v2.sh"],
                "env": {
                    "LEADLAG_SHADOW_ONLY": "1",
                    "LEADLAG_CAPTURE_0910": "0",
                },
                "note": "capture is disabled here to prevent a second 09:10 capture",
            },
            {
                "name": "acceptance_report",
                "command": [
                    python_bin,
                    "tools/validation/build_readonly_shadow_acceptance_report.py",
                    "--trade-date",
                    trade_date,
                    "--capture-dir",
                    str(capture_dir),
                    "--preflight",
                    str(preflight_path),
                    "--output-dir",
                    str(report_dir),
                ],
            },
        ],
    }


def _run_command(command: Sequence[str], *, env: Mapping[str, str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env),
        check=False,
    )
    if completed.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Stage-1 command failed with exit {completed.returncode}: {rendered}"
        )


def _wait_for_scheduled_capture(
    capture_dir: Path,
    *,
    trade_date: str,
) -> dict[str, Any]:
    """Wait for the registered LaunchAgent artifact without starting a second capture."""
    now = _jst_now()
    start = now.replace(
        hour=CAPTURE_START_HOUR,
        minute=CAPTURE_START_MINUTE,
        second=0,
        microsecond=0,
    )
    deadline = start + timedelta(seconds=CAPTURE_RESULT_DEADLINE_SECONDS)
    if now > deadline:
        raise RuntimeError(
            "scheduled 09:10 capture deadline already passed; refusing a late manual capture"
        )
    if now < start:
        wait = start - now
        if wait > MAX_EARLY_WAIT:
            raise RuntimeError(
                "Stage-1 execution may only wait for the capture from 08:55 JST onward"
            )
        time.sleep(wait.total_seconds())

    compact = trade_date.replace("-", "")
    capture_path = capture_dir / f"capture_{compact}.json"
    while _jst_now() <= deadline:
        if capture_path.exists():
            return verify_capture(capture_dir, trade_date=trade_date)
        time.sleep(CAPTURE_POLL_SECONDS)
    raise RuntimeError(
        "registered 09:10 LaunchAgent did not produce a capture artifact before deadline"
    )


def verify_capture(
    capture_dir: Path,
    *,
    trade_date: str,
) -> dict[str, Any]:
    """Fail closed before gap/shadow if terminal capture evidence is incomplete."""
    compact = trade_date.replace("-", "")
    capture_path = capture_dir / f"capture_{compact}.json"
    if not capture_path.exists():
        raise RuntimeError(f"capture artifact missing: {capture_path}")
    payload = json.loads(capture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("capture artifact must contain a JSON object")
    attempts = payload.get("attempts")
    run_id = payload.get("run_id")
    successful_attempt = (
        isinstance(attempts, list)
        and any(
            isinstance(attempt, dict) and attempt.get("status") == "CAPTURED"
            for attempt in attempts
        )
    )
    if (
        payload.get("status") != "CAPTURED"
        or not isinstance(run_id, str)
        or not run_id.strip()
        or not successful_attempt
    ):
        raise RuntimeError("09:10 capture did not finish with complete success evidence")

    snapshot = load_frozen_quote_snapshot(capture_dir, trade_date=trade_date)
    return {
        "run_id": run_id,
        "snapshot_id": snapshot.snapshot_id,
        "available_at": snapshot.as_of.isoformat(),
        "available_at_semantics": snapshot.timestamp_source,
        "source": snapshot.source,
        "observed_count": len(snapshot.prices),
    }


def _load_report(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "report.json"
    if not path.exists():
        raise RuntimeError(f"acceptance report missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("acceptance report must contain a JSON object")
    return payload


def run_preflight_only(*, capture_dir: Path) -> Path:
    """Run the network-free readiness preflight and save its non-secret record."""
    capture_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = capture_dir / "preflight.json"
    env = stage1_environment(capture_dir)
    _run_command(
        [
            _python_bin(),
            "tools/validation/check_readonly_acceptance_preflight.py",
            "--output",
            str(preflight_path),
        ],
        env=env,
    )
    return preflight_path


def execute_stage1(
    *,
    capture_dir: Path,
    report_dir: Path,
    require_risk_pass: bool = False,
) -> dict[str, Any]:
    """Execute the permitted Stage-1 market→shadow sequence for today's session."""
    now = _jst_now()
    if not is_trading_day(now):
        raise RuntimeError(
            f"{now.date().isoformat()} is not a trading day; no Stage-1 broker call was made"
        )
    trade_date = now.date().isoformat()
    capture_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    env = stage1_environment(capture_dir)
    preflight_path = capture_dir / "preflight.json"
    _run_command(
        [
            _python_bin(),
            "tools/validation/check_readonly_acceptance_preflight.py",
            "--output",
            str(preflight_path),
        ],
        env=env,
    )

    capture = _wait_for_scheduled_capture(
        capture_dir,
        trade_date=trade_date,
    )

    decision_env = dict(env)
    decision_env["LEADLAG_SHADOW_ONLY"] = "1"
    decision_env["LEADLAG_CAPTURE_0910"] = "0"
    _run_command(
        ["bash", "scripts/batch/run_decision_v2.sh"],
        env=decision_env,
    )

    _run_command(
        [
            _python_bin(),
            "tools/validation/build_readonly_shadow_acceptance_report.py",
            "--trade-date",
            trade_date,
            "--capture-dir",
            str(capture_dir),
            "--preflight",
            str(preflight_path),
            "--output-dir",
            str(report_dir),
        ],
        env=env,
    )
    report = _load_report(report_dir)
    if report.get("market_to_shadow_status") != "PASS":
        raise RuntimeError(
            "market→shadow acceptance did not PASS; inspect report.json before retrying"
        )
    if require_risk_pass and report.get("risk_inclusive_stage1_status") != "PASS":
        raise RuntimeError(
            "risk-inclusive Stage 1 is not PASS; #25 evidence remains required"
        )
    return {
        "trade_date": trade_date,
        "capture": capture,
        "acceptance_id": report.get("acceptance_id"),
        "market_to_shadow_status": report.get("market_to_shadow_status"),
        "risk_inclusive_stage1_status": report.get("risk_inclusive_stage1_status"),
        "report_dir": str(report_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only network-free host/scheduler/environment readiness checks.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute the permitted read-only 09:10 capture and V2 shadow sequence.",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--require-risk-pass",
        action="store_true",
        help="Return failure unless #25-backed risk-inclusive Stage 1 also passes.",
    )
    args = parser.parse_args(argv)

    capture_dir = _absolute(args.capture_dir)
    now = _jst_now()
    default_report = (
        DEFAULT_REPORT_ROOT
        / f"{now.date().strftime('%Y%m%d')}_readonly_shadow_acceptance"
    )
    report_dir = _absolute(args.report_dir) if args.report_dir else default_report

    if args.preflight_only:
        path = run_preflight_only(capture_dir=capture_dir)
        print(
            json.dumps(
                {
                    "status": "READY",
                    "mode": "preflight_only",
                    "preflight": str(path),
                    "note": "Repeat preflight on the actual trade date before acceptance.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not args.execute:
        print(
            json.dumps(
                build_execution_plan(
                    capture_dir=capture_dir,
                    report_dir=report_dir,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    summary = execute_stage1(
        capture_dir=capture_dir,
        report_dir=report_dir,
        require_risk_pass=args.require_risk_pass,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
