#!/usr/bin/env python3
"""Emit a safe, read-only preflight record for Issue #27 stage-1 acceptance."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import urllib.parse
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
JST = ZoneInfo("Asia/Tokyo")
SCHEMA_VERSION = "readonly-shadow-acceptance-preflight-v1"
DEFAULT_CAPTURE_OUTPUT = ROOT / "var/shadow_runs/ml_overlay_value/microstructure"
EXPECTED_CAPTURE_PROGRAM = (
    ROOT / "scripts/batch/run_0910_microstructure_capture.sh"
).resolve(strict=False)


def _validated_api_url(api_url: str) -> str:
    normalized = api_url.strip()
    if "e_api_v4r9" in normalized.casefold():
        raise ValueError(
            "TACHIBANA_API_URL points to retired v4r9; configure the current v4r10 URL"
        )
    return normalized


def _inspect_installed_scheduler() -> dict[str, Any]:
    path = Path.home() / "Library/LaunchAgents/com.leadlag.microstructure-0910.plist"
    if not path.exists():
        return {
            "status": "NOT_INSTALLED",
            "path": str(path),
            "label": "com.leadlag.microstructure-0910",
        }
    try:
        with path.open("rb") as handle:
            payload = plistlib.load(handle)
        try:
            registered = subprocess.run(
                ["launchctl", "print", f"gui/{os.getuid()}/com.leadlag.microstructure-0910"],
                capture_output=True,
                check=False,
                timeout=3,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            registered = False
        return {
            "status": "REGISTERED" if registered else "PLIST_PRESENT_NOT_REGISTERED",
            "path": str(path),
            "label": payload.get("Label"),
            "registered": registered,
            "schedule": payload.get("StartCalendarInterval"),
            "program": payload.get("ProgramArguments", []),
            "run_at_load": payload.get("RunAtLoad"),
            "keep_alive": payload.get("KeepAlive"),
            "capture_output_dir": payload.get("EnvironmentVariables", {}).get(
                "LEADLAG_CAPTURE_OUTPUT_DIR"
            ),
        }
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        return {
            "status": "UNREADABLE",
            "path": str(path),
            "label": "com.leadlag.microstructure-0910",
            "error": str(exc),
        }


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _read_git_state() -> dict[str, Any]:
    sha = _run_git("rev-parse", "HEAD")
    porcelain = _run_git("status", "--porcelain")
    changed_paths = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "sha": sha,
        "dirty": bool(changed_paths),
        "changed_path_count": len(changed_paths),
    }


def _safe_api_metadata(api_url: str) -> dict[str, Any]:
    normalized = _validated_api_url(api_url)
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("TACHIBANA_API_URL must be an absolute HTTP(S) URL")
    path = parsed.path.rstrip("/") or "/"
    if "e_api_v4r10" not in path.casefold():
        raise ValueError("TACHIBANA_API_URL must point to the current v4r10 API")
    return {
        "valid": True,
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "path": path,
        "version": "v4r10",
    }


def _capture_output_path(env: Mapping[str, str]) -> Path:
    raw = env.get("LEADLAG_CAPTURE_OUTPUT_DIR", "").strip()
    path = Path(raw).expanduser() if raw else DEFAULT_CAPTURE_OUTPUT
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=False)


def build_preflight(
    *,
    env: Mapping[str, str] | None = None,
    git_state: dict[str, Any] | None = None,
    scheduler: dict[str, Any] | None = None,
    checked_at: datetime | None = None,
) -> dict[str, Any]:
    source = os.environ if env is None else env
    now = checked_at or datetime.now(JST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=JST)
    else:
        now = now.astimezone(JST)

    resolved_git = git_state if git_state is not None else _read_git_state()
    resolved_scheduler = scheduler if scheduler is not None else _inspect_installed_scheduler()

    api_error: str | None = None
    try:
        api = _safe_api_metadata(source.get("TACHIBANA_API_URL", ""))
    except ValueError as exc:
        api_error = str(exc)
        api = {
            "valid": False,
            "scheme": None,
            "host": None,
            "path": None,
            "version": None,
        }

    capture_output = _capture_output_path(source)
    scheduler_registered = resolved_scheduler.get("status") == "REGISTERED"
    scheduler_output_raw = resolved_scheduler.get("capture_output_dir")
    scheduler_output_matches = True
    if scheduler_registered:
        if not isinstance(scheduler_output_raw, str) or not scheduler_output_raw.strip():
            scheduler_output_matches = False
        else:
            scheduler_output = Path(scheduler_output_raw).expanduser()
            if not scheduler_output.is_absolute():
                scheduler_output = ROOT / scheduler_output
            scheduler_output_matches = (
                scheduler_output.resolve(strict=False) == capture_output
            )

    expected_schedule = [
        {"Weekday": weekday, "Hour": 9, "Minute": 10}
        for weekday in range(1, 6)
    ]
    scheduler_program = resolved_scheduler.get("program")
    scheduler_program_readonly = (
        scheduler_registered
        and scheduler_program == ["/bin/bash", str(EXPECTED_CAPTURE_PROGRAM)]
    )
    scheduler_schedule_0910 = (
        scheduler_registered
        and resolved_scheduler.get("schedule") == expected_schedule
    )
    scheduler_no_auto_start = (
        scheduler_registered
        and resolved_scheduler.get("run_at_load") is False
        and resolved_scheduler.get("keep_alive") is False
    )

    checks = {
        "git_sha_present": bool(resolved_git.get("sha")),
        "git_clean": resolved_git.get("dirty") is False,
        "api_v4r10": api["valid"] is True,
        "capture_output_resolved": capture_output.is_absolute(),
        "scheduler_registered": scheduler_registered,
        "scheduler_capture_output_matches": scheduler_output_matches,
        "scheduler_program_readonly_capture": scheduler_program_readonly,
        "scheduler_schedule_0910": scheduler_schedule_0910,
        "scheduler_no_auto_start": scheduler_no_auto_start,
        "shadow_only_enabled": source.get("LEADLAG_SHADOW_ONLY") == "1",
        "capture_0910_enabled": source.get("LEADLAG_CAPTURE_0910", "1") != "0",
    }
    blocking = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at": now.isoformat(),
        "status": "READY" if not blocking else "BLOCKED",
        "blocking_checks": blocking,
        "git": resolved_git,
        "api": {**api, "error": api_error},
        "capture_output_dir": str(capture_output),
        "mode": {
            "shadow_only_env": source.get("LEADLAG_SHADOW_ONLY"),
            "capture_0910_env": source.get("LEADLAG_CAPTURE_0910"),
        },
        "scheduler": resolved_scheduler,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Secrets are never included in the payload.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    payload = build_preflight()
    rendered = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
