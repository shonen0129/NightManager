"""Run one batch phase with a process-group deadline.

The outer :mod:`leadlag.execution.job_guard` owns the durable lease and the
whole-job deadline.  This small wrapper adds a deadline to an individual
phase.  It starts the phase in its own process group and forwards signals from
the outer guard, so a phase timeout or an interrupted parent cannot leave a
network worker running after the shell has exited.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TIMEOUT_EXIT_CODE = 124


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _terminate_group(process: subprocess.Popen[bytes], grace_seconds: float) -> str:
    """Terminate a phase process group and return the final action."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - production is macOS
            process.terminate()
    except ProcessLookupError:
        return "already_exited"
    except PermissionError:
        # Some macOS test/sandbox environments deny signal delivery to a
        # process group even when the child was started in a new session.
        # Preserve the deadline with a leader-level fallback; normal
        # production runs still terminate the complete process group.
        process.terminate()

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        process.poll()
        try:
            if os.name == "posix":
                os.killpg(process.pid, 0)
            elif process.poll() is not None:  # pragma: no cover
                return "term"
        except ProcessLookupError:
            return "term"
        except PermissionError:
            pass
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover
            process.kill()
    except PermissionError:
        process.kill()
    except ProcessLookupError:
        pass
    process.wait()
    return "kill"


class _PhaseInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


def _interrupt(signum: int, _frame: object) -> None:
    raise _PhaseInterrupted(signum)


def run_phase(
    command: Sequence[str],
    *,
    label: str,
    timeout_seconds: float,
    grace_seconds: float = 10.0,
    log_path: str | Path | None = None,
) -> int:
    """Run *command* with a phase deadline and optional JSON evidence."""
    if not command:
        raise ValueError("A phase command is required")
    timeout = max(0.01, float(timeout_seconds))
    grace = max(0.0, float(grace_seconds))
    payload: dict[str, object] = {
        "schema_version": 1,
        "label": label,
        "command": list(command),
        "started_at": _now(),
        "timeout_seconds": timeout,
        "grace_seconds": grace,
        "status": "starting",
    }
    destination = Path(log_path) if log_path is not None else None
    if destination is not None:
        _write_json(destination, payload)

    old_handlers: dict[int, Any] = {}
    if signal.getsignal(signal.SIGTERM) is not None:
        old_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, _interrupt)
    if signal.getsignal(signal.SIGINT) is not None:
        old_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, _interrupt)

    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            start_new_session=(os.name == "posix"),
            env=dict(os.environ),
        )
        payload["status"] = "running"
        payload["pid"] = process.pid
        if destination is not None:
            _write_json(destination, payload)
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            action = _terminate_group(process, grace)
            return_code = TIMEOUT_EXIT_CODE
            payload.update({"status": "timed_out", "termination": action})
        except _PhaseInterrupted as exc:
            action = _terminate_group(process, grace)
            return_code = 128 + exc.signum
            payload.update({"status": "interrupted", "termination": action})
        else:
            # A phase command that exits cleanly may still have daemon-like
            # children.  Reap the group if it remains alive.
            action = _terminate_group(process, grace)
            payload["cleanup"] = action
            payload["status"] = "completed" if return_code == 0 else "failed"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _terminate_group(process, grace)
        payload.update({"status": "wrapper_error", "error": repr(exc)})
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    payload.update(
        {
            "finished_at": _now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "return_code": int(return_code),
        }
    )
    if destination is not None:
        _write_json(destination, payload)
    return int(return_code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--grace", type=float, default=10.0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    return run_phase(
        command,
        label=args.label,
        timeout_seconds=args.timeout,
        grace_seconds=args.grace,
        log_path=args.log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
