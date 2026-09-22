"""Process-group deadline and single-flight guard for production batch jobs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

from leadlag.config.paths import execution_state_path, logs
from leadlag.execution.state_store import ExecutionStateConflict, ExecutionStateStore

TIMEOUT_EXIT_CODE = 124
LEASE_CONFLICT_EXIT_CODE = 73


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float) -> str:
    """Terminate the child process group and return the final action."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - the production host is macOS
            process.terminate()
    except ProcessLookupError:
        return "already_exited"
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader, but keep checking its descendants.
        try:
            if os.name == "posix":
                os.killpg(process.pid, 0)
            elif process.poll() is not None:  # pragma: no cover
                return "term"
        except ProcessLookupError:
            return "term"
        except PermissionError:
            # Sandboxed hosts may forbid signal 0 even for our own group.
            # Conservatively keep the grace period and perform the final kill.
            pass
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()
    return "kill"


class _GuardInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


def _interrupt(signum: int, _frame: object) -> None:
    raise _GuardInterrupted(signum)


def execution_lease(store: ExecutionStateStore, *, metadata: dict) -> AbstractContextManager:
    """Use the enclosing guard's verified lease or acquire a direct CLI lease."""
    from contextlib import nullcontext

    scope = "live:production_v2"
    owner = os.environ.get("LEADLAG_LEASE_OWNER")
    if owner:
        if (
            os.environ.get("LEADLAG_LEASE_SCOPE") != scope
            or Path(os.environ.get("LEADLAG_LEASE_DB", "")).resolve() != store.path.resolve()
            or not store.owns_lease(scope, owner)
        ):
            raise ExecutionStateConflict("Enclosing job lease is missing, expired, or belongs to another store")
        return nullcontext()
    return store.acquire_lease(scope, ttl_seconds=21600.0, metadata=metadata)


def run_guarded_command(
    command: Sequence[str],
    *,
    scope: str,
    timeout_seconds: float,
    grace_seconds: float = 10.0,
    state_db: str | Path | None = None,
    guard_log: str | Path | None = None,
    cwd: str | Path | None = None,
) -> int:
    """Run *command* with a cross-process lease and process-group deadline.

    A timeout is returned as 124.  A lease conflict is returned as 73.  The
    child's exit code is returned unchanged otherwise.  No retry is attempted;
    an interrupted trading process requires the durable state reconciliation
    path before another submission.
    """
    if not command:
        raise ValueError("A guarded command is required")
    timeout = max(0.01, float(timeout_seconds))
    grace = max(0.0, float(grace_seconds))
    store = ExecutionStateStore(state_db or execution_state_path())
    log_path = Path(guard_log) if guard_log else logs("job_guard", f"{scope.replace('/', '_')}.json")
    payload: dict = {
        "schema_version": 1,
        "scope": scope,
        "command": list(command),
        "started_at": _now(),
        "timeout_seconds": timeout,
        "grace_seconds": grace,
        "status": "starting",
    }
    _write_json(log_path, payload)
    try:
        lease = store.acquire_lease(scope, ttl_seconds=timeout + grace + 30.0, metadata={"command": list(command)})
    except ExecutionStateConflict as exc:
        payload.update({"status": "lease_conflict", "finished_at": _now(), "error": str(exc)})
        _write_json(log_path, payload)
        return LEASE_CONFLICT_EXIT_CODE

    with lease:
        payload["status"] = "running"
        payload["lease_owner"] = lease.owner
        _write_json(log_path, payload)
        started = time.monotonic()
        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                old_handlers[signum] = signal.signal(signum, _interrupt)
        child_env = dict(os.environ)
        child_env.update({
            "LEADLAG_LEASE_OWNER": lease.owner,
            "LEADLAG_LEASE_SCOPE": scope,
            "LEADLAG_LEASE_DB": str(store.path.resolve()),
        })
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd) if cwd is not None else None,
                start_new_session=(os.name == "posix"),
                env=child_env,
            )
        except BaseException as exc:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            payload.update({"status": "guard_error", "finished_at": _now(), "error": repr(exc)})
            _write_json(log_path, payload)
            raise
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            action = _terminate_process_group(process, grace)
            payload.update(
                {
                    "status": "timed_out",
                    "finished_at": _now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "termination": action,
                    "return_code": TIMEOUT_EXIT_CODE,
                }
            )
            _write_json(log_path, payload)
            return TIMEOUT_EXIT_CODE
        except _GuardInterrupted as exc:
            action = _terminate_process_group(process, grace)
            payload.update({"status": "interrupted", "finished_at": _now(),
                            "termination": action, "return_code": 128 + exc.signum})
            _write_json(log_path, payload)
            return 128 + exc.signum
        except BaseException as exc:
            _terminate_process_group(process, grace)
            payload.update({"status": "guard_error", "finished_at": _now(), "error": repr(exc)})
            _write_json(log_path, payload)
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        # A successful shell leader must not leave background workers alive.
        _terminate_process_group(process, grace)
        payload.update(
            {
                "status": "completed" if return_code == 0 else "failed",
                "finished_at": _now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "return_code": int(return_code),
            }
        )
        _write_json(log_path, payload)
        return int(return_code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a production batch command with a lease and deadline.")
    parser.add_argument("--scope", required=True, help="Lease scope shared by mutually exclusive jobs.")
    parser.add_argument("--timeout", type=float, required=True, help="Whole process-group deadline in seconds.")
    parser.add_argument("--grace", type=float, default=10.0, help="TERM-to-KILL grace period in seconds.")
    parser.add_argument("--state-db", default=None, help="Execution state SQLite path.")
    parser.add_argument("--guard-log", default=None, help="JSON guard result path.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after `--`.")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    return run_guarded_command(
        command,
        scope=args.scope,
        timeout_seconds=args.timeout,
        grace_seconds=args.grace,
        state_db=args.state_db,
        guard_log=args.guard_log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
