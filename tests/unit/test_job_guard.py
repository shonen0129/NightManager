from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from leadlag.execution.job_guard import (
    LEASE_CONFLICT_EXIT_CODE,
    TIMEOUT_EXIT_CODE,
    execution_lease,
    run_guarded_command,
)
from leadlag.execution.state_store import ExecutionStateConflict, ExecutionStateStore


def _script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_guard_returns_child_code_and_writes_completion_record(tmp_path: Path) -> None:
    child = _script(tmp_path / "child.py", "raise SystemExit(7)\n")
    guard_log = tmp_path / "guard.json"
    code = run_guarded_command(
        [sys.executable, str(child)],
        scope="batch:test",
        timeout_seconds=2,
        state_db=tmp_path / "state.sqlite",
        guard_log=guard_log,
    )
    assert code == 7
    payload = guard_log.read_text(encoding="utf-8")
    assert '"status": "failed"' in payload
    assert '"return_code": 7' in payload


def test_guard_terminates_slow_process_group_at_deadline(tmp_path: Path) -> None:
    child = _script(tmp_path / "slow.py", "import time\ntime.sleep(30)\n")
    code = run_guarded_command(
        [sys.executable, str(child)],
        scope="batch:timeout",
        timeout_seconds=0.15,
        grace_seconds=0.2,
        state_db=tmp_path / "state.sqlite",
        guard_log=tmp_path / "guard.json",
    )
    assert code == TIMEOUT_EXIT_CODE
    assert '"status": "timed_out"' in (tmp_path / "guard.json").read_text(encoding="utf-8")


def test_guard_conflict_does_not_start_child(tmp_path: Path) -> None:
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    lease = store.acquire_lease("batch:shared", ttl_seconds=30)
    child = _script(tmp_path / "marker.py", f"from pathlib import Path\nPath({str(tmp_path / 'started').__repr__()}).touch()\n")
    try:
        code = run_guarded_command(
            [sys.executable, str(child)],
            scope="batch:shared",
            timeout_seconds=1,
            state_db=tmp_path / "state.sqlite",
            guard_log=tmp_path / "guard.json",
        )
    finally:
        lease.release()
    assert code == LEASE_CONFLICT_EXIT_CODE
    assert not (tmp_path / "started").exists()


def test_unverified_environment_marker_cannot_bypass_live_lease(tmp_path, monkeypatch):
    store = ExecutionStateStore(tmp_path / "state.sqlite")
    monkeypatch.setenv("LEADLAG_JOB_GUARDED", "1")
    monkeypatch.delenv("LEADLAG_LEASE_OWNER", raising=False)
    with store.acquire_lease("live:production_v2", ttl_seconds=30):
        with pytest.raises(ExecutionStateConflict):
            execution_lease(store, metadata={})


@pytest.mark.parametrize("batch", ["run_decision_v2.sh", "run_gap_distribution.sh", "run_close_positions.sh", "run_pnl_report.sh"])
@pytest.mark.parametrize("nested", [False, True])
def test_batch_entry_reuses_parent_lease_or_acquires_its_own(tmp_path, monkeypatch, batch, nested):
    """Exercise the real shell guard prefix, replacing only its live payload."""
    root = Path(__file__).resolve().parents[2]
    prefix = (root / "scripts/batch" / batch).read_text().split('mkdir -p "${LOG_DIR}"', 1)[0]
    assert prefix.endswith("fi\n\n")
    (tmp_path / "scripts/batch").mkdir(parents=True)
    (tmp_path / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    (tmp_path / "src").symlink_to(root / "src", target_is_directory=True)
    state_db = tmp_path / "var/live/pipeline_data/execution/execution_state.sqlite"
    marker = tmp_path / "payload_ran"
    payload = _script(tmp_path / "payload.py", (
        "import os\nfrom pathlib import Path\n"
        "from leadlag.execution.job_guard import execution_lease\n"
        "from leadlag.execution.state_store import ExecutionStateStore\n"
        "assert os.environ.get('LEADLAG_LEASE_OWNER')\n"
        f"with execution_lease(ExecutionStateStore({str(state_db)!r}), metadata={{}}):\n"
        f"    Path({str(marker)!r}).touch()\n"
    ))
    script = _script(tmp_path / "scripts/batch" / batch, prefix +
                     'exec env PYTHONPATH="${PROJECT_DIR}/src" "${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/payload.py"\n')
    assert payload.exists()
    for key in ("LEADLAG_LEASE_OWNER", "LEADLAG_LEASE_SCOPE", "LEADLAG_LEASE_DB"):
        monkeypatch.delenv(key, raising=False)
    # The obsolete Boolean marker must not suppress a standalone guard.
    monkeypatch.setenv("LEADLAG_JOB_GUARDED", "1")
    if nested:
        code = run_guarded_command(
            ["bash", str(script)], scope="live:production_v2", timeout_seconds=10,
            grace_seconds=0.1, state_db=state_db, guard_log=tmp_path / "outer.json",
        )
    else:
        code = subprocess.run(["bash", str(script)], timeout=15, check=False).returncode
    assert code == 0
    assert marker.exists()
    with ExecutionStateStore(state_db).acquire_lease("live:production_v2", ttl_seconds=1):
        pass


def test_pnl_batch_reports_even_if_reconciliation_is_incomplete(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "scripts/batch").mkdir(parents=True)
    (tmp_path / ".venv/bin").mkdir(parents=True)
    script = _script(tmp_path / "scripts/batch/run_pnl_report.sh",
                     (root / "scripts/batch/run_pnl_report.sh").read_text())
    # Replace interpreter only: run the full shell orchestration with harmless
    # stand-ins for broker reconciliation and the email-capable report tool.
    interpreter = _script(tmp_path / ".venv/bin/python", (
        '#!/bin/bash\n'
        'if [ "$1" = "-m" ] && [ "$2" = "leadlag.execution.reconcile" ] && [ "$3" = "--pending" ]; then\n'
        '  touch "reconciliation_attempted"\n  exit 2\nfi\n'
        'if [ "$1" = "tools/production/send_daily_close_pnl_report.py" ]; then\n'
        '  touch "report_attempted"\n  exit 0\nfi\nexit 99\n'
    ))
    interpreter.chmod(0o700)
    monkeypatch.setenv("LEADLAG_LEASE_OWNER", "isolated-shell-test")
    assert subprocess.run(["bash", str(script)], timeout=10, check=False).returncode == 2
    assert (tmp_path / "reconciliation_attempted").exists()
    assert (tmp_path / "report_attempted").exists()


def test_guard_kills_descendant_even_when_leader_exits_on_term(tmp_path):
    ready = tmp_path / "ready"
    survived = tmp_path / "survived"
    descendant = _script(tmp_path / "descendant.py", (
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).write_text(str(os.getpgrp()))\n"
        f"time.sleep(1.6)\nPath({str(survived)!r}).touch()\n"
    ))
    leader = _script(tmp_path / "leader.py", (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(descendant)!r}])\n"
        "time.sleep(30)\n"
    ))
    try:
        assert run_guarded_command(
            [sys.executable, str(leader)], scope="test:descendant", timeout_seconds=0.8,
            grace_seconds=0.1, state_db=tmp_path / "state.sqlite", guard_log=tmp_path / "guard.json",
        ) == TIMEOUT_EXIT_CODE
        assert ready.exists(), "Descendant must have started for this regression to be meaningful"
        time.sleep(1.0)
        assert not survived.exists()
    finally:
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_guard_forwards_external_termination_and_releases_lease(tmp_path):
    ready = tmp_path / "ready"
    survived = tmp_path / "survived"
    child = _script(tmp_path / "child.py", (
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).write_text(str(os.getpgrp()))\n"
        f"time.sleep(2)\nPath({str(survived)!r}).touch()\n"
    ))
    state_db = tmp_path / "state.sqlite"
    guard = subprocess.Popen([
        sys.executable, "-m", "leadlag.execution.job_guard", "--scope", "test:signal",
        "--timeout", "30", "--grace", "0.1", "--state-db", str(state_db),
        "--guard-log", str(tmp_path / "guard.json"), "--", sys.executable, str(child),
    ])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        guard.terminate()
        assert guard.wait(timeout=3) == 128 + signal.SIGTERM
        time.sleep(2.1)
        assert not survived.exists()
        with ExecutionStateStore(state_db).acquire_lease("test:signal", ttl_seconds=1):
            pass
    finally:
        if guard.poll() is None:
            guard.kill()
            guard.wait()
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
