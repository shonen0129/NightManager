"""Run audit checks with whole-process-group deadlines and retained logs."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
PYTHON = str(ROOT / ".venv/bin/python")


def run(name: str, args: list[str], limit: int) -> dict:
    start = time.monotonic()
    env = dict(os.environ, PYTHONPYCACHEPREFIX="/private/tmp/leadlag-audit-pycache")
    with (OUT / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            code = 124
    result = {"name": name, "command": args, "deadline_seconds": limit, "exit": code,
              "timed_out": timed_out, "elapsed_seconds": round(time.monotonic() - start, 2)}
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    if sys.argv[1] == "tests":
        checks = [
            ("pytest_baseline", [PYTHON, "-m", "pytest", "tests/regression/test_v2_baseline.py", "-q", "--junitxml=" + str(OUT / "baseline.xml")], 300),
            ("pytest_full", [PYTHON, "-m", "pytest", "tests", "--ignore=tests/regression/test_v2_baseline.py", "-n", "4", "-q", "--junitxml=" + str(OUT / "tests.xml")], 2400),
        ]
    else:
        checks = [
            ("compileall", [PYTHON, "-m", "compileall", "-q", "src/leadlag", "tests", "tools", "scripts", "src/research"], 180),
            ("ruff_ci", [str(ROOT / ".venv/bin/ruff"), "check", "src/leadlag", "tests", "tools/production", "tools/validation", "--output-format", "json"], 180),
            ("ruff_all", [str(ROOT / ".venv/bin/ruff"), "check", "src/leadlag", "src/research", "tests", "tools", "scripts", "--output-format", "json"], 180),
            ("mypy", [PYTHON, "-m", "mypy", "--config-file", "pyproject.toml", "src/leadlag"], 300),
            ("import_linter", [str(ROOT / ".venv/bin/lint-imports")], 180),
            ("docs_ci", [PYTHON, "scripts/ci/validate_docs.py", "docs/ARCHITECTURE.md", "docs/CI.md", "docs/SCHEDULER_SETUP.md", "README.md", "docs/refactor_roadmap.md", "docs/RESEARCH_ENV.md", "docs/decisions/2026-10-04-kiss-canonical-apis.md", "docs/decisions/2026-09-15-structural-improvement-boundaries.md", "reports/20260915_structural_improvement/plan.md"], 120),
        ]
    results = []
    for name, args, limit in checks:
        results.append(run(name, args, limit))
        (OUT / f"checks_{sys.argv[1]}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
