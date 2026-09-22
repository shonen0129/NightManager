"""Run the hosted CI checks locally with bounded subprocess deadlines."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    commands = [
        ["uv", "lock", "--check"],
        ["uv", "run", "--locked", "python", "-m", "compileall", "-q", "src/leadlag", "tests", "tools", "scripts", "src/research"],
        ["uv", "run", "--locked", "ruff", "check", "src/leadlag", "tests", "tools/production", "tools/validation"],
        ["uv", "run", "--locked", "ruff", "check", "src/research/experiments/ml_overlay_training.py"],
        ["uv", "run", "--locked", "mypy", "--config-file", "pyproject.toml", "src/leadlag"],
        ["uv", "run", "--locked", "lint-imports"],
        ["uv", "run", "--locked", "python", "scripts/ci/validate_docs.py", "docs/ARCHITECTURE.md", "docs/CI.md", "docs/SCHEDULER_SETUP.md", "docs/decisions/2026-09-15-structural-improvement-boundaries.md", "reports/20260915_structural_improvement/plan.md"],
        ["uv", "run", "--locked", "--with", "build>=1.2,<2", "python", "-m", "build", "--wheel", "--outdir", "dist"],
    ]
    report: list[dict[str, object]] = []
    for command in commands:
        try:
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=300, check=False)
            report.append({"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]})
        except subprocess.TimeoutExpired as exc:
            report.append({"command": command, "returncode": 124, "stdout_tail": str(exc.stdout)[-3000:], "stderr_tail": str(exc.stderr)[-3000:]})
            break
        if completed.returncode != 0:
            break
    if report and all(item["returncode"] == 0 for item in report):
        wheels = sorted((root / "dist").glob("*.whl"))
        if wheels:
            wheel = str(wheels[-1])
            commands = [
                ["uv", "run", "--locked", "python", "scripts/ci/verify_wheel.py", wheel],
                ["uv", "pip", "install", "--no-deps", "--target", "/tmp/leadlag-wheel-smoke", wheel],
                ["uv", "run", "--locked", "python", "scripts/ci/smoke_installed_wheel.py", "/tmp/leadlag-wheel-smoke",],
                ["uv", "run", "--locked", "python", "-m", "pytest", "tests", "-n", "auto", "--junitxml=var/ci/tests.xml"],
            ]
            for command in commands:
                try:
                    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=1800, check=False)
                    report.append({"command": command, "returncode": completed.returncode, "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]})
                except subprocess.TimeoutExpired as exc:
                    report.append({"command": command, "returncode": 124, "stdout_tail": str(exc.stdout)[-3000:], "stderr_tail": str(exc.stderr)[-3000:]})
                    break
                if completed.returncode != 0:
                    break
    out = root / "reports/20260920_structural_completion/evidence/r6_local_ci.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps([{"command": item["command"], "returncode": item["returncode"]} for item in report], indent=2, ensure_ascii=False))
    return 0 if report and all(item["returncode"] == 0 for item in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
