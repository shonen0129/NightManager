#!/usr/bin/env python3
"""Keep scheduled operational entry points independent of research packages.

Import-linter owns the transitive boundary below ``leadlag``. This small
checkout-level check covers Python batch producers plus the explicit tools
launched by scheduled shell entry points; it is intentionally not a general
dependency framework.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SHELL_LAUNCHED_PYTHON = (
    ROOT / "tools/production/publish_gap_distribution.py",
)
OPERATIONAL_SHELLS = (
    ROOT / "scripts/batch/run_gap_distribution.sh",
    ROOT / "scripts/batch/run_decision_v2.sh",
)
FORBIDDEN_SHELL_REFERENCES = (
    "tools/research/",
    "src/research/",
    "research.",
)


def forbidden_imports(path: Path) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return [name for name in imports if name == "research" or name.startswith("research.")]


def main() -> int:
    failures: list[str] = []
    python_paths = [
        *sorted((ROOT / "scripts/batch").glob("*.py")),
        *SHELL_LAUNCHED_PYTHON,
    ]
    for path in python_paths:
        if not path.exists():
            failures.append(f"{path.relative_to(ROOT)}: missing operational entry point")
            continue
        failures.extend(
            f"{path.relative_to(ROOT)}: {name}" for name in forbidden_imports(path)
        )

    for path in OPERATIONAL_SHELLS:
        if not path.exists():
            failures.append(f"{path.relative_to(ROOT)}: missing operational shell")
            continue
        content = path.read_text(encoding="utf-8")
        for reference in FORBIDDEN_SHELL_REFERENCES:
            if reference in content:
                failures.append(
                    f"{path.relative_to(ROOT)}: forbidden research reference {reference}"
                )

    if failures:
        print("\n".join(failures))
        return 1
    print(
        "Operational import boundary: "
        f"{len(python_paths)} Python entry points and {len(OPERATIONAL_SHELLS)} shells passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
