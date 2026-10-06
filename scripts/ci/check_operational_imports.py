#!/usr/bin/env python3
"""Keep scheduled Python batch producers independent of research packages.

The leadlag package's transitive boundary is checked by import-linter. This
check adds checkout-level batch entry points, which import-linter cannot own.
Shell-launched diagnostic tools remain a separately tracked extraction task.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def forbidden_imports(path: Path) -> list[str]:
    imports = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return [name for name in imports if name == 'research' or name.startswith('research.')]


def main() -> int:
    failures = []
    paths = sorted((ROOT / 'scripts/batch').rglob('*.py'))
    for path in paths:
        failures.extend(f'{path.relative_to(ROOT)}: {name}' for name in forbidden_imports(path))
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Operational Python import boundary: {len(paths)} entry points passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
