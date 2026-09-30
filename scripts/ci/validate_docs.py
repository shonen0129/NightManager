"""Check relative Markdown links and documented repository paths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
PATH_TABLE_PATTERN = re.compile(r"^\|\s*Path\s*\|", re.IGNORECASE)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def iter_links(path: Path) -> list[tuple[str, Path]]:
    """Return relative link targets and their source file."""
    text = path.read_text(encoding="utf-8")
    links: list[tuple[str, Path]] = []
    for raw_target in LINK_PATTERN.findall(text):
        target = raw_target.strip().split("#", 1)[0].split("?", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        links.append((unquote(target), path))
    return links


def iter_repository_paths(path: Path) -> list[tuple[str, Path]]:
    """Return root-relative paths declared in Markdown tables headed by Path."""
    paths: list[tuple[str, Path]] = []
    in_path_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_path_table:
            in_path_table = bool(PATH_TABLE_PATTERN.match(line))
            continue
        if not line.startswith("|"):
            in_path_table = False
            continue
        cell = line.strip().strip("|").split("|", maxsplit=1)[0].strip().strip("`")
        if cell and not set(cell) <= {"-", ":", " "}:
            paths.append((cell, path))
    return paths


def validate(paths: list[Path]) -> int:
    """Validate Markdown links and documented repository paths."""
    checked = 0
    missing: list[str] = []
    for path in paths:
        if not path.is_file():
            missing.append(f"input document missing: {path}")
            continue
        for target, source in iter_links(path):
            checked += 1
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{source}: {target}")
        for target, source in iter_repository_paths(path):
            checked += 1
            relative = Path(target)
            if relative.is_absolute() or ".." in relative.parts:
                missing.append(f"{source}: repository path must stay inside the repository: {target}")
                continue
            resolved = (REPOSITORY_ROOT / relative).resolve()
            if not resolved.exists():
                missing.append(f"{source}: repository path does not exist: {target}")
    if missing:
        for item in missing:
            print(f"missing documentation reference: {item}")
        return 1
    print(f"documentation references verified: {checked}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path)
    args = parser.parse_args()
    return validate(args.documents)


if __name__ == "__main__":
    raise SystemExit(main())
