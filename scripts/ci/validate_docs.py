"""Check relative Markdown links in the current architecture documents."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


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


def validate(paths: list[Path]) -> int:
    """Validate all relative links in *paths* and return their count."""
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
    if missing:
        for item in missing:
            print(f"missing documentation link: {item}")
        return 1
    print(f"documentation links verified: {checked}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="+", type=Path)
    args = parser.parse_args()
    return validate(args.documents)


if __name__ == "__main__":
    raise SystemExit(main())
