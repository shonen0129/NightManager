"""Check relative Markdown links and documented repository paths."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from urllib.parse import unquote

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
PATH_TABLE_PATTERN = re.compile(r"^\|\s*Path\s*\|", re.IGNORECASE)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Only maintained instructions are checked against today's API. Historical
# reports/ADRs passed on the CLI retain their relative-link/path-table checks.
CURRENT_DOCUMENTS = (
    "AGENTS.md", "docs/日次運用手順書.md", "docs/モデル技術仕様書.md",
    "README.md", "docs/ARCHITECTURE.md", "docs/CI.md", "docs/SCHEDULER_SETUP.md",
    "docs/RESEARCH_ENV.md", "docs/refactor_roadmap.md",
)
CURRENT_GLOBS = (
    ".agents/skills/**/*.md", ".devin/skills/**/*.md",
    ".windsurf/workflows/*.md", ".windsurf/plans/*.md",
)
LITERAL_PATTERN = re.compile(
    r"(?<![\w/])(?:src|tools|scripts|configs|tests|docs|\.agents|\.devin|\.windsurf)/"
    r"[^\s`<>\"'()\[\]|,;、。]+"
)
SYMBOL_PATTERN = re.compile(r"^([\w]+(?:\.[\w]+)*)")


def current_documents() -> list[Path]:
    """Discover maintained harness documents, including newly added skills."""
    paths = {REPOSITORY_ROOT / name for name in CURRENT_DOCUMENTS}
    for pattern in CURRENT_GLOBS:
        paths.update(REPOSITORY_ROOT.glob(pattern))
    return sorted(paths)


def iter_literal_references(path: Path) -> list[tuple[str, str | None]]:
    """Read concrete repository paths in prose, inline code and fenced commands.

    Runtime var/data/model outputs and parameterized paths (<name>, YYYY, glob)
    are not static source references. A historical section is explicitly marked
    with paired docs:historical / docs:current HTML comments.
    """
    text = re.sub(
        r"<!-- docs:historical -->.*?(?:<!-- docs:current -->|\Z)",
        "", path.read_text(encoding="utf-8"), flags=re.DOTALL,
    )
    references = []
    for match in LITERAL_PATTERN.finditer(text):
        raw = match.group().rstrip(".:，；")
        # A placeholder may immediately follow the captured prefix.
        if match.end() < len(text) and text[match.end()] == "<":
            continue
        target, _, symbol = raw.partition("::")
        if any(token in target for token in ("*", "{", "}", "YYYY", "...")):
            continue
        target = target.split("#", 1)[0]
        symbol_match = SYMBOL_PATTERN.match(symbol)
        references.append((target, symbol_match.group(1) if symbol_match else None))
    return references


def symbol_exists(path: Path, symbol: str) -> bool:
    """Inspect declared Python API without importing code or triggering I/O."""
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    for part in symbol.split("."):
        found = None
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == part:
                    found = node
                    break
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if any((alias.asname or alias.name.split(".")[0]) == part for alias in node.names):
                    found = node
                    break
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == part for target in targets):
                    found = node
                    break
        if found is None:
            return False
        nodes = found.body if isinstance(found, ast.ClassDef) else []
    return True


def validate_current_references(paths: list[Path]) -> int:
    """Check concrete paths and path.py::Class.method in maintained docs."""
    errors = []
    checked = 0
    for source in paths:
        if not source.is_file():
            errors.append(f"input document missing: {source}")
            continue
        for target, symbol in iter_literal_references(source):
            checked += 1
            resolved = (REPOSITORY_ROOT / target).resolve()
            if not resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
                errors.append(f"{source}: repository path escapes root: {target}")
            elif not resolved.exists():
                errors.append(f"{source}: repository path does not exist: {target}")
            elif symbol:
                if resolved.suffix != ".py" or not symbol_exists(resolved, symbol):
                    errors.append(f"{source}: Python symbol does not exist: {target}::{symbol}")
    for error in errors:
        print(f"invalid current documentation reference: {error}")
    if not errors:
        print(f"current documentation paths/symbols verified: {checked}")
    return int(bool(errors))


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
    parser.add_argument("documents", nargs="*", type=Path)
    parser.add_argument("--current", action="store_true", help="Check maintained operations/skills/IDE paths and Python symbols")
    args = parser.parse_args()
    if not args.documents and not args.current:
        parser.error("provide documents or --current")
    maintained = current_documents() if args.current else []
    links_result = validate(sorted(set(args.documents + maintained)))
    references_result = validate_current_references(maintained) if args.current else 0
    return max(links_result, references_result)


if __name__ == "__main__":
    raise SystemExit(main())
