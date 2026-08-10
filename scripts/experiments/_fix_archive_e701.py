"""Split remaining one-line compound statements (E701) in archive .py files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "archive"

COMPOUND_RE = re.compile(
    r"^(\s*)"
    r"((?:async\s+)?(?:if|elif|else|for|while|try|except|finally|with|match|case))"
    r"(\s.*)?"
    r":\s+"
    r"(.+)$"
)


def split_line(line: str) -> str | None:
    m = COMPOUND_RE.match(line)
    if not m:
        return None
    indent, keyword, rest, body = m.groups()
    if keyword in ("try", "finally"):
        header = f"{indent}{keyword}:"
    else:
        header = f"{indent}{keyword}{rest}:" if rest else f"{indent}{keyword}:"
    body_indent = indent + "    "
    return f"{header}\n{body_indent}{body}"


def fix_file(path: Path, dry_run: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    new_lines = []
    changed = False
    in_multiline_string = False

    for line in lines:
        stripped = line.lstrip()
        # Toggle multiline-string state on each line that contains a triple
        # quote. This is a simple guard; it can miss pathological cases but
        # prevents the regex from rewriting strings like docstrings.
        quote_marks = ['"""', "'''"]
        if any(q in stripped for q in quote_marks):
            in_multiline_string = not in_multiline_string

        if not in_multiline_string:
            replacement = split_line(line.rstrip("\n\r"))
            if replacement is not None:
                new_lines.append(replacement + "\n")
                changed = True
                continue
        new_lines.append(line)

    if changed and not dry_run:
        path.write_text("".join(new_lines), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Split remaining one-line compound statements (E701) in archive .py files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would change without writing them.",
    )
    args = parser.parse_args()

    changed = 0
    for py_path in sorted(ARCHIVE.rglob("*.py")):
        if fix_file(py_path, dry_run=args.dry_run):
            changed += 1
    action = "would be fixed" if args.dry_run else "fixed"
    print(f"E701 {action} in {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
