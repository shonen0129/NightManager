"""Validate that a built production wheel contains only production packages."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


def _source_python_members(project_root: Path) -> set[str]:
    source_root = project_root / "src"
    return {
        path.relative_to(source_root).as_posix()
        for path in (source_root / "leadlag").rglob("*.py")
    }


def verify_wheel(wheel_path: Path, project_root: Path | None = None) -> None:
    """Fail if the wheel contains research or stale production modules."""
    if not wheel_path.is_file():
        raise FileNotFoundError(wheel_path)

    with ZipFile(wheel_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]

    top_level = {name.split("/", 1)[0] for name in members}
    if "leadlag" not in top_level:
        raise RuntimeError(f"{wheel_path} does not contain the leadlag package")
    if "research" in top_level:
        raise RuntimeError(f"{wheel_path} contains the research package")

    root = Path(__file__).resolve().parents[2] if project_root is None else project_root
    source_members = _source_python_members(root)
    wheel_python_members = {
        name for name in members if name.startswith("leadlag/") and name.endswith(".py")
    }
    unexpected = sorted(wheel_python_members - source_members)
    missing = sorted(source_members - wheel_python_members)
    if unexpected:
        raise RuntimeError(f"{wheel_path} contains stale production modules: {unexpected}")
    if missing:
        raise RuntimeError(f"{wheel_path} is missing production modules: {missing}")

    print(f"wheel verified: {wheel_path} ({len(members)} files; source manifest and research boundary pass)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify_wheel(args.wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
