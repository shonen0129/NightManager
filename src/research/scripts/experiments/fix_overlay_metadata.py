#!/usr/bin/env python3
"""Report legacy ML overlay artifacts that must be retrained.

Overlay artifacts are immutable model/provenance pairs selected by a CURRENT
pointer. Updating root-level metadata would recreate the unpaired legacy
layout, so this migration helper deliberately performs no writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
BASE_DIR = ROOT / "models" / "ml_order_overlay"


def inspect_one(model_dir: Path) -> bool:
    if (model_dir / "CURRENT").exists():
        print(f"[ok] {model_dir}: versioned artifact")
        return True
    if (model_dir / "model.pkl").exists() or (model_dir / "metadata.json").exists():
        print(
            f"[retrain] {model_dir}: legacy root artifact is rejected; "
            "run the training command to publish a versioned replacement"
        )
        return False
    return True


def _is_artifact_root(path: Path) -> bool:
    return any(
        (path / name).exists() for name in ("CURRENT", "model.pkl", "metadata.json")
    )


def _iter_artifact_roots(base_dir: Path):
    """Yield roots while treating ``versions`` and staging as internal data."""
    if not base_dir.is_dir():
        return
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir() or path.name == "versions" or path.name.startswith(".staging-"):
            continue
        if _is_artifact_root(path):
            yield path
            continue
        yield from _iter_artifact_roots(path)


def main() -> int:
    if not BASE_DIR.exists():
        print(f"[error] {BASE_DIR} does not exist")
        return 1

    all_versioned = True
    for subdir in _iter_artifact_roots(BASE_DIR):
        all_versioned &= inspect_one(subdir)
    return 0 if all_versioned else 1


if __name__ == "__main__":
    sys.exit(main())
