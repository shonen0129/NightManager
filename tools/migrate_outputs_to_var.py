"""Migrate runtime output directories into the consolidated ``var/`` tree.

This is a one-time migration helper. It moves (not copies) data from the old
root-level output directories into ``var/`` and leaves symlinks behind for
backward compatibility.

Usage::

    python3 tools/migrate_outputs_to_var.py --dry-run
    python3 tools/migrate_outputs_to_var.py

The tool is idempotent: re-running it on an already-migrated tree is a no-op.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping: old root directory -> new var/ subdirectory
MIGRATIONS = {
    "outputs": "var/outputs",
    "results": "var/results",
    "logs": "var/logs",
    "shadow_runs": "var/shadow_runs",
    "live": "var/live",
    "artifacts": "var/artifacts",
    "market_data": "var/market_data",
}


def _resolve_symlink_target(path: Path) -> Path:
    """Return the final target of a symlink chain, or the path itself."""
    if path.is_symlink():
        return Path(os.path.realpath(path))
    return path


def migrate(dry_run: bool = True) -> None:
    root = Path(__file__).resolve().parent.parent
    for old_name, new_name in MIGRATIONS.items():
        old_dir = root / old_name
        new_dir = root / new_name

        if not old_dir.exists():
            logger.info("Old directory %s does not exist; skipping", old_name)
            continue

        old_target = _resolve_symlink_target(old_dir)
        if old_target == new_dir:
            logger.info("%s is already migrated to %s", old_name, new_dir)
            continue

        new_dir.mkdir(parents=True, exist_ok=True)

        # Move contents of old_dir into new_dir, preserving subdirectories.
        for item in old_target.iterdir():
            dest = new_dir / item.name
            if dest.exists():
                logger.warning("Destination already exists, skipping: %s", dest)
                continue
            if dry_run:
                logger.info("Would move %s -> %s", item, dest)
            else:
                logger.info("Moving %s -> %s", item, dest)
                shutil.move(str(item), str(dest))

        if not dry_run:
            # Replace old directory with a symlink so legacy paths still work.
            if old_dir.is_dir() and not any(old_dir.iterdir()):
                old_dir.rmdir()
            if old_dir.exists():
                # If rmdir failed (non-empty or not a dir), do not overwrite.
                logger.warning("Could not replace %s with a symlink; left in place", old_dir)
            else:
                old_dir.symlink_to(new_dir, target_is_directory=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Consolidate runtime output directories into var/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without making changes",
    )
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
