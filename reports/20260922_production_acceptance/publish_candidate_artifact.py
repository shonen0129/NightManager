"""Publish the approved acceptance artifact as a stable production root."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from leadlag.models.ml_overlay_artifact import load_overlay_model

SOURCE = Path("var/results/20260922_production_acceptance/artifacts/evaluation_2025")
DEST = Path("models/ml_order_overlay/production_20260923")


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def main() -> None:
    if DEST.exists():
        raise SystemExit(f"destination already exists; refusing overwrite: {DEST}")
    source_current = SOURCE / "CURRENT"
    version = source_current.read_text(encoding="utf-8").strip()
    source_version = SOURCE / "versions" / version
    for path in (source_current, source_version / "metadata.json", source_version / "model.pkl"):
        if not path.is_file():
            raise SystemExit(f"source artifact is incomplete: {path}")

    staging = DEST.parent / f".staging-{DEST.name}"
    if staging.exists():
        raise SystemExit(f"staging path already exists; refusing overwrite: {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "versions" / version).mkdir(parents=True)
        shutil.copy2(source_version / "metadata.json", staging / "versions" / version / "metadata.json")
        shutil.copy2(source_version / "model.pkl", staging / "versions" / version / "model.pkl")
        (staging / "CURRENT").write_text(f"{version}\n", encoding="utf-8")
        fsync_file(staging / "versions" / version / "metadata.json")
        fsync_file(staging / "versions" / version / "model.pkl")
        fsync_file(staging / "CURRENT")
        loaded = load_overlay_model(staging)
        promotion = {
            "promotion_date": "2026-09-23",
            "source_artifact": str(SOURCE),
            "source_current": version,
            "published_root": str(DEST),
            "metadata_status": loaded.metadata["metadata_status"],
            "model_sha256": loaded.metadata["model_sha256"],
            "approval": "explicit user approval in current session",
            "numerical_promotion_criteria_pass": False,
            "operator_override": True,
        }
        (staging / "PROMOTION.json").write_text(
            json.dumps(promotion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fsync_file(staging / "PROMOTION.json")
        os.replace(staging, DEST)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    load_overlay_model(DEST)
    print(json.dumps({"published_root": str(DEST), "active_version": version}, ensure_ascii=False))


if __name__ == "__main__":
    main()
