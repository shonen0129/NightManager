"""Read-only production artifact acceptance probes."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.config import load_config_from_yaml
from leadlag.models.ml_overlay_artifact import load_overlay_model
from leadlag.runner.model_factory import build_v2_model_bundle


def outcome(fn):
    try:
        value = fn()
        return {"status": "success", "detail": value}
    except Exception as exc:  # evidence probe: preserve exact blocking reason
        return {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-4:],
        }


def main() -> None:
    config_path = ROOT / "configs/production/production.yaml"
    artifact_root = ROOT / "var/results/20260920_structural_completion/ml_overlay_retrained"
    app = load_config_from_yaml(config_path, strict=True)
    current = (artifact_root / "CURRENT").read_text(encoding="utf-8").strip()
    metadata_path = artifact_root / "versions" / current / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def load_active():
        model = load_overlay_model(artifact_root)
        return {
            "model_class": type(model).__name__,
            "artifact_version": model.metadata.get("artifact_version"),
            "metadata_status": model.metadata.get("metadata_status"),
            "train_start": model.metadata.get("train_start"),
            "train_end": model.metadata.get("train_end"),
        }

    def build_production():
        bundle = build_v2_model_bundle(app)
        return {
            "overlay_enabled": bundle.overlay_enabled,
            "overlay_path": str(bundle.overlay_path) if bundle.overlay_path else None,
            "overlay_loaded": bundle.overlay_model is not None,
        }

    def build_explicit():
        bundle = build_v2_model_bundle(app, overlay_model_dir=artifact_root)
        return {
            "overlay_enabled_from_production_config": bundle.overlay_enabled,
            "overlay_path": str(bundle.overlay_path) if bundle.overlay_path else None,
            "overlay_loaded": bundle.overlay_model is not None,
            "decision_model_class": type(bundle.decision_model).__name__,
        }

    result = {
        "config": {
            "path": str(config_path),
            "ml_overlay_enabled": bool(app.v2.ml_overlay_enabled),
            "configured_overlay_path": str(app.v2.ml_overlay_model_dir),
        },
        "versioned_artifact": {
            "root": str(artifact_root),
            "current": current,
            "metadata_status": metadata.get("metadata_status"),
            "train_start": metadata.get("train_start"),
            "train_end": metadata.get("train_end"),
            "label_asof_end": metadata.get("label_asof_end"),
            "has_oos_evidence": any(
                key in metadata for key in ("oos_report", "oos_report_hash", "walk_forward_oos")
            ),
            "load": outcome(load_active),
            "explicit_bundle": outcome(build_explicit),
        },
        "production_bundle": outcome(build_production),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
