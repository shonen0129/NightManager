"""Record source hashes and non-secret resolved settings for this review."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.config import load_config_from_yaml


def main():
    paths = subprocess.run(
        ["rg", "--files", "src/leadlag", "src/research", "tests", "tools", "scripts",
         "configs", "docs", ".github", "pyproject.toml", "uv.lock"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
              for path in sorted(paths)}
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml")
    fields = ["mh_blend_enabled", "mh_horizons", "mh_weights", "ml_overlay_enabled",
              "macro_kappa_enabled", "macro_direction_enabled", "cs_overlay_enabled",
              "fallback_on_audit_failure", "ondemand_fallback_enabled"]
    old_wheel = ROOT / "reports/20260921_structural_rereview_round3/dist/leadlag-2.1.0-py3-none-any.whl"
    changes = []
    if old_wheel.exists():
        with ZipFile(old_wheel) as archive:
            old_files = {name: archive.read(name) for name in archive.namelist()
                         if name.startswith("leadlag/") and name.endswith(".py")}
        new_files = {path.removeprefix("src/"): (ROOT / path).read_bytes()
                     for path in paths if path.startswith("src/leadlag/") and path.endswith(".py")}
        changes = sorted(path for path in set(old_files) | set(new_files)
                         if old_files.get(path) != new_files.get(path))
    result = {
        "reviewed_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip(),
        "subject": "current working tree; HEAD is not the reviewed source snapshot",
        "sha256": hashes,
        "resolved_v2": {key: getattr(cfg.v2, key) for key in fields},
        "resolved_risk": cfg.risk.model_dump(mode="json"),
        "production_python_changed_since_round3_wheel": changes,
    }
    (Path(__file__).parent / "source_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"files_hashed": len(hashes), "production_changes_since_round3": changes},
                     indent=2))


if __name__ == "__main__":
    main()
