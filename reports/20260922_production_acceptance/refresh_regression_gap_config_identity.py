"""Update regression gap provenance after the approved production config change."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from leadlag.execution.config import load_config_from_yaml
from leadlag.utils.gap_provenance import config_version

ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = ROOT / "tests" / "regression" / "baselines" / "matrices"


def main() -> None:
    cfg = load_config_from_yaml(ROOT / "configs" / "production" / "production.yaml", strict=True)
    version = config_version(cfg.v2)
    candidates = sorted(METADATA_DIR.glob("*20260814*.json"))
    candidates.append(METADATA_DIR / ".mu_gap_20260814.bundle.json")
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_version") is None:
            continue
        payload["config_version"] = version
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))
        if path.name.startswith("gap_metadata_"):
            manifest = path.with_name("." + path.name.removeprefix("gap_metadata_").removesuffix(".json"))
            # The commit marker stores the canonical metadata digest, not the
            # presentation formatting of the sidecar JSON.
            manifest = path.with_name(
                ".mu_gap_" + path.name.removeprefix("gap_metadata_").removesuffix(".json") + ".bundle.json"
            )
            if manifest.exists():
                marker = json.loads(manifest.read_text(encoding="utf-8"))
                marker["config_version"] = version
                marker["metadata_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest.write_text(
                    json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(manifest.relative_to(ROOT))
    print("config_version", version)


if __name__ == "__main__":
    main()
