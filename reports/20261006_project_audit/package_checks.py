"""Build and smoke the current production wheel offline in the existing venv."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(OUT))
from run_checks import run


def main():
    os.environ["UV_OFFLINE"] = "1"
    os.environ["UV_CACHE_DIR"] = "/tmp/leadlag-audit-uv-cache"
    results = [run("lock_offline", ["/opt/homebrew/bin/uv", "lock", "--check", "--offline"], 120)]
    sys.path.insert(0, str(ROOT / "scripts/ci"))
    try:
        from build import ProjectBuilder
        from build_wheel_clean import _without_stale_build_outputs
        from verify_wheel import verify_wheel
        wheel_dir = OUT / "wheel"
        wheel_dir.mkdir(exist_ok=True)
        with _without_stale_build_outputs(ROOT):
            wheel_path = Path(ProjectBuilder(str(ROOT)).build("wheel", str(wheel_dir)))
        verify_wheel(wheel_path)
        installation = OUT / "wheel_install"
        with ZipFile(wheel_path) as archive:
            archive.extractall(installation)
        results.append({"name": "offline_wheel_build_and_manifest", "exit": 0, "wheel": str(wheel_path), "note": "Existing locked environment; no isolated build dependency installation"})
        results.append(run("wheel_smoke", [str(ROOT / ".venv/bin/python"), "-I", "scripts/ci/smoke_installed_wheel.py", str(installation)], 120))
    except Exception as exc:
        results.append({"name": "offline_wheel_build", "exit": 1, "error": str(exc)})
    (OUT / "checks_package.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
