"""Fresh isolated production wheel, stale-output exclusion and import smoke."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="leadlag-full-review-wheel-") as directory:
        source = Path(directory) / "source"
        site = Path(directory) / "site"
        (source / "docs").mkdir(parents=True)
        shutil.copy2(ROOT / "pyproject.toml", source)
        shutil.copy2(ROOT / "docs/ARCHITECTURE.md", source / "docs")
        shutil.copytree(ROOT / "src/leadlag", source / "src/leadlag",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        stale = source / "build/lib/leadlag/models/v2/distribution_resolver.py"
        stale.parent.mkdir(parents=True)
        stale.write_text("# stale review sentinel\n")
        marker = source / "src/leadlag.egg-info/review-marker"
        marker.parent.mkdir(parents=True)
        marker.write_text("preserve review sentinel\n")
        environment = os.environ.copy()
        environment.update(
            UV_OFFLINE="1", UV_CACHE_DIR="/private/tmp/leadlag-uv-build-cache",
            PYTHONPATH="/private/tmp/leadlag-uv-build-cache/archive-v0/1B3674zAN42ZRFn9/lib/python3.12/site-packages",
            PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR="/private/tmp/leadlag-review-matplotlib",
        )
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts/ci/build_wheel_clean.py"),
            "--outdir", str(OUT / "dist"), "--project-dir", str(source),
        ], env=environment, check=True)
        assert stale.read_text() == "# stale review sentinel\n"
        assert marker.read_text() == "preserve review sentinel\n"
        wheel = OUT / "dist/leadlag-2.1.0-py3-none-any.whl"
        subprocess.run([str(PYTHON), "scripts/ci/verify_wheel.py", str(wheel)], check=True)
        with ZipFile(wheel) as archive:
            members = [name for name in archive.namelist() if name.startswith("leadlag/") and name.endswith(".py")]
            for name in members:
                assert archive.read(name) == (ROOT / "src" / name).read_bytes(), name
        subprocess.run([
            "uv", "--cache-dir", "/private/tmp/leadlag-review-uv-cache", "pip", "install",
            "--offline", "--no-deps", "--target", str(site), str(wheel),
        ], check=True)
        subprocess.run([
            str(PYTHON), "-I", "scripts/ci/smoke_installed_wheel.py", str(site),
        ], check=True, env=environment)
        print(json.dumps({"wheel": str(wheel), "source_bytes_equal": len(members),
                          "stale_outputs_excluded_and_restored": True,
                          "isolated_cli_artifact_smoke": "pass"}))


if __name__ == "__main__":
    main()
