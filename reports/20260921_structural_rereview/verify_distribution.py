"""Build the current package in a fresh tree, then smoke an isolated install."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[2]
    report = Path(__file__).resolve().parent
    python = root / ".venv/bin/python"
    cache = Path("/private/tmp/leadlag-completion-uv-cache")
    with tempfile.TemporaryDirectory(prefix="leadlag-wheel-final-") as directory:
        source = Path(directory) / "source"
        site = Path(directory) / "site"
        (source / "docs").mkdir(parents=True)
        shutil.copy2(root / "pyproject.toml", source)
        shutil.copy2(root / "docs/ARCHITECTURE.md", source / "docs")
        shutil.copytree(root / "src/leadlag", source / "src/leadlag",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        with (report / "wheel_build.log").open("w") as log:
            subprocess.run([
                "uv", "--cache-dir", str(cache), "build", "--offline", "--wheel",
                "--python", str(python), "--out-dir", str(report / "dist"), str(source),
            ], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        wheel = report / "dist/leadlag-2.1.0-py3-none-any.whl"
        with ZipFile(wheel) as archive:
            files = set(archive.namelist())
        removed = {"leadlag/data/cache.py", "leadlag/models/v2/distribution_resolver.py",
                   "leadlag/models/blpx.py"}
        assert not files.intersection(removed), "Stale removed modules were included in the wheel"
        with (report / "wheel_smoke.log").open("w") as log:
            commands = [
                [str(python), "scripts/ci/verify_wheel.py", str(wheel)],
                ["uv", "--cache-dir", str(cache), "pip", "install", "--offline",
                 "--no-deps", "--target", str(site), str(wheel)],
                [str(python), "-I", "scripts/ci/smoke_installed_wheel.py", str(site)],
            ]
            for command in commands:
                subprocess.run(command, cwd=root, check=True, stdout=log,
                               stderr=subprocess.STDOUT, timeout=120)
            log.write(json.dumps({"removed_modules_absent": sorted(removed)}) + "\n")
    print("Fresh wheel build, removed-module inspection, isolated install and smoke: PASS")


if __name__ == "__main__":
    main()
