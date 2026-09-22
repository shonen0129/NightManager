"""Build the current package in a fresh tree, then smoke an isolated install."""

import json
import os
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
        stale = source / "build/lib/leadlag/models/v2/distribution_resolver.py"
        stale.parent.mkdir(parents=True)
        stale.write_text("# deleted module from an old build\n")
        egg = source / "src/leadlag.egg-info/review-marker"
        egg.parent.mkdir(parents=True)
        egg.write_text("preserve old ignored output\n")
        environment = dict(os.environ)
        environment.update({
            "UV_OFFLINE": "1", "UV_CACHE_DIR": str(cache),
            "PYTHONDONTWRITEBYTECODE": "1",
            # Reuse the existing cached build tool; do not install into .venv.
            "PYTHONPATH": "/Users/shonen/.cache/uv/archive-v0/uK8ZeLAu3l2QqA8E/lib/python3.12/site-packages",
        })
        with (report / "wheel_build.log").open("w") as log:
            subprocess.run([
                str(python), str(root / "scripts/ci/build_wheel_clean.py"),
                "--outdir", str(report / "dist"), "--project-dir", str(source),
            ], check=True, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=120)
        assert stale.read_text() == "# deleted module from an old build\n"
        assert egg.read_text() == "preserve old ignored output\n"
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
    print("Clean-wrapper build with seeded stale outputs, restoration, manifest, isolated install and smoke: PASS")


if __name__ == "__main__":
    main()
