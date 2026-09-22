"""Build the production wheel with an isolated setuptools build directory.

Setuptools keeps ``build/lib`` between invocations.  When a module is removed
from ``src`` that stale directory can otherwise put the deleted module back into
the next wheel.  This wrapper keeps the build output outside the worktree while
leaving the wheel output directory under the caller's control.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from build import ProjectBuilder
from build.env import DefaultIsolatedEnv


@contextlib.contextmanager
def _without_stale_build_outputs(project_dir: Path) -> Iterator[None]:
    """Temporarily quarantine ignored setuptools outputs from older builds."""
    with tempfile.TemporaryDirectory(prefix="leadlag-wheel-quarantine-") as quarantine:
        original_paths = (
            project_dir / "build",
            project_dir / "src" / "leadlag.egg-info",
        )
        backups: dict[Path, Path] = {}
        for index, path in enumerate(original_paths):
            if path.exists():
                backup = Path(quarantine) / f"original-{index}"
                shutil.move(str(path), str(backup))
                backups[path] = backup
        try:
            yield
        finally:
            # The build outputs created during this invocation are disposable.
            for index, path in enumerate(original_paths):
                if path.exists():
                    shutil.move(str(path), str(Path(quarantine) / f"generated-{index}"))
            for path, backup in backups.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup), str(path))


def build_wheel(output_dir: Path, project_dir: Path) -> Path:
    """Build one wheel and return the generated path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with _without_stale_build_outputs(project_dir):
        with DefaultIsolatedEnv(installer="uv") as environment:
            builder = ProjectBuilder.from_isolated_env(environment, str(project_dir))
            environment.install(builder.build_system_requires, _fresh=True)
            environment.install(builder.get_requires_for_build("wheel"))
            builder = ProjectBuilder.from_isolated_env(environment, str(project_dir))
            wheel_path = builder.build("wheel", str(output_dir))
    return Path(wheel_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    wheel = build_wheel(args.outdir.resolve(), args.project_dir.resolve())
    print(f"built clean wheel: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
