"""Unit tests for leadlag.config.paths."""

from __future__ import annotations

from leadlag.config.paths import (
    artifacts,
    gap_distribution_latest,
    live,
    logs,
    market_data,
    outputs,
    project_root,
    results,
    shadow_runs,
    var_dir,
)


def test_project_root_is_directory() -> None:
    root = project_root()
    assert root.exists() and root.is_dir()
    assert (root / "src" / "leadlag" / "config" / "paths.py").exists()


def test_var_dir_created() -> None:
    v = var_dir()
    assert v.exists() and v.is_dir()
    assert v.name == "var"


def test_subdir_helpers_are_under_var() -> None:
    for helper, name in [
        (results, "results"),
        (live, "live"),
        (artifacts, "artifacts"),
        (logs, "logs"),
        (shadow_runs, "shadow_runs"),
        (outputs, "outputs"),
        (market_data, "market_data"),
    ]:
        path = helper()
        assert path == var_dir() / name


def test_gap_distribution_latest() -> None:
    assert gap_distribution_latest() == var_dir() / "live" / "pipeline_data" / "gap_adjusted_distribution" / "latest"


def test_subdir_with_parts() -> None:
    assert results("foo", "bar") == var_dir() / "results" / "foo" / "bar"


def test_market_data_uses_canonical_path_when_old_root_exists(tmp_path, monkeypatch) -> None:
    import leadlag.config.paths as paths

    old_root = tmp_path / "market_data"
    old_root.mkdir()
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "var_dir", lambda: tmp_path / "var")

    canonical = tmp_path / "var" / "market_data"
    assert market_data("prices.sqlite") == canonical / "prices.sqlite"
    assert canonical.is_dir()
    assert list(old_root.iterdir()) == []


def test_installed_paths_require_explicit_deployment_root(tmp_path, monkeypatch):
    """Evaluate the real path module as if imported from a wheel."""
    import pytest

    from leadlag.config import paths

    source = paths.__file__
    namespace = {'__file__': str(tmp_path / 'lib/python3.12/site-packages/leadlag/config/paths.py')}
    from pathlib import Path
    exec(compile(Path(source).read_text(), source, 'exec'), namespace)
    resolve = namespace['project_root']
    monkeypatch.delenv('LEADLAG_RUNTIME_ROOT', raising=False)
    with pytest.raises(RuntimeError, match='LEADLAG_RUNTIME_ROOT'):
        resolve()
    monkeypatch.setenv('LEADLAG_RUNTIME_ROOT', 'relative')
    with pytest.raises(ValueError, match='absolute'):
        resolve()
    monkeypatch.setenv('LEADLAG_RUNTIME_ROOT', str(tmp_path))
    assert resolve() == tmp_path.resolve()
    assert namespace['results']('run') == tmp_path.resolve() / 'var/results/run'
