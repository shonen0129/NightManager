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


def test_market_data_can_handle_legacy_root() -> None:
    root = project_root()
    legacy = root / "market_data"
    canonical = var_dir() / "market_data"
    # If the canonical exists or legacy does not, we should get canonical.
    if canonical.exists() or not legacy.exists():
        assert market_data() == canonical
