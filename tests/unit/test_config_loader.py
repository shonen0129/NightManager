"""Tests for the YAML composition layer extracted in S1a."""

from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.config.loader import deep_merge, load_yaml_with_base
from leadlag.execution import config as execution_config
from leadlag.models import production_v2


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"model": {"window": 20, "nested": {"keep": True}}, "list": [1]}
    override = {"model": {"nested": {"replace": 2}}, "list": [3]}

    merged = deep_merge(base, override)

    assert merged == {
        "model": {"window": 20, "nested": {"keep": True, "replace": 2}},
        "list": [3],
    }
    assert base == {"model": {"window": 20, "nested": {"keep": True}}, "list": [1]}
    assert override == {"model": {"nested": {"replace": 2}}, "list": [3]}


def test_load_yaml_with_base_resolves_relative_chain(tmp_path: Path) -> None:
    parent = tmp_path / "base.yaml"
    child = tmp_path / "production.yaml"
    parent.write_text("model:\n  window: 20\nrisk:\n  stop: 0.03\n", encoding="utf-8")
    child.write_text("__base__: base.yaml\nmodel:\n  window: 60\n", encoding="utf-8")

    expected = {"model": {"window": 60}, "risk": {"stop": 0.03}}
    assert load_yaml_with_base(child) == expected


def test_load_yaml_with_base_rejects_circular_reference(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("__base__: second.yaml\n", encoding="utf-8")
    second.write_text("__base__: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular __base__"):
        load_yaml_with_base(first)


def test_execution_config_has_no_private_composition_aliases() -> None:
    for name in ("_deep_merge", "_load_yaml_with_base", "_resolve_config_path"):
        assert not hasattr(execution_config, name)


def test_production_v2_does_not_reexport_config_parser() -> None:
    assert not hasattr(production_v2, "parse_run_config")
