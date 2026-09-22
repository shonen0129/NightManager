"""YAML configuration loading primitives.

This module owns file-level configuration composition only. It deliberately
does not import Pydantic schemas, environment settings, brokers, or execution
code. Validation and construction of ``AppConfig`` remain at the explicit
application boundary in ``leadlag.execution.config``; callers must not use
this module as a second application-config entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "deep_merge",
    "load_yaml_with_base",
    "resolve_config_path",
]

_BASE_KEY = "__base__"


def resolve_config_path(path: str, relative_to: Path) -> Path:
    """Resolve an include path relative to the containing YAML file."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (relative_to.parent / candidate).resolve()


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge *override* into *base*.

    Mapping values are merged recursively; scalar and sequence values in the
    override replace the corresponding base value.  The input dictionaries are
    never mutated.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value) if key in merged else value
        return merged
    return override


def load_yaml_with_base(
    yaml_path: str | Path,
    _seen: set[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML file and recursively merge its ``__base__`` includes.

    A path can include one parent through ``__base__``; the parent can include
    another parent, and so on.  Circular references are rejected before any
    partially merged configuration is returned.
    """
    # Preserve the existing private helper's handling of an empty set while
    # making the recursive state explicit in this lower-level module.
    seen = _seen or set()
    resolved_path = Path(yaml_path).resolve()
    key = str(resolved_path)
    if key in seen:
        raise ValueError(f"Circular __base__ reference detected: {resolved_path}")
    seen.add(key)

    with resolved_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    base_path = data.pop(_BASE_KEY, None)
    if base_path:
        parent_path = resolve_config_path(str(base_path), resolved_path)
        parent_data = load_yaml_with_base(parent_path, seen)
        data = deep_merge(parent_data, data)

    return data
