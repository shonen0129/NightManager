"""Immutable configuration helpers.

Prevents the classic experiment bug where ``base_cfg.copy()`` (shallow copy)
makes two model runs share nested dicts, so a mutation in one run silently
propagates to another.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConfigMutationError(AttributeError):
    """Raised when an attempt is made to modify a frozen config object."""


class FrozenConfigDict:
    """Read-only wrapper around a configuration dict.

    Mutations raise ``ConfigMutationError``; nested dict access returns
    another ``FrozenConfigDict``. Supports read access and iteration.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return FrozenConfigDict(value)
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return self[key]

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable deep copy of the underlying config."""
        return copy.deepcopy(self._data)

    def __setitem__(self, key: str, value: Any) -> None:
        raise ConfigMutationError(f"Config is frozen; cannot set {key!r}")

    def __delitem__(self, key: str) -> None:
        raise ConfigMutationError(f"Config is frozen; cannot delete {key!r}")


def safe_config_copy(config: Any) -> Any:
    """Return a deep, independent copy of a config object.

    - Pydantic ``BaseModel`` instances use ``model_copy(deep=True)``.
    - Plain dicts use ``copy.deepcopy``.
    - Other types fall back to ``copy.deepcopy``.

    This should be used instead of ``cfg.copy()`` or ``dict.copy()`` for
    configuration objects, because shallow copies share nested mutable values.
    """
    if isinstance(config, BaseModel):
        return config.model_copy(deep=True)
    if isinstance(config, FrozenConfigDict):
        return FrozenConfigDict(copy.deepcopy(config.to_dict()))
    if isinstance(config, dict):
        return copy.deepcopy(config)
    return copy.deepcopy(config)


def freeze_config_dict(config: dict[str, Any]) -> FrozenConfigDict:
    """Wrap a config dict in a read-only, immutable view."""
    return FrozenConfigDict(copy.deepcopy(config))
