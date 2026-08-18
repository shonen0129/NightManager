"""Domain types for cross-sectional signal generation."""

from __future__ import annotations

from collections.abc import (
    Callable,
    ItemsView,
    KeysView,
    MutableMapping,
    ValuesView,
)
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SignalTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SignalComponent:
    """Named signal source and its raw series (one value per JP asset)."""

    name: str
    series: np.ndarray
    weight: float = 1.0
    enabled: bool = True
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class SignalPackage(MutableMapping[str, Any]):
    """Complete output of the signal-generation phase.

    Wraps the dict historically returned by ``BLPXOutputAdapter.adapt`` so that
    existing call sites and tests can keep using ``pred["signals"]``,
    ``pred.get(...)``, and ``for k in pred: ...``.  The package also exposes
    the same entries as typed attributes.
    """

    _data: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def raw_pca_signals(self) -> pd.DataFrame:
        return self._data["raw_pca_signals"]

    @property
    def residual_pca_signals(self) -> pd.DataFrame:
        return self._data["residual_pca_signals"]

    @property
    def p4_signals(self) -> pd.DataFrame:
        return self._data["p4_signals"]

    @property
    def raw_blpx_signals(self) -> pd.DataFrame:
        return self._data["raw_blpx_signals"]

    @property
    def residual_blpx_signals(self) -> pd.DataFrame:
        return self._data["residual_blpx_signals"]

    @property
    def signals(self) -> pd.DataFrame:
        return self._data["signals"]

    @property
    def normalized_signals(self) -> pd.DataFrame:
        return self._data["normalized_signals"]

    @property
    def y_jp_oc_df(self) -> pd.DataFrame:
        return self._data["y_jp_oc_df"]

    @property
    def blp_diagnostics(self) -> pd.DataFrame:
        return self._data["blp_diagnostics"]

    @property
    def sigma_yy(self) -> np.ndarray | None:
        return self._data.get("sigma_yy")

    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Any:
        return iter(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self) -> ItemsView[str, Any]:
        return self._data.items()

    def values(self) -> ValuesView[Any]:
        return self._data.values()

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy as a plain ``dict``."""
        return dict(self._data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SignalPackage:
        """Build a ``SignalPackage`` from a plain ``dict``."""
        if isinstance(d, cls):
            return d
        return cls(_data=dict(d))

    def __repr__(self) -> str:
        return f"SignalPackage({list(self._data.keys())})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SignalPackage):
            return False
        return self._data is other._data
