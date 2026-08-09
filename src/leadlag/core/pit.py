"""Point-in-time data access layer.

Provides read-only views of arrays that refuse access to rows
beyond a fixed as-of index. This turns the "no look-ahead" invariant from a
convention checked by ComplianceAuditor into a structural guarantee at the
point of data access.
"""

from __future__ import annotations

from typing import cast

import numpy as np


class PITAccessError(ValueError):
    """Raised when a point-in-time view is asked for data after its as-of row."""


class PITMatrixView:
    """Read-only view of an ndarray that blocks access beyond a fixed as-of row.

    Parameters
    ----------
    values
        Underlying data (copied by reference; the caller must not mutate it).
    as_of
        Last observable row index. Rows with index > ``as_of`` are the future
        and are inaccessible through this view.
    name
        Identifier used in error messages.
    """

    __slots__ = ("_values", "_as_of", "name", "shape", "ndim")

    def __init__(self, values: np.ndarray, as_of: int, *, name: str = "data") -> None:
        self._values = np.asarray(values)
        self._as_of = int(as_of)
        self.name = str(name)
        self.shape = self._values.shape
        self.ndim = self._values.ndim

    @property
    def as_of(self) -> int:
        return self._as_of

    def _validate_end(self, end: int) -> int:
        if end > self._as_of:
            raise PITAccessError(
                f"{self.name}: requested rows up to {end}, but as-of row is "
                f"{self._as_of}. Look-ahead access is structurally blocked."
            )
        return end

    def historical_slice(self, window: int) -> np.ndarray:
        """Return rows ``[max(0, as_of - window) : as_of]``.

        The as-of row itself is excluded, which is the convention used by
        ``signal.compute_signal`` when building rolling windows.
        """
        start = max(0, self._as_of - window)
        return self._values[start : self._as_of]

    def historical_range(self, start: int, end: int) -> np.ndarray:
        """Return rows ``[start:end]`` with ``end <= as_of`` and ``start >= 0`` enforced."""
        self._validate_end(end)
        if start < 0:
            raise PITAccessError(
                f"{self.name}: requested start={start}, but negative starts are not allowed."
            )
        return self._values[start:end]

    def asof_row(self) -> np.ndarray:
        """Return the full row at the as-of index."""
        return cast(np.ndarray, self._values[self._as_of])

    def asof_col(self, col: int) -> float:
        """Return a single value at ``(as_of, col)``."""
        return cast(float, self._values[self._as_of, col])

    def __len__(self) -> int:
        return cast(int, self._values.shape[0])


def maybe_as_pit(
    values: np.ndarray | PITMatrixView,
    as_of: int,
    *,
    name: str = "data",
) -> PITMatrixView:
    """Wrap a plain ndarray in a PIT view unless it already is one.

    If ``values`` is already a ``PITMatrixView`` with a different ``as_of``
    than the one requested, raise ``ValueError`` to avoid silent stale-data
    bugs.
    """
    if isinstance(values, PITMatrixView):
        if values.as_of != as_of:
            raise ValueError(
                f"{name}: existing PIT view has as_of={values.as_of}, "
                f"but requested as_of={as_of}. Re-wrap or align the as_of."
            )
        return values
    return PITMatrixView(values, as_of=as_of, name=name)
