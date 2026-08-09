"""Typed schema for the execution DataFrame (``df_exec``).

This module turns the column-name conventions described in ``AGENTS.md`` and
``data/preprocessor.py`` into a machine-enforceable contract.  It provides:

* ``ColumnFamily`` — an enum for every column family used by the strategy.
* ``column_name`` / ``family_columns`` — deterministic, registry-driven column
  name construction so string interpolation ``f"us_cc_{tk}"`` can be replaced
  by a single helper.
* ``ExecutionFrame`` — a thin, frozen wrapper around ``pd.DataFrame`` that
  exposes type-safe accessors for each block of returns and for the PIT-aware
  matrix views used by signal generation.

Existing callers may continue to use bare ``pd.DataFrame`` objects; the
registry helpers here are an optional, additive layer for new code and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, US_TICKERS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from leadlag.core.pit import PITMatrixView


class ColumnFamily(StrEnum):
    """Column families in ``df_exec``.

    Values are the exact prefixes / column names used in ``preprocessor.py``.
    """

    US_CC = "us_cc"
    JP_CC = "jp_cc"
    JP_OC = "jp_oc"
    JP_GAP = "jp_gap"
    JP_CLOSE_SIG = "jp_close_sig"
    JP_OPEN_TRADE = "jp_open_trade"
    JP_BETA = "jp_beta"
    TOPIX_NIGHT = "topix_night_return"
    TOPIX_OC = "topix_oc_return"
    TOPIX_CC = "topix_cc_trade"


FAMILY_TO_TICKERS: dict[ColumnFamily, list[str]] = {
    ColumnFamily.US_CC: US_TICKERS,
    ColumnFamily.JP_CC: JP_TICKERS,
    ColumnFamily.JP_OC: JP_TICKERS,
    ColumnFamily.JP_GAP: JP_TICKERS,
    ColumnFamily.JP_CLOSE_SIG: JP_TICKERS,
    ColumnFamily.JP_OPEN_TRADE: JP_TICKERS,
    ColumnFamily.JP_BETA: JP_TICKERS,
}

SCALAR_FAMILIES: set[ColumnFamily] = {
    ColumnFamily.TOPIX_NIGHT,
    ColumnFamily.TOPIX_OC,
    ColumnFamily.TOPIX_CC,
}


def column_name(family: ColumnFamily, ticker: str) -> str:
    """Return the canonical ``df_exec`` column name for *family* and *ticker*."""
    return f"{family.value}_{ticker}"


def family_columns(family: ColumnFamily, tickers: Sequence[str] | None = None) -> list[str]:
    """Return the list of column names belonging to *family*.

    For per-ticker families, *tickers* defaults to the universe defined in
    ``data/tickers.py``.
    """
    if family in SCALAR_FAMILIES:
        return [family.value]
    tickers = tickers or FAMILY_TO_TICKERS.get(family, [])
    return [column_name(family, tk) for tk in tickers]


def all_expected_columns() -> list[str]:
    """Return the full ordered list of columns produced by ``preprocessor.py``.

    Order: metadata columns first, then US returns, JP returns, TOPIX scalars,
    and finally JP betas.  This matches ``preprocessor.py``'s record layout.
    """
    return (
        ["sig_date", "is_provisional"]
        + family_columns(ColumnFamily.US_CC)
        + family_columns(ColumnFamily.JP_CC)
        + family_columns(ColumnFamily.JP_OC)
        + family_columns(ColumnFamily.JP_GAP)
        + family_columns(ColumnFamily.JP_CLOSE_SIG)
        + family_columns(ColumnFamily.JP_OPEN_TRADE)
        + [ColumnFamily.TOPIX_NIGHT.value]
        + [ColumnFamily.TOPIX_OC.value]
        + [ColumnFamily.TOPIX_CC.value]
        + family_columns(ColumnFamily.JP_BETA)
    )


@dataclass(frozen=True)
class ExecutionFrame:
    """A frozen, type-safe view over an execution DataFrame.

    The underlying ``pd.DataFrame`` is exposed as ``.df`` for full
    compatibility, but convenience accessors return the per-family numpy
    blocks that the signal pipeline consumes.  Missing columns raise
    ``KeyError`` at access time instead of silently propagating string typos.
    """

    df: pd.DataFrame

    @property
    def n_rows(self) -> int:
        return int(len(self.df))

    @property
    def n_us(self) -> int:
        return len(US_TICKERS)

    @property
    def n_jp(self) -> int:
        return len(JP_TICKERS)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.df.index)

    def _array(self, cols: list[str]) -> np.ndarray:
        return cast(np.ndarray, self.df[cols].to_numpy(dtype=float))

    def us_cc(self) -> np.ndarray:
        """US close-to-close returns, shape ``(T, N_US)``."""
        return self._array(family_columns(ColumnFamily.US_CC))

    def jp_cc(self) -> np.ndarray:
        """JP close-to-close returns, shape ``(T, N_JP)``."""
        return self._array(family_columns(ColumnFamily.JP_CC))

    def jp_oc(self) -> np.ndarray:
        """JP open-to-close (target) returns, shape ``(T, N_JP)``."""
        return self._array(family_columns(ColumnFamily.JP_OC))

    def jp_gap(self) -> np.ndarray:
        """JP overnight gap returns, shape ``(T, N_JP)``."""
        return self._array(family_columns(ColumnFamily.JP_GAP))

    def jp_close_sig(self) -> np.ndarray:
        """JP close used for signal, shape ``(T, N_JP)``."""
        return self._array(family_columns(ColumnFamily.JP_CLOSE_SIG))

    def jp_open_trade(self) -> np.ndarray:
        """JP open used for execution, shape ``(T, N_JP)``."""
        return self._array(family_columns(ColumnFamily.JP_OPEN_TRADE))

    def jp_beta(self) -> np.ndarray:
        """JP rolling TOPIX betas, shape ``(T, N_JP)``."""
        return self._array(family_columns(ColumnFamily.JP_BETA))

    def topix_night(self) -> np.ndarray:
        """TOPIX overnight return, shape ``(T,)``."""
        return cast(np.ndarray, self.df[ColumnFamily.TOPIX_NIGHT.value].to_numpy(dtype=float))

    def topix_oc(self) -> np.ndarray:
        """TOPIX open-to-close return, shape ``(T,)``."""
        return cast(np.ndarray, self.df[ColumnFamily.TOPIX_OC.value].to_numpy(dtype=float))

    def topix_cc(self) -> np.ndarray:
        """TOPIX close-to-close return, shape ``(T,)``."""
        return cast(np.ndarray, self.df[ColumnFamily.TOPIX_CC.value].to_numpy(dtype=float))

    def as_pit_view(self, family: ColumnFamily, as_of: int) -> PITMatrixView:
        """Return a point-in-time view of *family* with the given *as_of* row.

        The as-of row itself is included in the underlying array, which lets
        callers use ``historical_slice(window)`` (which excludes the as-of row)
        or access ``asof_row()`` directly.
        """
        from leadlag.core.pit import PITMatrixView

        arr = self._array(family_columns(family))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return PITMatrixView(arr, as_of=as_of, name=family.value)


def validate_frame(df: pd.DataFrame, *, required: bool = True) -> list[str]:
    """Check that *df* contains the expected ``df_exec`` columns.

    Returns a list of alerts (missing columns).  When *required* is ``True``,
    raises ``KeyError`` if any are missing.
    """
    expected = all_expected_columns()
    missing = [col for col in expected if col not in df.columns]
    if required and missing:
        raise KeyError(f"ExecutionFrame missing columns: {missing}")
    return missing
