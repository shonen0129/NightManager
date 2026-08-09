"""Tests for ``leadlag.data.schema``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.core.pit import PITAccessError
from leadlag.data.schema import (
    ColumnFamily,
    ExecutionFrame,
    all_expected_columns,
    column_name,
    family_columns,
    validate_frame,
)
from leadlag.data.tickers import JP_TICKERS, N_JP, N_US, US_TICKERS


def _make_df(rows: int = 10) -> pd.DataFrame:
    """Return a synthetic ``df_exec`` with the expected schema."""
    dti = pd.date_range("2020-01-01", periods=rows, freq="B")
    data: dict[str, np.ndarray | list] = {
        "sig_date": [d.strftime("%Y-%m-%d") for d in dti],
        "is_provisional": [False] * rows,
    }
    rng = np.random.default_rng(0)
    for tk in US_TICKERS:
        data[column_name(ColumnFamily.US_CC, tk)] = rng.normal(0, 0.01, rows)
    for tk in JP_TICKERS:
        for family in (
            ColumnFamily.JP_CC,
            ColumnFamily.JP_OC,
            ColumnFamily.JP_GAP,
            ColumnFamily.JP_CLOSE_SIG,
            ColumnFamily.JP_OPEN_TRADE,
            ColumnFamily.JP_BETA,
        ):
            data[column_name(family, tk)] = rng.normal(0, 0.01, rows)
    data[ColumnFamily.TOPIX_NIGHT.value] = rng.normal(0, 0.01, rows)
    data[ColumnFamily.TOPIX_OC.value] = rng.normal(0, 0.01, rows)
    data[ColumnFamily.TOPIX_CC.value] = rng.normal(0, 0.01, rows)
    return pd.DataFrame(data, index=dti)


@pytest.mark.unit
def test_column_name_construction():
    assert column_name(ColumnFamily.US_CC, "XLB") == "us_cc_XLB"
    assert column_name(ColumnFamily.JP_GAP, "1629.T") == "jp_gap_1629.T"


@pytest.mark.unit
def test_family_columns_count():
    assert len(family_columns(ColumnFamily.US_CC)) == N_US
    assert len(family_columns(ColumnFamily.JP_OC)) == N_JP
    assert family_columns(ColumnFamily.TOPIX_NIGHT) == ["topix_night_return"]


@pytest.mark.unit
def test_all_expected_columns():
    cols = all_expected_columns()
    # Metadata + US + 6 JP families (cc/oc/gap/close_sig/open_trade/beta) + 3 TOPIX
    expected_count = 2 + N_US + (6 * N_JP) + 3
    assert len(cols) == expected_count
    assert all(isinstance(c, str) for c in cols)


@pytest.mark.unit
def test_execution_frame_accessors():
    df = _make_df(rows=20)
    frame = ExecutionFrame(df)
    assert frame.n_rows == 20
    assert frame.us_cc().shape == (20, N_US)
    assert frame.jp_gap().shape == (20, N_JP)
    assert frame.jp_oc().shape == (20, N_JP)
    assert frame.topix_night().shape == (20,)


@pytest.mark.unit
def test_execution_frame_pit_view_blocks_future():
    df = _make_df(rows=20)
    frame = ExecutionFrame(df)
    view = frame.as_pit_view(ColumnFamily.US_CC, as_of=10)
    assert view.as_of == 10
    # Historical slice excludes the as-of row and does not include rows > as_of.
    hist = view.historical_slice(5)
    assert hist.shape == (5, N_US)
    with pytest.raises(PITAccessError):
        view.historical_range(0, 12)


@pytest.mark.unit
def test_validate_frame():
    df = _make_df()
    assert validate_frame(df) == []

    bad = df.drop(columns=["us_cc_XLB"])
    with pytest.raises(KeyError, match="us_cc_XLB"):
        validate_frame(bad)


@pytest.mark.unit
def test_validate_frame_non_required():
    bad = _make_df().drop(columns=["jp_oc_1617.T"])
    missing = validate_frame(bad, required=False)
    assert "jp_oc_1617.T" in missing
