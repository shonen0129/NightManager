"""Regression tests for the shared multi-day return transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.data.horizon_returns import compute_cumulative_returns


def test_cumprod_and_sum_are_explicit_and_leave_input_unchanged() -> None:
    frame = pd.DataFrame({"us_cc_XLB": [0.1, 0.2, -0.1], "other": [1, 2, 3]})
    original = frame.copy(deep=True)

    compounded = compute_cumulative_returns(frame, 2)
    summed = compute_cumulative_returns(frame, 2, method="sum")

    assert np.isnan(compounded.loc[0, "us_cc_XLB"])
    assert compounded.loc[1, "us_cc_XLB"] == pytest.approx(0.32)
    assert summed.loc[1, "us_cc_XLB"] == pytest.approx(0.3)
    pd.testing.assert_frame_equal(frame, original)
    assert list(compounded["other"]) == [1, 2, 3]


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_horizon_must_be_a_positive_integer(horizon: object) -> None:
    with pytest.raises(ValueError):
        compute_cumulative_returns(pd.DataFrame({"x": [1.0]}), horizon)  # type: ignore[arg-type]


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method"):
        compute_cumulative_returns(pd.DataFrame({"x": [1.0]}), 2, method="bad")  # type: ignore[arg-type]
