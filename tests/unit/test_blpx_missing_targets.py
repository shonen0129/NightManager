"""Regressions for missing realized labels in BLPX training windows."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from leadlag.models.blpx.correlation import _prepare_window_returns


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        blp_window=5,
        exec_adjustment="none",
        n_u=1,
        winsor_sigma=None,
    )


def test_blpx_window_drops_invalid_label_rows_instead_of_zero_filling() -> None:
    returns = np.array(
        [
            [0.01, 0.02],
            [0.02, np.nan],
            [0.03, 0.00],
            [0.04, 0.01],
        ],
        dtype=float,
    )

    window = _prepare_window_returns(_model(), returns, current_index=4, rolling_std=None)

    np.testing.assert_allclose(
        window,
        np.array(
            [
                [0.01, 0.02],
                [0.03, 0.00],
                [0.04, 0.01],
            ]
        ),
    )


def test_blpx_window_fails_closed_when_no_complete_history_remains() -> None:
    returns = np.array(
        [
            [0.01, np.nan],
            [0.02, np.inf],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="No complete finite rows"):
        _prepare_window_returns(_model(), returns, current_index=2, rolling_std=None)
