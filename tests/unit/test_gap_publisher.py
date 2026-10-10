from __future__ import annotations

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.pipeline.gap_publisher import compute_rank_reversal_signal


def test_rank_reversal_signal_ignores_current_day_close_label() -> None:
    dates = pd.date_range("2026-09-24", periods=4, freq="B")
    columns = [f"jp_oc_{ticker}" for ticker in JP_TICKERS]
    values = np.vstack(
        [
            np.arange(len(JP_TICKERS), dtype=float),
            np.arange(len(JP_TICKERS), dtype=float)[::-1],
            np.roll(np.arange(len(JP_TICKERS), dtype=float), 3),
            np.linspace(-100.0, 100.0, len(JP_TICKERS)),
        ]
    )
    frame = pd.DataFrame(values, index=dates, columns=columns)
    changed = frame.copy()
    changed.loc[dates[-1], columns] = np.linspace(1000.0, -1000.0, len(JP_TICKERS))

    first = compute_rank_reversal_signal(frame, dates[-1].date().isoformat())
    second = compute_rank_reversal_signal(changed, dates[-1].date().isoformat())

    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
