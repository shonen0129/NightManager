"""Run-owned rank-reversal inputs.

The rank-reversal overlay used to open one date-specific file from inside the
model.  This adapter loads the complete requested snapshot before a decision
run and hands the model an immutable copy through :mod:`leadlag.domain.inputs`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.utils.gap_matrix_io import load_gap_npy
from leadlag.utils.timestamps import normalize_jst_date


def load_rank_reversal_frame(
    gap_input_dir: Path | None,
    dates: Iterable[pd.Timestamp | str],
    *,
    file_pattern: str = "matrices/rank_reversal_{date}.npy",
) -> pd.DataFrame | None:
    """Load rank-reversal vectors for *dates* into one run-owned frame.

    Missing vectors are deliberately omitted.  The overlay can then apply its
    existing safe skip behaviour, while the model itself performs no implicit
    filesystem access on a typed decision path.
    """

    if gap_input_dir is None:
        return None
    rows: dict[pd.Timestamp, np.ndarray] = {}
    for value in dates:
        date = normalize_jst_date(value)
        signal, _alerts = load_gap_npy(gap_input_dir, date.strftime("%Y-%m-%d"), file_pattern)
        if signal is None:
            continue
        array = np.asarray(signal, dtype=float).reshape(-1)
        if len(array) != len(JP_TICKERS) or not np.all(np.isfinite(array)):
            continue
        rows[date] = array.copy()
    if not rows:
        return None
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(JP_TICKERS)).sort_index()


__all__ = ["load_rank_reversal_frame"]
