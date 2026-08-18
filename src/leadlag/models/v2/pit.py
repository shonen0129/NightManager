"""V2 PIT IR history loader."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_pit_ir_history(
    gap_input_dir: Path,
    trade_date: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load historical ex-ante IR series for PIT binning.

    Reads ``portfolio_gap_distribution_diagnostics.csv`` and returns only
    rows strictly before *trade_date* to preserve point-in-time integrity.

    Args:
        gap_input_dir: Root directory of the gap distribution output.
        trade_date: Trade execution date (rows >= this date are excluded).

    Returns:
        Tuple of (history_ir array, alerts list, history_trade_dates array).
    """
    alerts: list[str] = []

    # Prefer the canonical full-history diagnostics file (maintained across runs)
    # over the per-run portfolio_gap_distribution_diagnostics.csv, which may
    # contain only the recent days computed in that run.
    canonical_file = gap_input_dir / "full_history_diagnostics.csv"
    if not canonical_file.exists():
        canonical_file = gap_input_dir.parent / "full_history_diagnostics.csv"
    if canonical_file.exists():
        diag_file = canonical_file
    else:
        diag_file = gap_input_dir / "portfolio_gap_distribution_diagnostics.csv"

    if not diag_file.exists():
        alerts.append(
            f"Diagnostics file missing: {diag_file}. PIT binning falls back to Medium/1.0."
        )
        return np.array([]), alerts, np.array([])

    df = pd.read_csv(diag_file)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df_hist = df[df["trade_date"] < trade_date]

    # Prefer pred_ir_gap_baseline_cost (computed with same weight construction
    # and cost formula as current_ir) over pred_ir_gap_exante_cost (legacy,
    # uses different weights and rolling realized cost).
    ir_col = "pred_ir_gap_baseline_cost"
    if ir_col not in df_hist.columns:
        ir_col = "pred_ir_gap_exante_cost"
        alerts.append(
            "pred_ir_gap_baseline_cost not found in diagnostics CSV, falling back to "
            "pred_ir_gap_exante_cost. Historical IR may be inconsistent with current_ir. "
            "Regenerate diagnostics with updated compute_gap_adjusted_distribution.py."
        )

    if ir_col not in df_hist.columns:
        alerts.append(
            "No IR column found in diagnostics. PIT binning falls back to Medium/1.0."
        )
        return np.array([]), alerts, np.array([])

    history_ir = df_hist[ir_col].values
    history_dates = pd.to_datetime(df_hist["trade_date"]).values
    return history_ir, alerts, history_dates
