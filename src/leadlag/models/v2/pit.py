"""V2 PIT IR history loader."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DIAGNOSTICS_CACHE: dict[Path, tuple[str, pd.DataFrame, str, tuple[str, ...]]] = {}


def _normalize_diagnostic_trade_date(value: object) -> str:
    """Normalize one diagnostics timestamp to a JST calendar-date key.

    Naive values in the diagnostics files are historical JST timestamps/date
    strings.  Timezone-aware values are converted to JST before the date is
    taken, so a UTC timestamp near midnight cannot move into the wrong PIT
    bucket.
    """
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("diagnostics trade_date contains NaT")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Tokyo").tz_localize(None)
    return str(parsed.normalize().strftime("%Y-%m-%d"))


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

    # Diagnostics are stored as date-only strings.  Normalize the caller's
    # boundary before filtering so a timestamp such as ``09:10`` cannot make
    # the same trade-date row compare as historical data.
    try:
        trade_dt = pd.Timestamp(trade_date)
        if pd.isna(trade_dt):
            raise ValueError("NaT")
        if trade_dt.tzinfo is not None:
            trade_dt = trade_dt.tz_convert("Asia/Tokyo").tz_localize(None)
        trade_date_key = trade_dt.normalize().strftime("%Y-%m-%d")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"trade_date must be a valid timestamp: {trade_date!r}") from exc

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

    cache_key = diag_file.resolve()
    # Use the content digest rather than only mtime/size.  A producer can
    # rewrite a file in place while preserving both metadata values, and that
    # must invalidate the cached PIT history.
    raw_contents = diag_file.read_bytes()
    content_digest = hashlib.sha256(raw_contents).hexdigest()
    cached = _DIAGNOSTICS_CACHE.get(cache_key)
    if cached is not None and cached[0] == content_digest:
        df, ir_col, cached_alerts = cached[1], cached[2], list(cached[3])
    else:
        df = pd.read_csv(io.BytesIO(raw_contents))
        # Normalize each row independently so files containing a mixture of
        # naive JST dates and timezone-aware timestamps remain well-defined.
        df["trade_date"] = df["trade_date"].map(_normalize_diagnostic_trade_date)

        # Prefer pred_ir_gap_baseline_cost (computed with same weight
        # construction and cost formula as current_ir) over the legacy
        # ex-ante series.
        cached_alerts = []
        ir_col = "pred_ir_gap_baseline_cost"
        if ir_col not in df.columns:
            ir_col = "pred_ir_gap_exante_cost"
            cached_alerts.append(
                "pred_ir_gap_baseline_cost not found in diagnostics CSV, falling back to "
                "pred_ir_gap_exante_cost. Historical IR may be inconsistent with current_ir. "
                "Regenerate diagnostics with updated compute_gap_adjusted_distribution.py."
            )
        if ir_col not in df.columns:
            cached_alerts.append(
                "No IR column found in diagnostics. PIT binning falls back to Medium/1.0."
            )
        _DIAGNOSTICS_CACHE[cache_key] = (
            content_digest,
            df,
            ir_col,
            tuple(cached_alerts),
        )
    alerts.extend(cached_alerts)

    df_hist = df[df["trade_date"] < trade_date_key]
    if ir_col not in df_hist.columns:
        return np.array([]), alerts, np.array([])

    history_ir = df_hist[ir_col].values
    history_dates = pd.to_datetime(df_hist["trade_date"]).values
    return history_ir, alerts, history_dates
