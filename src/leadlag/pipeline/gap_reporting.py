"""Output adapters for Step 2 gap-distribution accumulators.

The accumulator is produced by the pure per-date loop, while this module owns
the reusable tabular diagnostic views and their file publication.  Research-only
portfolio metrics, PIT comparisons, plots, and report prose live under
``research.diagnostics`` and are deliberately kept out of the production path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GapDiagnosticFrames:
    """Tabular views emitted from one Step 2 accumulator."""

    ticker_summary: pd.DataFrame
    gap_long: pd.DataFrame
    gap_daily: pd.DataFrame
    distribution_long: pd.DataFrame
    distribution_daily: pd.DataFrame
    omega_daily: pd.DataFrame


def build_gap_diagnostic_frames(
    accumulator: Any,
    *,
    tickers: Iterable[str],
) -> GapDiagnosticFrames:
    """Build the stable CSV views without touching the filesystem."""
    ticker_records: list[dict[str, Any]] = []
    records_by_ticker = getattr(accumulator, "omega_gap_ticker_records", {})
    for ticker in tickers:
        frame = pd.DataFrame(records_by_ticker.get(ticker, []))
        if frame.empty:
            ticker_records.append(
                {
                    "ticker": ticker,
                    "mean_omega_raw_diag": float("nan"),
                    "mean_omega_gap_diag": float("nan"),
                    "mean_ratio": float("nan"),
                    "time_series_correlation": float("nan"),
                }
            )
            continue
        raw = frame["omega_raw_diag"]
        gap = frame["omega_gap_diag"]
        correlation = gap.corr(raw) if len(frame) > 1 else float("nan")
        ticker_records.append(
            {
                "ticker": ticker,
                "mean_omega_raw_diag": raw.mean(),
                "mean_omega_gap_diag": gap.mean(),
                "mean_ratio": (gap / raw.replace(0.0, 1e-10)).mean(),
                "time_series_correlation": correlation,
            }
        )

    return GapDiagnosticFrames(
        ticker_summary=pd.DataFrame(ticker_records),
        gap_long=pd.DataFrame(getattr(accumulator, "gap_long_records", [])),
        gap_daily=pd.DataFrame(getattr(accumulator, "gap_daily_records", [])),
        distribution_long=pd.DataFrame(getattr(accumulator, "dist_long_records", [])),
        distribution_daily=pd.DataFrame(getattr(accumulator, "dist_daily_records", [])),
        omega_daily=pd.DataFrame(getattr(accumulator, "omega_gap_daily_records", [])),
    )


def write_gap_diagnostic_frames(frames: GapDiagnosticFrames, out_dir: Path) -> None:
    """Publish the stable Step 2 CSV views under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames.ticker_summary.to_csv(out_dir / "omega_gap_summary_by_ticker.csv", index=False)
    frames.gap_long.to_csv(out_dir / "gap_components_long.csv", index=False)
    frames.gap_daily.to_csv(out_dir / "gap_components_daily.csv", index=False)
    frames.distribution_long.to_csv(
        out_dir / "gap_adjusted_distribution_long.csv", index=False
    )
    frames.distribution_daily.to_csv(
        out_dir / "gap_adjusted_distribution_daily.csv", index=False
    )
    frames.omega_daily.to_csv(out_dir / "omega_gap_summary_daily.csv", index=False)


__all__ = [
    "GapDiagnosticFrames",
    "build_gap_diagnostic_frames",
    "write_gap_diagnostic_frames",
]
