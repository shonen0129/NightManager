#!/usr/bin/env python3
"""Exploratory wallet-snapshot Sharpe proxy; never account P&L."""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

from leadlag.config.paths import results

TRADING_DAYS_PER_YEAR = 245


def observations(root: Path, start: date | None, end: date | None) -> list[tuple[date, float]]:
    """Read close snapshots, de-duplicated by date, without network access."""
    found: dict[date, float] = {}
    for path in sorted(root.glob("*_production_close_positions/wallet_close_*.json")):
        raw_date = path.stem.removeprefix("wallet_close_")[:8]
        try:
            day = date.fromisoformat(f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}")
        except ValueError:
            continue
        if (start and day < start) or (end and day > end):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            value = float(data["ukeire_hosyoukin"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if math.isfinite(value) and value > 0:
            found[day] = value
    return sorted(found.items())


def report(rows: list[tuple[date, float]]) -> str:
    disclaimer = (
        "WARNING: ukeire_hosyoukin is a margin/cash proxy, NOT realized net P&L "
        "or complete daily account returns. Missing trading days, deposits, "
        "withdrawals and unequal observation intervals are not adjusted; "
        "245-day annualization is indicative only."
    )
    if not rows:
        return "No close wallet snapshots found.\n" + disclaimer
    if len(rows) < 3:
        return "Need at least three positive equity observations to compute Sharpe.\n" + disclaimer
    changes = [b[1] / a[1] - 1 for a, b in zip(rows, rows[1:])]
    average = sum(changes) / len(changes)
    variance = sum((x - average) ** 2 for x in changes) / (len(changes) - 1)
    volatility = math.sqrt(variance)
    sharpe = average / volatility * math.sqrt(TRADING_DAYS_PER_YEAR) if volatility else float("nan")
    peak = rows[0][1]
    drawdown = 0.0
    for _, equity in rows:
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
    lines = [
        "=== Close wallet proxy statistics (research only) ===",
        f"Period: {rows[0][0]} -> {rows[-1][0]}",
        f"Close snapshots: {len(rows)}",
        f"Return intervals: {len(changes)}",
        f"Proxy change: {(rows[-1][1] / rows[0][1] - 1) * 100:+.2f}%",
        f"Annualized interval mean: {average * TRADING_DAYS_PER_YEAR * 100:+.2f}%",
        f"Annualized interval volatility: {volatility * math.sqrt(TRADING_DAYS_PER_YEAR) * 100:.2f}%",
        f"Max observed drawdown: {drawdown * 100:.2f}%",
        f"Proxy Sharpe (245-day assumption): {sharpe:.2f}",
        disclaimer,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None, help="Snapshot root; defaults to canonical var/results")
    parser.add_argument("--start-date", type=date.fromisoformat, default=None, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None, help="YYYY-MM-DD inclusive")
    args = parser.parse_args(argv)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    root = args.results_dir if args.results_dir is not None else results()
    print(report(observations(root, args.start_date, args.end_date)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
