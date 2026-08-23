"""Build and validate the subsector -> TOPIX-17 aggregation matrix.

Reads `configs/research/subsector_mapping_generated.yaml` and the PIT
market-cap panel, then computes:

- A: 17 x 79 row-normalised aggregation matrix.  A[s, k] is the share of
  sector s's covered market cap that belongs to subsector k.
- coverage_s: covered market cap for sector s divided by the total market
  cap of all JPX-listed names in 17-sector s.  When the expanded cap panel
  is unavailable we fall back to the count-based ratio from JPX master.

All outputs are written to `configs/research/subsector_aggregation_vw.yaml`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("research.subsector.taxonomy_map")


def _topix17_index(ticker: str) -> int:
    """Return 0-based index for a TOPIX-17 ETF ticker."""
    return int(ticker.replace(".T", "")) - 1617


def load_mapping(mapping_path: Path) -> dict[str, Any]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_aggregation_matrix(
    stock_mapping: list[dict[str, Any]],
    cap_panel: pd.DataFrame,
    date: pd.Timestamp | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute aggregation matrix A and sector/subsector label lists.

    Args:
        stock_mapping: list of {ticker, subsector, topix17_etf} dicts.
        cap_panel: date x ticker market-cap panel.
        date: optional snapshot date.  If None, uses the last available row.

    Returns:
        A (17 x n_subsectors), row labels (TOPIX-17 ETFs), column labels
        (subsectors in stock_mapping order).
    """
    if date is None:
        cap = cap_panel.iloc[-1].copy()
    else:
        try:
            cap = cap_panel.loc[date].copy()
        except KeyError:
            # Use last available date on or before *date*
            valid = cap_panel.index[cap_panel.index <= date]
            if len(valid) == 0:
                raise ValueError(f"No market-cap data on or before {date}")
            cap = cap_panel.loc[valid[-1]].copy()

    subsectors = sorted({m["subsector"] for m in stock_mapping})
    sectors = [f"{1617 + i}.T" for i in range(17)]
    subsector_index = {s: i for i, s in enumerate(subsectors)}

    A = np.zeros((17, len(subsectors)))
    for m in stock_mapping:
        tk = m["ticker"]
        sub = m["subsector"]
        etf = m["topix17_etf"]
        if etf not in sectors or sub not in subsector_index:
            continue
        s_idx = _topix17_index(etf)
        k_idx = subsector_index[sub]
        val = float(cap.get(tk, 0.0))
        if not np.isfinite(val):
            continue
        A[s_idx, k_idx] += val

    # Row-normalise.  Empty rows remain zero (non-covered sectors).
    row_sums = A.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(row_sums > 1e-12, A / row_sums, 0.0)
    return A, sectors, subsectors


def build_coverage(
    stock_mapping: list[dict[str, Any]],
    cap_panel: pd.DataFrame,
    jpx_master_path: Path | None = None,
    date: pd.Timestamp | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-TOPIX-17 sector market-cap coverage.

    Returns a dict keyed by sector ETF with two metrics:
    - cap_coverage: covered market cap / covered market cap within sector
      (1.0 when no expanded panel is available; placeholder for PIT mcap).
    - count_coverage: number of covered tickers / number of JPX-listed names
      in the 17-sector code (from jpx_master_path).
    """
    if date is None:
        cap = cap_panel.iloc[-1].copy()
    else:
        try:
            cap = cap_panel.loc[date].copy()
        except KeyError:
            valid = cap_panel.index[cap_panel.index <= date]
            cap = cap_panel.loc[valid[-1]].copy()

    sectors = [f"{1617 + i}.T" for i in range(17)]
    covered_cap = {s: 0.0 for s in sectors}
    covered_count = {s: 0 for s in sectors}
    for m in stock_mapping:
        tk = m["ticker"]
        etf = m["topix17_etf"]
        covered_cap[etf] += cap.get(tk, 0.0)
        covered_count[etf] += 1

    # Map TOPIX-17 ETF code to the 17-sector code in JPX master
    # ETF 1617=食品 -> JPX code 1, 1618=エネルギー資源 -> 2, ..., 1633=不動産 -> 17
    etf_to_jpx17 = {f"{1617 + i}.T": i + 1 for i in range(17)}

    jpx_count = {s: 0 for s in sectors}
    if jpx_master_path is not None and jpx_master_path.exists():
        df = pd.read_csv(jpx_master_path)
        # 17業種コード is the numeric TOPIX-17 sector code
        for etf, code in etf_to_jpx17.items():
            jpx_count[etf] = int((df["17業種コード"] == str(code)).sum())

    total_covered_cap = {s: 0.0 for s in sectors}
    for m in stock_mapping:
        etf = m["topix17_etf"]
        total_covered_cap[etf] += cap.get(m["ticker"], 0.0)

    coverage = {}
    for s in sectors:
        cap_cov = (
            covered_cap[s] / total_covered_cap[s]
            if total_covered_cap[s] > 1e-12
            else 1.0
        )
        if jpx_count[s] > 0:
            count_cov = covered_count[s] / jpx_count[s]
        else:
            count_cov = 1.0
        coverage[s] = {
            "cap_coverage": float(cap_cov),
            "count_coverage": float(count_cov),
            "covered_count": int(covered_count[s]),
            "total_jpx_count": int(jpx_count[s]),
            "covered_cap": float(covered_cap[s]),
        }
    return coverage


def save_aggregation(
    output_path: Path,
    A: np.ndarray,
    rows: list[str],
    cols: list[str],
    coverage: dict[str, dict[str, float]],
    meta: dict[str, Any],
) -> None:
    """Save aggregation matrix, coverage, and metadata to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "aggregation_matrix_A": {
            "rows": rows,
            "cols": cols,
            "data": A.tolist(),
        },
        "coverage": coverage,
        "meta": meta,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def validate_convexity(A: np.ndarray) -> bool:
    """Check that each non-empty row sums to 1."""
    row_sums = A.sum(axis=1)
    empty = np.abs(row_sums) < 1e-10
    return bool(np.allclose(row_sums[~empty], 1.0))
