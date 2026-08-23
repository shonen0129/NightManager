#!/usr/bin/env python3
"""Build PIT market-cap panel for subsector weighting.

Outputs:
  var/research/subsector/cap_panel.parquet
  var/research/subsector/cap_panel.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from research.experiments.subsector.market_cap import build_market_cap_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> int:
    mapping_path = ROOT / "configs" / "research" / "subsector_mapping_generated.yaml"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    tickers = [m["ticker"] for m in mapping["stock_mapping"]]
    # Add any canonical taxonomy tickers not in mapping (unmapped)
    tax_path = ROOT / "configs" / "taxonomy_subsectors.yaml"
    with open(tax_path, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    canonical = set()
    for _, tks in taxonomy.items():
        canonical.update(tks)
    extra = sorted(canonical - set(tickers))
    tickers = sorted(set(tickers) | canonical)
    print(f"Building market-cap panel for {len(tickers)} tickers ({len(extra)} extras)")

    out_path = ROOT / "var" / "research" / "subsector" / "cap_panel.parquet"
    build_market_cap_panel(
        tickers=tickers,
        subsector_dir=ROOT / "var" / "research" / "subsector",
        output_path=out_path,
        chunk_size=50,
        timeout_per_ticker=30,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
