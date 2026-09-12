#!/usr/bin/env python3
"""Extract subsector details including tickers and sensitivity labels for analysis."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

# Load subsector mapping
with open(ROOT / "configs" / "research" / "subsector_mapping_generated.yaml", "r", encoding="utf-8") as f:
    mapping = yaml.safe_load(f)

# Load sensitivity labels
with open(ROOT / "configs" / "research" / "subsector_sensitivity_labels_heuristic.yaml", "r", encoding="utf-8") as f:
    heuristic_labels = yaml.safe_load(f)["sensitivity_labels"]

# Group tickers by subsector
subsector_tickers = defaultdict(list)
for entry in mapping["stock_mapping"]:
    subsector = entry["subsector"]
    ticker = entry["ticker"]
    name = entry["name"]
    topix17_etf = entry["topix17_etf"]
    subsector_tickers[subsector].append({
        "ticker": ticker,
        "name": name,
        "topix17_etf": topix17_etf
    })

# Sort subsectors alphabetically
sorted_subsectors = sorted(subsector_tickers.keys())

# Generate markdown content
md_lines = [
    "# サブセクター詳細：銘柄構成と感応度ラベル",
    "",
    "各サブセクターの銘柄構成と感応度ラベルの詳細",
    "",
    f"**総サブセクター数**: {len(sorted_subsectors)}",
    f"**総銘柄数**: {sum(len(tickers) for tickers in subsector_tickers.values())}",
    "",
    "---",
    ""
]

for subsector in sorted_subsectors:
    tickers = subsector_tickers[subsector]
    tickers_sorted = sorted(tickers, key=lambda x: x["ticker"])

    # Get sensitivity labels
    if subsector in heuristic_labels:
        labels = heuristic_labels[subsector]
        labels_str = f"w3={labels['w3']:.2f}, w4={labels['w4']:.2f}, w5={labels['w5']:.2f}, w6={labels['w6']:.2f}"
    else:
        labels_str = "未設定"

    # Count ETF distribution
    etf_counts = defaultdict(int)
    for t in tickers:
        etf_counts[t["topix17_etf"]] += 1
    etf_dist_str = ", ".join([f"{etf}({count})" for etf, count in sorted(etf_counts.items())])

    md_lines.extend([
        f"## {subsector}",
        "",
        f"**銘柄数**: {len(tickers)}",
        f"**感応度ラベル**: {labels_str}",
        f"**ETF分布**: {etf_dist_str}",
        "",
        "### 登録銘柄",
        "",
        "| Ticker | 銘柄名 | TOPIX17 ETF |",
        "|---|---|---|",
    ])

    for t in tickers_sorted:
        md_lines.append(f"| {t['ticker']} | {t['name']} | {t['topix17_etf']} |")

    md_lines.extend(["", "---", ""])

# Write to markdown file
output_file = ROOT / "docs" / "analysis" / "subsector_details_with_tickers.md"
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text("\n".join(md_lines), encoding="utf-8")

print(f"Generated: {output_file}")
print(f"Total subsectors: {len(sorted_subsectors)}")
print(f"Total tickers: {sum(len(tickers) for tickers in subsector_tickers.values())}")
