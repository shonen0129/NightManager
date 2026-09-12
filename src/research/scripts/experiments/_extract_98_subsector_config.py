#!/usr/bin/env python3
"""Extract 98 subsector configuration from the reviewed md file."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

# Read the md file
md_file = ROOT / "docs" / "analysis" / "TOPIX_subsector_details_98_final_reviewed.md"
content = md_file.read_text(encoding="utf-8")

# Parse subsectors
subsectors = []
current_subsector = None
current_tickers = []
current_labels = None

lines = content.split("\n")
for i, line in enumerate(lines):
    # Match subsector header
    header_match = re.match(r"## (\d+)\. (.+)", line)
    if header_match:
        # Save previous subsector
        if current_subsector:
            sub_data = {
                "name": current_subsector,
                "tickers": current_tickers.copy()
            }
            if current_labels:
                sub_data["sensitivity_labels"] = current_labels
            subsectors.append(sub_data)
        current_subsector = header_match.group(2)
        current_tickers = []
        current_labels = None
        continue

    # Match sensitivity labels (handle both +0.6 and 0.6 formats)
    sens_match = re.match(r"\*\*感応度ラベル\*\*: w3=([+-]?\d+\.?\d*), w4=([+-]?\d+\.?\d*), w5=([+-]?\d+\.?\d*), w6=([+-]?\d+\.?\d*)", line)
    if sens_match and current_subsector:
        current_labels = {
            "w3": float(sens_match.group(1)),
            "w4": float(sens_match.group(2)),
            "w5": float(sens_match.group(3)),
            "w6": float(sens_match.group(4))
        }
        continue

    # Match ticker rows (format: | 3001.T | 片倉工業 | 繊維製品 |)
    # Handle both numeric (3001.T) and alphanumeric (543A.T) tickers
    ticker_match = re.match(r"\| ([\dA-Z]+\.T) \| (.+) \| (.+) \|", line)
    if ticker_match and current_subsector:
        ticker = ticker_match.group(1)
        name = ticker_match.group(2)
        jpx33 = ticker_match.group(3)
        current_tickers.append({
            "ticker": ticker,
            "name": name,
            "jpx33": jpx33
        })

# Save last subsector
if current_subsector:
    sub_data = {
        "name": current_subsector,
        "tickers": current_tickers.copy()
    }
    if current_labels:
        sub_data["sensitivity_labels"] = current_labels
    subsectors.append(sub_data)

# Create configuration
config = {
    "meta": {
        "source_file": "docs/analysis/TOPIX_subsector_details_98_final_reviewed.md",
        "total_subsectors": len(subsectors),
        "total_tickers": sum(len(s["tickers"]) for s in subsectors)
    },
    "subsectors": subsectors
}

# Write YAML file
output_file = ROOT / "configs" / "research" / "subsector_98_config.yaml"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"Generated: {output_file}")
print(f"Total subsectors: {len(subsectors)}")
print(f"Total tickers: {sum(len(s['tickers']) for s in subsectors)}")

# Show sample
print("\nSample subsectors:")
for i, sub in enumerate(subsectors[:3]):
    print(f"{i+1}. {sub['name']}")
    if "sensitivity_labels" in sub:
        print(f"   Labels: {sub['sensitivity_labels']}")
    print(f"   Tickers: {len(sub['tickers'])}")
