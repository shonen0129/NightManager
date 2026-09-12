#!/usr/bin/env python3
"""Build aggregation matrix for 98 subsectors to 17 TOPIX ETFs."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

# Load 98 subsector configuration
with open(ROOT / "configs" / "research" / "subsector_98_config.yaml", "r", encoding="utf-8") as f:
    subsector_config = yaml.safe_load(f)

# Load JPX master data to get TOPIX17 ETF mapping
jpx_master = pd.read_csv(ROOT / "var" / "research" / "subsector" / "jpx_master" / "jpx_master.csv")

# 17 TOPIX ETF codes
TOPIX17_ETFS = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービス",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

# Extract subsector names
SUBSECTOR_NAMES = [s["name"] for s in subsector_config["subsectors"]]

# Build ticker to ETF mapping from subsector config
ticker_to_etf = {}
for sub in subsector_config["subsectors"]:
    for ticker_info in sub["tickers"]:
        ticker = ticker_info["ticker"]
        # We need to determine which ETF this ticker belongs to
        # For now, use JPX33 industry to infer ETF
        ticker_to_etf[ticker] = {
            "ticker": ticker,
            "name": ticker_info["name"],
            "jpx33": ticker_info["jpx33"],
            "subsector": sub["name"]
        }

# Map JPX33 industries to TOPIX17 ETFs
# This is a simplified mapping - in reality would need proper classification
JPX33_TO_TOPIX17 = {
    "食料品": "1617.T",
    "石油・石炭製品": "1618.T",
    "鉱業": "1618.T",
    "建設業": "1619.T",
    "ガラス・土石製品": "1619.T",
    "化学": "1620.T",
    "医薬品": "1621.T",
    "繊維製品": "1620.T",
    "パルプ・紙": "1620.T",
    "輸送用機器": "1622.T",
    "鉄鋼": "1623.T",
    "非鉄金属": "1623.T",
    "金属製品": "1624.T",
    "機械": "1624.T",
    "電気機器": "1625.T",
    "情報・通信業": "1626.T",
    "サービス業": "1626.T",
    "電気・ガス業": "1627.T",
    "陸運業": "1628.T",
    "海運業": "1628.T",
    "空運業": "1628.T",
    "倉庫・運輸関連業": "1628.T",
    "卸売業": "1629.T",
    "小売業": "1630.T",
    "銀行業": "1631.T",
    "証券・商品先物業": "1632.T",
    "保険業": "1632.T",
    "その他金融業": "1632.T",
    "不動産業": "1633.T",
    "その他製品": "1630.T",
}

# Assign ETF to each ticker
for ticker in ticker_to_etf:
    jpx33 = ticker_to_etf[ticker]["jpx33"]
    if jpx33 in JPX33_TO_TOPIX17:
        ticker_to_etf[ticker]["etf"] = JPX33_TO_TOPIX17[jpx33]
    else:
        # Default assignment for unmapped industries
        ticker_to_etf[ticker]["etf"] = "1630.T"  # Default to retail

# Build aggregation matrix
# A[i, j] = weight of subsector j in ETF i
subsector_to_etf_weights = defaultdict(lambda: defaultdict(float))

for sub in subsector_config["subsectors"]:
    subsector_name = sub["name"]
    etf_counts = defaultdict(int)

    for ticker_info in sub["tickers"]:
        ticker = ticker_info["ticker"]
        if ticker in ticker_to_etf:
            etf = ticker_to_etf[ticker]["etf"]
            etf_counts[etf] += 1

    # Normalize to get weights (value-weighted would need market cap, using equal-weight for now)
    total = sum(etf_counts.values())
    if total > 0:
        for etf, count in etf_counts.items():
            subsector_to_etf_weights[subsector_name][etf] = count / total

# Create aggregation matrix
etf_list = sorted(TOPIX17_ETFS.keys())
subsector_list = SUBSECTOR_NAMES

A = np.zeros((len(etf_list), len(subsector_list)))

for i, etf in enumerate(etf_list):
    for j, subsector in enumerate(subsector_list):
        A[i, j] = subsector_to_etf_weights[subsector].get(etf, 0.0)

# Normalize each column to sum to 1 (ensure aggregation works)
col_sums = A.sum(axis=0)
for j in range(A.shape[1]):
    if col_sums[j] > 0:
        A[:, j] = A[:, j] / col_sums[j]

# Save aggregation matrix
agg_config = {
    "meta": {
        "source": "98 subsector configuration",
        "etf_count": len(etf_list),
        "subsector_count": len(subsector_list),
        "description": "Aggregation matrix from 98 subsectors to 17 TOPIX ETFs"
    },
    "etf_list": etf_list,
    "subsector_list": subsector_list,
    "aggregation_matrix": A.tolist()
}

output_file = ROOT / "configs" / "research" / "subsector_98_aggregation.yaml"
with open(output_file, "w", encoding="utf-8") as f:
    yaml.dump(agg_config, f, default_flow_style=False, allow_unicode=True)

print(f"Generated aggregation matrix: {output_file}")
print(f"Shape: {A.shape}")
print(f"Column sums (should be ~1.0): {col_sums}")
print(f"Row sums (ETF total weights): {A.sum(axis=1)}")

# Show sample distribution
print("\nSample subsector ETF distribution:")
for i, subsector in enumerate(subsector_list[:5]):
    weights = A[:, i]
    non_zero = [(etf_list[j], weights[j]) for j in range(len(etf_list)) if weights[j] > 0]
    print(f"{subsector}: {non_zero}")
