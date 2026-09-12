#!/usr/bin/env python3
"""Check sensitivity label assignment for subsector experiment.

Verifies that each subsector inherits its dominant ETF's sensitivity labels
correctly and identifies any potential issues with the mapping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.tickers import SENSITIVITY_LABELS

# Canonical 17 TOPIX-17 ETF tickers
_ORIGINAL_JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T",
]

def main() -> int:
    # Load aggregation matrix
    with open(ROOT / "configs" / "research" / "subsector_aggregation_vw.yaml", "r", encoding="utf-8") as f:
        agg = yaml.safe_load(f)
    A = np.array(agg["aggregation_matrix_A"]["data"])
    ETF_ROWS = agg["aggregation_matrix_A"]["rows"]
    SUBSECTOR_NAMES = agg["aggregation_matrix_A"]["cols"]

    print("=" * 80)
    print("Sensitivity Label Assignment Analysis")
    print("=" * 80)
    print(f"Subsectors: {len(SUBSECTOR_NAMES)}")
    print(f"ETFs: {len(ETF_ROWS)}")
    print(f"Aggregation matrix shape: {A.shape}")
    print()

    # Analyze dominant ETF for each subsector
    dominant_mapping = {}
    for k, sub in enumerate(SUBSECTOR_NAMES):
        etf_idx = int(np.argmax(A[:, k]))
        etf = ETF_ROWS[etf_idx]
        dominant_weight = A[etf_idx, k]
        dominant_mapping[sub] = {
            "dominant_etf": etf,
            "dominant_weight": float(dominant_weight),
            "total_weight": float(A[:, k].sum()),
            "num_nonzero": int(np.sum(A[:, k] > 0)),
        }

    # Check if any subsector has no sensitivity labels assigned
    missing_labels = []
    for sub, info in dominant_mapping.items():
        etf = info["dominant_etf"]
        if etf not in SENSITIVITY_LABELS:
            missing_labels.append((sub, etf))

    if missing_labels:
        print("⚠️  WARNING: Subsectors with missing sensitivity labels:")
        for sub, etf in missing_labels:
            print(f"  - {sub}: dominant ETF {etf} not in SENSITIVITY_LABELS")
        print()
    else:
        print("✓ All dominant ETFs have sensitivity labels defined")
        print()

    # Analyze weight distribution
    print("Weight Distribution Analysis:")
    print("-" * 80)
    dominant_weights = [info["dominant_weight"] for info in dominant_mapping.values()]
    print(f"Mean dominant weight: {np.mean(dominant_weights):.4f}")
    print(f"Median dominant weight: {np.median(dominant_weights):.4f}")
    print(f"Min dominant weight: {np.min(dominant_weights):.4f}")
    print(f"Max dominant weight: {np.max(dominant_weights):.4f}")
    print()

    # Identify subsectors with low dominant weight (potential issues)
    low_dominance_threshold = 0.3
    low_dominance = [
        (sub, info["dominant_etf"], info["dominant_weight"])
        for sub, info in dominant_mapping.items()
        if info["dominant_weight"] < low_dominance_threshold
    ]

    if low_dominance:
        print(f"⚠️  Subsectors with dominant weight < {low_dominance_threshold} (potential issues):")
        for sub, etf, weight in sorted(low_dominance, key=lambda x: x[2]):
            print(f"  - {sub}: {etf} ({weight:.4f})")
        print()
    else:
        print(f"✓ All subsectors have dominant weight >= {low_dominance_threshold}")
        print()

    # Sample a few subsectors to show full assignment
    print("Sample Sensitivity Label Assignments:")
    print("-" * 80)
    for sub in SUBSECTOR_NAMES[:10]:
        info = dominant_mapping[sub]
        etf = info["dominant_etf"]
        if etf in SENSITIVITY_LABELS:
            labels = SENSITIVITY_LABELS[etf]
            print(f"{sub}:")
            print(f"  Dominant ETF: {etf} (weight={info['dominant_weight']:.4f})")
            print(f"  Labels: w3={labels['w3']:.2f}, w4={labels['w4']:.2f}, w5={labels['w5']:.2f}, w6={labels['w6']:.2f}")
        else:
            print(f"{sub}: NO LABELS (dominant ETF {etf} not in SENSITIVITY_LABELS)")
    print()

    # Check if the current assignment makes sense semantically
    print("Semantic Validation (sample):")
    print("-" * 80)
    # Technology-related subsectors should have technology-like labels
    tech_keywords = ["Semiconductor", "Electronic", "Software", "Internet", "IT"]
    tech_subsectors = [s for s in SUBSECTOR_NAMES if any(kw in s for kw in tech_keywords)]

    if tech_subsectors:
        print("Technology-related subsectors:")
        for sub in tech_subsectors[:5]:
            info = dominant_mapping[sub]
            etf = info["dominant_etf"]
            if etf in SENSITIVITY_LABELS:
                labels = SENSITIVITY_LABELS[etf]
                print(f"  {sub}: {etf} (w3={labels['w3']:.2f}, w4={labels['w4']:.2f})")
        print()

    # Save detailed mapping for inspection
    output_file = ROOT / "var" / "research" / "subsector" / "sensitivity_label_mapping.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dominant_mapping, f, ensure_ascii=False, indent=2)
    print(f"Detailed mapping saved to: {output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
