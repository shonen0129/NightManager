#!/usr/bin/env python3
"""Compute orthogonalization residuals w6' for baseline and user-provided w6.
Saves per-component residuals and prints summary (norms, angle, top diffs).
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

# Ensure src is importable
import sys

sys.path.insert(0, os.path.abspath("src"))

from domain.signals import lead_lag as signals
from data_loader import US_TICKERS, JP_TICKERS


def main():
    n_u = 11
    n_j = 17

    # Build base vectors v1, v2
    base = signals.build_base_vectors(n_u, n_j)
    v1 = base["v1"]
    v2 = base["v2"]

    # v3 (as in implementation)
    us_labels = np.array([1, 0, 1, 1, 0, -1, -1, 1, -1, -1, 0])
    jp_labels = np.array(
        [
            -1,
            1,
            0,
            0,
            -1,
            0,
            0,
            0,
            1,
            0,
            -1,
            0,
            1,
            -1,
            1,
            0,
            0,
        ]
    )
    w3 = np.concatenate([us_labels, jp_labels])
    v3 = w3 / 4.0

    # w4 (copy from lead_lag.py)
    w4 = np.array(
        [
            0.4,
            0.0,
            0.1,
            0.2,
            0.7,
            0.8,
            -0.5,
            -0.4,
            -0.7,
            -0.4,
            0.6,
            -0.6,
            0.2,
            0.2,
            0.5,
            -0.2,
            1.0,
            0.6,
            0.8,
            1.0,
            -0.2,
            -0.8,
            -0.4,
            0.8,
            -0.7,
            0.3,
            0.0,
            -0.9,
        ],
        dtype=float,
    )

    # w5 (copy from lead_lag.py)
    w5 = np.array(
        [
            0.4,
            0.0,
            1.0,
            0.0,
            0.2,
            0.0,
            -0.3,
            0.0,
            -0.8,
            0.0,
            -0.3,
            -0.3,
            1.0,
            -0.1,
            0.3,
            0.0,
            -0.2,
            0.2,
            0.0,
            0.0,
            0.0,
            -0.9,
            -0.1,
            0.7,
            -0.2,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    # baseline w6 (copy from lead_lag.py)
    w6_base = np.array(
        [
            0.0,
            -0.2,
            0.1,
            1.0,
            0.0,
            -0.6,
            -0.2,
            -0.9,
            -0.7,
            -0.2,
            -0.3,
            -0.1,
            0.0,
            -0.2,
            0.0,
            -0.1,
            -0.2,
            0.0,
            0.0,
            -0.3,
            -0.2,
            -0.5,
            -0.1,
            0.1,
            -0.2,
            1.0,
            0.5,
            -0.9,
        ],
        dtype=float,
    )

    # new w6 (from user)
    w6_new = np.array(
        [
            +0.8,
            -0.3,
            +1.0,
            +0.3,
            +0.3,
            -0.5,
            -0.2,
            +0.4,
            -0.7,
            -0.2,
            -0.4,
            -0.4,
            +1.0,
            +0.3,
            +0.7,
            -0.2,
            -0.1,
            +0.6,
            +0.2,
            -0.3,
            -0.3,
            -0.8,
            -0.3,
            +0.8,
            -0.5,
            +0.2,
            +0.1,
            +0.3,
        ],
        dtype=float,
    )

    # compute v4, v5 using the same orthogonalize+normalize routine
    v4 = signals._orthogonalize_and_normalize(w4, [v1, v2, v3])
    v5 = signals._orthogonalize_and_normalize(w5, [v1, v2, v3, v4])

    bases = [v1, v2, v3, v4, v5]

    def residual(w, bases):
        w = np.asarray(w, dtype=float).reshape(-1)
        R = w.copy()
        # use the same formula as in docs: subtract (w^T v_i) * v_i
        proj = np.zeros_like(w)
        for b in bases:
            proj += (w @ b) * b
        return w - proj

    r_base = residual(w6_base, bases)
    r_new = residual(w6_new, bases)

    norm_base = float(np.linalg.norm(r_base))
    norm_new = float(np.linalg.norm(r_new))
    norm_diff = norm_new - norm_base
    norm_rel = (norm_new / norm_base - 1.0) * 100.0 if norm_base > 0 else float("nan")

    # angle / similarity
    if norm_base > 0 and norm_new > 0:
        cos = float(np.dot(r_base, r_new) / (norm_base * norm_new))
        cos = max(-1.0, min(1.0, cos))
        angle_deg = float(np.degrees(np.arccos(cos)))
    else:
        cos = float("nan")
        angle_deg = float("nan")

    # per-component differences
    diff = r_new - r_base
    abs_diff = np.abs(diff)
    order = np.argsort(-abs_diff)

    # Prepare output DataFrame with tickers
    tickers = list(US_TICKERS) + list(JP_TICKERS)
    if len(tickers) != len(r_base):
        tickers = [f"i{i}" for i in range(len(r_base))]

    df = pd.DataFrame(
        {
            "ticker": tickers,
            "resid_base": r_base,
            "resid_new": r_new,
            "diff": diff,
            "abs_diff": abs_diff,
        }
    )
    df = df.set_index("ticker")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", f"compare_w6_residuals_{ts}")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "w6_residuals.csv")
    df.to_csv(csv_path, encoding="utf-8-sig")

    # Print concise summary
    print("=== w6 orthogonalization residuals summary ===")
    print(f"output: {csv_path}")
    print(f"||w6'_baseline|| = {norm_base:.6f}")
    print(
        f"||w6'_new||      = {norm_new:.6f} (Δ = {norm_diff:.6f}, {norm_rel:.2f}% relative) "
    )
    print(f"cosine similarity = {cos:.6f}")
    print(f"angle (deg) = {angle_deg:.3f}")

    print(
        "\nTop 6 components by absolute change in residual (ticker, base, new, diff):"
    )
    for idx in order[:6]:
        t = df.index[idx]
        a = df.iloc[idx]
        print(f"{t}: {a['resid_base']:.6f} -> {a['resid_new']:.6f}  Δ={a['diff']:.6f}")

    print("\nSaved per-component residuals to:", csv_path)


if __name__ == "__main__":
    main()
