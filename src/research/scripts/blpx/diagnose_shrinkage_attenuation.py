"""Diagnose two-stage shrinkage attenuation.

With lambda_lw=0.5, lambda_reg=0.75:
  Stage 1: c_lw = 0.5*c_t + 0.5*C_LW
  Stage 2: c_reg = 0.25*c_lw + 0.75*c_0
  => c_reg = 0.125*c_t + 0.125*C_LW + 0.75*c_0

Only 12.5% of raw sample correlation survives.
This script quantifies the problem with real data and tests alternatives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from research.backtest_common import (
    load_cached_df_exec,
)

from leadlag.core.correlation import (
    build_c0_from_v0,
    build_lw_target_correlation,
    compute_correlation,
    regularize_correlation,
    build_v3_static,
    compute_baseline_correlation,
)
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.models.sre import compute_jp_target_returns


def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Frobenius distance (off-diagonal only)."""
    n = a.shape[0]
    mask = ~np.eye(n, dtype=bool)
    diff = (a - b)[mask]
    norm_b = np.linalg.norm(b[mask])
    if norm_b < 1e-12:
        return 0.0
    return float(np.linalg.norm(diff) / norm_b)


def off_diag_mean(a: np.ndarray) -> float:
    n = a.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(a[mask]))


def effective_weights(lambda_lw: float, lambda_reg: float) -> dict[str, float]:
    """Compute effective weights of c_t, C_LW, c_0 in final c_reg."""
    w_ct = (1 - lambda_lw) * (1 - lambda_reg)
    w_lw = lambda_lw * (1 - lambda_reg)
    w_c0 = lambda_reg
    return {"c_t": w_ct, "C_LW": w_lw, "c_0": w_c0}


def main():
    print("=" * 80)
    print("Two-Stage Shrinkage Attenuation Diagnosis")
    print("=" * 80)

    # --- Effective weight analysis ---
    lambda_lw_default = 0.5
    lambda_reg_default = 0.75

    w = effective_weights(lambda_lw_default, lambda_reg_default)
    print(f"\n[Effective Weights] lambda_lw={lambda_lw_default}, lambda_reg={lambda_reg_default}")
    print(f"  c_t  (raw sample) : {w['c_t']:.4f}  ({w['c_t']*100:.1f}%)")
    print(f"  C_LW (LW target)  : {w['C_LW']:.4f}  ({w['C_LW']*100:.1f}%)")
    print(f"  c_0  (prior)      : {w['c_0']:.4f}  ({w['c_0']*100:.1f}%)")
    print(f"  Sum               : {sum(w.values()):.4f}")

    # --- Load real data ---
    print("\n[Loading data...]")
    df_exec = load_cached_df_exec()
    n_u = len(US_TICKERS)
    n_j = len(JP_TICKERS)
    n_total = n_u + n_j

    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    us_returns_raw = df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].values
    all_returns = np.column_stack([us_returns_raw, y_jp_target])
    date_index = df_exec.index.values

    print(f"  Data shape: {all_returns.shape}")
    print(f"  US assets: {n_u}, JP assets: {n_j}")
    print(f"  NaN count in all_returns: {np.isnan(all_returns).sum()}")
    print(f"  Date range: {df_exec.index.min()} to {df_exec.index.max()}")

    # Replace NaN with 0 for correlation computation
    all_returns = np.nan_to_num(all_returns, nan=0.0)

    # --- Build prior structures ---
    v0_static = build_v3_static(n_u, n_j, include_v4=True)
    c_full = compute_baseline_correlation(all_returns, date_index, ewma_half_life=45)
    print(f"  c_full NaN count: {np.isnan(c_full).sum()}")
    if np.isnan(c_full).any():
        # Try without baseline period restriction
        _, _, c_full = compute_correlation(all_returns[:252], ewma_half_life=45)
        print(f"  c_full (first 252 rows) NaN count: {np.isnan(c_full).sum()}")
    c0_t = build_c0_from_v0(v0_static, c_full)
    print(f"  c0_t NaN count: {np.isnan(c0_t).sum()}")

    # --- Sample multiple time steps ---
    corr_window = 60
    ewma_half_life = 45
    sample_indices = list(range(corr_window + 100, len(all_returns), max(1, len(all_returns) // 20)))
    sample_indices = sample_indices[:20]

    print(f"\n[Sampling {len(sample_indices)} time steps]")

    results = []
    for idx in sample_indices:
        window = all_returns[idx - corr_window: idx]
        mu_w, sigma_w, c_t = compute_correlation(window, ewma_half_life)

        # Stage 1: LW shrinkage
        lw_mat = build_lw_target_correlation(c_t, "equicorrelation")
        c_lw = (1 - lambda_lw_default) * c_t + lambda_lw_default * lw_mat

        # Stage 2: Regularization
        c_reg = regularize_correlation(c_t, c0_t, lambda_reg_default, lambda_lw_default, "equicorrelation")

        # Distances
        d_ct_c0 = frobenius_distance(c_t, c0_t)
        d_reg_c0 = frobenius_distance(c_reg, c0_t)
        d_reg_ct = frobenius_distance(c_reg, c_t)
        d_lw_ct = frobenius_distance(c_lw, c_t)
        d_lw_c0 = frobenius_distance(c_lw, c0_t)

        results.append({
            "idx": idx,
            "d_ct_c0": d_ct_c0,
            "d_reg_c0": d_reg_c0,
            "d_reg_ct": d_reg_ct,
            "d_lw_ct": d_lw_ct,
            "d_lw_c0": d_lw_c0,
            "off_diag_ct": off_diag_mean(c_t),
            "off_diag_c0": off_diag_mean(c0_t),
            "off_diag_reg": off_diag_mean(c_reg),
            "off_diag_lw": off_diag_mean(c_lw),
        })

    # --- Summary statistics ---
    print("\n[Distance Summary (normalized Frobenius, off-diagonal)]")
    print(f"{'Metric':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("-" * 65)

    for key, label in [
        ("d_ct_c0", "d(c_t, c_0)"),
        ("d_reg_c0", "d(c_reg, c_0)"),
        ("d_reg_ct", "d(c_reg, c_t)"),
        ("d_lw_ct", "d(c_lw, c_t)"),
        ("d_lw_c0", "d(c_lw, c_0)"),
    ]:
        vals = np.array([r[key] for r in results])
        print(f"{label:<25} {vals.mean():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f} {vals.std():>8.4f}")

    # Ratio: how much closer is c_reg to c_0 vs c_t to c_0?
    ratios = np.array([r["d_reg_c0"] / max(r["d_ct_c0"], 1e-12) for r in results])
    print(f"\n[Attenuation Ratio] d(c_reg, c_0) / d(c_t, c_0)")
    print(f"  Mean: {ratios.mean():.4f}  (1.0=no attenuation, 0.0=fully collapsed to prior)")
    print(f"  If ratio ~0.125, confirms 12.5% raw info survival")

    # Off-diagonal means
    print(f"\n[Off-diagonal Mean Correlation]")
    print(f"{'Source':<20} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for key, label in [
        ("off_diag_ct", "c_t (raw)"),
        ("off_diag_lw", "c_lw (stage1)"),
        ("off_diag_reg", "c_reg (final)"),
        ("off_diag_c0", "c_0 (prior)"),
    ]:
        vals = np.array([r[key] for r in results])
        print(f"{label:<20} {vals.mean():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

    # --- Alternative parameter combinations ---
    print(f"\n[Alternative Parameter Combinations]")
    print(f"{'lambda_lw':>10} {'lambda_reg':>11} {'w_c_t':>7} {'w_C_LW':>7} {'w_c_0':>7} {'d_reg/c_0':>10} {'d_reg/c_t':>10}")
    print("-" * 75)

    alternatives = [
        (0.5, 0.75),   # current
        (0.5, 0.50),   # reduce reg
        (0.3, 0.50),   # reduce both
        (0.3, 0.25),   # minimal
        (0.0, 0.75),   # no LW, only reg
        (0.5, 0.00),   # no reg, only LW
        (0.0, 0.00),   # no shrinkage
        (0.25, 0.50),  # moderate
        (0.15, 0.30),  # light
    ]

    for lam_lw, lam_reg in alternatives:
        w = effective_weights(lam_lw, lam_reg)
        d_reg_c0_list = []
        d_reg_ct_list = []
        for idx in sample_indices:
            window = all_returns[idx - corr_window: idx]
            _, _, c_t = compute_correlation(window, ewma_half_life)
            c_reg = regularize_correlation(c_t, c0_t, lam_reg, lam_lw, "equicorrelation")
            d_reg_c0_list.append(frobenius_distance(c_reg, c0_t))
            d_reg_ct_list.append(frobenius_distance(c_reg, c_t))

        d_c0 = np.mean(d_reg_c0_list)
        d_ct = np.mean(d_reg_ct_list)
        print(f"{lam_lw:>10.2f} {lam_reg:>11.2f} {w['c_t']:>7.3f} {w['C_LW']:>7.3f} {w['c_0']:>7.3f} {d_c0:>10.4f} {d_ct:>10.4f}")

    # --- Eigenvalue spectrum comparison ---
    print(f"\n[Eigenvalue Spectrum (top-6, averaged over samples)]")
    print(f"{'Source':<20} {'eig1':>8} {'eig2':>8} {'eig3':>8} {'eig4':>8} {'eig5':>8} {'eig6':>8}")
    print("-" * 70)

    for label, mat_fn in [
        ("c_t (raw)", lambda c_t: c_t),
        ("c_lw (stage1)", lambda c_t: (1 - lambda_lw_default) * c_t + lambda_lw_default * build_lw_target_correlation(c_t, "equicorrelation")),
        ("c_reg (final)", lambda c_t: regularize_correlation(c_t, c0_t, lambda_reg_default, lambda_lw_default, "equicorrelation")),
        ("c_0 (prior)", lambda c_t: c0_t),
    ]:
        eigvals_top = []
        for idx in sample_indices:
            window = all_returns[idx - corr_window: idx]
            _, _, c_t = compute_correlation(window, ewma_half_life)
            mat = mat_fn(c_t)
            eigvals = np.linalg.eigvalsh(mat)
            eigvals_top.append(np.sort(eigvals)[::-1][:6])
        avg = np.mean(eigvals_top, axis=0)
        print(f"{label:<20}", " ".join(f"{v:>8.3f}" for v in avg))

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
