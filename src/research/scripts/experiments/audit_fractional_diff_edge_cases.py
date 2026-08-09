#!/usr/bin/env python
"""Audit fractional diff edge cases and integration correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.core.pipeline import build_common_inputs
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.features.fractional_diff import (
    adf_test,
    apply_fractional_diff_to_df_exec,
    compute_weights,
    find_optimal_d,
    fractional_diff,
    hurst_exponent,
)


def test_weight_properties():
    """Verify binomial weights for d=0.1 match known properties."""
    print("=== Weight properties ===")
    for d in [0.0, 0.1, 0.5, 1.0]:
        w = compute_weights(d, threshold=1e-5)
        s = np.sum(w)
        print(f"d={d}: n_weights={len(w)}, sum={s:.6f}")
        if d == 0.0:
            assert len(w) == 1 and w[0] == 1.0
        if d == 1.0:
            assert len(w) == 2 and np.isclose(w, [1.0, -1.0]).all()


def test_warmup_behavior():
    """Check expanding-window warmup values for constant and trend series."""
    print("\n=== Warmup behavior ===")
    n = 200
    const = pd.Series(np.ones(n) * 5.0, name="const")
    trend = pd.Series(np.arange(n) * 0.01, name="trend")

    for d in [0.1, 0.5]:
        r_const = fractional_diff(const, d=d, threshold=1e-5, window=100)
        r_trend = fractional_diff(trend, d=d, threshold=1e-5, window=100)
        print(f"d={d}: const last={r_const.iloc[-1]:.6f}, trend last={r_trend.iloc[-1]:.6f}")
        print(f"  const first 5: {r_const.head(5).values}")
        print(f"  const last 5: {r_const.tail(5).values}")


def test_nan_handling():
    """Check how NaNs in input propagate."""
    print("\n=== NaN handling ===")
    s = pd.Series(np.random.randn(100), name="test")
    s.iloc[50] = np.nan
    result = fractional_diff(s, d=0.5, threshold=1e-5, window=100)
    nan_count = result.isna().sum()
    print(f"Input has 1 NaN at idx 50 -> output has {nan_count} NaNs")
    # NaN propagates for window days due to lookback
    assert nan_count > 0, "NaN should propagate through lookback window"


def test_extreme_values():
    """Check numerical stability with extreme returns."""
    print("\n=== Extreme returns ===")
    s = pd.Series(np.random.randn(500) * 0.01, name="ret")
    # Insert a few very large returns
    s.iloc[100] = 10.0
    s.iloc[200] = -10.0
    result = fractional_diff(s, d=0.1, threshold=1e-5, window=100)
    assert np.isfinite(result).all() or result.isna().sum() < 10
    print(f"Max abs output: {result.abs().max():.4f}")
    print(f"Finite ratio: {result.notna().mean():.4f}")


def test_adf_edge_cases():
    """Check adf_test on edge cases."""
    print("\n=== ADF edge cases ===")
    # Constant series
    const = pd.Series(np.ones(100))
    r = adf_test(const)
    print(f"Constant: {r}")
    # Very short
    short = pd.Series([1.0, 2.0, 3.0])
    r = adf_test(short)
    print(f"Short: {r}")
    # Large values
    large = pd.Series(np.random.randn(500) * 1e6)
    r = adf_test(large)
    print(f"Large scale: {r}")


def test_hurst_edge_cases():
    """Check hurst_exponent edge cases."""
    print("\n=== Hurst edge cases ===")
    # Constant
    const = pd.Series(np.ones(1000))
    h = hurst_exponent(const)
    print(f"Constant: {h}")
    # Short
    short = pd.Series(np.random.randn(10))
    h = hurst_exponent(short)
    print(f"Short: {h}")


def test_find_optimal_d():
    """Check find_optimal_d on random walk."""
    print("\n=== find_optimal_d ===")
    np.random.seed(42)
    s = pd.Series(np.cumsum(np.random.randn(500)))
    result = find_optimal_d(s, d_range=np.arange(0.1, 1.01, 0.1))
    print(f"best_d: {result['best_d']}")
    for r in result["results"][:3]:
        print(f"  d={r['d']:.1f}: adf_p={r['adf_p']:.3f}, hurst={r['hurst']:.3f}, stationary={r['is_stationary']}")


def test_build_common_inputs_integration():
    """Verify build_common_inputs applies fractional diff when enabled."""
    print("\n=== build_common_inputs integration ===")
    n = 3000
    dates = pd.date_range("2010-01-04", periods=n, freq="B")
    np.random.seed(42)
    us_rets = np.random.randn(n, len(US_TICKERS)) * 0.01
    jp_rets = np.random.randn(n, len(JP_TICKERS)) * 0.01

    df_exec = pd.DataFrame(
        {f"us_cc_{tk}": us_rets[:, i] for i, tk in enumerate(US_TICKERS)}
        | {f"jp_oc_{tk}": jp_rets[:, i] for i, tk in enumerate(JP_TICKERS)}
        | {f"jp_cc_{tk}": jp_rets[:, i] for i, tk in enumerate(JP_TICKERS)}
        | {f"jp_gap_{tk}": jp_rets[:, i] for i, tk in enumerate(JP_TICKERS)}
        | {f"jp_beta_{tk}": np.ones(n) * 0.5 for i, tk in enumerate(JP_TICKERS)}
        | {"topix_night_return": np.zeros(n), "topix_oc_return": np.zeros(n)}
        | {"sig_date": dates},
        index=dates,
    )

    inputs_off = build_common_inputs(
        df_exec,
        jp_rets,
        n_u=len(US_TICKERS),
        n_j=len(JP_TICKERS),
        ewma_half_life=45,
        beta_window=60,
        include_v4_prior=True,
        frac_diff_enabled=False,
    )
    inputs_on = build_common_inputs(
        df_exec,
        jp_rets,
        n_u=len(US_TICKERS),
        n_j=len(JP_TICKERS),
        ewma_half_life=45,
        beta_window=60,
        include_v4_prior=True,
        frac_diff_enabled=True,
        frac_diff_d=0.1,
        frac_diff_threshold=1e-5,
        frac_diff_window=100,
    )

    us_off = inputs_off.all_returns_raw[:, :len(US_TICKERS)]
    us_on = inputs_on.all_returns_raw[:, :len(US_TICKERS)]

    diff = np.abs(us_on - us_off)
    print(f"US returns off vs on: max diff={diff.max():.6f}, mean diff={diff.mean():.6f}")
    assert diff.max() > 1e-6, "Fractional diff should change US returns when enabled"


def test_apply_fractional_diff_nan_fill():
    """Check apply_fractional_diff_to_df_exec NaN fill behavior."""
    print("\n=== apply_fractional_diff_to_df_exec NaN fill ===")
    n = 100
    df = pd.DataFrame({"us_cc_XLB": np.cumsum(np.random.randn(n))})
    result = apply_fractional_diff_to_df_exec(df, ["XLB"], d=0.5)
    print(f"No input NaNs -> output NaNs: {result['us_cc_XLB'].isna().sum()}")
    assert not result["us_cc_XLB"].isna().any()

    # With input NaN
    df2 = df.copy()
    df2.iloc[50, 0] = np.nan
    result2 = apply_fractional_diff_to_df_exec(df2, ["XLB"], d=0.5)
    print(f"Input NaN at idx 50 -> output NaNs: {result2['us_cc_XLB'].isna().sum()}")
    print(f"Values around idx 50: {result2['us_cc_XLB'].iloc[48:53].values}")


if __name__ == "__main__":
    test_weight_properties()
    test_warmup_behavior()
    test_nan_handling()
    test_extreme_values()
    test_adf_edge_cases()
    test_hurst_edge_cases()
    test_find_optimal_d()
    test_build_common_inputs_integration()
    test_apply_fractional_diff_nan_fill()
    print("\nAll edge-case checks completed.")
