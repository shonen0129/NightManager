# マルチホライズン累積リターン近似の修正と比較検証

## 概要

マルチホライズン信号ブレンド（Phase 2A）で使用される `compute_cumulative_returns` 関数が `rolling(horizon).sum()`（単純足し算）で累積リターンを近似していた問題を修正し、正確な複利累積リターン `(1+r₁)(1+r₂)…(1+rₕ)-1` を計算するよう変更した。両方式で比較バックテストを実施し、パフォーマンスへの影響を検証した。

## 修正内容

### 対象ファイル

- `tools/production/compute_gap_adjusted_distribution.py` — `compute_cumulative_returns` 関数
- CLI引数 `--cumulative-method` を追加（`cumprod`（デフォルト）/ `sum`（旧来））

### 従来の近似

```python
df_mod[col] = df_exec[col].rolling(horizon).sum()
```

### 修正後（厳密な複利累積）

```python
df_mod[col] = df_exec[col].rolling(horizon, min_periods=horizon).apply(
    lambda x: np.prod(1.0 + x) - 1.0, raw=True
)
```

## 比較バックテスト結果

### 条件

- **期間**: 2020-01-06 ～ 2026-07-10（1,532営業日）
- **モデル**: ProductionV2 (Residual-BLPX-RA v2)
- **ホライズン**: h=1, 3, 5（ブレンド重み: 0.8, 0.1, 0.1）
- **コスト**: なし（gross リターン、PIT binning は Medium 固定）

### 結果

| 指標 | sum（旧来） | cumprod（修正） | 差分 |
|------|------------:|----------------:|-----:|
| n_success | 1,476 | 1,476 | 0 |
| n_fallback | 56 | 56 | 0 |
| total_return | 8.2836 | 8.2807 | -0.0030 |
| mean_return | 0.005407 | 0.005405 | -0.000002 |
| std_return | 0.008124 | 0.008121 | -0.000002 |
| **Sharpe** | **10.5661** | **10.5651** | **-0.0009** |
| max_dd | -0.04178 | -0.04173 | +0.00006 |
| avg_turnover | 2.7926 | 2.7917 | -0.0008 |
| avg_gross | 1.9269 | 1.9269 | 0.0000 |
| daily_return_corr | — | — | 0.99989 |
| max_abs_daily_ret_diff | — | — | 0.00289 |

## 分析

### 影響が極小である理由

1. **日次リターンの大きさ**: セクターETFの日次リターンは通常 ±1% 程度。h=3 の場合、`sum` と `cumprod` の差は交叉項 `r₁·r₂ + r₁·r₃ + r₂·r₃` ≈ 3 × 0.01² = 0.0003 程度
2. **ブレンド重み**: h>1 の重みは各 0.1（h=1 は 0.8）。h>1 の信号変動が最終スコアに与える影響は限定的
3. **クロスセクショナル z-score**: `apply_multi_horizon_blend` で各ホライズンのスコアを z-score 化してからブレンドするため、レベル差が吸収される
4. **ギャップ調整**: `mu_gap = (1 + mu_raw) / denom - 1` の変換も差を縮小する方向に作用

### 結論

`sum` → `cumprod` の修正は理論的に正しいが、パフォーマンスへの影響は統計的に無視できるレベル（Sharpe 差: 0.0009、日次リターン相関: 0.9999）。修正を本番に適用してもリスクはなく、数理的な正確性が向上する。

## 推奨

- `cumprod` をデフォルトとして採用（既に実装済み）
- 既存の `sum` 方式は `--cumulative-method sum` で後方互換性を確保
