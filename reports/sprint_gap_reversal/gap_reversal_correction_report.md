# Gap-Reversal Correction Experiment Report

## 概要

ギャップ観測後の条件付き期待値・分散の修正がバックテスト性能に与える影響を検証した。

## 背景と仮説

現行のgap調整ロジックは、米国データ条件付き期待値 `E[r_cc | US_data]` を用い、delta-approximationで `mu_gap = (1 + mu_raw) / (1 + gap_filt) - 1` に変換する。しかし、ギャップ観測後の真の条件付き期待値は `E[r_cc | US_data, gap]` であり、ギャップとr_ccの相関がゼロでない場合、バイアスが生じる。

特に日本市場で知られる**ギャップリバーサル現象**（朝方の大きなギャップが当日の残りリターンと負に相関）は、idiosyncraticギャップ `gap_idio` とintradayリターン `r_oc` の負の相関として現れる。

### 修正案

1. **平均補正**: `mu_gap_corrected = mu_gap + beta_rev_oc * gap_idio`
   - `beta_rev_oc = Cov(r_oc, gap_idio) / Var(gap_idio)` （gap-reversal係数、負値期待）
2. **分散補正**: `Sigma_YY_corrected` の対角成分を `Var(r_cc | gap) / Var(r_cc)` 倍に縮小
   - `reduction_std = beta_rev_cc^2 * Var(gap_idio) / Var(r_cc)` （standardized空間）

### 重要な知見: `beta_rev` の計算には `r_oc` を使用すべき

| 係数 | 定義 | 平均値 | 意味 |
|------|------|--------|------|
| `beta_rev_oc` | `Cov(r_oc, gap_idio) / Var(gap_idio)` | **-0.23** | ギャップリバーサル（負値期待通り） |
| `beta_rev_cc` | `Cov(r_cc, gap_idio) / Var(gap_idio)` | **+0.73** | 恒等式 `r_cc ≈ gap + r_oc` による機械的正相関 |

`r_cc = (1+gap)(1+r_oc) - 1 ≈ gap + r_oc` なので、`beta_rev_cc ≈ 1 + beta_rev_oc` となる。`r_cc`で計算すると機械的正相関が支配的になり、ギャップリバーサルの真の効果（負の相関）が見えなくなる。初回実行でSharpe 4.60→0.39の大幅劣化を引き起こした原因はこれであった。

## 実験設計

- **期間**: 2015-01-05 〜 2026-07-10（2733営業日）
- **モデル**: `SectorRelativeEnsembleBLPEnhancedModel`（現行）vs `GapReversalCorrectedModel`（3バリアント）
- **3バリアント**:
  - **mean-only**: 平均補正のみ
  - **variance-only**: 分散補正のみ
  - **both**: 両方適用
- **推定窓**: 252営業日（strictly historical、ルックアヘッドなし）
- **コスト**: 片道5bps + 金利・貸株・逆日歩含むnet評価

## 結果

### Key Metrics

| Metric | Baseline | Mean-only | Variance-only | Both |
|--------|----------|-----------|---------------|------|
| **Sharpe (net)** | **4.604** | 4.530 | 4.547 | 4.493 |
| Annual Return | 147.15% | 152.30% | 147.30% | 152.98% |
| Annual Risk | 31.96% | 33.62% | 32.40% | 34.05% |
| Max Drawdown | -6.89% | -6.93% | -7.14% | -6.77% |
| Avg Turnover | 1.639 | 1.636 | 1.653 | 1.652 |
| Avg Gross | 2.000 | 2.000 | 2.000 | 2.000 |

### Baseline vs 修正版の相関

| Variant | Signal Corr | Weight Corr | Return Corr |
|---------|-------------|-------------|-------------|
| mean-only | 0.9722 | 0.9606 | 0.9788 |
| variance-only | 1.0000 | 0.9895 | 0.9951 |
| both | 0.9722 | 0.9614 | 0.9794 |

### Per-Ticker `beta_rev_oc` 統計

| Ticker | mean beta_rev_oc | mean beta_rev_cc |
|--------|------------------|------------------|
| 1617.T | -0.257 | 0.677 |
| 1618.T | -0.196 | 0.760 |
| 1619.T | -0.353 | 0.581 |
| 1620.T | -0.206 | 0.735 |
| 1621.T | -0.262 | 0.671 |
| 1622.T | -0.286 | 0.672 |
| 1623.T | -0.152 | 0.800 |
| 1624.T | -0.181 | 0.770 |
| 1625.T | -0.247 | 0.711 |
| 1626.T | -0.251 | 0.656 |
| 1627.T | -0.303 | 0.633 |
| 1628.T | -0.117 | 0.874 |
| 1629.T | -0.158 | 0.795 |
| 1630.T | -0.181 | 0.789 |
| 1631.T | -0.223 | 0.760 |
| 1632.T | -0.285 | 0.734 |
| 1633.T | -0.244 | 0.738 |

全ティッカーで `beta_rev_oc < 0`（ギャップリバーサル確認）。

## 分析

### 全バリアントでSharpe劣化

- **Mean-only**: Sharpe -1.6%（4.604→4.530）。リターン増加（+5.15pp）だがリスク増加（+1.66pp）が上回る
- **Variance-only**: Sharpe -1.2%（4.604→4.547）。信号は不変、weight corr 0.9895でminvar最適化への影響は軽微
- **Both**: Sharpe -2.4%（4.604→4.493）。平均・分散の補正が複合して最も劣化

### なぜ現行モデルが勝るか

現行のdelta-approximation `mu_gap = (1 + mu_raw) / (1 + gap_filt) - 1` は、1次近似で `mu_gap ≈ mu_raw - gap_filt` となり、`gap_filt = c * gap_idio + (c-b) * gap_syst`（c=0.70, b=0.6）で既にgap_idioを0.70倍して引いている。

一方、修正版はdelta-approximation後にさらに `beta_rev_oc * gap_idio`（≈-0.23 * gap_idio）を加算する。これにより実質的なgap_idio調整が `0.70 + 0.23 = 0.93` 倍になり、現行の0.70よりも過剰調整となる。

`gap_open_coef=0.70` は過去のチューニングで最適化された値であり、ギャップリバーサル効果を暗黙的に織り込み済みと考えられる。明示的な `beta_rev_oc` 追加は二重カウントになる。

### 分散補正の効果が限定的

`Sigma_YY_reg` は相関行列空間（対角≈1）であり、`reduction_std = beta_rev_cc^2 * var_gap_idio / var_r_cc` で計算すると、平均で `0.73^2 * 0.0001 / 0.0003 ≈ 0.18`（18%削減）となる。minvar_alpha=0.8のブレンドでは、この分散変化がweightに与える影響は限定的（weight corr 0.9895）。

## 結論

**不採用**。ギャップリバーサル補正は全バリアントでSharpeを劣化させた。

- `beta_rev_oc` は全ティッカーで負値であり、日本市場のギャップリバーサル現象を定量的に確認できた
- しかし、現行の `gap_open_coef=0.70` を用いたdelta-approximationが既にこの効果を暗黙的に取り込んでおり、明示的な補正は過剰調整となる
- 分散補正はminvar最適化への影響が軽微で、Sharpe改善に寄与しない

### 今後の方向性

- `gap_open_coef` の再チューニング（例: 0.70→0.93）でギャップリバーサル効果をより明示的に取り込む可能性があるが、過学習リスクが高い
- ウォークフォワード検証で `gap_open_coef` の感度分析を実施することが推奨される

## 実験コード・データ

- スクリプト: `scripts/experiments/gap_reversal_correction.py`
- 結果: `results/gap_reversal_experiment/`
  - `metrics_comparison.csv`: 4モデルの指標比較
  - `{variant}_daily_returns.csv`: 日次リターン
  - `{variant}_equity_curve.csv`: 損益曲線
  - `beta_rev_oc_array.npy`, `beta_rev_cc_array.npy`: ティッカー別係数
