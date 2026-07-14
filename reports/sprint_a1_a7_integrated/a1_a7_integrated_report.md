# A1-A7 実験結果統合レポート

## 実行日: 2026-07-13
## データ: 4149行 (2010-01 ~ 2026-07), バックテスト期間: 2015-01-05 ~ 2026-07-13

---

## サマリー

| 実験 | 内容 | 結果 | 判定 |
|------|------|------|------|
| **A1** | 適応的ギャップ係数 c_t | Sharpe -0.2% (best λ=0.25: 8.703 vs 8.719) | **不採用** |
| **A2** | BLP Empirical Bayes λ推定 | Sharpe +0.5% (best combo: 8.765 vs 8.719) | **不採用** (感度範囲内) |
| **A3** | PITビニング isotonic診断 | Low tertile Sharpe 6.63 < Mid 8.29 < High 9.60 | **採用維持** (現行RuleD妥当) |
| **A4** | アンサンブル重み IC最適化 | Sharpe +0.8% vs equal-weight (8.685 vs 8.615) | **不採用** (production単体が8.72で上位) |
| **A5** | BayesianBLPX Kalman修正 | ic -2.6%, kalman -2.5%, cs_var -23.3% vs baseline | **不採用** (R3リーク修正は正しい) |
| **A6** | マクロ感度行列 | スキップ（外部データ依存） | — |
| **A7** | ウォークフォワード + DSR | DSR=1.0000≥0.95, 9/9期間正, 感度0.04% | **PASS** (戦略は統計的に有意) |

**結論: 現行のProductionV2モデル（Residual-BLPX-RA v2）は理論的改善の余地がほぼない水準で最適化されている。全ての理論的改善（A1-A5）はSharpe比を有意に改善しなかった。ウォークフォワード検証（A7）は戦略の統計的有意性を確認。**

---

## A1: 適応的ギャップ係数 c_t

### 設計
- β_rev_oc（ローリング252日）から理論値 c_t = 1 + β_rev_oc を計算
- 固定値 0.70 と理論値を λ でブレンド: c_t = (1-λ)·0.70 + λ·(1+β_rev_oc)
- β_rev_oc mean = -0.2319 (理論予測 ~-0.23 と一致)
- 理論値 1+β_rev_oc ≈ 0.77 vs 現行 0.70 (差0.07)

### 結果
| λ | c_t mean | Sharpe | MDD |
|---|----------|--------|-----|
| baseline (fixed 0.70) | 0.70 | 8.719 | -6.31% |
| 0.25 | 0.716 | 8.703 | -6.33% |
| 0.50 | 0.732 | 8.704 | -6.35% |
| 0.75 | 0.748 | 8.685 | -6.37% |

### 判定: **不採用**
理論値0.77と現行0.70の差0.07は推定ノイズ範囲内。適応的調整による恩恵なし。

### レポート: `reports/sprint_a1_adaptive_gap/a1_adaptive_gap_report.md`

---

## A2: BLP Empirical Bayes λ推定

### 設計
- Tikhonov正則化パラメータ λ_pca, λ_sector, ρ を3点log-scaleグリッドで摂動
- Stage 1: 1パラメータずつ摂動（6 backtests + baseline）
- Stage 2: Stage 1の最良組合せ（1 backtest）
- 感度分析: best comboの±20%摂動（27 backtests）

### Stage 1結果
| パラメータ | 値 | Sharpe | vs baseline |
|-----------|-----|--------|-------------|
| baseline | λ_pca=0.10, λ_sec=0.60, ρ=0.01 | 8.719 | — |
| λ_pca=0.05 | 0.05, 0.60, 0.01 | 8.709 | -0.1% |
| λ_pca=0.20 | 0.20, 0.60, 0.01 | 8.768 | +0.6% |
| λ_sector=0.30 | 0.10, 0.30, 0.01 | 8.710 | -0.1% |
| λ_sector=1.20 | 0.10, 1.20, 0.01 | 8.693 | -0.3% |
| ρ=0.005 | 0.10, 0.60, 0.005 | 8.722 | +0.03% |
| ρ=0.020 | 0.10, 0.60, 0.020 | 8.716 | -0.03% |

### Stage 2結果
| パラメータ | 値 | Sharpe | vs baseline |
|-----------|-----|--------|-------------|
| best combo | λ_pca=0.20, λ_sec=0.60, ρ=0.005 | 8.765 | +0.5% |

### 感度分析（一部）
| variant | Sharpe |
|---------|--------|
| sens_lp0.160_ls0.480_rho0.0040 | 8.758 |
| sens_lp0.160_ls0.480_rho0.0050 | 8.754 |

### 判定: **不採用**
最良組合せで+0.5%のみ。±20%摂動でもSharpeは8.75-8.77に収束し、現行との差は感度範囲内。過学習リスクを考慮し現行λを維持。

### レポート: `reports/sprint_a2_empirical_bayes/a2_empirical_bayes_report.md`

---

## A3: PITビニング isotonic診断

### 設計
- バックテスト日次リターンからローリング252日Sharpe（IR proxy）を計算（shift(1)でリーク防止）
- PIT percentileを計算（strictly historical window）
- 翌日リターンとのisotonic regression + decile/tertile分析

### 結果
- **Spearman corr(pit_pct, next_return) = 0.1607 (p<0.0001)** — 有意な正の相関

### Tertile Sharpe貢献
| bin | n | mean_ret | sharpe_contrib |
|-----|---|----------|----------------|
| Low | 833 | 0.00343 | 6.63 |
| Mid | 817 | 0.00530 | 8.29 |
| High | 831 | 0.00716 | 9.60 |

### 判定: **採用維持**
Low tertileのSharpe貢献がMid/Highより明確に低い → RuleDの0.75x乗数は妥当。現行3分位方式を維持。

### レポート: `reports/sprint_a3_pit_isotonic/pit_isotonic_report.md`

---

## A4: アンサンブル重み IC最適化

### 設計
- 4成分シグナル（raw_pca, residual_pca, raw_blpx, residual_blpx）の日次Spearman ICを計算
- ローリング504日IC平均・共分散からIC最適重みを計算（Ledoit-Wolf shrinkage blend）

### Mean IC per Component
| component | mean IC |
|-----------|---------|
| raw_pca | 0.2178 |
| residual_pca | 0.2176 |
| raw_blpx | 0.2270 |
| residual_blpx | 0.2275 |

### 結果
| variant | Sharpe | MDD |
|---------|--------|-----|
| baseline (equal weight 4 comp) | 8.615 | -6.95% |
| IC-optimal δ=0.25 | 8.643 | -6.93% |
| IC-optimal δ=0.50 | 8.685 | -6.82% |
| **production (residual_blpx only)** | **8.719** | **-6.31%** |

### 判定: **不採用**
IC最適化重みは等重に対し+0.8%のみ。production（residual_blpx単体）が最高。メタ学習アンサンブルは非推奨候補。

### レポート: `reports/sprint_a4_ensemble_ic/a4_ensemble_ic_report.md`

---

## A5: BayesianBLPX Kalman修正

### 修正内容
1. **R3リーク修正**: `y_jp_target[i]` → `y_jp_target[i-1]`（当日ターゲット使用を1日ラグ化）
2. **Q推定修正**: 非重複サブサンプリング（blp_window間隔）でautocorrelation bias除去
3. **R推定修正**: 理論式 `(Var(ΔB̂) - Q) * blp_window / 2` でbias-corrected推定

### 結果
| variant | Sharpe | MDD | vs baseline |
|---------|--------|-----|-------------|
| baseline (BLPEnhanced) | 8.719 | -6.31% | — |
| Bayesian[ic] post-fix | 8.496 | -7.30% | -2.6% |
| Bayesian[kalman] post-fix | 8.498 | -7.30% | -2.5% |
| Bayesian[cs_var] post-fix | 6.688 | -11.58% | -23.3% |

### 判定: **不採用**（BayesianBLPX自体）
- 全Bayesianモードがbaseline BLPEnhancedに劣る
- R3リーク修正は正しい（コード品質改善として保持）
- BayesianBLPXの非推奨化を推奨（コード複雑性削減）
- 注: 本番configはBLPEnhancedを使用するため、本番への性能影響なし

### レポート: `reports/sprint_a5_bayesian_fix/a5_bayesian_fix_report.md`
### コード変更: `src/leadlag/models/bayesian_blpx.py` (R3修正 + Q/R修正)

---

## A7: ウォークフォワード検証 + Deflated Sharpe Ratio

### 設計
- 2018-2026の年次ロール（9区間）でOOS backtest
- purge = 61日, embargo = 5日
- Deflated Sharpe Ratio (Bailey & López de Prado 2014) with N_trials=55
- ±20%パラメータ摂動の感度分析

### Per-Period Sharpe
| 年 | Sharpe | MDD | n_days |
|----|--------|-----|--------|
| 2018 | 6.15 | -6.31% | 251 |
| 2019 | 7.39 | -3.00% | 234 |
| 2020 | 9.40 | -3.20% | 235 |
| 2021 | 6.69 | -5.53% | 238 |
| 2022 | 8.64 | -2.96% | 236 |
| 2023 | 4.70 | -5.56% | 239 |
| 2024 | 7.26 | -5.66% | 237 |
| 2025 | 7.68 | -3.31% | 232 |
| 2026 | 8.77 | -5.30% | 122 |

### Pooled Statistics
- **Sharpe (net): 7.29**
- Annual Return: 113.85%
- Max DD: -6.31%
- Skewness: 1.01, Excess Kurtosis: 5.11

### Deflated Sharpe Ratio
- **DSR = 1.0000 (threshold: 0.95) → PASS**
- N_trials = 55, Trials Sharpe Std = 0.5

### Per-Period Stability
- Mean: 7.41, Std: 1.45
- Range: [4.70, 9.40]
- **Positive periods: 9/9** (0 negative)

### Sensitivity (±20% all params)
| variant | Sharpe |
|---------|--------|
| -20% | 8.715 |
| base | 8.719 |
| +20% | 8.718 |

**感度範囲: 0.04%** — 極めて安定

### 判定: **PASS (ROBUST)**
- DSR=1.0 ≥ 0.95: 55試行補正後も統計的に有意
- 全9期間正: レジーム非依存
- 感度0.04%: パラメータ堅牢性極めて高い

### レポート: `reports/sprint_a7_walkforward/a7_walkforward_dsr_report.md`

---

## テスト結果

### 並列実行 (7プロセス, 約8.5分)
```
tests/unit/test_sprint0_diagnostics.py: 1 passed (510s)
tests/unit/test_sprint0_qa.py: 1 passed (512s)
tests/unit/test_sprint1.py::test_backtest_simulation: 1 passed (515s)
tests/unit/test_sprint1.py::test_calibration_rolling: 1 passed (513s)
tests/unit/test_sprint1.py (rest): 2 passed (13s)
tests/integration/: 106 passed (45s)
tests/unit/ (rest): 291 passed (21s)
---
Total: 403 passed, 0 failed
```

**`bayesian_blpx.py`の修正による回帰なし。全テスト合格。**

---

## 総合判定

### 採用・維持
- **A3**: 現行RuleD（PIT 3分位、Low=0.75x）は統計的に妥当 → 変更不要
- **A5 (R3修正のみ)**: リーク修正は正しい → コード品質として保持
- **A7**: 戦略は統計的に有意（DSR=1.0） → 本番継続

### 不採用
- **A1**: 適応的c_t → 理論値と現行値の差がノイズ範囲内
- **A2**: λ最適化 → +0.5%は感度範囲内、過学習リスク
- **A4**: IC最適化アンサンブル → production単体が最高
- **A5 (BayesianBLPX)**: 全モードがbaselineに劣る → 非推奨候補

### 今後の方向性
1. **現行モデル維持**: ProductionV2Model（Residual-BLPX-RA v2）は理論的改善の余地がほぼない水準で最適化済み
2. **BayesianBLPX非推奨化**: コード複雑性削減のため`bayesian_blpx.py`をarchiveへ移行検討
3. **メタ学習アンサンブル検証**: A4の結果から、4成分アンサンブルよりresidual_blpx単体が優位 → 設計を見直すか削除検討
4. **外部データ依存改善**: A6（マクロ感度行列）は外部データ取得後に再検討
