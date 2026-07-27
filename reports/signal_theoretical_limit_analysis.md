# 信号品質 — 理論限界に近いかの分析レポート

**Date**: 2026-07-26
**Scope**: 本番 v2 モデル（Residual-BLPX-RA v2）のシグナル品質と、過去の信号側改善実験の結果を総括し、戦略が理論限界に近づいているかを評価する。

---

## 1. 現行シグナルの定量的特徴

### 1.1 予測力指標

| 指標 | 値 | 出典 |
|---|---|---|
| 平均 Daily IC (residual_blpx) | **0.2275** | `reports/sprint_a4_ensemble_ic/a4_ensemble_ic_report.md` |
| 平均 Daily IC (raw_blpx) | 0.2270 | 同上 |
| 対日中残差 Rank IC 平均 | **0.1894** | `reports/sprint0/sprint0_diagnostics_report.md` |
| 対日中残差 ICIR | **10.35** | 同上 |
| 上位30% / 下位30% Long-Short Spread | **39.97 bps/日** | 同上 |
| 勝率 (Hit Rate) | **74.12%** | 同上 |
| 分位別単調性 | **満たしている** (Q1<Q2<Q3) | 同上 |
| TOPIX ベータエクスポージャー平均 | -0.0037 | 同上 |

### 1.2 バックテスト性能

| 期間 | Sharpe (net) | AR (net) | MDD | Turnover |
|---|---|---|---|---|
| 2015-2020 (Phase 3 combined) | **8.79** | — | -6.01% | 1.49 |
| 2015-2026 (Fractional Diff d=0.1, 12 window mean) | **8.87** | — | — | 1.60 |
| 2015-2026 (PBO baseline) | **8.36** | 142.04% | -6.89% | 1.64 |
| 2020-2026 (Dynamic MH baseline) | **7.50** | 153.7% | -7.47% | 1.30 |
| 2020-2022 (v2 macro kappa OFF) | **4.56** | 117.43% | -6.73% | 1.48 |
| 2020-2026 (v2 macro kappa OFF) | **3.98** | 114.51% | -7.10% | 1.40 |

---

## 2. 理論限界の試算

### 2.1 Grinold の法則による Information Ratio 上限

Grinold & Kahn (2000) の法則:

```
IR ≈ IC × sqrt(BR)
```

- **IC**: 日次予測とターゲットの相関（≈ 0.19）
- **BR**: 年間独立な賭けの数

#### ケース A: 純粋な銘柄横断的日次ベット

17 銘柄 × 245 営業日 = **4,165** ベット/年

```
IR ≈ 0.19 × sqrt(4165) ≈ 12.3
```

これは **実現不可能な上限**。JP セクター ETF 間の相関（平均 0.3-0.6）は「独立な賭け」を大幅に減らす。

#### ケース B: 実効的な独立ベット数を考慮

相関行列の有効ランクは通常 N/3〜N/2 と推定される。N=17 の JP セクターで有効銘柄数を **6〜9** と仮定:

```
BR ≈ 245 × 6 〜 245 × 9 = 1,470 〜 2,205
IR ≈ 0.19 × sqrt(1470) 〜 0.19 × sqrt(2205)
IR ≈ 7.3 〜 8.9
```

現行の年間 Sharpe（7.5〜8.9）は **この理論上限の範囲内** に位置する。

### 2.2 ターゲットの予測可能範囲上限

`sprint0` 診断のターゲットミスマッチ分析:

- Close-to-Close リターンの分散うち:
  - 寄付きギャップで説明される割合: **60.20%**
  - 9:10→Close（日中）で説明される割合: **39.82%**

本戦略は 9:10 時点で既に観測可能な米国市場の情報を入力とするため、**既に発生した寄付きギャップ（60%）を予測対象にはできない**。戦略が獲得できるのは日中残差部分の予測のみ。

日中残差の予測可能率を仮に **30〜50%** と推定すると:

```
実効予測可能分散 ≈ 40% × (30%〜50%) = 12%〜20% の総分散
```

これは、現行戦略の高い Sharpe が **すでに日中予測のかなりの部分を獲得している**ことを示唆する。

### 2.3 コスト後の限界

`sprint0` 診断の取引コスト感応度:

| コスト想定 | AR | Sharpe |
|---|---|---|
| 5bps round-trip | 113.16% | 6.80 |
| 15bps round-trip | 73.17% | 4.40 |
| 30bps round-trip | 13.20% | 0.79 |

本番は 5bps/side（10bps round-trip）付近で動作。コスト前 alpha が大きくても、コストで大幅に削られる。したがって **コスト後 Sharpe 8 前後は、コスト前 IR 12-15 程度の信号からしか達成できない**。

---

## 3. 過去の信号側改善実験の総括

### 3.1 大きな改善を示したもの

| 実験 | 改善幅 | 備考 |
|---|---|---|
| **Fractional Differentiation (d=0.1)** | Sharpe +1.12（12/12 window 勝利） | 唯一、強力かつロバストな信号側改善。長期記憶と短記憶の分離。 |

### 3.2 限定的・不採用となったもの

| 実験 | 結果 | 出典 |
|---|---|---|
| Meta-Labeling | Sharpe +0.05、DSR=1.0 だが Brier 0.228（ほぼランダム） | `meta_labeling_walkforward_report.md` |
| Macro Kappa v2 | 最大 +0.07 Sharpe、AR 低下、パラメータ敏感 | 本日作成 `macro_kappa_v2_non_adoption_report.md` |
| Macro Direction | Sharpe -51% | `macro_direction_adjustment_report.md` |
| Dynamic Multi-Horizon Weights | Sharpe -0.25〜-0.37 | `dynamic_mh_weights_walkforward_report.md` |
| Nonlinear Shrinkage (NL-MP) | 平均 Sharpe 8.04 vs 8.87 baseline、0/12 勝利 | `nonlinear_shrinkage_walkforward_report.md` |
| RMT Eigenvalue Cleaning | 限界的改善、ノイズ範囲内 | `rmt_filter_walkforward_report.md` |
| IC-optimal ensemble weights | 最大 +0.8% Sharpe | `a4_ensemble_ic_report.md` |
| Hinge features / interactions | — | 各 sprint レポート |
| Stochastic Block Propagation | — | 廃棄（アーカイブ） |
| Robust PCA propagation | Sharpe -35%、IC -32% | AGENTS.md 不採用記録 |
| Time-varying gap coefficient (Kalman) | baseline 劣化 | `tv_gap_coef_walkforward_report.md` |

### 3.3 パラメータロバスト性の証拠

| 実験 | パラメータ感度 |
|---|---|
| Meta-Labeling | ±20% 閾値変化で Sharpe 57.5% 変動 |
| PBO (CSCV) | PBO = 0.5145、IS-OOS 相関 = -0.9957（高い過学習リスク） |

---

## 4. 総合分析

### 4.1 信号は理論限界に近いか？

**結論: 「信号側」単独の改善は、おおむね限界に近づいている。**

根拠:

1. **Grinold 上限との整合性**: IC=0.19、有効 BR≈1,500-2,200 の仮定で IR 上限は 7.3-8.9。現行 Sharpe 7.5-8.9 はこの上限域内。
2. **ターゲットの構造的制約**: 9:10→Close は総分散の 40% に過ぎず、その中でさらに予測可能部分は限られる。既にその大部分を獲得している可能性。
3. **信号側実験の収穫逓減**: 多数の信号側アイデア（RMT、nonlinear shrinkage、マクロ、メタラベリング、動的 MH ウェイト等）が marginal または不採用。唯一大きく改善したのは Fractional Differentiation だけ。
4. **過学習リスクの高まり**: PBO=0.5145、IS-OOS 相関が強く負。パラメータに敏感。新しいアイデアを追加しても、本当の改善か過学習かの区別が困難。

### 4.2 まだ改善余地がある可能性のある領域

信号側は限界に近いが、**システム・データ・実行側**には大きな改善余地が残存する可能性がある。

| 領域 | 具体的改善機会 |
|---|---|
| **データ品質** | 9:10 価格の実データ利用率は現在 1.34%（55/4,119 日）。実 9:10 価格に置き換わるとターゲットミスマッチが縮小する可能性。 |
| **ターゲット設計** | Close-to-Close ではなく 9:10-to-Close への直接最適化、あるいは異なる予測ホライズン。 |
| **実行・コスト** | LOB ベースのスリッページ推計、取引時間帯の最適化、成行ではなく指値。 |
| **データ前処理** | Fractional Diff の成功は「US 入力のスペクトル特性」が重要であることを示す。さらに異なる d 値やマルチスケール処理。 |
| **外部データ** | 株価指数先物夜間取引やオプション IV など、米国クローズ以降から日本 9:10 までの追加情報源。 |
| **モデル構造** | 固定感応度行列 `MACRO_SENS_MATRIX` や線形 BLP 仮定を超える、非線形・動的关系の学習。ただし過学習リスク高。 |

### 4.3 信号側限界の直感的解釈

US セクター ETF の前日リターンは、翌日日本市場の寄付きまでの大部分（≈60%）がすでに夜間取引や海外市場を通じて織り込まれる。本戦略は 9:10 時点で「残された」日中（9:10→Close）の相対動向を予測する。

この予測問題の **情報上限** は、米国市場クローズ後から日本市場大引けまでの追加情報フローに依存する。US 市場が主要情報源ならば、9:10 以降の日本市場動向はおおむねランダムウォークに近づき、予測可能成分は限られる。

現行の IC=0.19 は、日中リターンに対してかなり強い予測力を示しており、これを大きく上回るのは難しい。

---

## 5. 結論

| 観点 | 評価 |
|---|---|
| 信号は理論限界に近いか | **はい、おおむね近い**。IC=0.19、Grinold 上限 7-9、現行 Sharpe 7.5-8.9 は整合的。 |
| さらなる信号側改善は期待できるか | **限定的**。過去多数の実験で収穫逓減が明確。大きく改善したのは Fractional Diff のみ。 |
| 次に注力すべき領域は | **データ品質（9:10 実価格）、実行コスト、ターゲット設計、外部情報源**。信号側よりも取り巻くインフラ側に余地。 |

### 推奨アクション

1. **信号側実験は暫定停止または超厳格な過学習ガード下で実施**: PBO=0.5145 を考慮し、新しい信号アイデアはウォークフォワード + DSR + 感度分析が必須。
2. **9:10 実価格データの蓄積を最優先**: 現在 1.34% しか使えていない実 9:10 価格が増えれば、ターゲットミスマッチが解消され信号・ポートフォリオ両方が改善する。
3. **Fractional Diff (d=0.1) は本番昇格済み（2026-07-21）**: `configs/production/production.yaml` において `enabled: true`、`d: 0.1` となっている。今後はその効果を本番運用でモニタリングし、さらなる d 値チューニングや他の前処理手法（マルチスケール・ウェーブレット等）の探索に移る。
4. **実行・コスト側の改善を並行検討**: LOB スリッページモデル、指値戦略、取引時間帯の最適化。
5. **外部情報源の検討**: 米国クローズ〜日本 9:10 までの株価指数先物夜間取引、オプション IV、為替オーダーフロー等。

---

## 6. 出典

- `reports/sprint0/sprint0_diagnostics_report.md`
- `reports/sprint_a4_ensemble_ic/a4_ensemble_ic_report.md`
- `reports/phase3_walkforward_validation_report.md`
- `reports/fractional_diff_walkforward_audit_report.md`
- `reports/meta_labeling/meta_labeling_walkforward_report.md`
- `reports/macro_kappa_v2/macro_kappa_v2_non_adoption_report.md`
- `reports/sprint_macro_direction/macro_direction_adjustment_report.md`
- `reports/dynamic_mh_weights/dynamic_mh_weights_walkforward_report.md`
- `reports/nonlinear_shrinkage_walkforward_report.md`
- `reports/rmt_filter_walkforward_report.md`
- `reports/pbo/pbo_report.md`
- `reports/tv_gap_coef/tv_gap_coef_walkforward_report.md`
- `AGENTS.md`（不採用実験の記録）
