# Macro Directional Adjustment 比較バックテストレポート

## 概要

`compute_macro_direction_adjustment`（符号付きマクロサプライズによる方向調整）を v2 production pipeline に統合し、現行モデルとの比較バックテストを実施した。

## 問題提起

現行の `compute_factor_kappa_scale` は `|surprise|` と `|sensitivity|` を使用するため、`scale_j >= 1.0` 常にポジションを縮小する。正のショック（例: USDJPY上昇＝円安）も「リスク増」として扱われ、輸出企業にとっての好材料を逃す非対称性がある。

`compute_macro_direction_adjustment` は符号付き surprise × 符号付き sensitivity を使用し、正のサプライズでポジション拡大、負で縮小を実現する。本関数は `macro.py:349-403` に実装済みだが、v2 production pipeline では未活性化であった。

## 実装内容

### 1. Schema 追加 (`src/leadlag/config/schemas.py`)

`ProductionV2RunConfig` に `macro_direction_enabled: bool` フィールドを追加。

### 2. Config parsing (`src/leadlag/models/production_v2.py`)

`parse_run_config` で `portfolio.macro_direction_enabled` を読み取り。

### 3. Pipeline 統合 (`src/leadlag/models/production_v2.py`)

`generate_v2_production_portfolio` 内で、`macro_kappa_enabled` または `macro_direction_enabled` が True の場合にマクロデータをダウンロード・処理。各効果は独立して適用:

- **macro_kappa_enabled=True**: `compute_sigma_yy_inflation` で Omega_gap を膨張（|surprise| × |sensitivity|、リスク調整）
- **macro_direction_enabled=True**: `compute_macro_direction_adjustment` で mu_gap を調整（signed surprise × signed sensitivity、シグナル調整）

### 4. 比較スクリプト (`scripts/experiments/compare_macro_direction.py`)

4バリアントで比較:
- **A**: No macro（現行デフォルト）
- **B**: Kappa only（|surprise| Omega_gap膨張）
- **C**: Direction only（signed surprise mu_gap調整）
- **D**: Kappa + Direction（両方）

## バックテスト条件

- 期間: 2020-01-06 ～ 2026-07-08（1,474営業日）
- データ: gap-adjusted distribution matrices（`results/gap_adjusted_distribution/latest`）
- コストモデル: 片道5bps slippage、overnight alpha=0（日次全額決済）
- macro kappas: [3.0, 0.5, 0.5]（USDJPY, CLF, TNX）
- macro surprise halflife: mean=20d, vol=60d
- macro prices: yfinance経由で事前ダウンロード（USDJPY, CLF, TNX）

## 結果

### サマリー指標

| Metric | A: NoMacro | B: KappaOnly | C: DirOnly | D: Both |
|--------|-----------|-------------|-----------|---------|
| **Sharpe (net)** | **4.4148** | **4.4963** | 2.1522 | 2.3357 |
| AR (net) | 85.37% | 83.73% | 37.54% | 38.86% |
| MDD | -4.77% | -4.33% | -18.11% | -16.76% |
| Hit Rate | 66.21% | 66.35% | 55.83% | 56.11% |
| Calmar | 17.89 | 19.34 | 2.07 | 2.32 |
| Turnover | 1.4917 | 1.4131 | 1.4836 | 1.4598 |
| GrossExp | 1.9294 | 1.8246 | 1.9023 | 1.8701 |

### 年次Sharpe

| Year | A: NoMacro | B: KappaOnly | C: DirOnly | D: Both |
|------|-----------|-------------|-----------|---------|
| 2020 | 6.4443 | 6.4862 | 2.0598 | 2.2370 |
| 2021 | 3.4273 | 3.6612 | 2.8823 | 3.0235 |
| 2022 | 6.2484 | 6.0818 | 2.9707 | 3.3142 |
| 2023 | 2.3519 | 2.5709 | 0.1669 | 0.5726 |
| 2024 | 5.5685 | 5.3547 | 3.2655 | 3.3575 |
| 2025 | 4.0441 | 4.0495 | 0.8435 | 0.9972 |
| 2026 | 7.2934 | 7.0670 | 4.3761 | 4.9086 |

### コスト内訳（日次平均、bps）

| Component | A: NoMacro | B: KappaOnly | C: DirOnly | D: Both |
|-----------|-----------|-------------|-----------|---------|
| Slippage | 19.29 | 18.25 | 19.02 | 18.70 |
| Financing | 0.00 | 0.00 | 0.00 | 0.00 |
| Borrow | 0.00 | 0.00 | 0.00 | 0.00 |
| Reverse | 0.00 | 0.00 | 0.00 | 0.00 |

※ overnight alpha=0のため金利・貸株・逆日歩コストは発生しない。

## 分析

### 方向調整（C）の大幅劣化

- Sharpe: 4.41 → 2.15（**-51%**）
- AR: 85.37% → 37.54%（**-56%**）
- MDD: -4.77% → -18.11%（**3.8倍悪化**）
- Hit Rate: 66.21% → 55.83%（**-10.4pp**）

**全期間・全年で一貫して劣化**。特に2023年はSharpe 0.17まで落ち込む。

### 劣化の原因

1. **ノイズ注入**: `compute_macro_direction_adjustment` は `mu_gap`（BLPXシグナル）に `1 + kappa * surprise * sensitivity` を乗算する。マクロサプライズのz-scoreと翌日JPセクターリターンの間に単純な線形関係はなく、シグナルにノイズを注入する

2. **シグナル歪曲**: BLPXシグナルは既にUS→JPリードラグ構造を捕捉している。マクロ調整はこれを上書きし、シグナルの予測力を低下させる

3. **感応度行列の限界**: `MACRO_SENS_MATRIX` の固定的な感応度仮定が実際の動的関係を反映していない

### Kappa only（B）の僅かな改善

- Sharpe: 4.41 → 4.50（+1.8%）
- MDD: -4.77% → -4.33%（改善）
- Turnover: 1.49 → 1.41（改善）

Omega_gap膨張はシグナルを変更せずリスク推定のみ調整するため、シグナル品質を維持したままリスク管理を改善する。ただし改善幅は微小。

## 結論

| バリアント | 判定 | 理由 |
|-----------|------|------|
| A: No Macro | **採用（現行維持）** | ベースラインとして最も安定 |
| B: Kappa Only | **検討余地あり** | 僅かなSharpe改善・MDD改善。ただし改善幅が小さく過学習リスクを要評価 |
| C: Direction Only | **不採用** | Sharpe -51%、MDD 3.8倍悪化。シグナル品質を破壊 |
| D: Both | **不採用** | Cと同様の劣化パターン。kappaの改善がdirectionの劣化を相殺できず |

`compute_macro_direction_adjustment` の v2 pipeline 有効化は**不採用**とする。符号付きマクロサプライズとJPセクターリターンの関係は、固定感応度行列では線形モデルとして表現できず、BLPXシグナルに対してノイズ注入として作用することが実証された。

## 注意事項

- 本バックテストは overnight alpha=0（日次全額決済）で実施。本番の `alpha_long=0.75, alpha_short=0.5` では金利・貸株・逆日歩コストが追加されるが、相対比較の結論は変わらないと予想される
- macro_kappas=[3.0, 0.5, 0.5]はv1 fallback用チューニング値。v2 pipeline向けの別チューニングは未実施
- ウォークフォワード検証は未実施（過学習リスク評価が必要）
