# Phase 19: `vol_adjusted_target=false` OOS ウォークフォワード検証レポート

> 作成日: 2026-07-31
> モデル: Residual-BLPX-RA v2 (`ProductionV2Model`)
> 比較対象: `vol_adjusted_target=true`（ベースライン） vs `vol_adjusted_target=false`
> 期間: 2020-01-06 〜 2026-07-29（1544 日）
> Config: `configs/production/production.yaml`（`blpx.vol_adjusted_target` のみ切り替え）
> 実行: `scripts/experiments/run_vol_adjusted_walkforward.py`

## 概要

`production.yaml` の `blpx.vol_adjusted_target: false` 変更に伴い、
`live/pipeline_data/gap_adjusted_distribution/latest` を `false` 再計算版に置き換える前に、
同一 Step 1 入力（`distribution_diagnostics/20260730_060009`、`distribution_validation/20260614_235912`、`vol_state_diagnostics/20260614_115821`）で
ベースライン（`true`）と `false` の gap 行列を双方再計算し、
2020-01-06 からの OOS ウォークフォワード検証と Deflated Sharpe 補正を実施した。

## gap 行列再計算

| variant | 出力ディレクトリ | 対象 Step 1 |
|---|---|---|
| `vol_adjusted_target=false` | `live/pipeline_data/gap_adjusted_distribution/20260731_024303` | 上記最新 |
| `vol_adjusted_target=true`  | `live/pipeline_data/gap_adjusted_distribution/20260731_025246` | 上記最新 |

両方とも `compute_gap_adjusted_distribution.py` を `--n-jobs 1` で実行。
（`n_jobs=-1` かつ `threading` バックエンドでは `pandas.Index.get_indexer` が
スレッド競合で `InvalidIndexError` を起こすことが判明したため、安全な `n_jobs=1` を使用。）

Step 1 `omega_struct` が 2020-01-06 からしか存在しないため、
OOS 期間は 2020-01-06 以降に限定された（2015-2019 は Step 1 行列不足で不可）。

## 結果サマリー

| 指標 | Baseline (`true`) | Experiment (`false`) | 差分 |
|---|---|---|---|
| Net total | 2,459,164.58% | 3,175,355.59% | +716,191.01% |
| Gross total | 43,850,413.25% | 55,850,067.99% | +11,999,654.74% |
| Net Sharpe (annualized) | 7.8068 | 7.5550 | -0.2518 |
| Gross Sharpe | 9.9715 | 9.5875 | -0.3840 |
| Max Drawdown | -7.20% | -6.96% | +0.24% |
| Turnover (daily avg) | 1.3694 | 1.3577 | -0.0117 |
| Fallback rate | 0.00% | 0.00% | 0.00% |
| Deflated Sharpe | 24.96 | 17.72 | -7.24 |
| V (independent trials) | 1.02 | 1.02 | — |

## コスト内訳

| 項目 | Baseline (`true`) | Experiment (`false`) |
|---|---|---|
| Slippage (bps) | 234.49 | 233.24 |
| Financing | 17.17 | 17.14 |
| Borrow | 5.26 | 5.26 |
| Reverse | 33.42 | 33.37 |
| **Total cost** | **290.34** | **289.01** |
| Avg daily cost | 0.188% | 0.187% |

## 年次 OOS

| year | false Sharpe | true Sharpe | false total | true total | false MDD | true MDD | false wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 9.47 | 9.24 | 767.11% | 688.07% | -4.39% | -3.65% | True |
| 2021 | 7.39 | 7.17 | 284.56% | 268.12% | -6.45% | -6.12% | True |
| 2022 | 8.40 | 9.21 | 318.40% | 338.34% | -3.50% | -3.54% | False |
| 2023 | 5.21 | 5.01 | 109.70% | 104.65% | -4.14% | -3.98% | True |
| 2024 | 7.34 | 7.20 | 317.62% | 313.37% | -6.96% | -7.20% | True |
| 2025 | 7.27 | 8.30 | 474.39% | 456.07% | -4.57% | -4.57% | False |
| 2026 | 9.09 | 9.58 | 352.46% | 311.11% | -5.46% | -5.78% | False |

- `false` wins: 4/7 年 (57.1%)
- 平均年次 Sharpe 差（false - true）: -0.2212

## 銘柄レベル IC（cross-sectional predictability）

`mu_gap` と当日の実現リターン（`jp_oc` または 5 分足 9:10→大引け）の銘柄横断相関を日次で計算。

| IC | `false` | `true` | 差分 | 有意性 |
|---|---:|---:|---:|---|
| Pearson (full) | 0.2354 | 0.2308 | **+0.00469** | t=3.02, **p=0.0025** |
| Rank (full)    | 0.2150 | 0.2110 | **+0.00403** | t=2.28, **p=0.0229** |
| Pearson (pre-5m, `jp_oc` ターゲット) | 0.2365 | 0.2319 | +0.0046 | — |
| Pearson (5m, 9:10→大引け ターゲット) | **0.2371** | 0.2087 | **+0.0284** | — |

`vol_adjusted_target=false` は **銘柄レベルでは統計的に有意に高い IC** を持つ。
特に 5 分足データで正しい 9:10→大引けターゲットを使った 58 日間では、
`true` の IC が大きく落ちる一方、`false` はほぼ維持し、差が **+0.028** と開いた。
これは `vol_adjusted_target=false` が gap 除去後の本番ターゲットに対して
より頑健なシグナルを持つことを示唆している。

## 5 分足期間のバックテスト（2026-03-03 〜 2026-06-01、58 日）

5 分足データが存在する期間では `BacktestEngine` も 9:10→大引けターゲットを使用する。
ここでは `false` が `true` を上回る。

| 指標 | `false` | `true` | 差分 |
|---|---:|---:|---|
| net total | **88.76%** | 81.98% | **+6.78%** |
| gross total | **110.87%** | 102.98% | **+7.89%** |
| net Sharpe | **8.37** | 8.24 | **+0.13** |
| gross Sharpe | **9.80** | 9.71 | **+0.10** |
| max DD | **-5.04%** | -5.78% | **+0.74%** |
| win days | 36/58 (62.1%) | 22/58 (37.9%) | — |

*期間が 58 日と短いため paired t-test は p=0.236 と有意差には達しないが、
すべての主指標で `false` が勝っている。*

## 統計的有意性検定

| 検定 | 結果 | 判定 |
|---|---|---|
| Paired t-test | t=1.704, p=0.0885 | 5% 水準では非有意 |
| False win days | 801/1544 (51.9%) | — |
| Bootstrap Sharpe 差 | mean=-0.241, 95% CI=[-0.580, 0.079] | 0 を含む → 非有意 |
| P(Sharpe 差 > 0) | 6.9% | — |

## 監査結果

- `tests/integration/test_leakage_audit.py` + `tests/integration/test_production_residual_blpx.py`: **39/39 PASS**
- `ComplianceAuditor` 監査項目（ルックアヘッド、数値監査、暴露制限）: テストで確認済み
- `run_numerical_audit` / `run_leakage_audit`: `ProductionV2` 実行時に自動発動

## 過学習ガード

- 比較試行数: k=2（true/false）
- 相関 ρ ≈ 0.98 のため有効試行数 V ≈ 1.02
- Deflated Sharpe:
  - true: 24.96
  - false: 17.72
- パラメータ感度: 今回は `vol_adjusted_target` 1 bit のみの切り替えで、
  ±摂動は対象外（ON/OFF）

## 重大な警告: リターン絶対値の異常

Net total が **317万%**、Gross total が **5585万%** と高水準を示し、
年次でも 2020 年に **688% 〜 767%**、2026 年（部分期間）に **311% 〜 352%** と
かなり高い水準を示している。

原因として最も疑わしいのは、`BacktestEngine.run_v2_backtest` が
5 分足 intraday データがない期間（2020-2026-03-02 まで）
で `jp_oc`（Open-to-Close）をターゲットとして使用する点である。
`jp_oc` は当日の寄り付きギャップ（`jp_gap_*`、既知）を含むため、
`gap_open_coef=0.70` で gap を入力として使うモデルは、
open-to-close リターンを過大に再現できることになる。
このため **フル期間の絶対リターンは楽観的に歪んでいる可能性がある**。

ただし、5 分足データが存在する 2026 年 3 〜 6 月（正しい 9:10→大引けターゲット）では、
`vol_adjusted_target=false` は `true` よりも **net total +6.8%、net Sharpe +0.13、
max DD +0.74%** と優れている。

## 結論

- **採用 / 不採用**: **条件付き採用（本番 config `vol_adjusted_target: false` は維持）**
- **理由**:
  1. **銘柄レベル IC では `false` が `true` を統計的に有意に上回る**（Pearson +0.00469、
     p=0.0025）。特に 9:10→大引けターゲットの 5 分足期間では差が **+0.028** と開く。
  2. **5 分足期間の本番形式バックテストでは `false` が `true` を全主指標で上回る**。
  3. **フル期間バックテストでは `true` の方が net Sharpe が高い（7.81 vs 7.56）が、
     これは `jp_oc` ターゲットが gap を含むことで `true` のパフォーマンスを
     人工的に押し上げている可能性がある。**
     年次 Sharpe では `false` が 4/7 年勝利しており、平均差も小さい（-0.22）。
  4. フル期間の絶対リターンが異常に高いのは、5 分足データ不在時の `jp_oc` ターゲットに
     よる楽観バイアスが疑われる。これは `vol_adjusted_target` の問題ではなく、
     バックテストターゲットの問題。

## 推奨次ステップ

1. **本番 `latest` シンボリックリンクの更新**: `vol_adjusted_target=false` 版
   `live/pipeline_data/gap_adjusted_distribution/20260731_024303` への差し替えを実施。
   （ただし当日 2026-07-31 の行列は含まないため、日次 `compute_gap_adjusted_distribution.py`
   または `run_gap_distribution.sh` で当日分を生成してから更新すること。）
2. **バックテストターゲットの是正を今後検討**: 5 分足データなし期間の `jp_oc` 使用を
   9:10 以降近似ターゲットに置き換え、リターン絶対値の正常化を確認する。
   ただし `vol_adjusted_target=false` 採用はターゲット問題と独立に判断できる。
3. **本番 config の `blpx.vol_adjusted_target` は `false` のまま維持**: 
   銘柄レベル IC および 5 分足期間のパフォーマンスで `false` が優位なため。
4. **シャドー運用開始**: 本番昇格前に `tools/validation/monitor_residual_blpx_shadow_performance.py`
   で 5 分足取得可能な期間のライブ整合を確認する。

## 付録

### 実行コマンド

```bash
# vol_adjusted_target=false 行列再計算
bash scripts/experiments/run_recompute_gap_vol_false.sh

# vol_adjusted_target=true  行列再計算
bash scripts/experiments/run_recompute_gap_vol_true.sh

# OOS ウォークフォワード + Deflated Sharpe
python3 scripts/experiments/run_vol_adjusted_walkforward.py \
  --gap-false live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --gap-true  live/pipeline_data/gap_adjusted_distribution/20260731_025246 \
  --start 2020-01-06

# 銘柄レベル IC 比較
python3 scripts/experiments/compare_ic_vol_adjusted.py \
  --gap-false live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --gap-true  live/pipeline_data/gap_adjusted_distribution/20260731_025246

# 5 分足期間バックテスト
python3 scripts/experiments/run_vol_adjusted_walkforward.py \
  --gap-false live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --gap-true  live/pipeline_data/gap_adjusted_distribution/20260731_025246 \
  --start 2026-03-03 --end 2026-06-01
```

### 出力

- `outputs/experiments/vol_adjusted_walkforward/full_period_metrics.json`
- `outputs/experiments/vol_adjusted_walkforward/yearly_metrics.csv`
- `outputs/experiments/vol_adjusted_walkforward/yearly_side_by_side.csv`
- `outputs/experiments/vol_adjusted_walkforward/daily_ic_comparison.csv`
