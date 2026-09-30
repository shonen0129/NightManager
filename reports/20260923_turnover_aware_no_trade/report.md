# Turnover-aware no-trade 研究結果

実行日: 2026-09-22T16:56:53.374056+00:00。本番設定・本番artifactは変更していない。

## 仮説と設計

現行の accepted production artifact の target weight を信号として固定し、前日実効weightから当日targetへの raw-weight L1 変更量を上限化すると、alphaを大きく失わずに slippage と turnover を減らせるかを検証した。候補は事前固定した cap=1.0/1.5/2.0。
選択期間は2020-01-06〜2024-12-31、holdoutは2025-01-01〜2026-08-14。全営業日を含み、flat日を除外していない。

## 集計結果（全期間）

| variant | net Sharpe | gross Sharpe | max DD | mean turnover | total cost |
|---|---:|---:|---:|---:|---:|
| baseline | 2.2201 | 2.4839 | -29.41% | 1.3083 | 3.872052 |
| cap_1.0 | 1.9427 | 2.0810 | -18.40% | 0.5000 | 1.923736 |
| cap_1.5 | 2.0426 | 2.2200 | -24.64% | 0.7476 | 2.529818 |
| cap_2.0 | 2.1187 | 2.3329 | -29.01% | 0.9766 | 3.085888 |

## 期間別比較

| period | baseline SR | selected SR | ΔSR | baseline DD | selected DD | Δ turnover |
|---|---:|---:|---:|---:|---:|---:|
| selection_2020_2024 | 6.5166 | 6.5166 | +0.0000 | -7.71% | -7.71% | +0.0000 |

### Holdout sensitivity (2025-01-01〜2026-08-14)

| variant | net Sharpe | max DD | mean turnover | mean daily Δ vs baseline | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| baseline | 3.7425 | -29.41% | 1.2423 | +0.00000000 | [+0.00000000, +0.00000000] |
| cap_1.0 | 3.5774 | -18.40% | 0.5000 | -0.00704260 | [-0.01643940, -0.00152229] |
| cap_1.5 | 3.6390 | -24.64% | 0.7461 | -0.00420616 | [-0.00995073, -0.00086088] |
| cap_2.0 | 3.6859 | -29.01% | 0.9666 | -0.00251919 | [-0.00616295, -0.00051550] |

## 判定

選択候補: `baseline`。選択期間の事前ゲートは `False`、未使用 holdout の確認は `True`、総合判定は **REJECTED**。

採用条件は、選択期間で baseline 以上の net Sharpe・max DD と turnover低下を満たし、holdoutでも net Sharpe と max DD を悪化させないこと。bootstrap区間は平均日次差の不確実性であり、Sharpe改善の有意性を意味しない。DSRは候補4試行（baselineを含む）の選択期間Sharpeに対する補正値であり、過去の未登録試行を完全に数えたものではない。

### コスト内訳（selected / 全期間）

- slip: 3.33370609
- financing: 0.16547517
- borrow: 0.05074572
- reverse: 0.32212500

再現結果: `reports/20260923_turnover_aware_no_trade/results.json`。実験registryにも結果を追記した。
