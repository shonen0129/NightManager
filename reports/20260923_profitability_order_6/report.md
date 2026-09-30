# 収益改善順序6：overlay逐次ablation

発注・取消・再送を行わない、同一OOS期間・同一費用のV2診断です。

判定: **COMPLETE_DIAGNOSTIC_PENDING_ADOPTION**
- 期間: 2024-12-23 -> 2026-07-29 (368日)
- 比較: current production artifactを全機能有効baselineとし、1機能だけ無効化
- 年率換算: 252営業日
- ブロックbootstrap: 20営業日、1000回、seed=42（平均差の不確実性のみ）
- 新パラメータ探索ではないため、DSRを採用判断の根拠にしない

## 結果

| variant | net Sharpe | max DD | turnover | fallback | mean差95%CI | 採否 |
|---|---:|---:|---:|---:|---|---|
| baseline | 5.163636 | -0.249799 | 1.287065 | 0.0000% | — | 現行baseline |
| mh_off | 0.809105 | -0.308138 | 1.395398 | 0.0000% | [-0.00781426, -0.00171528] | RETAIN_COMPONENT |
| ml_off | 5.179883 | -0.237011 | 1.275906 | 0.0000% | [-0.000222649, -1.35162e-05] | RETAIN_COMPONENT |
| macro_off | 4.938812 | -0.286714 | 1.344442 | 0.0000% | [-3.85364e-05, 0.000861027] | RETAIN_COMPONENT |
| fracdiff_off | 5.084448 | -0.272357 | 1.270585 | 0.0000% | [-0.000298682, 8.37238e-05] | RETAIN_COMPONENT |
| copula_off | 5.220363 | -0.249808 | 1.288518 | 0.0000% | [-2.40554e-05, 0.000161448] | RETAIN_COMPONENT |
| cs_off | 5.093990 | -0.251568 | 1.286158 | 0.0000% | [-0.000160279, 7.20568e-05] | RETAIN_COMPONENT |
| minvar_off | 5.197070 | -0.253632 | 1.299802 | 0.0000% | [3.54941e-05, 0.000246923] | RETAIN_COMPONENT |
| rule_d_off | 5.031115 | -0.303434 | 1.461339 | 0.0000% | [0.000142058, 0.00119543] | RETAIN_COMPONENT |

## 完了条件と採否

| 条件 | 結果 |
|---|---|
| MH/ML/macro/fracdiff/copula/CS/minvar/RuleDを1つずつ外す | PASS |
| 同一期間・同一費用・全評価日 | PASS |
| raw net/gross制約・4費用identity | 各variantで検査 |
| 逐次選択のOOS採用 | 保留（本番価格・実約定未証明） |
| 本番config/artifact変更 | 実施なし |

削除採用の基準は、net Sharpe・最大DD・turnover・fallback率が悪化せず、20日block bootstrap平均差CI下限が非負であること。全variantでこの基準を満たす削除案はなく、現行componentを維持する。さらにstage 2の実約定・9:10 quoteが不足しているため、現行componentの本番継続自体も収益性の最終証明ではない。

再現JSON: reports/20260923_profitability_order_6/ablation_summary.json
