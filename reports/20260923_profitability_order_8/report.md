# 収益改善順序8：容量・集中・β制約

発注・取消・再送を行わない、Stage 5の同一OOS weightsを使ったモデル側診断です。

判定: **PENDING_NOT_ADOPTED**

## 固定条件

- 期間: 2024-12-23 -> 2026-07-29 (368日)
- model gross上限2.0、model net±0.05、side leverage 1.5を分けて集計
- 5/10/20bps片道は事前固定のストレス表示であり、採用パラメータ探索ではない

## 制約と集中

- max model net: 4.163e-16、max model gross: 2.000000
- max effective net: 6.245e-16、max effective gross: 3.000000
- model制約（net/gross）: PASS
- 単一銘柄weight abs 平均 / p95 / 最大: 0.331021 / 0.455283 / 0.677408
- top1 abs share平均: 18.854%、top3 abs share p95: 55.650%
- abs-weight HHI最大: 0.182674、平均long/short銘柄数: 5.00 / 5.00

## β診断

- historical jp_betaによるweighted gap-beta proxy: available
- weighted gap-beta 平均 / abs p95 / abs最大: 0.021940265364076426 / 0.5358643406182112 / 1.0469183526088313
- これは `sum(model_weight * jp_beta)` であり、約定後の実現βやTOPIXへの日中βを証明しない。

## スリッページ・資金規模の表示

| 片道slippage | 追加費用合計 | net合計 | net Sharpe | max DD |
|---:|---:|---:|---:|---:|
| 5.0bps | 0.000000 | 1.964607 | 5.163636 | -0.249799 |
| 10.0bps | 0.778427 | 1.186180 | 3.115212 | -0.368060 |
| 20.0bps | 2.335280 | -0.370673 | -0.970627 | -0.580122 |

AUM表はweightから得た理論notionalで、ADV・板厚・約定率を含まないためcapacityとは呼ばない。詳細は `stage8_summary.json` と `daily_capacity_beta.csv` を参照。

## 実行可能性の未充足

- true 09:10 quote: False、17銘柄5段板: False
- 約定・手数料明細: False、runtime数量gap: 199
- OOSのlot/約定価格: False、銘柄別borrow: False
- よって、資金規模別の実容量、spread/impact、fill rate、貸株制約を判定できない。

## 判定と次の依存

- **COMPLETE_MODEL_DIAGNOSTIC_PENDING_EXECUTION_CAPACITY / PENDING_NOT_ADOPTED**
- モデルweight制約・集中・beta proxy・固定stressは診断したが、順序8の経済的完了条件（板厚、約定率、口数、実約定後exposure）は未充足。
- 順序9は、実行可能baselineと容量制約を通過した独立OOSがないため、新特徴量/universeの追加実験へ進めない。

## 参照

- `stage8_summary.json`
- `daily_capacity_beta.csv`
- `reports/20260923_profitability_order_2/microstructure.json`
- `reports/20260923_profitability_order_5/report.md`
