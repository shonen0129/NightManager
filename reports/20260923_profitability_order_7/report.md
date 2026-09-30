# 収益改善順序7：コスト後のML取引価値

発注・取消・再送を行わない、Stage 5のML有効とStage 6のML無効を同一OOSで比較する診断です。

判定: **PENDING_NOT_ADOPTED**

## 固定条件

- 期間: 2024-12-23 -> 2026-07-29 (368日)
- 年率換算: 252営業日
- 同一のV2入力・費用モデル・全営業日で比較
- ML有効: `reports/20260923_profitability_order_5/v2_current_overlay_oos`
- ML無効: `reports/20260923_profitability_order_6/ml_off`

## 取引価値の分解

| 系列 | ML有効 | ML無効 | ML有効−無効 |
|---|---:|---:|---:|
| net Sharpe | 5.163636 | 5.179883 | -0.016247 |
| net 合計 | 1.964607 | 1.921412 | 0.043195 |
| gross 合計 | 2.871635 | 2.824956 | 0.046679 |
| cost 合計（負値表示） | -0.907028 | -0.903543 | -0.003485 |
| 平均turnover | 1.287065 | 1.275906 | 0.011158 |

- 日次net差分の20日block bootstrap 95%区間: [0.000013516, 0.000222649]
- 日次net差分の平均: 0.000117377、合計: 0.043194810
- `delta_net = delta_gross - delta_cost` 最大誤差: 2.006e-16
- ML有効は総収益を増やす一方、同じOOSではML無効よりnet Sharpeがわずかに低く、費用後のリスク調整改善は証明されていない。

## 在庫・約定の証拠

- 重みが変わった日: 368 / 368日 (100.000%)
- 平均weight L1差分: 0.079124、最大: 0.246337
- 上記はbacktestのweight proxyであり、brokerの実在庫・注文数量・約定価格ではない。
- broker再照会成功: 0件、失敗: 27件
- 公式CSVのfee components: なし
- したがって、実約定費用・在庫を反映したML targetの校正ゲートは未充足。

## ML target / artifact

- artifact: `20260922T011723828589Z-7e1a81ab2617`、target_type: `raw`、metadata: `verified`
- 学習期間: 2015-01-05 -> 2024-12-20、label_asof_end: 2024-12-20
- 学習コード上の固定round-trip cost: 0.001000 (5.0bps/side)
- raw targetは固定round-trip costを控除するが、観測されたbroker fee/fill costをtargetに接続していない。
- p_tradeの校正曲線と実約定費用の校正サンプルはartifact/保存成果物から確認できない。

## 判定と次の依存

- **PENDING_NOT_ADOPTED**: 現行MLを削除・再学習・本番昇格する判断はしない。
- Stage 7の数値分解は完了したが、実費・実在庫・校正証拠不足のため、収益改善順序7を経済的採用完了とは扱わない。
- Stage 8（容量・集中・beta）とStage 9（新規特徴量・universe）は、Stage 2の実行証拠およびStage 3の実行可能baselineが未完了のため保留する。

## 参照

- `stage7_summary.json`
- `daily_ml_value.csv`
- `reports/20260923_profitability_order_5/report.md`
- `reports/20260923_profitability_order_6/report.md`
- `models/ml_order_overlay/production_20260923/versions/20260922T011723828589Z-7e1a81ab2617/metadata.json`
