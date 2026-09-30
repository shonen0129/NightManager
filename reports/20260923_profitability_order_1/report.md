# 収益改善順序1：本番再生・約定台帳照合

実行は読み取り専用。brokerへの発注・取消・再送は行っていない。

判定: **COMPLETE_WITH_LIMITATIONS** / 次段階: **PROCEED_TO_STAGE_2**

## 固定snapshotの照合

| 項目 | 結果 |
|---|---|
| μ/Ω cache vs on-demand | True |
| production vs BT weights | True |
| production vs collector weights | True |
| 数値・リーク監査 | True |
| scores・PITの直接比較 | True |

## 約定照合

| 指標 | 値 |
|---|---:|
| local close order | 27 |
| local group | 23 |
| quantity一致group | 23 |
| requested quantity | 207 |
| runtime log confirmed quantity | 8 |
| CSV confirmed quantity for local keys | 207 |
| runtime log quantity gap | 199 |
| observed execution price一致group | 7 |

## 未解決事項

- The CSV has no broker order IDs, so one-to-one order mapping is unavailable.
- The supplied runtime logs record zero fill quantity for 199 of 207 shares; the CSV confirms those executions but cannot repair the original logs.
- The broker detail re-query was unavailable for the historical records and fee components are not present in the CSV; these are carried into stages 2 and 3 and are not treated as zero.

この結果により、固定snapshotの重複計算とscores/PIT配列の直接比較はPASS、実約定は公式CSVの数量・日付・銘柄・売買単位でローカル23/23 group、207/207株を照合できた。順序1はこの範囲で完了とする。ただしbroker order ID・費用内訳・実行ログの199株分は未解決のまま順序2・3へ持ち越し、約定済み・手数料済みとは扱わない。

再現JSON: `reports/20260923_profitability_order_1/reconciliation.json`
