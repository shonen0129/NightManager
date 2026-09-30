# 収益改善順序2：9:10価格・spread・板・約定の蓄積

発注・取消・再送を行わない読み取り専用の検証です。

判定: **INSTRUMENTED_HISTORICAL_DATA_INSUFFICIENT** / 次段階: **HOLD_BEFORE_STAGE_3**

## 観測状況

- historical 5分足 proxy: `PROXY_ONLY`, quote days=102, complete cross-section days=8
- live capture: `STORED_CAPTURE_SUMMARY`
- stored timestamped captures: 4 (append-only JSONL)
- 9:10 hook: `run_decision_v2.sh` + `LEADLAG_CAPTURE_0910=1`（リポジトリ側に追加済み、インストール済みschedulerは旧template）
- installed scheduler: `INSTALLED_OLD_TEMPLATE`; capture flag=None
- live quote spread bps: count=17, min=3.2567985670086306, median=53.475935828877, max=121.89029873114198
- fill detail: `UNAVAILABLE_991012`; fees observed=True
- local confirmed fills: orders=8, quantity=8, fee-detail rows=8

## 完了条件

| 条件 | 結果 |
|---|---|
| timestamped 09:10 quote | False |
| 17銘柄の5段板 | False |
| broker約定・手数料明細 | False |
| proxyの格上げなし | True |

## 未解決事項

- Historical five-minute bars are proxy observations, not executable 09:10 quotes.
- The official CSV has no broker order IDs and the historical detail re-query is unavailable.
- No fill quantity/price/fee is inferred from a market quote snapshot.

順序2は収集器と欠損管理を実装済みだが、過去期間の真の9:10 quote/板/約定明細がまだ不足しているため完了扱いにしない。09:10 JSTの稼働日に `--collect-live` を実行し、得られた観測を蓄積してから順序3へ進める。

再現JSON: `reports/20260923_profitability_order_2/microstructure.json`
