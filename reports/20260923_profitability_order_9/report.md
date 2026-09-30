# 収益改善順序9：新特徴量・universe

判定: **HOLD_DEPENDENCY_GATE / NOT_STARTED**

順序9の新特徴量・universe追加実験は実施していない。順序2の真の9:10 quote・板・約定明細、順序3の実行可能baseline、順序7の実費/在庫/校正、順序8の容量・約定率・実約定後exposureが未完了のためである。

## 実施しなかったこと

- 新しい特徴量、銘柄、分類、パラメータ、ML target、universeは追加していない。
- 新規OOS、感度分析、DSR、production artifactの作成は行っていない。
- 本番config、モデルartifact、発注経路は変更していない。

## 保留ゲート

| 前提 | 状態 | 根拠 |
|---|---|---|
| 実行可能な9:10 quote/板/約定 | 未達 | `reports/20260923_profitability_order_2/microstructure.json`：true 9:10 capture=0、runtime数量gap=199 |
| コスト後baseline | 保留 | `reports/20260923_profitability_order_5/report.md`：実約定・9:10価格不足 |
| ML取引価値の実費校正 | 未達 | `reports/20260923_profitability_order_7/report.md`：observed fee/fill costなし |
| 容量・β・約定後exposure | 未達 | `reports/20260923_profitability_order_8/report.md`：板厚・fill rate・lot roundingなし |

## 再開条件

前提ゲートを満たした後、既存の不採用索引・registryを確認し、仮説・期間・費用・独立OOS・±感度・試行数補正を先に固定する。そのうえで新特徴量またはuniverseを一度に混ぜず、baselineとの差分を全評価日で検証する。

順序9を未実施のまま、見かけのbacktest改善を理由に本番採用・設定変更は行わない。
