# S2b 実行記録：版付き入力・read-only所有権・PIT入力分離

- 実施日: 2026-09-16
- 判定: **実装完了（S2b範囲 PASS）**
- 対象: 修正済み作業ツリー（未commit差分を保持）

## 実施内容

| 境界 | 実施内容 |
|---|---|
| 入力契約 | `domain.inputs`に`KnownMarketInputs`、`HistoricalInputs`、`EvaluationInputs`、`DecisionInputs`、`InputVersion`を追加 |
| 版付け | schema・既知入力・履歴・評価入力の内容をSHA-256で束ね、入力訂正を次runで検知可能にした |
| PIT | `MarketSnapshot.to_known_inputs`と`PITDataLake.build_decision_inputs`で、当日既知値とrun所有履歴を一つの契約へ変換 |
| 所有権 | snapshot配列・価格mapping・`PITMatrixView`をコピーしてread-only化。履歴は契約生成時に一度だけdeep copyし、BTの日付ループで再コピーしない |
| 時点分離 | `sig_date >= trade_date`を拒否。今日のUS/gap/価格と未確定JP targetを同じ既知入力へ入れない。2010-01-01〜2014-12-31 priorを固定 |
| 利用者切替 | ProductionRunner、ProductionV2Model/V2 decision、BacktestEngine、v2 bridgeを`DecisionInputs`経路へ切替 |
| 互換 | 旧parallel引数はRunner/modelの入口adapterに限定。入力契約経路では複数候補の優先順位解決を行わない |

## 撤去状況

S2bで内部decision経路の`df_exec`/`lake`/`snapshot`/`current_prices`の多重入力を撤去した。
外部利用者が確認できるまで、`preprocessor.compute_jp_target_returns` compatibility entry、
`data/cache.py` shim、`models/blpx.py`再export、`generate_v2_production_portfolio`の公開dict
wrapperは残す。これは無期限の互換層ではなく、S2/S3の利用者切替・artifact確認後に削除する対象である。

## 検証

| 検証 | 結果 |
|---|---|
| S2b input contract・PIT・S2a boundary | **15 passed** |
| V2 / Stage A-C 回帰 | **76 passed** |
| S2b input contract（model新経路を含む） | **8 passed** |
| `compileall`（変更対象） | PASS |
| `tests/` 全体 | **673 passed / 18 warnings / 722.77秒**（逐次）。その後のRunnerのcache指定伝播の最終1行変更は対象65件で再確認 |
| ruff（変更対象） | PASS |
| plan / ADR / architecture のリンク・fence・空白検証 | PASS（60リンク） |

## 判定と残件

S2bの版付き入力、PIT分離、read-only所有権、主要4経路の切替はPASSとする。S2全体の完了には、
残存compatibility entry・cache shim・公開wrapperの具体的利用者切替と撤去、macro/ADRの
available_at証拠を伴う厳密化が必要であり、後続S2/S3の対象として残す。
