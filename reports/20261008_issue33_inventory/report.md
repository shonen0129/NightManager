# Issue #33 在庫会計の修正・受入

## 結果と未了

持越しの寄付gap/open→09:10分解、値洗い数量の翌朝rebalance、終端cash/残在庫/費用、artifact間の状態引継ぎ、実効執行量を実装した。
手計算のlong/short・逆方向gap・翌日flat・週末/連休・初期在庫・配列終端を回帰で固定した。
V2入口→価格区間→PnL→CSV/SQLite保存と、日次riskの実際のVaR履歴入口を通る連続replayも検証した。

**Issue全体は未完了。** ユーザー確認により、2026-09-25までの元df_exec・5分足、269日版versioned_var_returns.pkl/paired_returns.pklはこの端末にない。
元の269日risk replayとversioned ML paired比較の完全再評価は保留。入力をネット取得・補間で置換して完了扱いにはしない。
実装済みの連続replayは元データが揃えば動くが、269日の実データ受入を宣言していない。

## 会計契約

[設計判断](../../docs/decisions/2026-10-08-inventory-accounting.md)を参照。契約はinventory-v3。
ターゲットnotionalは前日close NAV×モデルweight×side leverage。在庫はsigned effective marked notional。
carry割合は数量へ適用し、前日close評価額から寄付と09:10へ値洗いして当日の損益へ計上する。
financing/borrow/reverseは前日close在庫×受入日までの暦日数。終端後の架空の保有日は足さない。
slipは(mark済みopening+closing売買notional)/前日close NAV×片道slip。
主turnoverは同volumeの1/2。旧目標weight差L1/2はtarget_weight_turnoverとして分離する。

通常BTのterminal_policy=liquidateは最終closeで全解消、open_inventoryは最終close markの在庫とcashを出す。
VaRは全artifact区間でopen_inventoryとして状態を渡す。standalone研究pairedは最終区間だけ全解消する。
日次markを表示するが、実約定時刻・通貨建ての実口座cashを再現した証拠ではない。

## 利用可能な保存入力での再評価

対象は2024-12-23〜2026-07-29の368営業日、保存済みML-on/offウェイト。
価格はtests/regression/baselines/df_exec_20260814.csv.gz。5分足/09:10実測はないため、全cellで寄付proxyを明示している。
両側・旧新会計・carryなしは同じ価格、日付、コストを使う。旧会計コードは監査commit b5b901e6d467fb000263f9178722e3954d652720から一時抽出した。
新会計の正本へ互換wrapper/旧会計実装を置いていない。

**このsnapshotは元レポートの入力とは異なる。** 以下は保存weightの条件付きrepricingであり、旧成績の再現や会計修正だけによる旧レポートとの差ではない。
同一入力の旧会計を併記して会計差だけを比較する。元保存netもdaily_comparison.csvへ別列で保持している。
未測定の朝区間0はproxyの仮定であって、実際の朝損益が0だった証拠ではない。

slip片道5bps、long carry .75/short .5、金利/貸株/逆日歩は継承解決済み本番cost設定。
side 1.50は旧比較基準、1.30は現行cost設定。新パラメータ探索・採用判断は行っていない。
フラット日を除かず252営業日でSharpeを年率化、DDは初期資本1込み、VaR/ESは最終250日/confidence .99/既存method、tail countは3。

| 系列 | 会計 | Net Sharpe | Gross Sharpe | 最大DD | 平均turnover | 複利net | VaR99 | ES99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_ml_on_side_1.5 | 旧会計/同入力 | 1.0529 | 3.0341 | -80.86% | 1.287065 | 50.97% | 5.398% | 5.701% |
| baseline_ml_on_side_1.5 | inventory-v3 | 1.0945 | 3.1529 | -80.97% | 2.118130 | 51.74% | 5.539% | 6.197% |
| baseline_ml_on_side_1.5 | carryなし/v3 | 1.0132 | 3.4448 | -79.05% | 2.627782 | 41.75% | 4.728% | 5.278% |
| baseline_ml_on_side_1.3 | 旧会計/同入力 | 1.0529 | 3.0341 | -76.04% | 1.287065 | 44.09% | 4.678% | 4.941% |
| baseline_ml_on_side_1.3 | inventory-v3 | 1.0968 | 3.1549 | -76.16% | 1.835696 | 44.76% | 4.794% | 5.367% |
| baseline_ml_on_side_1.3 | carryなし/v3 | 1.0132 | 3.4448 | -74.12% | 2.277411 | 36.16% | 4.097% | 4.574% |
| ml_off_side_1.5 | 旧会計/同入力 | 1.0713 | 3.0949 | -79.91% | 1.275906 | 50.94% | 5.225% | 5.603% |
| ml_off_side_1.5 | inventory-v3 | 1.1169 | 3.2239 | -79.95% | 2.108632 | 51.76% | 5.388% | 6.033% |
| ml_off_side_1.5 | carryなし/v3 | 1.0358 | 3.5276 | -78.12% | 2.627799 | 41.98% | 4.594% | 5.181% |
| ml_off_side_1.3 | 旧会計/同入力 | 1.0713 | 3.0949 | -75.02% | 1.275906 | 44.00% | 4.528% | 4.856% |
| ml_off_side_1.3 | inventory-v3 | 1.1191 | 3.2259 | -75.07% | 1.827468 | 44.72% | 4.664% | 5.225% |
| ml_off_side_1.3 | carryなし/v3 | 1.0358 | 3.5276 | -73.13% | 2.277426 | 36.31% | 3.981% | 4.491% |

旧turnover列はtarget-weight差、新turnover列は実効執行volume/2であり、同じ意味の改善率として比較しない。
モデルgross上限は2、abs(net)最大は約4.2e-16。side1.30の目標実効gross上限は2.60、side1.50は3.00。
継承後riskのgross cap 3.00に適合。保存fallback率は両側0で、モデルや監査の再実行をPASSとした値ではない。
このproxy条件のVaR/ESは本番stop .03/.04を超えており、現行ライブriskがpassする根拠には使わない。

費用は日次return fractionの単純合計（実broker費用や複利費用ではない）。

| 系列 | slip | financing | borrow | reverse | total |
|---|---:|---:|---:|---:|---:|
| baseline_ml_on_side_1.5 | 0.779472 | 0.039446 | 0.012064 | 0.076578 | 0.907560 |
| baseline_ml_on_side_1.3 | 0.675536 | 0.034191 | 0.010456 | 0.066375 | 0.786559 |
| ml_off_side_1.5 | 0.775976 | 0.039445 | 0.012063 | 0.076575 | 0.904060 |
| ml_off_side_1.3 | 0.672508 | 0.034191 | 0.010456 | 0.066373 | 0.783528 |

side1.30の新会計ML-on−off平均日次net差は +0.00000800。
凍結weightの会計再計算であり、特徴量providerのPIT証明、未使用OOS、実約定/板/金利等の完全照合は未実施。
採否は保留。旧成績・本番cost/risk設定・発注経路は変更していない。

再現: `/usr/bin/timeout -k 10s 180s .venv/bin/python src/research/scripts/experiments/reevaluate_inventory_accounting.py --output /private/tmp/issue33-repricing-new`
09:10入力がある場合は `--df-exec` と `--open-910` に当時の不変snapshotを指定する。
成果物は[summary](revaluation/summary.json)、[日次比較](revaluation/daily_comparison.csv)、各variantのcash/holdings/terminal CSV・JSON。
標準ExperimentRegistryへ記録し、record_idはsummaryへ保存した。revaluation/registry.jsonlは今回recordのexport証跡。

## 検証

最終差分の対象回帰・全tests・compileall・CI Ruff/import契約・文書参照の結果を以下に追記する。
Mypy 2.3.0は未変更HEADでも同じ4件が失敗する（gap_adjustment.py:195、market_calendar.py:111/187、model_meta.py:82）。
現行端末はpandas/numpy等がlockfileの完全なCI環境と一致していない。新規型エラーはないが、Mypy全体成功とは報告しない。
全コマンドは外側timeout（TERM後10秒でKILL）を使用。テストは既存の直列環境を使い、既存assertion/監査を弱めていない。


- 最終対象回帰: **84 passed**、20.66秒。
- 最終全tests: **1162 passed / 0 failed / 0 skipped / 2 warnings**、208.28秒。外側1800秒、個別test 300秒。
- compileall、CI対象の全Ruff、import契約7件、operational import境界2入口: 成功。
- CI文書集合＋今回ADR/報告の参照161件: 成功。
- Mypy全体: 変更前と同じ4件のエラー。新規エラー0。成功扱いにはしない。
- 保存済み4系列×368日と最終コードの再計算: 最大日次net誤差1.04e-16。
- 既存warningは定数系列の相関とDataFrame断片化。会計監査やテスト閾値は緩和していない。

詳細は[verification.json](verification.json)、[全テストログ](tests.log)、[JUnit](tests.xml)、[型検査](mypy.log)。
元データ不足による完全269日再評価は未実行として残す。本番設定の更新・注文・決済・再送は実施していない。
