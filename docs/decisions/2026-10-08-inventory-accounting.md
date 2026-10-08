# 日次在庫会計 inventory-v3（Issue #33）

## 判断

`core.pnl.simulate_daily_pnl` を目標weightからcash・値洗い在庫を接続する日次会計の正本とする。
在庫はsigned effective notional（initial_cashと同じ通貨、標準は初期資本1）で表現する。
独立した実約定FIFO ledgerは観測Fillを評価する用途であり、バックテストの価格・費用仮定を実約定へ二重適用しない。

前日close NAVを当日の目標notionalの基準とする。entry時の持越し値洗いを反映してから差分売買する。
carry fractionは保有数量の割合なので、翌日に渡す値は `alpha * target_notional * (1 + entry_to_close_return)`。
weightの割合だけを次日へ渡すと価格変動を消し、翌朝売買量と残高を実在庫に接続できない。

close→openの損益は前日close在庫、open→09:10はそのopen評価額に掛ける。
2区間を受入日の損益へ計上し、当日のentry→close損益と連結する。翌日flatでもcarry損益と決済費用は残る。
金利・貸株・逆日歩は前日close在庫に対して受入日までの暦日数を課金し、評価終端の後へ費用を仮計上しない。
従来の持越し元への帰属と最終日の架空の1日分費用は廃止する。

初期在庫が非zeroなら `initial_mark_date` を必須とする。`initial_cash`、`initial_holdings`、補助turnoverの
`initial_target_weights` を明示入力し、最初の受入日にも値洗い・暦日費用・リバランスを適用する。

## 終端とartifact境界

標準 `terminal_policy=liquidate` は最終観測closeで全解消し、そのnotionalのslippageを最終行へ計上する。
`open_inventory` はcarry数量を最終closeで値洗いした状態で保持する。いずれも `terminal_inventory` に
holdings、cash、equity、評価日・TSE close mark、最終日費用・closing volumeを出す。
観測のない翌日flat行を足さない。入力は日次markなので、実約定timestampを証明する出力ではない。

VaR履歴は各artifact区間を `open_inventory` で評価し、終端holdings/cash/mark date/target weightsを次区間へ渡す。
最終区間もopen inventoryで終え、継続運用の標本へ人為的な最終全解消を混ぜない。
研究paired replayは中間区間を引き継ぎ、評価全体の最終区間だけ決済する。artifact変更と在庫resetを区別する。

## 指標と価格境界

opening/closing volumeは価格で値洗いした実効売買notionalの絶対値合計を前日close NAVで割った値。
`daily_execution_volume` は両者の合計、`daily_turnover` はその1/2。
slippageはexecution volume × 片道slippageであり、target-weight差には掛けない。
`daily_target_weight_turnover` はモデル目標weight差L1/2で、初期行は指定されたinitial target（標準zero）から測る。
モデルnet/gross exposureとside leverage後の目標effective exposureも分けて出す。値洗い後のcarry在庫は目標exposureとは別の量である。

`BacktestEngine._compute_price_intervals` はtarget、寄付gap、open→entryを別配列で返す。
09:10実測がないcellはtargetとcarryの双方で同じ寄付proxyを使う。欠損gap/targetがactive在庫にかかる場合は非有限を残し、
リスク評価では失敗として扱う。ゼロ在庫に掛かる欠損だけを価格損益0とする。
未来価格は会計へだけ渡し、decisionへcash・carryリターンを戻す経路は追加しない。
既存のstrictly historical統計、2010–2014 prior、監査・fallback・riskの閾値は変更しない。

CSV・SQLiteのfull result cacheに日次cash/holdings/分解損益と会計versionを保存する。
旧成績・旧turnoverの無印更新はせず、[再評価・検証](../../reports/20261008_issue33_inventory/report.md)に比較と不足入力を記録する。
