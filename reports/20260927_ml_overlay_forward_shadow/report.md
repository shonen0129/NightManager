# ML overlay の前向き増分価値測定

状態: **PENDING_REAL_EXECUTION_COSTS**

まず既知の368日OOSを片道5/10/20bpsで再価格付けし、次の市場日から同じ本番入力でML有効/無効の判断を記録する仕組みを追加した。既知期間の再集計はML有効の累積net差が5/10/20bpsでそれぞれ+0.0432 / +0.0397 / +0.0327だった一方、net Sharpeは5bpsでML有効の方が0.016低い。10/20bpsのpaired差の95%区間は0をまたぎ、実約定費用は含まない。詳細は[コスト感度レポート](../20260927_ml_overlay_cost_sensitivity/report.md)。

## 前向き記録

`ProductionRunner`で現在の本番ML有効判断を一度計算し、その同一 `DecisionInputs` からoverlayだけを無効にした対照判断を計算する。対照結果は発注経路へ渡さない。日次ペアを `var/shadow_runs/ml_overlay_value/daily.jsonl` に追記し、日付、09:10時点、known/history fingerprint、effective gap distribution hash、両config fingerprint、artifact metadata、価格、資本、weights、scores、監査/fallbackを保持する。同じ内容の再実行は重複追記しない。

`run_decision_v2.sh` は09:10:00〜09:10:30 JSTの読み取り専用17 ETF quote/板captureを `var/shadow_runs/ml_overlay_value/microstructure/quote_snapshots.jsonl` に保存する設定である。captureや対照計算が失敗しても、ML有効の本番decisionはそのまま続く。shadowフック自体はlive価格を伴うAPI有効run以外ではskipする。

## 採否と未完了

現行artifact `20260926T192555935698Z-ee306a32f3ec` の固定比較として250以上の完全paired forward取引日を待つ。paired net PnL差の20日non-circular block bootstrap 95%区間下限>0、net Sharpe差>0、最大DD悪化≤1.0pp、turnover増加≤10%、全日の監査・fallback/market-neutral条件を事前ゲートとする。仕様は[前向きシャドー判断記録](../../docs/decisions/2026-09-27-ml-overlay-forward-shadow.md)。

この日次レコードは予測と対照weightsの保存までで、実現収益を付与しない。candidateの約定・手数料・金利/貸株/逆日歩はfill ledgerと照合し、baseline側は同じ資本・ロット条件と当日板を使ったcounterfactual replayで推定する必要がある。baseline側費用は実約定ではないため、実測と推定を分けて表示する。現状の公式CSVは注文IDや費用明細が全約定分そろわず、日次シャドーだけで「実費込みの増分」を確定できない。

2026-09-28 00:39 JSTの確認で、`com.leadlag.decision` がすでに平日09:10のLaunchAgentとしてロード済みと分かった。独立のshadow Agentを同時刻に起動すると、既存ジョブと同じ `live:production_v2` single-flight guardを競合するため、試験登録した `com.leadlag.ml-overlay-shadow` は直ちにbootoutし、生成したplistも削除した。既存の `run_decision_v2.sh` にはペア記録フックがあり、同一decision入力からML有効/無効を追記する。`--shadow-only` のCLI経路は発注・建玉照会・ポートフォリオ出力前に終了する安全な実行モードとして追加したが、同時刻の別Agentは登録していない。

既存decision Agentの `launchctl print` は `runs=2`、`last exit code=1`。直近の9/25ログではgap分布計算が終了コード2となり、続くVaR/ES履歴計算が300秒でtimeout、リスク停止が発注をブロックしていた。新しいペアフックが入る前のログなので、9/28の実行がシャドー行を残せるかは初回ログとJSONLで確認する。現在、`daily.jsonl` はまだなく、実約定照合と実費評価も未着手。既存のorder-capable Agentの挙動は変更していない。

変更検証: 対象8テストと全913テスト通過（16 warnings、全体115.06秒）、compileall、変更対象ruff、151ファイルのmypy、architecture import contracts、bash構文、`git diff --check`が成功した。全体ruffにはリポジトリ内102件の既存指摘があるため、変更対象ファイルを個別に確認した。全体pytestは終了猶予10秒付きの900秒プロセス期限内に完了した。
