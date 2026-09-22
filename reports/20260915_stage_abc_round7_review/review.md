# 段階A〜C 第7回レビュー — 2026-09-15

**判定：CONDITIONAL PASS。今回確認したP2（X01/X02）は修正済みです。** 無効な本番ML artifactの再生成と本番/BT weights照合は運用上の未完了として残ります。

対象はHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285`上の未commit作業ツリーです。[第6回レビュー](../20260914_stage_abc_round6_review/review.md)と[修正報告](../20260914_stage_abc_round6_fix/fix_report.md)を照合し、前回manifestから変更された7ファイルと利用経路をレビューしました。今回の依頼に従い、実装4ファイルと回帰テスト2ファイルを修正し、元のF01〜F22・後続R/S/T/U/Vの関連再現も再実行しています。変更対象とhashは[source_snapshot.json](./source_snapshot.json)・[review_manifest.json](./review_manifest.json)に記録しています。

**X01（解消確認） — 文字列`"NaT"`・空文字の日付をvalidatorが例外化しない。**

対象は[distribution_provenance.py:31](/Users/shonen/leadlag/src/leadlag/utils/distribution_provenance.py:31)です。`pd.to_datetime`後の値を`pd.Timestamp`へ変換した直後、timezone除去と`normalize()`の前に共通の`_missing()`で検査するよう修正しました。`"NaT"`と`""`は`pd.NaT`へ変換されても`AttributeError`にならず、`"sig_date is invalid"`として返ります。SQLite/.npyのloaderと`distribution_source`は同じvalidatorを使うため、strictでは`DataValidationError`、non-strictでは配列なしの拒否、computeでは許可されたon-demand/fallbackへ統一されます。

受入確認は[provenance.json](./provenance.json)の72ケース（2保存方式×複数日付・h=1/3/5）で、**72/72 pass、失敗0**です。[boundaries.json](./boundaries.json)でも`nat_signal_*`/`empty_signal_*`はloader拒否・on-demand 1回・h3/h5 blendの警告付きskipを確認しました。回帰テストには文字列`"NaT"`と空文字を追加しています。

**X02（解消確認） — VaR/ESの総deadlineを準備から結果採用まで適用した。**

対象は[var_history.py:33](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:33)、[snapshot:163](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:163)、[cache hit:532](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:532)です。設定値から絶対時刻のdeadlineを一度だけ作り、cache初期化、`df_exec`/effective config読込、gap snapshot、overlay・dataframe・code hash、cache read、BT、cache write、最終系列返却に残時間を渡しました。期限切れは空系列へ戻し、cache hitでも返却前に残時間を確認します。
cache writeは終了可能なfork/spawn子プロセスへ隔離し、期限超過時にterminate/killして遅延した書込みを共有SQLiteへ公開しないようにしました。子プロセスには残り時間をSQLiteのbusy timeoutとして渡しています。

ディレクトリ入力は`shutil.copytree/copy2`の一括呼出しを廃止し、`_copy_directory_with_deadline`/`_copy_file_with_deadline`でファイルをチャンク単位にコピーし各チャンク前後で期限を確認します。hashも1MiBチャンクごとに確認します。BT timeout後のsnapshot所有権は従来どおりworkerへ移し、worker終了時にcleanupします。これにより遅延workerが使用中の一時入力をcallerが先に削除しません。

[deadline.json](./deadline.json)の故障注入結果は、1秒設定で1.5秒のcode hash、cache get遅延のいずれも約1.01秒で空系列へ戻り、backtestを追加実行しません。0.1秒の遅延cache writeは約0.11秒で子プロセスを終了し、遅れて値が公開されないことを確認しました。directory copyは期限確認3回目で`TimeoutError`となり、`copytree_bypassed=true`です。SQLite lock、cache/config/worker例外、worker遅延後のsnapshot回収も同一証跡で確認しました。これは実ストレージの速度保証ではなく、期限切れ結果を採用しないことと、チャンク境界で停止することの検証です。OSレベルのread自体が停止不能な場合に備え、外側watchdogは維持します。

**確認できた修正**

| 対象 | 今回の結果 |
|---|---|
| X01/W01の来歴validator | `NaT`/空文字を含む72ケースが全て契約どおり拒否・fallback。strict/non-strict、SQLite/.npy、h1/3/5を確認 |
| X02/W02の総deadline | cache初期化・入力load・snapshot・hash・cache read/write・BT・結果採用を同一絶対deadlineで制限。遅延hitは空系列、directory copyは期限確認で停止 |
| W02のSQLite排他ロック | 1秒設定で約1.06秒で空系列、BT 0回。旧35秒待機は解消 |
| W02のworker寿命 | timeout後も一時DBが存在し、worker終了後に削除。設定例外・cache例外・worker例外でも回収 |
| V01のbundle保存障害 | Ω/metadata/manifest保存前の失敗を実注入し、不整合配列を使わずon-demandへ。正常bundleは非flat |
| V02の注文状態 | 部分失効はCANCELLED、訂正/取消失敗はpending。通常の部分約定は最後までpoll、取消後の30株を保持 |
| V03のschema cache | 実4135行×122列fixtureで列名交換後の再計算を確認。fresh builderとの差0 |
| V04/U02の入力版 | gap A→B競合でもBTはsnapshot A、overlayも選択済みobjectを固定。cacheの版不一致は再発せず |

これらは[reproductions.json](./reproductions.json)、[schema_cache.json](./schema_cache.json)、[var_gap_version.json](./var_gap_version.json)、[boundaries.json](./boundaries.json)、[deadline.json](./deadline.json)で確認できます。版識別の0.01/0.02は実運用の収益率ではありません。

**元のF01〜F22と後続レビューの確認範囲**

| ID | 今回の確認 |
|---|---|
| F01 | 注文状態変換、部分約定poll継続、失効・取消終端を確認 |
| F02 | 全拒否でaccepted=0/failed=2、正常終了しない。close CLI=2 |
| F03 | 実risk stop条件でflat・削減を許可、増加・反転・新規は停止 |
| F04 | 同日signalのリークFAILEDでgross=0・audit_failure=true。未来h3・metadata欠落もflat。未来target摂動h1/3/5のμ/Ω差0 |
| F05 | 要求取引日がdf_execにない場合は実bridgeが拒否 |
| F06 | 実Step 2の継承設定・baseline IRパラメータ一致。VaR入力版固定と総deadlineを確認 |
| F07 | 偽APIの始値1000/現在値1050でcurrentを取得、実約定照合は未実施 |
| F08 | 日米休日対応と休場padding後の有効日保持 |
| F09 | 正常80営業日rawからstrict前処理79行、全signal日<取引日 |
| F10 | 相関入力訂正、horizon切替、未来target摂動でfreshとの一致 |
| F11 | 数値・target・列名訂正後にcommon-input cacheを無効化 |
| F12 | SQLite書込rollback/読込snapshot、`.npy`混在拒否と文字列NaT/空文字の拒否を確認 |
| F13 | 実BT入口で正常な一時versioned artifactを自動ロード・転送。実本番artifactの照合は運用未了 |
| F14 | flat/符号反転/同weightの費用が在庫フロー手計算と一致 |
| F15 | 初期−10%,0%のMDD=-10%、関連回帰を実行 |
| F16 | fallback込み4日、総収益-8.2%、MDD-10%、fallback率50% |
| F17 | DSRの245日/252日が各同頻度の参照式に一致 |
| F18 | A7は明示legacyフラグなしで停止、データ読込0回 |
| F19 | MLの不正日付・期間・hash・in-sampleを拒否。実本番legacy artifactは引き続き拒否 |
| F20 | FILLED/取消後部分約定の数量収集と、detail・初回ログ失敗後の後続照合 |
| F21 | PIT fallback_multiplier=0.25を実適用 |
| F22 | 固定baselineが非flat。同一processでpytest終了後のmacro関数identity復元 |

R/S/T/Uの関連確認も再実行しました。休日padding、SQLiteの同一snapshot、部分失敗後のfills/positions/wallet/journal、close終了コード、ML来歴・公開atomic性、fixtureの汚染防止、WAL fingerprint、overlay版固定、検証・移行ツールの旧再現は維持しています。旧probeの不具合名は当時の識別名で、結果は今回のコードで取得しています。

**検証結果と限界**

全`tests/`は**650 passed / 17 warnings / 706.66秒**、10workersで成功しました。watchdog exit=0、全体期限1800秒です。Ruff・mypy・compileall・import-linter・`git diff --check`・変更batchの`bash -n`も成功しました。

確定結果は[pytest_full.log](./pytest_full.log)と[review_summary.json](./review_summary.json)、静的検査は[ruff.log](./ruff.log)・[mypy.log](./mypy.log)・[compileall.stderr](./compileall.stderr)・[import_linter.log](./import_linter.log)に保存しています。Ruffは本番・tests・指定した変更研究コード/tools、mypyは本番122 files、import-linterは4 contractsが対象です。研究コード全域のlint/型保証ではありません。pytestの17 warningsは研究テストのゼロ除算・定数列相関です。

来歴72ケースは72件すべて契約を満たしました。probeのexit=0は調査結果を正常に収集したことを示し、外部APIや実市場状態まで保証するものではありません。[validation.json](./validation.json)は証跡と修正内容の整合を検査します。

段階A〜Cのコード上の確認範囲ではX01/X02を解消し、条件付きで通過と判定します。運用未了のML artifact再生成と本番/BT weights照合を完了するまで、本番昇格可とは判定しません。`docs/refactor_roadmap.md`と対象ADRを照合し、対象外Phaseの改修や新規戦略実験は追加していません。

本番設定はML有効・`models/ml_order_overlay/phase2_8`ですが、そのartifactはlegacyとして実runnerが拒否します。検証可能な学習データからのversioned artifact再生成、本番/BTで同じ入力・同じartifactを使ったweights照合は引き続き未了です。この安全側の拒否を新規不具合に数えていません。

固定fixtureのmodel gross=1.5、net≒0、side_leverage=1.5適用の対応値は実効gross=2.25、net≒0です。モデル上限gross 2.0/net±0.05と設定の実効gross上限3.0/net±0.05以内です。片道slippage 5bps等の解決済みcost設定は[contracts.json](./contracts.json)に記録しています。実口数丸め・実約定後exposure・新規OOS成績は検証していません。未来target摂動はmacro無効の合成入力で、全外部系列の公表時刻を証明するものではありません。

**再現と追跡**

リポジトリルートから実行します。watchdogはprocess group全体の停止期限を設定します。実口座への発注・決済・照会、本番DBのロック・編集は行っていません。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 240 .venv/bin/python reports/20260915_stage_abc_round7_review/probe_review.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 70 .venv/bin/python reports/20260915_stage_abc_round7_review/probe_boundaries.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260915_stage_abc_round7_review/probe_provenance.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260915_stage_abc_round7_review/probe_deadline.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260915_stage_abc_round7_review/probe_schema_cache.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260915_stage_abc_round7_review/probe_var_gap_version.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260915_stage_abc_round7_review/probe_contracts.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_model.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260913_stage_abc_rereview/probe_remaining.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n auto --tb=short -q
```

開始・終了時hashは[source_snapshot.json](./source_snapshot.json)と[review_manifest.json](./review_manifest.json)、リンクと証跡の整合は[validation.json](./validation.json)を参照してください。
