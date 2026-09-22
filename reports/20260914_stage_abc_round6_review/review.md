# 段階A〜C 第6回レビュー — 2026-09-14

**判定：BLOCK。P2が2件残っています。** 前回V01〜V04の元の再現条件はいずれも改善しました。追加した来歴検証とVaR snapshotに、別の入力・失敗条件で問題が残ります。

対象はHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285`上の未commit作業ツリーです。[前回レビュー](../20260914_stage_abc_round5_review/review.md)と[今回の修正報告](../20260914_stage_abc_round5_fix/fix_report.md)を照合し、前回manifestから変更された14ファイルを中心に利用側まで追跡しました。元のF01〜F22と後続R/S/T/Uの関連再現も再実行しました。今回のために本番ソース・設定・テストを変更していません。

**W01 / P2 — 来歴validatorの契約が分かれ、矛盾した日付や不正な型を拒否しきれていません。**

対象：[gap_matrix_io.py:75](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:75)、[同:92](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:92)。比較対象：[distribution_source.py:141](/Users/shonen/leadlag/src/leadlag/models/v2/distribution_source.py:141)。利用側：[gap_io.py:293](/Users/shonen/leadlag/src/leadlag/models/v2/gap_io.py:293)、[signal_enhancement.py:89](/Users/shonen/leadlag/src/leadlag/models/signal_enhancement.py:89)。V01の保存整合性の修正を取り消す必要はありません。

新設した`_bundle_provenance_error`は、`sig_date`があると`signal_date`を確認しません。例えば取引日2026-08-14に対し、以下のmetadataを持つ、digestの正しいbundleを実writerで保存します。

```json
{"sig_date": "2026-08-13", "signal_date": "2026-08-17", "trade_date": "2026-08-14", "horizon": 1}
```

既存の`_validate_distribution_metadata`は日付の矛盾として拒否しますが、新しいloaderのvalidatorは受理します。実`load_gap_matrices`と`compute_distribution`が配列を返し、on-demandは0回です。h=1/3/5で同じ結果でした。h=3/5の`apply_multi_horizon_blend`も警告なしで分布を混ぜます。再現用h1 scoreからの最大変化は2.0でした。この差は無効分布が混合されたことを示すマーカーで、運用成績の推計ではありません。

来歴の型の処理にも差があります。

| metadataの条件 | 互換loader / compute_distribution | 既存validator / 下流 |
|---|---|---|
| 正常なscalar日付 | cacheを利用、on-demand 0回 | 正常に受理 |
| 上記の矛盾する日付alias | cacheを利用、警告なし | 既存validatorは拒否 |
| `trade_date=["2026-08-14"]` | 1要素配列を受理 | 既存validatorはscalar違反として拒否 |
| `trade_date=null` | `AttributeError`、on-demand 0回 | 既存validatorにも同じ例外処理漏れ |
| `horizon=Infinity` | `OverflowError`、on-demand 0回 | 既存validatorにも同じ例外処理漏れ |

InfinityはPython側のwriterが実際に保存できる非有限値の境界として確認しました。null・配列・alias矛盾は通常のJSONで表現可能です。いずれも手作業でmanifestを偽造していません。

**実害と範囲：** `.npy`のdigest不一致・manifest欠落・metadata欠落は今回正しく拒否できています。残る問題はmetadataの意味の検証です。通常`decide`の単一/複数horizon経路には既存validatorによる再検証があるため、今回の矛盾aliasがそのまま通常発注へ通ったとは判定していません。しかし互換API・研究の`_multi_horizon_scores`（内部で`require_provenance=False`）・[Step 2 baseline IR:1015](/Users/shonen/leadlag/tools/research/compute_gap_adjusted_distribution.py:1015)のblendは同じ保証を持ちません。null等では許可されたon-demandへ到達する前に処理が中断します。

**具体修正と受入条件：**

1. 2つのvalidatorを、下位の共通モジュールに置く1つの正規化・検証関数へ統合します。`utils`から`models`をimportして依存を逆転させず、I/Oに依存しない関数を両方から呼びます。例えば`src/leadlag/utils/distribution_provenance.py`を設け、正常なmetadataまたは明示的な拒否理由を返す契約にします。
2. `sig_date`と`signal_date`が両方あれば両方をscalar日付へ正規化し、一致を要求します。null/NaTを`normalize()`より前に拒否します。`trade_date`もscalar・有限な日付であることを確認してから比較し、1要素配列を暗黙にbool化して受理しません。
3. horizonは有限な整数として検証し、不正値を単純な`int(...)`で処理しません。型の許容範囲を決め、想定する入力不正は拒否結果へ変換します。広い`except Exception`で正常扱いする修正は不要です。
4. 非strictの意思決定用loaderは拒否時に配列を返さず、strictでは`DataValidationError`へ揃えます。compute側は無効cacheから許可されたon-demandへ進め、blendは不正horizonを利用せず警告を保持します。既存の正常cache優先・監査失敗時のflat/停止は維持します。
5. 共通テーブルで、正常、alias一致/矛盾、null/NaT、1要素/複数要素配列、非有限horizonを検査します。関数単体だけでなく実loader→compute/blend、通常decideへ通し、h=1/3/5で判断が一致することを確認します。新規unit testの「単独sig_dateが未来なら拒否」だけでは今回の条件を検出できません。

証跡：[boundaries.json](./boundaries.json)の`metadata`、[再現コード](./probe_boundaries.py)。元のbundle保存障害の結果は[reproductions.json](./reproductions.json)の`U01_bundle_boundaries`にあります。

**W02 / P2 — gap snapshotの取得に停止期限がなく、SQLiteのロック待ちがVaRのtimeoutを迂回します。**

対象：[var_history.py:143](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:143)、[snapshot呼出:255](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:255)、[BT timeout:384](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:384)。V04の「同じsnapshotをkeyとBTで使う」という方向は正しく、元のA/B取り違えは解消しています。

`source.backup(target)`は既存の`run_with_timeout`より前に、呼出スレッドで実行されます。3回のretry上限は1回のbackupが返ってからしか働きません。`sqlite3.connect(timeout=30.0)`もbackup全体の停止期限にはなっていません。

実VaR関数・実設定loader・実SQLiteを使い、一時DBの排他ロックを別接続で保持しました。`var_history_timeout=1`を指定しても35秒時点で処理が戻らず、ロックを解除すると35.263秒で正常系列を返し、BTを1回実行しました。snapshot取得不能時に空系列を返すexcept節には、待機中は到達しません。BT数値ループのみマーカーへ置き換えています。

このためDBの排他操作や外部プロセスのロックが長引くと、[日次bridge:478](/Users/shonen/leadlag/src/leadlag/execution/v2_bridge.py:478)のrisk準備も待ち続け、9:10の判定を遅延させます。デフォルト300秒を実時間で超過させる試験は行っていません。1秒設定・30秒の接続timeoutを超えた実測と、timeoutの外にある呼出位置を根拠としています。通常のWAL writerが常にbackupをブロックするという主張でもありません。

**同じsnapshot管理で確認した寿命の問題：** `_gap_snapshot_temp`はcallerのローカル変数で、BT workerのclosureには保持されていません。BTで1秒timeoutを発生させるとcallerは安全側に空系列を返しますが、その時点で一時DBが削除されます。まだ生きているworkerを再開すると、実際に渡されたsnapshot pathは存在しませんでした。コメントの「timeout worker threadを含めて保持する」という保証は成立していません。これにより誤ったVaRがcallerへ返ったとは判定していませんが、継続中の計算の入力を先に回収してしまいます。

**具体修正と受入条件：**

1. risk履歴準備の入口でmonotonicな総deadlineを決め、snapshotのコピー・hash・ロック待ち・retry・BTをその残時間で制限します。backupにもdeadlineを伝え、期限超過時は一貫して空系列/停止へ進めます。各retryに同じ満額timeoutを与えて総待機時間を伸ばしません。
2. SQLite backupを期限付きのworker processで実行して停止・回収する方法、または小さなpage単位のbackupとprogress callbackでdeadlineを検査する方法などを選びます。後者ではbusy待ちも残時間以下に抑える必要があります。`connect(timeout=...)`の変更や、呼出側だけをdaemon threadで打ち切る対応では完了条件を満たしません。
3. snapshotの所有者を実計算workerに持たせ、使用終了後に`finally`で回収します。timeoutでworker自体を停止する設計なら、停止・回収してからsnapshotを削除します。workerが生きている間にcallerのローカル変数の破棄で入力が消えないようにします。
4. 実SQLiteの排他ロックで、総deadline以内に停止側の結果が返ること、ロック解除後に期限切れrunが再開・cache保存しないことを回帰で確認します。正常WAL更新、元のA→B競合、directory snapshotの更新割込み、timeout中のsnapshot寿命も検査します。実運用プロセス・本番DBのロック解除をテスト手順にしません。
5. 既存のsnapshot unit testは「取得完了後にwriterが更新してもAを読む」を確認しています。今回不足するのは取得中のロックと、計算中のtimeoutです。正常snapshotの固定を維持しながら追加します。

証跡：[boundaries.json](./boundaries.json)の`locked_snapshot`と`snapshot_timeout_lifetime`、[再現コード](./probe_boundaries.py)。ロックはprobeが作成した一時DBだけに設定し、解除・接続close・子process回収まで完了しています。

**前回V01〜V04の再確認**

| ID | 今回の結果 |
|---|---|
| V01 bundle拒否の伝播 | default/3/5のΩ・metadata・manifest保存前障害に到達。実3値利用経路では無効配列を使わずon-demandを1回呼ぶ。h3/5 blendはh1を維持し警告を返す。通常decideはgross=0。元の発生条件は解消、追加の来歴境界がW01 |
| V02 注文状態 | 部分失効11→CANCELLED、訂正/取消失敗5/8→SUBMITTED。通常の部分約定は最後までpoll、取消後30株の収集、全拒否の失敗伝播を維持。追加14状態ケースを含むunit回帰も確認 |
| V03 schema cache | 固定4135行×122列・実builderで列名交換後のcache再利用なし。同一モデルとfreshモデルの`all_returns_raw`最大差0。元の数値訂正・target訂正の再計算も維持 |
| V04 VaR gap版 | Aのkey作成後に元DBをBへ更新しても、BTはsnapshot Aを使用。元DBを一時的にAへ戻した再取得もAのマーカー0.01を返す。元の版不一致は解消、snapshotの待機・寿命がW02 |

schemaは[schema_cache.json](./schema_cache.json)、VaRの版は[var_gap_version.json](./var_gap_version.json)を参照してください。VaRの0.01/0.02は版識別の値で、実PnLではありません。

**元のF01〜F22と後続レビューの再発確認**

以下は今回実行した同じ再現条件での結果です。元の全機能の完全性や、全市場データに対する保証ではありません。

| ID | 確認した挙動・残る条件 |
|---|---|
| F01 注文状態 | 元の誤分類とV02は解消。通常SUBMITTED→PARTIALLY_FILLED→FILLEDはsubmit/closeとも3回poll |
| F02 全拒否 | accepted=0、failed=2で正常終了しない。close CLIは2 |
| F03 risk stop | 実stop条件でもflat・削減を許可。増加・反転・新規は停止 |
| F04 監査失敗 | 同日signalのリークFAILEDでgross=0、audit_failure=true。未来h3・metadata欠落もflat。未来target摂動h1/3/5のμ/Ω差0 |
| F05 過去日流用 | 指定取引日がない実bridgeは例外。過去日のrunnerへ置換しない |
| F06 継承設定/VaR | 実Step 2はstrict loaderで本番v2と同じ設定・baseline IRパラメータ。VaRのoverlay/gap版固定は維持。snapshot待機はW02 |
| F07 9:10価格 | 始値1000/現在値1050の偽APIでcurrentだけを取得。実約定照合は未実施 |
| F08 日米休日 | 日本休場08-11のUS+10%が08-12へ対応し、翌08-13は0%。padding後の有効日を保持 |
| F09 strict前処理 | 正常80営業日rawから79行を生成、signal日<取引日 |
| F10 相関/horizon | 数値変更後の相関はfreshと一致。h1→h3とfresh h3のμ/Ω差0 |
| F11 common-input cache | 数値・target・列名訂正を反映。V03の実builder差は0 |
| F12 分布atomic性 | SQLite default/3/5の途中書込みはrollback、読込中commitでも同一snapshot。`.npy`の混在拒否を維持。来歴の追加条件はW01 |
| F13 BTのML | 正常な一時versioned artifactを実BT入口が自動ロードし転送。実本番artifactでの照合は運用未了 |
| F14 決済費用 | flat遷移20.625→9.375bps、符号反転・同weightの費用が在庫フロー手計算と一致 |
| F15 初期DD | −10%,0%のMDD=-10%。関連unit/regressionを再実行 |
| F16 全日集計 | fallback込み4日、総収益-8.2%、MDD-10%、fallback率50% |
| F17 DSR | 245日/252日の両方で同頻度の参照式に一致 |
| F18 A7/V1 | 実mainは明示legacyフラグなしで停止し、データ読込0回 |
| F19 ML来歴 | 不正日付・hash・期間・in-sample適用を拒否。既存本番legacy artifactは拒否され、再生成が必要 |
| F20 fill収集 | FILLED/取消後部分約定の数量保持、detail失敗・初回ログ障害後も後続照合を実施し失敗を伝播 |
| F21 PIT multiplier | 設定0.25を適用。分布取得不能のflatとは区別 |
| F22 fixture | 2026-08-14の固定baselineは非flat。別途同一processでpytest終了後のmacro関数identity復元を確認 |

後続指摘も、R01/S05の休日padding、R02/S02/T04/T05の部分失敗後の照合、R03/S01/T03の分布来歴拒否、R04のSQLite読込snapshot、R05/U02のWAL・overlay版固定、R06/S04/T01/T02のML来歴・公開atomic性、R07の初期DD、R08のテスト汚染、S03のclose終了コード、U03/U04の検証・移行ツールを再確認しました。旧probeの関数名には当時の不具合名が残りますが、結果値は今回のコードで再取得したものです。

**検証範囲と完了判定**

全`tests/`は**639 passed / 17 warnings / 382.22秒**、10workersで成功しました。watchdog exit=0、全体期限1800秒です。Ruff・mypy・compileall・import-linter（4 contracts kept）も成功しました。

結果は[review_summary.json](./review_summary.json)、[pytest_full.log](./pytest_full.log)、[ruff.log](./ruff.log)、[mypy.log](./mypy.log)、[compileall.stderr](./compileall.stderr)、[import_linter.log](./import_linter.log)へ保存しています。pytestの17 warningsは研究テストのゼロ除算・定数列相関です。Ruffは本番・tests・指定した変更研究コード/tools、mypyは本番121 source filesが対象で、研究全域の型保証ではありません。

前回の主要な修正は維持されていますが、W01/W02が残るため段階A〜C全体の完了・リリース可とは判定しません。`docs/refactor_roadmap.md`の部分完了と対象ADRも照合しました。レビュー範囲外のPhaseや新規戦略実験は追加していません。

継承解決後の本番設定はML有効・`models/ml_order_overlay/phase2_8`ですが、同artifactはlegacyとして実runnerが拒否します。検証可能な学習データからのversioned artifact再生成、本番/BTで同一入力・同一artifactを使ったweights照合は引き続き運用未了です。この安全側の拒否を新規不具合に数えていません。

正常固定fixtureのmodel gross=1.5、net≒0、設定side_leverage=1.5を適用した対応値は実効gross=2.25、net≒0です。モデル上限gross 2.0/net±0.05および設定の実効gross上限3.0/net±0.05以内です。片道slippage 5bps等の解決済みcost設定は[contracts.json](./contracts.json)に残しています。実口数丸め・実約定後exposure・新たなOOS収益評価は今回検証していません。未来target摂動はmacro無効の合成入力で、外部系列全体の公表時刻を証明するものではありません。

**再現方法と証跡の整合**

リポジトリルートで以下を実行します。各watchdogはprocess group全体に停止期限を設定します。追加のロック再現は自分で作成した一時DBだけを使用します。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 240 .venv/bin/python reports/20260914_stage_abc_round6_review/probe_review.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 70 .venv/bin/python reports/20260914_stage_abc_round6_review/probe_boundaries.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260914_stage_abc_round6_review/probe_schema_cache.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python reports/20260914_stage_abc_round6_review/probe_var_gap_version.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260914_stage_abc_round6_review/probe_contracts.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_model.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260913_stage_abc_rereview/probe_remaining.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n auto --tb=short -q
```

開始・終了時のソースhashは[source_snapshot.json](./source_snapshot.json)と[review_manifest.json](./review_manifest.json)、リンク・証跡・判定の整合は[validation.json](./validation.json)を参照してください。意図的な保存障害・risk stop・timeoutのログは異常系を検証した証跡で、全テストの失敗と混同しません。
