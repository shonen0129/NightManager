**段階A〜C 第8回レビュー — 2026-09-15**

**新規の指摘はありません。今回の修正X01・X02は、下記の受入条件でPASSです。段階A〜C全体の判定はCONDITIONAL PASSとします。** 以前から残る本番ML artifactの再生成と本番/BT weights照合が未了のため、本番昇格可とは判定していません。

対象はHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285`上の未commit作業ツリーです。第7回の開始時点から変更されたのは本番ソース2ファイルとテスト2ファイルです。今回のレビューではソース・設定・既存テストを変更していません。新しい証拠をこのディレクトリに保存しました。

比較した4ファイルは`src/leadlag/utils/distribution_provenance.py`、`src/leadlag/execution/var_history.py`、`tests/unit/test_gap_matrix_io.py`、`tests/unit/test_stage_abc_followup_fixes.py`です。[第7回レポート](../20260915_stage_abc_round7_review/review.md)と終了manifestはユーザーの修正時に更新されています。そのため「前回開始時のsnapshotとの差分」と「更新済み終了manifestとの差分」を分けて記録しています。後者との差分はありません。[開始時snapshot](./source_snapshot.json)と[終了時manifest](./review_manifest.json)を参照してください。

**X01：文字列NaT・空文字の扱いは解消を確認しました。**

[共通validator](/Users/shonen/leadlag/src/leadlag/utils/distribution_provenance.py:31)では、文字列をTimestampへ変換した直後に欠損を判定し、timezone処理・`normalize()`の前に拒否しています。前回の`AttributeError`は再現しませんでした。拒否時に配列を渡さないnon-strict loader、`DataValidationError`を返すstrict loader、許可されたon-demandを呼ぶmodel、無効な追加horizonを警告付きで除外するblendまで追跡しました。

| 受入条件 | 今回の確認結果・証拠 |
|---|---|
| SQLite / `.npy`、h=1/3/5で契約を統一する | 従来の来歴72ケースは72件ともPASS。[provenance.json](./provenance.json) |
| 正常日付・一致するtimezone付きaliasを維持する | 正常値、aliasのみ、一致aliasを両保存方式で受理。矛盾alias・null・配列・Infは拒否 |
| NaT・空文字でも下流fallbackに進む | 各horizonでon-demandを1回実行。h3/h5 blendは既存scoresを保ち、警告を返す。[boundaries.json](./boundaries.json) |
| 他の日付フィールドでも同じ原因が再発しない | `sig_date` / `signal_date` / metadata `trade_date`にNaT文字列・空文字を入れた追加loader36ケースは全件PASS。[additional.json](./additional.json) |
| 要求日付・欠損オブジェクトでも例外化しない | 共通validatorに対し4フィールド×5欠損表現の20ケースを確認。None、pd.NaT、NumPy NaT、NaT文字列、空文字を拒否 |

追加36/20ケースは従来ケースと一部重複しています。要求日付の確認は共通validatorの契約であり、すべての公開CLIが不正な要求日付を同じ例外型にすることを保証するものではありません。

**X02：VaRの待機期限と遅延処理の扱いは、前回の再現条件で解消を確認しました。**

[VaR入口](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:326)は絶対時刻の期限を一度作り、cache初期化・入力読込・snapshot・hash・cache readを残時間付きの待機に通しています。cache hitも返却前に期限を確認します。[保存処理](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:286)は子プロセスに分離され、待機期限を超えた子を終了します。コピー・hashはチャンク境界でも期限を確認しています。

| 受入条件 | 今回の実測結果 |
|---|---|
| cold/warm双方で遅いhashを採用しない | 1秒設定・1.5秒遅延を注入し、いずれも約1.006秒で空系列。coldのBTは0回、warmでもBT追加なし |
| 遅いcache hitを採用しない | 1秒設定・1.5秒get遅延で約1.005秒で空系列。同一キーのcache hitでも遅れた値を返さない |
| コピー自体が待機しても呼出側が止まらない | `.npy`入力のファイルコピーをEventで保持。0.25秒設定で約0.260秒で空系列。解除後に一時入力を回収し、BT・cache保存なし |
| config読込・cache初期化の待機を制限する | 各境界をEventで保持し、0.25秒設定で約0.260秒で空系列。解除後も計算結果を保存しない |
| SQLite snapshotの排他ロックを制限する | 一時DBに実排他ロックを保持したまま、1秒設定で空系列・BT 0回。従来の35秒待機は再発せず |
| BT timeout後もworkerの入力を壊さない | 呼出側が戻ってもsnapshotは存在。worker再開時も読め、終了後に削除。config/cache/worker例外でも回収 |
| 保存用子プロセスが正常に動く | 3000行のDataFrameを3回保存。途中で親がcacheを読み込んだ後も、保存値は元の値と完全一致 |
| 実SQLiteの書込ロック後に遅延保存しない | 一時DBで`BEGIN IMMEDIATE`を保持。0.15秒設定で約0.155秒後にTimeoutError。解除後も対象キーは存在せず |
| 子プロセスの失敗で既存値を壊さない | シリアライズ不能値を渡す故障注入で親へRuntimeError。既存cache値を保持し、新たな子プロセス残存なし |

証拠は[deadline.json](./deadline.json)、[boundaries.json](./boundaries.json)、[additional.json](./additional.json)です。遅延・例外は一時入力への故障注入です。実ストレージの応答速度を計測した結果ではありません。`additional.stderr`内の子プロセスのTypeErrorは、シリアライズ障害の受入確認で意図したものです。

この期限は呼出側の待機を打ち切るもので、すべての背景threadを強制終了する機構ではありません。OS内のreadやBTが永久停止する場合は外側watchdogが必要です。今回確認した「解除後の回収」と「遅延結果を採用しない」範囲でX02を解消と判定します。実行環境はmacOS / Python 3.12です。Windowsのspawn経路は実行していません。

**元の問題と後続レビューの再発条件も再実行しました。**

以下は今回のコードから再取得した証拠です。以前のJSONを成功結果として転用していません。過去のprobe名に不具合名が残っていても、現在も不具合があるという意味ではありません。

| ID | 確認内容と結果 |
|---|---|
| F01 | 注文状態変換、部分約定のpoll継続、取消・失効の終端判定を維持 |
| F02 | 全拒否はaccepted=0 / failed=2で正常終了しない。close CLIも終了コード2 |
| F03 | risk stop時もflat・削減は可能。増加・反転・新規は停止 |
| F04 | 同日signalによるリーク失敗はgross=0 / audit_failure=true。未来h3も拒否。未来target摂動h1/3/5でμ/Ω差0 |
| F05 | 要求取引日がdf_execに存在しない場合、実bridgeが拒否 |
| F06 | 実Step 2の継承解決済み設定とbaselineパラメータが本番と一致。VaR入力の版固定を維持 |
| F07 | 偽APIで始値1000・現在値1050を返し、現在値を使用することを確認 |
| F08 | 日米休日対応、休場padding後の有効日を保持 |
| F09 | 正常raw 80日からstrict前処理79行、全signal日が取引日より前 |
| F10 | 数値訂正後の相関cacheを更新。horizon切替後もfresh計算と一致 |
| F11 | 数値・target・列名訂正後にcacheを無効化。実4135行×122列fixtureの列名交換後もfreshとの差0 |
| F12 | SQLite書込rollback・同一snapshot読込、`.npy`保存の部分失敗・混在拒否を維持 |
| F13 | 実BT入口で有効な一時versioned artifactを自動ロード・転送。本番artifactについては下記の運用未了あり |
| F14 | flat・符号反転・同weight時の費用が在庫フローの手計算と一致 |
| F15 | 初期−10%・次日0%のMDDが−10% |
| F16 | flatを含む4日を集計、総収益−8.2% / MDD−10% / fallback率50% |
| F17 | DSRの245日・252日が、それぞれ同頻度の参照式と一致 |
| F18 | A7の既定入口がlegacy実行を拒否し、データ読込0回 |
| F19 | ML来歴・学習期間・hash・in-sampleを検証。不正artifactを拒否 |
| F20 | FILLED・取消後部分約定の数量を収集。detail/初回ログ失敗後も必要な照合を継続 |
| F21 | PIT履歴不足時のfallback_multiplier=0.25を適用 |
| F22 | 固定baselineは非flat。同一process内のpytest後にmacro関数のidentityを復元 |

主な証拠は[reproductions.json](./reproductions.json)、[model_probes.log](./model_probes.log)、[prior_probes.log](./prior_probes.log)、[contracts.json](./contracts.json)、[schema_cache.json](./schema_cache.json)です。IDごとの参照は[review_summary.json](./review_summary.json)に記録しています。

R〜Wの関連条件では、部分失敗後のfills/positions/wallet/journal収集、close失敗の終了コード、artifactのatomic公開・検証・移行、WAL更新のfingerprint、alias整合を再確認しました。V01の不整合bundleはon-demandへ進み、h3/h5 blendは除外します。V02の部分失効はCANCELLED、訂正/取消失敗はpendingのままです。V03の列名訂正によるcache混同も再発しませんでした。

V04/U02の競合では、元入力をA→Bに更新してもBTは取得済みsnapshot Aを使用し、overlayも選択済みobjectを保持しました。Aを復元した2回目は同じキーのAのcacheを再利用し、BT追加なしです。[var_gap_version.json](./var_gap_version.json)の0.01等は入力版の識別値で、実運用の収益ではありません。

**検証の終了条件と結果を記録します。**

全`tests/`は **650 passed / 17 warnings / 705.34秒**、10workersで成功しました。全体期限1800秒のwatchdogもexit=0です。[pytest_full.log](./pytest_full.log)・[停止期限と終了コード](./pytest_full.stderr)に記録しています。

Ruffは本番・tests・指定した変更研究コード/toolsの範囲で成功、mypyは本番122ファイルで成功、compileallは`src/leadlag tests tools scripts src/research`で成功しました。import-linterは4 contracts kept / 0 brokenです。`git diff --check`と変更batchの`bash -n`も成功しました。[Ruff](./ruff.log)・[mypy](./mypy.log)・[compileall](./compileall.stderr)・[依存関係検査](./import_linter.log)を参照してください。研究コード全域のlint・型を保証する結果ではありません。

[validation.json](./validation.json)では、probeの終了コードだけでなく値・例外・fallback・cache identity等の期待値を検査し、レポートのリンクと証拠hashも確認しています。全テスト、前回の再現条件、正常系の反証、後始末の確認をこの修正の終了条件としました。この範囲では追加修正を要求する根拠は見つかっていません。

**残るのは、以前から分けている本番運用の確認です。**

本番設定はML有効、artifactは`models/ml_order_overlay/phase2_8`ですが、現行loader/runnerはこれをlegacyとして拒否します。この拒否が続くことを実確認しました。検証可能な学習データからversioned artifactを再生成し、その同一artifact・同一入力で本番/BT weightsを照合する作業が残ります。今回の修正で生じた新規不具合には数えていません。

本番設定は監査失敗fallbackとon-demandを有効に保ち、片道slippageは5bps、side_leverageは1.5です。固定fixtureではモデルgross=1.5 / net≒0、倍率適用の対応値は実効gross=2.25 / net≒0で、モデル上限gross≤2.0 / net±0.05と設定の実効gross上限3.0 / net±0.05を満たします。解決済みcost設定は[contracts.json](./contracts.json)に保存しています。口数丸め・実約定後のexposureを証明した結果ではありません。

実口座の発注・決済・照会、本番DBへの故障注入、新規OOS収益実験は行っていません。VaRのprobeでは実cache・一時DBを使用し、重いBT数値ループだけを版識別markerまたは待機workerに置換しています。未来target摂動はmacro無効の合成入力であり、すべての外部系列の公表時刻を検証したものではありません。`docs/refactor_roadmap.md`と対象ADRも照合しましたが、範囲外Phaseの未了を今回の追加修正にはしていません。

「他にバグが存在しない」とは断言できません。今回言えるのは、**X01・X02の修正と記載した過去の再発条件は検証を通過し、このレビューで追加の不具合は確認されなかった**ということです。
