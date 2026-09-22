# 段階A〜C 第3回レビュー — 2026-09-13

**判定：BLOCK。主要な再現は改善しましたが、P1が1件、P2が4件残っています。** S01〜S05を全て修正完了とは判定できません。特にML artifactは、必要キーの存在確認に加え、その値と本体との対応を全読込み経路で検証する必要があります。

対象はHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285`上の未commit作業ツリーです。[前回レビュー・実装引継ぎ・ユーザーの修正報告](../20260913_stage_abc_rereview/review.md)と、実装・テスト・更新されたADR/技術文書を照合しました。新しい番号T01〜T05は今回残った問題です。本番コード・設定・テストは変更していません。実口座・外部APIへの照会や注文は行わず、偽broker・一時SQLite・一時artifactで検証しました。

> **修正追記（2026-09-13）:** T01〜T05とVaR/ESのactive artifact fingerprintを修正し、全テストは614 passedで完了した。実装・回帰・運用上のartifact再学習要件は[fix_validation.md](./fix_validation.md)を正本とする。この本文のBLOCK判定と再現内容は、修正前の監査証跡として保持する。

**T01 / P1 — versioned ML artifactでも、train_endがnull/NaTだと学習時点制約を通過します（S04）。**

対象：[ml_order_overlay.py:441](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:441)、[同:789](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:789)。保存側の入口は[同:344](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:344)です。

loaderはrequired keysが存在するかを検査しますが、日付として有効かは検査しません。`train_end=None`なら適用時の比較自体を省略し、`train_end="NaT"`なら日付比較がFalseになって拒否を通過します。CURRENT・model digest・内外metadataの一致検証が成功していても、この値の不適合は残ります。

新しい`save_overlay_model`で作成したversioned artifactを実際のloaderから読み、2021-01-04に`apply_overlay`しました。null、文字列NaTの両方でロード・適用が成功し、最終gross=2.0でした。同じ入力で有効な`train_end=2026-08-14`を渡すと、正しくin-sample例外になります。実モデルを2021年へ遡及して取引した証拠ではなく、「学習終端を検証できないartifactを適用できる」境界再現です。

**具体修正：** save/load/applyに共通する来歴validatorを設け、train_start/train_endをscalarな有効日付へ正規化して、空・null・NaT・不正順序を拒否します。metadata_versionは対応する版かを検査し、data/config hashも非空等の所定の契約を満たすか確認します。loader経由でない直接渡しのモデルも適用時に同じvalidatorへ通してください。拒否すべきartifactを「MLなしの通常判断」に変換してはいけません。

既存の`test_in_sample_overlay_application_fails_closed`は有効な終端日だけの比較です。追加回帰は、実save→load→apply経由でnull/NaT/空文字/開始日>終端日を拒否し、正常artifactのtrain_end当日は拒否・翌取引日は適用可能とするものです。公開前検証で失敗した場合は旧CURRENTを維持します。

証跡：[reproductions.json](./reproductions.json) の `invalid_artifact_dates`。

**T02 / P2 — legacy ML artifactの読込みでは、本体と異なる学習期間を依然として受け入れます（S04）。**

対象：[ml_order_overlay.py:456](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:456)、[同:473](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:473)。

CURRENTがある場合だけ本体digestと内外metadataを検査します。CURRENTなしのroot直下model.pkl/metadata.jsonは、必要キーがあればロードし、外部metadataで本体のmetadataを上書きします。新保存形式のatomic性は改善していますが、既存ファイルを読む互換経路には元の混在問題が残ります。

モデル内のtrain_end=2026-08-14、外部metadataのtrain_end=2020-12-31というlegacy一組を一時ディレクトリに作ると、正常ロードされ、返るモデルは2020-12-31まで学習したものとして扱われました。現行phase2_8がこの状態だという主張ではありません。現行phase2_8は開示どおりキー不足で拒否されます。

**具体修正：** 本番loaderでは未検証legacyを拒否し、検証可能な学習データから新形式で再生成する方針を推奨します。legacyを残すなら、少なくとも内外の学習期間・data/config識別の不一致や本体側の欠落を拒否し、同一性を裏付けられないファイルを成功扱いしないでください。root形式を読めることを、来歴が正しいことの代用にできません。

更新されたADR/ARCHITECTUREはlegacyを既存guardで許可すると記載していますが、そのguardが本体とmetadataを結び付けられないことが今回の問題です。規約を緩めて元のS04を閉じることはできません。既存の版改変テストに、CURRENTなし・内外終端不一致・本体metadataなしを追加し、拒否を確認してください。

証跡：`legacy_artifact_mix`。

**T03 / P2 — metadataなしの.npy互換経路は、実signal日を確認せずPASSEDの通常判断を返します（S01）。**

対象：[decision_engine.py:351](/Users/shonen/leadlag/src/leadlag/models/v2/decision_engine.py:351)。実際の利用元として、[ML学習の_collect_training_data:503](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:503)が`generate_v2_production_portfolio`を呼んでいます。

単一horizon分岐は`dist.metadata is not None`の場合だけ来歴を検査します。SQLiteのmetadataなしは空dictとして拒否されますが、.npyはNoneを返すため検査しません。後続で前営業日を推定し、実入力の日付が分からないまま監査PASSEDになります。

同じμ・Ωを2026-08-14のファイル名で保存し、実際の互換wrapperを呼ぶと、.npyはgross=2.0 / leakage=PASSED / diagnostics=nullでした。同じ数値のSQLite bundleは、未来metadataでも欠落metadataでもgross=0、audit_failure=trueです。**現行の通常MH経路で未来のh=3を通す問題は修正されています。** 今回は、まだML学習でも使う互換経路だけに限定した指摘です。

**具体修正：** Noneを検査の免除条件にせず、同じvalidatorへ渡します。来歴不明cacheは許可されたon-demandで再計算するかflatへ進めます。元の実signal日が追跡できる入力を供給し、ファイル名や前営業日から日付を捏造しません。テスト専用互換が必要なら研究/fixtureとして明示し、本番用・学習用の監査PASSEDに混ざらないよう分離します。

既存の単一horizon未来metadataテストはこのNone分岐を通りません。実.npy→wrapperと実ML学習入口の回帰を追加し、「Noneの.npy」「空dictのSQLite」「有効metadataのSQLite」の結果を比較してください。全fixtureをflatへ変更して正常計算の回帰を失う修正は避けます。

証跡：`missing_distribution_metadata` の `future` / `missing` / `npy_missing`。

**T04 / P2 — 約定詳細の取得失敗が上位へ伝わらず、照合不完了でも正常終了します（S02/S03）。**

対象：[pricing.py:165](/Users/shonen/leadlag/src/leadlag/execution/pricing.py:165)、[post_decision.py:258](/Users/shonen/leadlag/src/leadlag/execution/post_decision.py:258)、[close.py:445](/Users/shonen/leadlag/src/leadlag/execution/close.py:445)。

呼出し元に追加したtry/exceptは、`fetch_fill_prices`内部で捕捉されるエラーを検出できません。同関数は注文へ`fill_status=FETCH_ERROR`を設定して正常returnするため、post-decisionの`reconciliation_errors`は空のままです。closeもFILLEDという注文状態だけから完了と判定し、CLIが0を返します。

偽brokerの注文結果はFILLED、詳細照会はOSErrorとしました。本物のsubmit・収集関数を通すと、2件ともFETCH_ERRORのままpost-decisionが正常returnしました。さらに本物の`close_all_positions`とCLIを通したところ、詳細照会1回が失敗していても`close_incomplete=false / reconciliation_errorsなし / exit=0`でした。休日の早期returnは無効化し、決済処理と詳細照会への到達を確認しています。約定していないことを証明したものではなく、数量・価格の照合が未完了なのに成功として通知される問題です。

**具体修正：** 注文ごとの詳細取得結果を上位へ返し、収集全体でFETCH_ERROR等を集約します。全件を試行した後に結果を持つ照合例外を出す方式か、型付きの収集結果を返す方式に統一してください。最初の1件の失敗で残りの照会を打ち切らず、既に確定した数量は保持します。close runnerは、close summaryに既に記録された`reconciliation_errors`も引き継ぎ、後続のposition/wallet/journalエラーと合わせて判定してください。

回帰ではfetch関数全体を例外モックへ置換するのではなく、brokerの詳細照会だけを失敗させます。FILLEDと取得失敗が併存する保存JSON、照合不完了の通知、client.close、その他snapshotの継続を検査します。既存の「確定数量を再照会失敗で消さない」テストは維持します。

証跡：`fill_failure_silent_success`。

**T05 / P2 — 最初のAPIログ保存が失敗すると、部分約定のsummaryを失って照合を飛ばします（S02）。**

対象：[broker_ops.py:686](/Users/shonen/leadlag/src/leadlag/execution/broker_ops.py:686)、[post_decision.py:232](/Users/shonen/leadlag/src/leadlag/execution/post_decision.py:232)。

`OrderExecutionIncomplete`はログ保存後にだけ作られます。保存がOSErrorになるとsummary付き例外へ到達せず、呼出し元もこの型しか捕捉しないため、約定・建玉・余力収集を飛ばします。注文結果自体は既に取得済みなので、記録の失敗と実行結果を分離する必要があります。

2注文がPARTIALLY_FILLEDという偽brokerで、初回の`_write_api_execution_log`だけをOSErrorにすると、上位へ返るのはそのOSErrorのみで、照合呼出しは0回でした。後処理側の保存エラーを個別に捕捉する修正は、この前段の保存失敗をカバーしません。

**具体修正：** 不完了判定とsummary付きの結果/例外の構築を、最初のログ保存成功に依存させません。ログ保存エラーを実行結果へ添付し、上位へ必ず取得済みsummaryを渡して照合を試みます。保存不能ならloggerにも原因を残し、後続保存も成功扱いせず、元の注文不完了とI/O失敗を両方報告します。ログが壊れたために同じ注文を再送する処理は追加しません。

回帰は部分約定後の初回ログ書込み失敗を注入し、summaryに注文ID・数量・状態が残ること、約定照会・snapshotの試行、元の不完了と保存エラーの保持を確認します。実ディスク障害を起こす必要はありません。

証跡：`initial_log_failure_skips_reconciliation`。

**修正を確認できた部分**

| 前回項目 | 今回の評価 |
|---|---|
| S01：MHの未来signal日 | h=3の未来日付はgross=0、audit_failure=true、h=3の拒否理由ありへ改善。許可されたon-demand再計算の追加回帰も全体テスト対象。ただしT03の互換経路が残る |
| S02：部分失敗後の収集 | 通常の不完了例外ではfills→positions→wallet→journalが実行される。取消済み部分約定は詳細照会され、約定数量30を保持。ただしT04/T05の異常系が残る |
| S03：全件拒否のclose | 休日判定を固定した実CLIで終了コード2へ改善。通常の拒否・不完了通知は修正済み。照合失敗のT04とは別 |
| S04：新形式のatomic公開 | CURRENT公開失敗後も旧version・旧本体・旧train_endをロードできる。digest不一致等の回帰を追加。ただしT01/T02とactive版の扱いが残る |
| S05：JP休場日のpadding | strict/non-strictとも8月12日・13日を保持し、休場行を除いた入力と同じ日付列へ改善。US側のcalendar拡張は行っていない |
| 前回R04：GapStoreの同時読込み | default/3/5の全てで、読込み中のcommitをまたいでもμ・Ω・metadataが同じ旧版、次回読込みは新版となることを再確認 |

上記の再現は[prior_probes.json](./prior_probes.json)に保存しています。旧probeの関数名に「skips」などが残っていても、今回の成否はJSONの実際の観測値で判定しています。例えば`partial_failure_skips_reconciliation`の収集リストは今回4処理あり、旧不具合がそのまま残るという意味ではありません。

**版管理に関する追加の未了確認**

S04の引継ぎにあった「VaRが実際に採用したartifact版へ固定する」接続はまだありません。[var_history.py:63](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:63)はroot以下を全走査し、stagingや非activeな履歴版もfingerprintに入れます。採用versionが同一でも`.staging-interrupted/model.pkl`を置くだけでkeyが変わることを再現しました（`inactive_staging_invalidates_var_cache`）。これは不必要なVaR用BTの再実行を起こす条件です。誤ったPnLを再利用したという実証ではないため、上記の主要5件とは分けています。

本番runnerが保持したモデル、VaR用BTがロードするモデル、cache keyの三つで同じactive version-idを利用する契約を完成させ、staging・未採用版ではkeyを変えない回帰を追加してください。run途中にCURRENTが変わる場合の同一性は、今回の全体テスト成功だけでは保証できません。

**段階A〜Cの判定と文書整合**

段階AはMH監査・全拒否通知・通常の不完了収集が改善した一方、T04/T05が残るため部分完了です。段階BのS05とGapStore境界は改善を確認しました。段階CはT01/T02/T03と、開示済みのartifact準備・本番/BT同一入力でのweights一致検証が残ります。

現行`models/ml_order_overlay/phase2_8`を解決済み本番設定のProductionRunnerで読み込むと、今回もprovenance不足で拒否されます。この安全側の拒否自体を新規不具合として数えていません。実artifactの再学習・移行をこのレビューで勝手に実行することもしていません。

`docs/ARCHITECTURE.md`、`docs/モデル技術仕様書.md`、安全境界ADRへの追記は確認しました。legacy許可はT02と、照合失敗の伝播という記述はT04/T05と整合させる必要があります。`docs/refactor_roadmap.md`の部分完了・未完了も照合しましたが、古いPhaseの状態を今回の修正完了へ読み替えていません。新規のOOS収益実験・段階D〜Fの整備を、今回のS01〜S05修正の追加条件にしているものではありません。

**検証**

全体`tests/`は **605 passed / 17 warnings / 700.81秒** で終了しました。4workers、プロセス全体deadline=1800秒で実行し、watchdogもexit=0（701.5秒）です。unit/integration/research/regressionを含む実行で、部分スイートを全体合格と扱ったものではありません。`git diff --check`もPASSです。

Ruff（src/leadlag、tests、変更対象の研究コード・tools）、mypy（本番120 source files）、compileall（指定roots）、import-linter（154 files / 380 dependencies、4 contracts kept）はPASSです。研究全体のRuffを合格したという意味ではありません。

ログは[pytest_full.log](./pytest_full.log)、[ruff.log](./ruff.log)、[mypy.log](./mypy.log)、[compileall.stderr](./compileall.stderr)、[import_linter.log](./import_linter.log)。再現コードは[probe_review.py](./probe_review.py)、結果は[reproductions.json](./reproductions.json)です。各長時間処理には既存watchdogでプロセスグループ全体の期限を設定しました。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260913_stage_abc_round3_review/probe_review.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260913_stage_abc_rereview/probe_remaining.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n 4 --tb=short -q
```

調査コードはその時点のworking treeをimportします。最終の対象ファイルhash・リンク検証は[review_manifest.json](./review_manifest.json)と[validation.json](./validation.json)、指摘の機械可読一覧は[review_summary.json](./review_summary.json)へ保存しています。今回の結果はA〜Cの修正差分と関連経路についてのレビューであり、実運用の収益性や全コードの無欠陥を保証するものではありません。
