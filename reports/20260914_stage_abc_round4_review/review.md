# 段階A〜C 第4回レビュー — 2026-09-14

**判定：BLOCK。前回T01〜T05の再現条件は解消しましたが、関連する保存・版選択・検証ツールにP2が3件、P3が1件残っています。** 前回のP1（無効な学習終端を持つML artifactの適用）は修正を確認しました。今回の残存問題を同じP1として扱っているわけではありません。

対象はHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285`上の未commit作業ツリーです。[第3回レビュー](../20260913_stage_abc_round3_review/review.md)、[修正報告](../20260913_stage_abc_round3_review/fix_validation.md)、実装・テスト・ADRを照合しました。追跡対象の変更は54ファイルで、前回manifestとの比較では文書・fixtureを含め22ファイルが更新・追加されています。レビュー中に本番ソース・設定・テストは変更していません。

全体テストは **614 passed / 17 warnings / 373.52秒**、Ruff・mypy・compileall・import-linterも成功しました。以下は既存テストに含まれていない条件を、一時ファイル・一時SQLite・偽brokerで再現した結果です。実口座への発注・決済・照会は行っていません。

**U01 / P2 — `.npy`の再保存に失敗すると、新しい行列を古い来歴で監査PASSEDにできます。**

対象：[gap_matrix_io.py:456](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:456)、[同:459](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:459)、[読込側:323](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:323)。T03の修正で追加されたsidecarと、従来の行列上書きの接続に問題があります。

`save_gap_matrices`は同じ日付のμ、Ωを順番に上書きし、その後にmetadataを書きます。旧sidecarを失効させず、読込時にもsidecarと実際の配列の同一性を検証しません。そのため、初回作成時には安全でも、同日再実行・部分書込み失敗・metadataなしの上書きでは別世代を一組として扱います。「sidecarを最後に書く」だけでは既存ファイルのある場合を保護できません。

2026-08-14の正常bundle（signal日=08-13）を先に保存し、符号を反転した新μと、新しいΩを再保存しました。新metadataはsignal日=08-17という拒否されるべき値にし、どの来歴が使われるかを識別しています。

| 再保存条件 | 保存関数の戻り値 | 読み込まれた組合せ | 実wrapperの結果 |
|---|---|---|---|
| 正常bundleの読込み | True | 旧μ・旧Ω・旧metadata | PASSED、gross≒2 |
| metadataを省略して上書き | True | 新μ・新Ω・旧metadata | PASSED、gross=2 |
| Ω保存直前にOSError | False | 新μ・旧Ω・旧metadata | PASSED、gross≒2 |
| metadata保存直前にOSError | False | 新μ・新Ω・旧metadata | PASSED、gross=2 |

いずれの異常ケースも読込alertは空で、`audit_failure`によるflatへ進みませんでした。μ・Ωは形状と数値検査を通る値です。単なる壊れたファイルの検出不足ではなく、正常な別世代のデータが混ざった場合の問題です。

SQLiteのtransaction経路はこの再現の対象ではありません。影響は`.npy`互換経路であり、[生成ツール:761](/Users/shonen/leadlag/tools/research/compute_gap_adjusted_distribution.py:761)と[ML学習用wrapper呼出し:589](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:589)が関連します。現行本番SQLiteで同じ混在が起きたという報告ではありません。

**具体修正・実装の順序：**

1. μ・Ω・metadataの対応を検証できる保存契約を決めます。既存の`.npy`名を維持するなら、sidecarに両ファイルのdigestを持たせ、読込時に実際に使用するbytesのdigestと照合する方法が取れます。ハッシュ計算のためにパスを読み直すと別世代を検査し得るため、読み込んだbytesを検証し、その同じbytesから配列を復元してください。
2. 書込前に旧sidecarの有効性を失わせ、各ファイルは一時ファイルから置換し、両配列に対応するsidecarを最後に公開します。`metadata=None`で上書きした場合も旧来歴を継承しません。別案は、μ・Ω・metadataをimmutableな一つの版へ保存し、読込側が一度選んだ版を最後まで読む方式です。
3. 不一致・欠落・読込中の更新は、検証不能な分布として既存のon-demand→flatへ接続します。各ファイルの`os.replace`を3回行うだけ、または旧sidecarを削除するだけで読込側の同一性検証を省く実装は、並行読込みを保護できません。
4. `save_gap_matrices`以外から同じ行列ファイルを更新する場合にも、古い来歴で受理されないことを確認します。生成ツール、horizon別出力、batchの確認、fixtureも新契約に合わせます。既存の正常計算を一律flatへ変えて回帰を通さないでください。

**追加する回帰・完了条件：** `tests/unit/test_gap_matrix_io.py`に上表の3異常ケースと、μ/Ω読込みの間に新版を公開するケースを追加します。default/h=3/h=5で、「完全な旧版」「完全な新版」「拒否してon-demand/flat」のいずれかになることを確認します。最後に実`generate_v2_production_portfolio`まで通し、混在状態が通常PASSEDにならないことと、正常bundleが非flatのままであることを検査します。現行の`test_npy_bundle_reads_published_provenance_sidecar`は初回正常保存だけなので、この不具合を検出しません。

証跡：[reproductions.json](./reproductions.json)の`npy_republication`。

**U02 / P2 — VaRのcache keyが表すモデル版と、実際にBTが使う版が固定されていません。**

対象：[var_history.py:180](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:180)、[同:258](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:258)、[同:292](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:292)。関連：[backtester.py:303](/Users/shonen/leadlag/src/leadlag/execution/backtester.py:303)、[v2_bridge.py:478](/Users/shonen/leadlag/src/leadlag/execution/v2_bridge.py:478)。前回の「採用版をrun全体で固定する」という未了事項です。

active versionだけをfingerprintする変更は正しく、staging作成による不要なcache missは解消しました。しかしkey計算時にCURRENTを読み、cache miss後のBT開始時にloaderがCURRENTを再読込みします。この間に新版が公開されると、A版のkeyへB版のreturn seriesを保存できます。本番runnerが既に保持しているA版も、VaRへ渡されていません。

再現では正常なversioned artifactをA/Bの2版用意しました。実`ProductionRunner`がAを保持した状態で、VaRがAのkeyを計算した直後、cache照会の境界でCURRENTをBへ切り替えました。実VaR関数・実artifact loader・実SQLite return cacheを使い、重いBTの数値計算部分だけを「実際にロードした版に対応する識別値を返す」処理に置換しています。

結果はrunner=A、BTが読んだ版=B、保存key=Aでした。CURRENTをAへ戻して再実行するとBTは再実行されず、AのkeyからBの識別値`0.02`を返しました。A用の識別値は`0.01`です。これらは再現用マーカーであり、実運用の収益率や損失を測定した値ではありません。モデルとrisk historyの対応が崩れることを確かめた結果です。

**具体修正・実装の順序：**

1. 1回のdecisionで使用する検証済みoverlay artifactを一度だけ選択します。runnerが保持するモデルと、`artifact_version`・本体digest等を取得できる明示的なインターフェースを設けます。
2. `v2_bridge`からその選択済みモデル/識別情報を`get_hist_returns_for_risk`へ渡します。VaRのcache keyは渡された版を使い、rootの現在のCURRENTを再読込みして決めません。
3. cache miss時は既存の`BacktestEngine.run_v2_backtest(overlay_model=...)`引数で同じモデルを渡します。VaR単独呼出しでも、入口で一度ロードしてからkey作成とBTの両方に使用します。モデルが与えられた場合に別のrootからロードし直さない契約にしてください。
4. cacheに版の識別情報を残して、利用時にも要求版との一致を確認します。保存直前だけCURRENTを再確認する修正では、runnerとVaRのずれやCURRENTの往復切替を解消できません。モデルを無効化して一致させる修正も行いません。

**追加する回帰・完了条件：** key計算後、BT開始直前、BT実行中のそれぞれでCURRENTをA→Bへ切り替えます。runner・cache key・BTが全てAを維持し、次の独立したrunからBへ移ることを確認します。Aへ戻した場合にもBの系列を返さないことを実cacheで検査します。現行`test_var_cache_fingerprint_tracks_only_current_overlay_version`はfingerprint関数単体の変化しか見ておらず、この呼出し間の不一致を検出しません。

証跡：`active_fingerprint_scope`、`var_version_race`。

**U03 / P2 — 検証スクリプトが、CURRENTの参照先欠落を成功終了にします。**

対象：[verify_overlay_models.py:48](/Users/shonen/leadlag/src/research/scripts/experiments/verify_overlay_models.py:48)。

CURRENTが存在しても、選択された`versions/<id>/model.pkl`がなければ`[skip]`を出してTrueを返し、loaderを呼びません。そのため、モデル本体の欠落やCURRENTの参照先ディレクトリ欠落を検証成功として扱います。

正常なartifactだけを検証対象にした`main()`はexit=0でした。同じ一時artifactからactiveなmodel.pklを削除した場合も、CURRENTを存在しない版へ変更した場合も、検証ツールはexit=0でした。一方、同じ各ディレクトリに対する本物の`load_overlay_model`は拒否しました。本番loaderが不正モデルを受け入れる問題ではありませんが、検証ツールでは成功し、本番初期化で失敗する状態を作れます。

**具体修正：** CURRENTがある時点で「検証対象のartifact」として扱い、欠落をskipにしません。共通loaderを呼び、`FileNotFoundError` / `ValueError`等を対象ディレクトリ名付きの`[fail]`として集約し、残りの対象も調査した上でmainを非0にします。構造比較にはロードしたモデルに付随する検証済みmetadataを使い、ツールがCURRENTとmetadataを別々に選び直す実装を避けてください。任意の未作成研究ディレクトリのskip方針とは区別します。

**追加する回帰・完了条件：** 正常版、active model欠落、active directory欠落、metadata欠落、不正pointer、digest不一致を一時artifactで作り、実`main()`の終了コードと対象ごとの結果を確認します。正常版だけなら0、検証対象に不正版が1つでもあれば非0です。loader自体を成功/失敗モックへ置換してはいけません。

証跡：`verification_dangling_pointer`。

**U04 / P3 — 移行確認スクリプトが正常なversionディレクトリをlegacy artifactと誤判定します。**

対象：[fix_overlay_metadata.py:38](/Users/shonen/leadlag/src/research/scripts/experiments/fix_overlay_metadata.py:38)、[同:23](/Users/shonen/leadlag/src/research/scripts/experiments/fix_overlay_metadata.py:23)。

`BASE_DIR.rglob("*")`で全階層を探索するため、正常artifact rootだけでなく、その内部の`versions/<id>`も独立したartifactとして検査します。versionディレクトリにはmodel.pklとmetadata.jsonがありますがCURRENTはないので、必ず`[retrain] legacy root artifact is rejected`と判定します。

正常にsave→loadできるartifactを1つだけ置いた一時BASE_DIRで、同じartifactについてrootは`[ok]`、active versionは`[retrain]`となり、mainはexit=1でした。これは運用データの破壊ではありませんが、移行後も移行未完了の診断が消えず、不要な再学習を案内します。

**具体修正：** artifact rootを列挙する処理と、root内の版を扱う処理を分離します。rootを発見したら、その配下の`versions/`と`.staging-*`を別artifactとして再帰検査しません。一方でwalk-forward用の親ディレクトリの下にある各fold rootは引き続き検出します。read-onlyという今回の方針を維持し、CURRENTがあるだけで健全性まで確認したと表示する場合はU03と同じloaderの検証結果に基づけてください。

**追加する回帰・完了条件：** 正常root＋複数inactive版＋stagingだけなら再学習対象0件・exit=0、本物のlegacy rootを追加した場合はそのrootだけを1件報告・exit=1とします。`wf/fold_1`、`wf/fold_2`も検出され、正常version内部を報告しないことを確認します。

証跡：`migration_report`。

**前回指摘の確認結果**

| 項目 | 今回の確認 |
|---|---|
| T01：無効な学習終端 | null、NaT文字列/オブジェクト、空文字、配列、不正な期間順序、空hash、未来label終端、非対応metadata版をsaveと直接applyで拒否。正常save→load後のtrain_end当日は拒否、最初のOOS取引日は適用可能 |
| T02：legacyの内外metadata不一致 | verifiedに見えるmetadataでもCURRENTなしのrootを拒否。digest・内外metadata比較の既存回帰も成功 |
| T03：metadataなし.npy | 実wrapperでgross=0、audit_failure=true、leakage=FLATへ改善。同じ数値のmetadataなしSQLite・未来metadataも拒否。正常sidecarで通常計算を維持。ただし再保存にU01が残る |
| T04：約定詳細取得失敗 | 実fetch関数で2注文とも照会し、エラーを保存してpost-decisionは例外へ。実close処理はclose_incomplete=true、実CLIは2。休日skipを無効にし、実際の照会到達も確認 |
| T05：初回APIログ書込み失敗 | 部分約定のsummaryを保持したOrderExecutionIncompleteへ改善し、fills・positions・walletの照合後に再送出。journal継続と再保存内容は追加unit testでも確認 |
| stagingによるVaR cache miss | inactive/stagingの変更でkeyは不変、CURRENT変更では変化。run内での版固定はU02として未了 |

[prior_probes.json](./prior_probes.json)では、MHの未来h=3拒否、SQLiteのdefault/3/5の一貫した読込み、JP休場日のNaN padding、通常の部分失敗後の4処理継続、取消済み部分約定の数量30保持、全拒否closeのCLI=2、ML公開失敗後の旧CURRENT維持も再確認しました。過去probeの関数名に`skips`等が残る箇所は、名前ではなく今回の観測値で判定しています。

**段階A〜C・運用準備の扱い**

今回のAに関わるT04/T05の再現条件は解消しました。分布の保存・来歴にはU01、モデル版とrisk historyの対応にはU02、移行・検証手順にはU03/U04が残るため、A〜C全体を完了とは判定しません。

さらに、継承をstrictに解決した本番設定の`ProductionRunner`は、現在の`models/ml_order_overlay/phase2_8`をlegacyとして拒否します。これは前回から開示されている運用準備の未了であり、今回の新規不具合数には含めていません。検証可能な学習データからのversioned artifact再生成と、本番/BTの同一入力・同一モデル版でのweights一致確認は未実施です。今回のレビューでモデルを再学習したり、本番設定を変更したりはしていません。

安全境界ADR・ARCHITECTURE・技術仕様書のlegacy拒否と共通validatorの記述は実装に対応しています。一方、bundleの一体性はU01、cacheのartifact同一性はU02を含めて成立させる必要があります。`docs/refactor_roadmap.md`の部分完了・未完了も照合しましたが、別Phaseや段階D〜Fの未了を今回の修正範囲へ追加してはいません。

**検証と再現方法**

全テストは10workers、プロセス全体の期限1800秒で実行し、watchdogはexit=0、374.6秒でした。`tests/`のunit/integration/research/regressionを含みます。17 warningsは研究テストのゼロ除算・定数列の相関に関するものです。成功件数に未実行テストやレビューprobeを加算していません。

| 検証 | 結果・証跡 |
|---|---|
| 全pytest | 614 passed、17 warnings：[pytest_full.log](./pytest_full.log) |
| Ruff | src/leadlag・tests・変更された研究共通コード/スクリプト・toolsで成功：[ruff.log](./ruff.log) |
| mypy | 本番120 source filesで成功：[mypy.log](./mypy.log) |
| compileall | src/leadlag・tests・tools・scripts・src/researchで成功：[compileall.log](./compileall.log) |
| import-linter | 154 files / 380 dependencies、4 contracts kept：[import_linter.log](./import_linter.log) |
| その他 | `git diff --check`、`bash -n scripts/batch/run_gap_distribution.sh`成功 |

Ruff/mypyを研究コード全体で成功させたという意味ではありません。変更batchは構文を検査し、実行はしていません。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260914_stage_abc_round4_review/probe_review.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260913_stage_abc_rereview/probe_remaining.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n auto --tb=short -q
```

再現コードは[probe_review.py](./probe_review.py)、結果は[reproductions.json](./reproductions.json)、例外注入のログは[probe.stderr](./probe.stderr)です。probeは現在のworking treeをimportするので、修正後はその時点の挙動を観測します。VaR検証の数値ループは識別値を返す代替処理であり、実バックテスト成績・本番risk閾値への影響量はこのレビューでは測定していません。

開始時と終了時のソースhashは[source_snapshot.json](./source_snapshot.json)と[review_manifest.json](./review_manifest.json)、指摘一覧は[review_summary.json](./review_summary.json)、証跡とリンクの整合検証は[validation.json](./validation.json)に保存しています。
