段階A〜Cの修正レビュー — 2026-09-13

**結論：段階A〜Cは部分完了です。現差分を完了扱いにすることはできません。** 新規の境界不具合と前回指摘の修正漏れを、P1 6項目・P2 2項目に整理しました。レビュー対象はHEAD `cf97bfaf`に対するworking treeの37追跡ファイルの変更、および追加されたテスト・固定fixture・修正報告です。段階D以降の運用基盤や新規収益実験を、今回の完了条件へ追加していません。

本番ソース・設定・ユーザーのテストは変更していません。調査用コードは`/private/tmp/`、証跡はこのディレクトリに置きました。broker検証は偽の応答・偽broker、DB検証は一時SQLiteを使い、実口座への発注・決済・照会は行っていません。

**R01 / P1 — provisional行が同日のUS終値を同日のJP判断に対応付けます（F08/F09）。**

[preprocessor.py:270](/Users/shonen/leadlag/src/leadlag/data/preprocessor.py:270)で`jp_all_dates >= last_joint`を選び、279行でも`>=`を許可しています。JPの最終行が未確定で、同日名のUS終値が入力にあると、`sig_date == trade_date`の行を作ります。

合成入力で2026-08-14のJP closeをNaNにすると、strict=Trueでもsignal日・trade日とも8月14日になります。同日のUS closeを100→110に変えたところ、8月14日の`us_cc_*`が0→+10%へ変わりました。米国8月14日終値は、日本8月14日9:10には利用できません。将来のUS情報がJP当日入力に入る条件を今回の変更で導入しています。

既存の`test_provisional_zero_open_is_kept`は当日行を残すことだけを検証し、その行のsignal日を検査しません。修正はUSの確定セッション日とJPの取引日をstrictに分け、provisional行も「そのJP取引日より前に確定したUSセッション」から構築することです。同日のUS終値がrawに追加されても当日判断が変わらない回帰が必要です。該当入力が本番で実際に売買された、という主張ではありません。

**R02 / P1 — PARTIALLY_FILLEDを追加した後の状態処理が揃っていません（F01/F02/F20）。**

[client.py:447](/Users/shonen/leadlag/src/leadlag/broker/tachibana/client.py:447)は数量による部分約定判定を取消・失効より先に行います。100株注文・30株約定・状態コード7（取消完了）が`PARTIALLY_FILLED`になります。未約定残の取消完了という状態が失われます。コード7とコード9の意味は[立花証券公式仕様](https://www.e-shiten.jp/e_api/mfds_json_api_ref_text.html)と照合しました。

さらに、[broker_ops.py:267](/Users/shonen/leadlag/src/leadlag/execution/broker_ops.py:267)は`SUBMITTED`以外をpendingから除きます。SUBMITTED→PARTIALLY_FILLED→FILLEDという偽brokerで、2回目の部分約定でpollが終わり、3回目の全部約定まで照会しません。[close.py:85](/Users/shonen/leadlag/src/leadlag/execution/close.py:85)も同じ条件です。

[broker_ops.py:653](/Users/shonen/leadlag/src/leadlag/execution/broker_ops.py:653)の終了判定は`first_batch_failed`/`close_failed`を使わず、部分約定をaccepted件数へ入れています。分割なしの2注文が両方部分約定でも、accepted=2、filled=0、failed=0、first_batch_failed=trueのまま正常returnしました。修正報告の「未約定は成功扱いしない」という契約を満たしません。

受付済み・部分約定は、照合待ちとして追跡してください。取消/失効等の注文状態と、累積約定量/残量を分離し、最終状態かdeadlineまでpollします。結果は完了・部分実行・照合待ち・失敗を区別し、正常returnだけで完了と扱わせない設計が必要です。単純に例外を出すだけでなく、約定・残存建玉の記録を残してから上位へ返す必要があります。追加enumを使う遷移の回帰テストが現差分にはありません。

**R03 / P1 — リークFAILEDの処理は直りましたが、実入力の時点を監査していません（F04）。**

[audit_comparator.py:130](/Users/shonen/leadlag/src/leadlag/models/v2/audit_comparator.py:130)のフラット化自体は改善です。ただし、[decision_engine.py:34](/Users/shonen/leadlag/src/leadlag/models/v2/decision_engine.py:34)は依然として前の日本営業日からsignal日を作り、gap metadataを読みません。[v2_auditor.py:49](/Users/shonen/leadlag/src/leadlag/compliance/v2_auditor.py:49)の可用性・計算窓の判定も変わっていません。

8月14日の分布bundleに`sig_date=2026-08-17`を保存し、分布からポートフォリオを作る共通経路を呼ぶと、リーク監査は全項目PASSED、gross=2でした。これは不正metadataを想定した境界再現で、実データに同じ混入があるという主張ではありません。前回F04で指摘した「監査対象の本当の日付を渡さない」問題が残っていることを示しています。

μ/Ωと同じ版のmetadataから実signal日・available_at・学習終端を受け取り、欠落/当日未確定/未来時点を実行可否へ接続する必要があります。FAILEDを返す監査のモックだけでは不足です。metadataを未来に変えたbundleが本番経路で拒否/flatになることを確認してください。

**R04 / P1 — atomicなGapStore.loadが本番の読込み経路で使われません（F12）。**

[GapStore.save/load](/Users/shonen/leadlag/src/leadlag/data/gap_store.py:197)はdefault horizonについて一つのtransactionになりました。しかし本番の`FileCacheSource`が使う[load_gap_matrices:253](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:253)は、μとΩを別々の`load_gap_npy`→`GapStore.get`で読みます。

μ読込み後に別のwriterが新bundle全体をatomicに保存するタイミングを模擬すると、μ=旧版1、Ω=新版2になり、strict=Trueでもalertsなしでした。writerだけatomicでも、readerが複数snapshotにまたがれば混在します。また、[Step2のh=3/5保存:836](/Users/shonen/leadlag/tools/research/compute_gap_adjusted_distribution.py:836)と[save_gap_matricesのhorizon保存:343](/Users/shonen/leadlag/src/leadlag/utils/gap_matrix_io.py:343)は別々のputのままです。

bundleのAPIにhorizonを持たせ、default/3/5すべてでμ・Ω・metadataを一括保存・一括取得してください。下流のreaderをそのAPIへ接続することが完了条件です。単体`GapStore.load`のテストだけでなく、実際の`load_gap_matrices`経由で更新割込み・途中書込み失敗を検査する必要があります。

**R05 / P1 — VaR cacheのfingerprintがSQLite WALとモデル更新を捉えません（F06）。**

[var_history.py:40](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:40)はSQLite本体のサイズ・mtimeだけを見ます。WALモードではcommit済みの変更が`-wal`へ入り、本体はcheckpointまで変わらないことがあります。一時DBで接続を維持してμ=1→2へ保存すると、実際の読込値は2、WALには16512byteある一方、fingerprintは不変でした。

同じdf_exec・取引日・configでgapを再生成した場合、異なる分布から計算した旧VaR履歴をcache hitで返し得ます。加えて[同:112](/Users/shonen/leadlag/src/leadlag/execution/var_history.py:112)でfingerprintするコードは`production_v2.py`だけで、処理本体の`models/v2/`・`blpx/`やML artifactを含みません。

DB内でtransactionとともに更新するbundle version/content hashを使用し、解決済み設定・ML artifact・計算コード版を含む識別にすべきです。ファイルmtimeによる推測では、要求される同一性を保証できません。WALにだけ更新がある場合、同じパスでML再学習した場合にcache missとなる回帰が必要です。

**R06 / P1 — MLの適用禁止日を、MLなしの通常成績として計上します（F13/F19）。**

[ml_order_overlay.py:685](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:685)は学習期間内の適用を検出すると、例外や無効runではなく元のポートフォリオを返します。train_end=8月14日のモデルを8月13日に適用した境界再現では、元のgross=2を保持し、fallback=false、summaryのfallback_triggered=0のままでした。

新しい標準BTの自動ロードと組み合わさると、学習終端より前はMLなし、後はMLありの混合戦略を、一つの有効設定の成績として計算できます。学習済みモデルの直接的なin-sample適用は防ぎますが、同じ設定・同じモデルで評価する契約は満たしません。

評価期間の開始時点で適用可能artifactを検査し、全期間を賄えないならrunを失敗にするか、foldごとに適用可能なartifactを供給してください。意図的にMLなし期間を比較する場合は、別設定/別run・明示的な集計にします。`train_end`をまたぐ期間で有効設定が密かに変わらないことを標準backtest経由で検査する必要があります。

**R07 / P2 — 初期wealthを含むDD修正がグラフとbacktesterへ反映されません（F15/F16）。**

[metrics.py:154](/Users/shonen/leadlag/src/leadlag/reporting/metrics.py:154)では、`concatenate([1], wealth)[1:]`としてから`maximum.accumulate`するため、追加した初期値を計算前に捨てています。[BacktestEngine:564](/Users/shonen/leadlag/src/leadlag/execution/backtester.py:564)のDD計算も旧式のままです。

−10%、0%の2日で、`calculate_metrics`のMDDは−10%ですが、グラフと`results['drawdown']`は両方 `[0,0]`でした。保存結果・図と主指標が一致しません。初期wealthを含めて累積最大を計算した後で表示用の初期行を取り除き、できれば共通関数を使ってください。追加テストは`calculate_metrics`だけでなく、engine出力とplotへ渡される系列まで比較する必要があります。

**R08 / P2 — regressionがグローバル関数を置換したまま戻しません（F22）。**

[test_v2_baseline.py:82](/Users/shonen/leadlag/tests/regression/test_v2_baseline.py:82)の`production_v2.download_macro_prices = lambda ...: None`は復元されません。独立プロセス内でこのテストを実行し、テスト自体はPASS、終了後の関数identityは元に戻らず`<lambda>`のままであることを確認しました。同じpytest workerの後続テストがmacro無効の状態を引き継ぐため、順序・worker割当で検証内容が変わります。

`monkeypatch.setattr` fixtureまたは`patch` contextに変更し、終了時の復元を保証してください。macroなしfallbackのsnapshotとして固定する方針自体は妥当です。ただしこのfixtureは本番macro/MLを含む完全一致の証明ではなく、明示した限定経路の回帰として扱う必要があります。

**解消が確認できた部分・完了判定の内訳**

| 前回項目 | 今回の評価 |
|---|---|
| F01/F02 | 標準の全部約定・全拒否の処理を修正。部分約定・取消後の遷移と完了判定はR02が残る |
| F03 | 既存建玉に対するreduction-only判定を追加。増加・反転を許可しない分岐を確認。実口座照合は未実施 |
| F04 | FAILED時のflat化を実装。実入力時点の監査はR03が残る |
| F05 | 当日行欠損を過去行へ置換する分岐を削除。履歴日指定/`latest`の扱い全体を本番の鮮度保証と同一視しない |
| F06 | gap/VaRの設定継承読込み、Step2 baseline IRの型付き設定参照を修正。cache版管理はR05が残る |
| F07 | 始値と現在値のcacheを分離。現在値取得へ変更。観測時刻付きsnapshotの全経路一致は未証明 |
| F08/F09 | 非対称休日のUS session対応・通常の初行NaNを修正。provisionalのR01が新たに残る |
| F10 | 前回と同じh1→h3対fresh h3の比較でμ差・Ω差とも0。修正を再現確認 |
| F11 | DataFrame値、target、9:10価格をcache keyへ追加。列名/入力版を含む契約のさらなる固定化は課題だが、今回の主要指摘とは分離 |
| F12 | default horizonのsave/load内部はtransaction化。本番readerとh3/5はR04が残る |
| F13/F19 | 設定からのML自動ロード・provenance検査を追加。期間不適合を通常成績にするR06と、旧artifactの移行が残る |
| F14 | 前日在庫に基づくslippage式とflat遷移テストを追加。完全な口数/価格/最終日清算の台帳へは未移行 |
| F15/F16 | 日次主指標・全評価日・複利の集計を修正。DDの複数出力はR07が残る |
| F17 | SRと試行間分散を同じ頻度へ変換する修正を確認。既存研究の全trial記録を新たに完了したという意味ではない |
| F18 | A7をlegacy V1と明示し、実行に明示フラグを要求。現行V2のOOS結果を新規に検証した変更ではない |
| F20 | FILLED/PARTIALLY_FILLEDと決済注文を約定情報収集へ追加。取消済み部分約定・失敗時照合はR02と合わせて検証が必要 |
| F21 | fallback_multiplierを呼出しへ追加し、0.25設定の既存追加テストPASS |
| F22 | 日付・df_execを固定化し、regression自体はPASS。グローバル置換のR08が残る |

**既知の未完了：現在の本番ML artifactは起動時に拒否されます。**

解決済みproduction configで`ProductionRunner`を構築すると、`models/ml_order_overlay/phase2_8 lacks temporal/data provenance metadata; retrain it before use`で失敗しました。[修正報告](/Users/shonen/leadlag/reports/20260912_workspace_audit/fix_report.md)に明記済みの挙動であり、未知artifactを拒否する方針自体を不具合とは数えていません。ただし、段階Cを「現行本番設定で動作確認まで完了」とは判定できません。必要な過去データでprovenance付きartifactを生成し、適用可能期間を含む本番/BT一致検証を通す必要があります。

**検証**

対象テスト86件PASS。全体`tests/`も**583 passed、17 warnings、705.67秒（11分45秒）**で終了しました。1800秒のプロセス全体deadline・4workersで実行し、watchdogの終了コードも0です。[全テストログ](/Users/shonen/leadlag/reports/20260913_stage_abc_review/pytest_full.log)を保存しています。Ruffは本番・tests・変更された研究/ツールの範囲でPASS、mypyは本番120ファイルPASS、import-linterは4契約PASS、指定rootsのcompileallもPASS、`git diff --check`もPASSです。研究全体のRuffをPASSとするものではありません。

[境界再現JSON](/Users/shonen/leadlag/reports/20260913_stage_abc_review/reproductions.json)、[horizon・未来摂動JSON](/Users/shonen/leadlag/reports/20260913_stage_abc_review/model_probes.json)、[対象テストログ](/Users/shonen/leadlag/reports/20260913_stage_abc_review/pytest_targeted.log)、[regression隔離性](/Users/shonen/leadlag/reports/20260913_stage_abc_review/regression_isolation.json)を保存しています。h=1,3,5の合成履歴に対する未来target摂動はいずれもμ・Ω差0です。これは、R01のraw→df_exec変換や、macro/ML全経路の可用性を保証する検証ではありません。

再現コマンドはリポジトリルートで次のとおりです。調査コードは一時ディレクトリにあり、実行時点のworking treeをimportします。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python /private/tmp/leadlag_stage_abc_review.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 90 .venv/bin/python /private/tmp/leadlag_regression_isolation_review.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_model.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n 4 --tb=short -q
```

レビュー対象41ファイルと再現スクリプトのSHA-256を[manifest](/Users/shonen/leadlag/reports/20260913_stage_abc_review/review_manifest.json)へ記録しています。レポート内のローカル参照先と行番号、再現JSONの構文・エラーの有無も検証しました。

`docs/refactor_roadmap.md`、ADR-P35、data validation、transactional cacheのADRを照合しました。今回の変更に対応する数理・時系列・artifact契約の正本文書は更新されておらず、修正報告だけでは恒久仕様が追跡できません。R01〜R08を直した後、変更した契約を対象の技術仕様/設計文書へ反映し、段階A/B/Cをそれぞれの合格条件で閉じるのが妥当です。
