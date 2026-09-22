# 段階A〜C 再レビュー — 2026-09-13

**判定：BLOCK。段階A〜Cを完了とは判定できません。P1が3件、P2が2件残っています。** 全体テストは今回も **591 passed / 17 warnings** ですが、追加の境界検証で監査・発注後処理・終了コードの問題を再現しました。既存のML artifactが本番runnerで拒否される、開示済みの運用準備未了もあります。

対象はHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285` 上のユーザー作業ツリーです。[元の監査](../20260912_workspace_audit/report.md)、[前回レビュー](../20260913_stage_abc_review/review.md)、[R01〜R08の修正報告](../20260913_stage_abc_review/fix_report.md)を照合し、修正後の呼出し経路を確認しました。今回の指摘番号S01〜S05は、前回のR番号と区別しています。

**修正担当向けの具体手順を末尾の「実装・統合の引継ぎ」に追記しました。** 変更対象、推奨する実装方式、テストの入力と期待値、統合順序を示しています。追記は実装指示の具体化であり、以下の不具合が修正済みになったという意味ではありません。

レビュー対象の実装・設定・テストは変更していません。追加したのはこのディレクトリの調査コード・証跡・報告です。再現は偽broker、一時SQLite、一時ML artifactを使っています。

**S01 / P1 — 複数horizonを混合する本番経路で、h=3/5の実signal日が監査されません。**

対象：[decision_engine.py:263](/Users/shonen/leadlag/src/leadlag/models/v2/decision_engine.py:263)、[gap_io.py:292](/Users/shonen/leadlag/src/leadlag/models/v2/gap_io.py:292)。段階A・F04、前回R03に対応します。

単一horizonの分布にはmetadataを渡す修正が入りました。しかし複数horizonの分岐は、各分布からμ・Ωだけを取得し、混合後も`distribution_metadata`を渡しません。[signal日の導出:48](/Users/shonen/leadlag/src/leadlag/models/v2/decision_engine.py:48)が改めて読むのはSQLiteのdefault horizonのmetadataだけです。h=3/5がscoreに寄与しても、その入力時点を検査できません。

解決済み本番設定のMH有効・horizons=[1,3,5]を維持し、監査対象を分離するためmacro/CS/ML overlayだけ無効化して、実際の`ProductionV2Model.decide`を呼びました。取引日2026-08-14に対し、h=1/5の`sig_date`を8月13日、h=3だけを未来の8月17日にしたbundleでも、リーク監査は全項目PASSED、モデルgross=2.0です。h=3のμだけ符号反転するとscoreの最大差は3.265986となり、問題の分布が実際に計算へ入ることも確認しました。

これは不正metadataを与えた境界再現であり、実履歴に未来データが混入したと断定するものではありません。ただし「実分布のsignal日を監査へ渡す」という修正が、本番で有効なMH経路には届いていません。

修正は、寄与する全horizonについてμ・Ωと同じ版のmetadataを保持し、混合前に入力可用性を検査することです。default horizonの再読込みや前営業日の推定で代用せず、metadata欠落時の扱いも明示してください。既存の`test_gap_metadata_is_used_for_leakage_signal_date`はこのMH経路を通りません。実際の`decide`経由でh=3/5の未来日付・欠落・cache/on-demand切替を検査する回帰が必要です。

証跡：[reproductions.json](./reproductions.json) の `future_h3_metadata_not_audited`。

**S02 / P1 — 発注の部分失敗で例外を出すと、その後の約定・建玉・余力記録を飛ばします。**

対象：[broker_ops.py:677](/Users/shonen/leadlag/src/leadlag/execution/broker_ops.py:677)、[post_decision.py:223](/Users/shonen/leadlag/src/leadlag/execution/post_decision.py:223)。段階A・F02/F20、前回R02に対応します。

部分約定・未確認の注文を正常完了にしない変更は正しい方向です。ただし`submit_orders_via_api`が例外を出すと、呼出し元の後続処理である約定情報取得、position/wallet snapshot、daily journalに到達しません。実際に約定した数量と残存建玉を把握すべき場面で、記録が不足します。

2注文がPARTIALLY_FILLEDのまま期限を迎える偽brokerを使い、実際のsubmit関数からpost-decisionまで通しました。`accepted=2/expected=2, failed=0, partial=2, unresolved=2`の例外は発生しますが、後続の収集呼出しは0回でした。最小限の`api_execution_log.json`は保存される一方、約定数量は両注文とも未記録です。「ログが全く残らない」という問題ではありません。

さらに[pricing.py:123](/Users/shonen/leadlag/src/leadlag/execution/pricing.py:123)はCANCELLEDを収集対象から除きます。100株注文のうち30株約定後に残数量を取り消した状態を模したところ、詳細取得を呼ばず`fill_quantity=null / NOT_FILLABLE`となりました。終端の取消と、既に成立した約定は別に扱う必要があります。

注文結果を保持したうえで、成功・部分失敗・期限到達のいずれでも可能な範囲の約定・建玉照合と記録を行い、その後に失敗を上位へ返してください。元の失敗原因を後処理の例外で隠さない設計も必要です。既存のpolling/件数テストに加え、post-decision入口から部分失敗・取消済み部分約定・期限到達を通して、数量・snapshot・journalが残ることを検査してください。自動再送を追加する提案ではありません。

証跡：`partial_failure_skips_reconciliation`、`cancelled_partial_not_collected`。

**S03 / P1 — 決済注文が全件拒否されても、close CLIが終了コード0を返します。**

対象：[close.py:614](/Users/shonen/leadlag/src/leadlag/execution/close.py:614)、[cli.py:354](/Users/shonen/leadlag/src/leadlag/cli.py:354)。段階A・F01/F02、前回R02に対応します。

決済summaryとログの不完了表示は修正されました。しかし`run_close_positions_mode`は`close_incomplete=true`でもエラーログを出すだけで正常returnし、CLIは無条件に0を返します。[バッチ:31](/Users/shonen/leadlag/scripts/batch/run_close_positions.sh:31)もその終了コードを転送します。終了コードで監視する呼出し元には、決済失敗が正常終了として伝わります。

`close_all_positions`が全件拒否のsummaryを返す条件で、実際の`leadlag.cli.main(['close'])`から検査しました。`filled=0 / pending=0 / failed=1 / close_incomplete=true`でも戻り値は0です。実broker呼出しは偽物へ置換しています。

journal保存とclientのcloseを行ったうえで、拒否・失効等の失敗をCLIの非ゼロ終了へ伝播してください。引け注文の受付後、オークション前にpendingである状態をどう扱うかは別途明示できますが、今回の再現は全件拒否なので、その正常待機とは区別できます。既存テストはsummary・保存JSONまでの検査で、CLI終了コードを検査していません。

証跡：[追記時の再確認](./reproductions_handoff.json) の `all_rejected_close_cli_success`。当初のprobeは実行日が休日だとCLIの早期returnに入るため、決済処理到達の証明として不十分でした。追記時に休日判定を営業日へ固定し、`close_all_positions`が1回呼ばれたこともassertするよう調査コードを補強しました。その条件でも終了コード0を確認しています。

**S04 / P2 — MLの学習時点metadataとモデル本体を別々に上書きし、異なる版を正常ロードできます。**

対象：[ml_order_overlay.py:327](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:327)、[同:348](/Users/shonen/leadlag/src/leadlag/models/ml_order_overlay.py:348)。段階C・F19に対応します。

保存はmetadata.jsonを先に更新し、次にmodel.pklを上書きします。読込みは別々に行い、外部metadataをモデルへ設定します。モデル本体とmetadataを結び付けるdigestや版一致検証がありません。同じ保存先への更新中断・読書き競合で、学習時点制約の根拠が別モデルの情報になります。

2026年まで学習したモデルを模したartifactに、2020年までの別モデルを保存しようとし、metadata更新直後・model.pklを開く直前でOSErrorを注入しました。ロードは成功し、本体の識別子は`trained-through-2026`なのに`train_end=2020-12-31`となります。現行の時点比較なら2021-01-04への適用を許す組合せです。この再現では数値予測や売買は実施していません。

immutableな版ごとのartifactを完成させてから参照をatomicに切り替えるなど、本体とmetadataを一組として公開してください。ロード時も本体digest、特徴量schema、学習終端の一致を検証し、混在版を拒否すべきです。legacy拒否・単体のtrain_end比較テストでは、この保存失敗を検出できません。途中失敗と同時読込みの回帰が必要です。

証跡：`overlay_binary_and_metadata_mix`。

**S05 / P2 — 休場日の全NaN行を含む入力では、翌営業日以降の有効なデータまで落とします。**

対象：[preprocessor.py:206](/Users/shonen/leadlag/src/leadlag/data/preprocessor.py:206)、[同:288](/Users/shonen/leadlag/src/leadlag/data/preprocessor.py:288)。段階B・F08/F09に対応します。

市場ごとに有効日を抽出していますが、`pct_change`と前日closeの`shift(1)`は、抽出前のDataFrame上で計算します。raw/cacheが共通営業日indexへ整列済みで、ある市場の休場日を全NaN行として保持している場合、次の実営業日のリターンやgapまで欠損になります。

日米の価格を一定100とし、日本休場日の2026-08-11だけJP open/closeを全NaNにした入力で、非strictは8月12日・13日の両行を出力から落とし、strictは8月12日の`jp_gap missing`で失敗しました。既知の休場日である全NaN行だけを取り除いた同じ入力はstrictでも通り、両日が復元されます。

**市場別の実営業日だけを持つ通常入力では、この条件は発生しません。** 今回の指摘は休場日をNaNで埋めた入力に限定します。全ての祝日対応が壊れているという評価ではありません。

取引所calendarで真の休場日とデータ欠損を区別してから、実セッション列上で前日比・gapを計算してください。営業日の予期しない欠損まで一律drop/fillしてstrict検証を緩める修正は避けるべきです。非対称休日のテストには、休場日の行が存在しない場合に加え、全NaN行として存在する場合も必要です。

証跡：`nontrading_nan_row_loses_next_return`。

**前回R01〜R08の修正確認**

| 前回指摘 | 今回の確認結果 |
|---|---|
| R01 provisionalの同日US close利用 | strictly laterなJP日へ対応付ける修正と追加回帰を確認。今回の全テストでもPASS。S05は別の休場日入力条件 |
| R02 部分約定・取消・成功扱い | 部分約定をpollingに残し、件数・不完了を分ける修正は確認。ただしS02の失敗後記録、S03のCLI契約が未了 |
| R03 実signal日の監査 | 単一horizonではmetadata伝播を確認。MHのS01が残るため全経路完了ではない |
| R04 GapStore readerの版混在 | **default/3/5すべて修正を再現確認**。読込み途中に新bundleをcommitしても、取得したμ・Ω・metadataは全て旧版、直後の再読込みは新版になる |
| R05 VaR fingerprint | SQLite WAL/SHM、ML artifact、src/leadlagの内容digestを含む修正を確認。関連回帰を含む全テストPASS |
| R06 ML in-sample適用 | 直接適用は例外となり、BT側はflat fallbackとして記録する。以前の「通常のMLなしウェイトを成功として保持」は解消。ただし、評価全期間を賄うartifactの準備・本番との同一weightsは未検証 |
| R07 DD系列不一致 | 初期wealth=1を含む共通関数をmetrics・engine・chartへ接続。関連回帰PASS |
| R08 global monkeypatch残留 | scopedな`monkeypatch.setattr`への変更を確認。固定fixtureによるregressionも今回の全テストでPASS |

元のF10についても、h=1の計算後に同じモデルでh=3を計算した結果とfresh modelのh=3は、μ差・共分散差とも0でした。h=1/3/5に対する未来target摂動でもμ・Ω差は全て0です。[model_probes.json](./model_probes.json)に保存しています。計算窓に関するこの合格と、S01の入力metadata検査不足は異なる境界です。

**段階ごとの完了判定**

| 段階 | 判定 | 残る条件 |
|---|---|---|
| A：注文・監査・日付 | 部分完了 / BLOCK | S01〜S03。削減専用判定・当日行欠損時の拒否・FAILED時のflat化は改善したが、実入力の監査と部分失敗処理が未完 |
| B：設定・snapshot・cache | 部分完了 | 設定継承、現在値cache分離、horizon別cache、μ/Ω bundle、PIT multiplierは改善。S05の休日入力境界が残る。全入口・同一snapshotの一致を包括的に証明したものではない |
| C：BT・損益・指標・artifact | 部分完了 / BLOCK | PnL/DD・全評価日・DSR・fixtureと全テストは改善。S04、利用可能artifact、および本番/BT同一入力のweights一致確認が残る |

解決済み本番設定から`ProductionRunner`を構築すると、現在の`models/ml_order_overlay/phase2_8`が`lacks temporal/data provenance metadata; retrain it before use`で拒否されます（`configured_runner`）。これは修正報告で開示済みで、未知artifactを拒否する方針自体を新しい不具合として数えていません。ただし、現行本番ML経路の稼働準備まで完了したとはいえません。検証可能な学習履歴を持つartifactを用意し、評価対象の各日で利用できることを確認する必要があります。過去artifactへ根拠のないmetadataを補う対応は適切ではありません。

新しい[安全境界ADR](../../docs/decisions/2026-09-13-stage-abc-safety-boundaries.md)の追加は確認しました。その「各分布の実signal日を監査へ渡す」という記述はS01の実装と一致していません。また、MLの期間不適合をflat fallbackとして継続する実装と、ADRのfold別artifactを要求する記述は、runの有効性・評価日数を含めた契約整理が必要です。`docs/refactor_roadmap.md`の未完了・部分完了、ADR-P35、関連data/cache/registry ADRも照合しました。acceptedや全テストPASSだけで完了とせず、今回の契約変更を技術仕様・構造文書と整合させてください。段階D〜Fの作業を今回へ追加する判定ではありません。

**実行した検証と再現方法**

| 検証 | 今回の結果 |
|---|---|
| tests/全体（unit/integration/research/regressionを含む） | **591 passed, 17 warnings, 699.38秒**。4workers、プロセス全体deadline=1800秒、watchdog exit=0 |
| Ruff：src/leadlag、tests、変更対象の研究・ツール | PASS。研究全体のlint合格を意味しない |
| mypy：src/leadlag | PASS、120 source files |
| import-linter | PASS、4 contracts kept / 0 broken |
| compileall：src/leadlag tests tools scripts src/research | PASS |
| git diff --check | PASS |
| 追加境界再現 | 上記5件を再現。SQLite atomic読込みとhorizon/未来target摂動は改善を確認 |

各ログは[pytest_full.log](./pytest_full.log)、[ruff.log](./ruff.log)、[mypy.log](./mypy.log)、[import_linter.log](./import_linter.log)、[compileall.log](./compileall.log)です。前回修正報告ではimport-linterが未実行でしたが、今回の環境では実行でき、4契約とも通っています。

リポジトリルートで実行します。調査コードはその時点のworking treeをimportします。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260913_stage_abc_rereview/probe_remaining.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python reports/20260912_workspace_audit/probe_model.py
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n 4 --tb=short -q
```

レビュー対象と証跡のhashは[review_manifest.json](./review_manifest.json)、参照先・再現結果の検証は[validation.json](./validation.json)へ保存しました。実口座の発注・決済・残高照会、外部schedulerの検査、ML再学習、修正後の長期収益性評価は実施していません。本報告は段階A〜Cの変更に対するレビューであり、リポジトリ全体の無欠陥や収益改善を保証するものではありません。

---

**実装・統合の引継ぎ：作業範囲と進め方**

以下はS01〜S05を修正する担当者への実装仕様案です。「新設案」と記した関数・フィールド・ファイルはまだ存在しません。既存の関数を検索してから着手し、提案名を既存APIとして呼び出さないでください。行番号は移動するため、上記リンクと下表の関数名を併用します。今回の文書追記では本番コードを変更していません。

| 順番 | 作業単位 | 同時に揃えるもの | 他の指摘との依存 |
|---|---|---|---|
| 1 | S01：分布の来歴を監査まで保持 | 分布取得・MH混合・監査・fallback・保存診断・入力fixture | 独立。S04と共通のMLテストを変更する際は差分を保持 |
| 2 | S02：不完了結果を保持して記録 | submitの例外契約・post-decision・約定情報取得・回帰 | S03でも使う約定取得の契約を先に固める |
| 3 | S03：決済結果をCLIへ返す | close runnerの戻り値・CLI・journal/cleanup・回帰 | S02の収集処理を利用。終了判定を別の意味で再実装しない |
| 4 | S04：ML artifactを一組で公開 | save/load・学習CLIの出力説明・読込み元・VaR cache識別・回帰 | 実artifact移行と同一weights検証は、この保存契約の完成後 |
| 5 | S05：休場行を正規化 | JP open/close/TOPIXのindex・前日比/gap・strict検証・回帰 | 独立。S01が使う`sig_date`の意味を変えない |

一つの作業単位を、入口から下流まで接続してから次へ進めてください。内部関数だけ直して呼出し元が旧契約のまま残る状態は未完了です。正常入力のweights、RuleD、gross/net制約、コスト設定、既存のcache→on-demand→flat順序は維持します。新しいα、パラメータ最適化、V1復帰、本番MLの無効化をこの修正に混ぜません。

まず既存の再現を回帰テストへ移し、修正前に期待する理由で失敗することを確認します。本ディレクトリの`reproductions*.json`は不具合発生時の証拠です。修正後の合格期待値ではありません。`validate_review.py`も過去のレビュー証拠を検証するスクリプトであり、修正後コードの回帰試験ではありません。修正後の結果は別ログ・別JSONに保存してください。

**S01の実装仕様：日付検査を分布の取得結果に結び付ける**

読む入口は[distribution_source.py](../../src/leadlag/models/v2/distribution_source.py)の`DistributionResult`、`FileCacheDistributionSource.resolve`、`OnDemandDistributionSource.resolve`、[fallback_policy.py](../../src/leadlag/models/v2/fallback_policy.py)の`FallbackPolicy.resolve`です。MHは[overlay_applier.py](../../src/leadlag/models/v2/overlay_applier.py)の`_multi_horizon_scores`、最終判定は`decision_engine.py`と`audit_comparator.py::_run_safety_audits`にあります。

1. **既存の`DistributionResult`を共通の取得結果として使う。** MHもhorizonごとに`FallbackPolicy.resolve(..., horizon=h, snapshot=snapshot)`から結果を受け取る方式を推奨します。現在の`compute_distribution`の2配列tupleをMHで使い続けるとmetadataが落ちます。互換用のtuple APIを残す場合も、本番判断はmetadataを持つ経路へ接続します。既存のshadow比較、`use_file_cache=False`時の順序、on-demand有効条件を移行時に確認してください。
2. **取得結果に紐付く最低限の来歴を検査する。** 必須は実`sig_date`（別名`signal_date`も正規化）です。NaT・空文字・パース不能は不適合、両キーが存在して異なる日付なら不適合にします。取引日・horizonは要求引数と一緒に保持し、metadataにも存在するなら一致を検査します。日付が存在するだけで合格にせず、既存の`run_leakage_audit`の鮮度上限等も維持します。h=1のDBキーは現行のdefault horizon=`None`との対応を維持し、別のh=1レコードを新設しません。
3. **cacheのmetadataを後から取り直さない。** `load_gap_bundle`が返したμ・Ω・metadataだけを一組として使います。h=3/5のmetadataをdefaultで代用しません。異なるhorizon同士が同じDB transactionで生成されていることまでは、この修正の必須条件ではありません。まず各horizon内の同一版と、全寄与分布の可用性を保証します。
4. **on-demandでも実入力から来歴を返す。** `_compute_ondemand`が実際に用いる`df_exec`の当日行と、その行の`sig_date`を同じ入力として保持して、結果へ添付する経路を追加します。`trade_date`から前営業日を推定して埋める処理は禁止です。`MarketSnapshot`には現在signal日フィールドがないため、存在しない属性を参照しないでください。snapshot由来の入力へ来歴を追加する場合は、生成元の同じrowから設定し、df_execと矛盾する場合に拒否します。履歴・モデル・価格・来歴が不足すればon-demand不可として終端flatへ進めます。
5. **各sourceを使えるか判定してから混合する。** cacheの来歴不適合はそのsourceを採用せず、許可されたon-demandで再計算・再監査します。採用する全horizonが合格した時だけ既存の混合式を実行します。一つでも解決不能ならポートフォリオ全体をflatとし、h=1だけで成功したことにはしません。現行MH分岐の広い`except Exception`→`_file_cache_or_flat`が、不適合なh=3/5を無視したh=1の通常判断を返さないよう接続を変える必要があります。
6. **監査結果を最終出力まで保持する。** horizon別のsource・signal日・判定・不採用理由は、既存`PortfolioDecision.diagnostics`内の`distribution_provenance`などへ保存する案を推奨します。これは新設する診断キーです。最終`leakage`は採用分布とPIT監査を反映し、監査不適合でflatにした場合は`fallback.audit_failure=true`、最終weights=0と原因を保持します。`_run_safety_audits`が事前fallbackを一律FLATへ置換する分岐で、既に判明したFAILEDを消さないでください。`diagnostics`はreport writerやoverlay通過後の保存物まで残ることを確認します。

既存監査へhorizon別結果を渡す引数を追加する場合は、`_run_safety_audits`の全呼出し元を検索して揃えます。共通監査を拡張することは許容されますが、日付チェックだけでローリング窓・入力の観測時刻も検証済みとしないでください。`available_at`や学習終端が記録されている入力はその意味に沿って検査し、未記録の時刻を便宜的に生成しません。今回の必須再現はsignal日の漏れを閉じるもので、時刻付きsnapshot全体の証明とは分けて報告します。

**旧入力への対応も同じ変更で決めます。** 現状の`.npy`読込みはmetadataを返しません。本番では来歴不明のcacheとして扱い、利用可能なon-demandかflatへ進む方式を推奨します。ファイル名の日付をsignal日にする互換策は不可です。テストが従来のmetadataなしfixtureで失敗したら、テスト用と明示した来歴を持つ一時GapStoreを作るか、来歴不足のflatを期待するケースに分けます。正常計算のテストを全てflat期待へ変えて通すことは避けてください。固定された実データfixtureへ来歴を補う場合は、元のdf_exec/生成記録から裏付けを残します。

| 回帰ケース（追加先案：tests/unit/test_stage_abc_fixes.py、tests/integration/test_production_v2.py） | 修正後の必須期待値 |
|---|---|
| 本番設定をdeep copyしMH=[1,3,5]のまま、h=3だけ未来signal日。on-demand不可 | `decide`の最終weightsが全0、audit failureとh=3の原因を記録。PASSEDの通常判断を返さない |
| h=5だけ未来、h=3は当日signal日、空/NaT/矛盾する別名キー | 各ケースで不適合を検出。前営業日推定で回避しない |
| cache不適合、on-demandに十分な正常入力あり | 不正cacheを使わず、再計算結果の実signal日を検査。source切替の理由が残る |
| 全horizon正常、通常の単一horizon、MH重みの既存境界 | 変更前とscore/weightsが許容誤差内で一致。監査やMHを無効にして比較しない |
| 読込み途中にwriter更新、default/3/5それぞれ | μ・Ω・metadataが同じ版。前回R04の修正を維持 |
| 不適合分布＋ML overlay有効 | flatからoverlayで建玉を復活させず、監査理由も保持 |

テストでは監査関数自体を「FAILEDを返すモック」にしません。日付入りの実bundleを作り、実際の`ProductionV2Model.decide`と監査を通してください。on-demandの数値計算を軽いfixtureへ差し替える場合も、来歴検証とfallback判定は本物を使います。**完了条件は、全寄与horizonの根拠が最終保存物で追跡でき、上記の拒否と正常計算の両方が通ることです。**

**S02の実装仕様：結果を持つ例外と、失敗しても進む収集処理**

変更対象は`broker_ops.py::submit_orders_via_api`、`post_decision.py::_write_decision_output_and_submit`、`pricing.py::fetch_fill_prices`です。推奨する最小変更は、submitの既存dict戻り値を維持し、不完了時の例外に同じsummaryを持たせる方式です。

1. `broker_ops.py`に`OrderExecutionIncomplete(RuntimeError)`を新設し、`summary`と`log_path`を属性で保持します。末尾で現在投げる`RuntimeError`をこの型へ置き換えます。件数の判定、先に最小ログを保存する順序、dry-runの扱いは維持します。エラーメッセージやログファイルを解析してsummaryを復元する方式は避けます。
2. post-decisionでこの例外だけを捕捉し、`order_summary=exc.summary`、`execution_error=exc`として保持します。正常returnと不完了例外の両方が、同じ収集処理へ進むようにします。`order_summary`・保存先・snapshotの戻り値はtryの前に初期化し、失敗時の未代入参照を防ぎます。
3. 収集は「各注文の約定詳細→約定情報を加えたAPIログ保存→position snapshot→wallet snapshot→daily journal」の順とします。各処理の失敗を個別に捕捉し、他の収集は続けます。エラーの処理名・原因は`reconciliation_errors`等の新設フィールドへ記録します。ファイル保存失敗ならloggerにも残し、存在しないsnapshotをjournalで保存成功と表現しません。収集に使う既存APIのtimeoutを外したり、無限再試行を追加したりしません。
4. 収集終了後、元の不完了例外があれば再送出します。収集失敗が重なっても元の発注不完了を失いません。発注成功でも照合が不完了なら、その状態を上位へ通知し、完全に記録済みと扱わない契約にします。この通知は再発注要求ではありません。transport例外等でsummaryを受け取れない経路は、空の成功summaryを作らず、取得できた情報と未確認状態を残して元の例外を維持します。
5. `fetch_fill_prices`は「FILLED等の状態だけ」という条件を外し、brokerで照会可能なorder_id/営業日がある注文を対象にします。CANCELLEDや失効後にも累積約定があり得ます。状態enumを約定数量の代用にせず、`quantity`は注文数量、`fill_quantity`は実際の累積約定数量として別々に保持します。詳細照会で得た0は0、未取得はnullとして区別します。詳細取得失敗で既存の確定した数量をnullに戻さないでください。

収集側の接続先として[output_ops.py](../../src/leadlag/execution/output_ops.py)も変更対象です。現在の`save_position_snapshot`は建玉なしと照会失敗の両方でNoneを返し、`save_wallet_snapshot`も照会例外を捕捉してNoneにします。そのまま外側へtry/exceptを追加しても取得失敗を検出できません。最小案は、両関数にkeyword-onlyの`raise_on_error: bool = False`を追加し、今回の収集処理ではTrueで呼ぶことです。また、照会成功・建玉0件なら空のpositionsと`position_count=0`をファイルへ保存してpathを返すようにします。既存呼出し元とそのテストも確認し、正常な0件照会と未確認を明確に区別してください。daily journalへ収集状態を追加する場合は、既存のartifact参照を保持して拡張します。

dry-runのSIMULATEDには実照会を行いません。order_idがなく照会不能なら明示的に未取得とし、broker未対応の場合も取得済みと表示しません。100株注文・30株約定・残数量取消の例では、最終注文状態CANCELLEDを保ったまま`fill_quantity=30`を保存します。`quantity=30`への上書きやFILLEDへの変更は誤りです。終端の取消済み未約定分を「未確認pending」と取り違えないよう、残数量を追加する場合は未約定・取消済み・未確認を区別します。

回帰は[既存の発注テスト](../../tests/unit/test_split_large_orders.py)と[境界テスト](../../tests/unit/test_stage_abc_fixes.py)を起点にします。必要なら`tests/unit/test_post_decision_reconciliation.py`を新設し、入口から保存までまとめて検査します。

| 偽brokerの応答 | 修正後に検査するもの |
|---|---|
| 2注文とも部分約定のままdeadline | 不完了例外を維持。両order_id・累積約定量・position/wallet/journalの呼出しと実ファイルが残る |
| 片側全部約定、反対側拒否 | 全部約定側の数量・価格が保存され、失敗件数が保持される。再送呼出しは0回 |
| 100株中30株約定後にCANCELLED | 詳細照会が1回以上行われ、状態CANCELLED・注文数量100・約定数量30を同時に保存 |
| 約定詳細取得失敗、position取得失敗をそれぞれ注入 | 他の収集を継続。元の不完了例外を保ち、収集失敗を別途記録 |
| 全部約定、dry-run | 既存の正常経路を維持。dry-runで実照会しない |

テストは`submit_orders_via_api`自体を単純に例外モックへ置換するだけでは不足です。偽broker応答から本物のsummary集計と新例外を通すケースを一つ以上置いてください。**完了条件は、不完了を上位へ返す前に、取得可能な約定・残存建玉の記録を試み、その成否も保存されることです。**

**S03の実装仕様：closeのsummaryを終了コードへ接続する**

この項目では、S02とは別の巨大な結果型を新設せず、`close_all_positions`の既存summaryを使う方式を推奨します。最低限の終了契約は次の表で固定します。数字は実装案なので、既存の運用契約が別にある場合は意味を揃えて文書化してください。

| 結果 | 推奨CLI終了コード | 意味 |
|---|---|---|
| 必要な決済が全て約定、または照会に成功して決済対象なし | 0 | 要求した処理が完了 |
| 拒否・取消・部分約定・deadline後の未確認が残る | 2 | 決済不完了。確認が必要で、自動再送を意味しない |
| broker初期化・照会・必須記録などの例外 | 1 | 処理エラー。判明済みの結果を可能な範囲で保存 |
| dry-run | 0 | simulationであることをログに明記。実約定完了の証明にはしない |

1. `run_close_positions_mode`の戻り値を`None`からsummaryへ変更します。`close_all_positions`を呼び、約定・建玉・余力・journalを記録した後に返し、clientのcloseは`finally`で実行します。`finally`内でreturnして例外を抑制しないでください。記録失敗時もclientを解放し、取得済みsummaryを失わないようにします。
2. `_handle_close`は返されたsummaryの`close_incomplete`から終了コードを決めます。loggerの文面で判定しません。例外をコード1へ変換するならここで対象例外を捕捉し、原因をログに残します。`main`・daily経由のcloseでも、このコードが伝播することを確認します。
3. `close_all_positions`の「建玉なし」という早期returnにも、件数0・`close_incomplete=false`等の一貫したキーを持たせます。ただし建玉照会失敗を「建玉なし」へ変換しません。現在の持越し率に従って決済しない建玉や丸めで注文不要なケースと、出すべき注文の拒否を区別します。
4. `scripts/batch/run_close_positions.sh`は既にCLIの終了コードを転送するので、不必要に書き換えません。待機して決済する`close.py::wait_and_auto_close`を変更対象に含める場合は、その呼出し元へ同じ不完了契約を接続します。直接CLI経路だけ直した段階で、自動決済経路も検証済みとは書きません。

オークション前の受付pendingを正常な受付結果として扱う必要があるなら、「受付」と「決済完了」を別の実行契約にする追加設計が必要です。今回の最小修正ではdeadline後の未確認をコード2とし、pendingを0へ戻す例外規則を安易に足しません。運用上コード2を受けた際に注文を再送する処理を追加しないでください。

回帰は[closeテスト](../../tests/unit/test_close_positions.py)へsummaryを、CLIの回帰用ファイルを新設して終了コードを追加します。**CLIテストでは`leadlag.core.market_calendar.is_market_closed`をFalseへpatchし、決済処理が1回呼ばれたことをassertします。** これがないと休日の「何もしないで0」を成功/失敗の検査と誤認します。全件拒否・混合結果・全部約定・建玉なし・dry-run・journal例外を通し、終了コード、保存された集計、client.closeの実行を同時に確認してください。**完了条件は、全件拒否が必ず非ゼロになり、その前後の記録とcleanupが失われないことです。**

**S04の実装仕様：版を固定して読み、完成したartifactだけを公開する**

変更の起点は`ml_order_overlay.py::save_overlay_model/load_overlay_model`です。保存・読込みを同じ変更で実装し、`tools/production/train_ml_order_overlay.py`、`runner/production.py`、`execution/backtester.py`、`execution/var_history.py`の利用経路を確認します。既存の`--output-dir`をartifactのルートとして維持する設計を推奨します。

保存形式の新設案は次のとおりです。`CURRENT`は完成済みの版IDを一つだけ記録する小さなファイルで、symlinkである必要はありません。

```text
<output-dir>/
  CURRENT
  versions/<version-id>/
    model.pkl
    metadata.json
  .staging-<unique-id>/  # 保存途中のみ。readerは参照しない
```

1. **metadataを確定してから本体を直列化する。** `train_start`、`train_end`、`label_asof_end`、data/configのhash、特徴量の名前と順序、ティッカー順序、分類/回帰等のモデル設定を、実際の学習結果から構築します。必要な値は非空・型・日付順序も検査し、`metadata_version`の新しい値を定義します。callerから渡されたmetadataで特徴量schema等を矛盾した値へ上書きできないようにします。
2. モデル内のmetadataへ、確定した来歴・schema・version-idを格納したコピーを直列化します。保存処理がcallerのモデルを中途半端な状態へ変更しないようにします。出来上がったmodel.pklのbytesからSHA-256を計算し、外部metadataへ`model_sha256`として格納します。モデル本体へ自身のhashを埋めようとすると循環するため、本体hashは外部manifest側だけに置きます。
3. artifactルートと同じファイルシステム内の一意なstagingディレクトリへ全ファイルを書き、flush/fsyncし、保存物を内部検証します。完成してから一意な`versions/<version-id>`へ移し、一時CURRENTファイルを`os.replace`で公開します。最後の参照切替が完了するまで旧CURRENTは維持します。2ファイルへ個別に`os.replace`するだけでは一組のatomic性にはなりません。
4. readerはCURRENTを**一度だけ**読み、その版のディレクトリへ固定してmetadataとmodelを読みます。metadataのschema・required fieldsとmodel digestを確認してから本体を復元し、モデル内のversion-id・来歴・特徴量順序とも一致することを検査します。外部metadataで不一致を上書きしてロード成功にしてはいけません。
5. 途中失敗時はwriterがエラーを返し、旧CURRENTの完成版を利用可能なまま保ちます。新規保存で旧版がない場合は、未完成版を発見して自動採用せずロード失敗にします。readerが保持している旧版をwriterが直後に削除しないでください。版の自動掃除はこの修正に含めず、再現可能な旧版を残します。
6. `var_history`のartifact識別を、ロードで選んだactive version-idと内容digestに対応させます。stagingや未採用の履歴版を読むたびcache keyが変わる設計にしません。fingerprint計算時とモデル読込み時に違うCURRENTを選ばないよう、run内で採用版を固定し、そのIDを結果へ残します。保存形式だけ変え、VaRが旧モデルの履歴を再利用する状態を作らないでください。

旧root直下のmodel.pkl/metadata.jsonには本体と来歴の結合根拠がないため、production loaderでは引き続き未検証として拒否する方針を推奨します。形式変換だけで学習履歴の信頼性は回復しません。既存artifactの再学習は、下記の運用準備作業で別途行います。ディレクトリ構成・loaderの戻り値（`MLOrderOverlayModel`）・学習CLIのhelpを揃え、呼出し元がroot/model.pklを直接読む経路が残っていないか`rg`で確認します。

| 回帰ケース（追加先：tests/unit/test_ml_order_overlay.py、tests/unit/test_pipeline_speed_caches.py） | 修正後の必須期待値 |
|---|---|
| 正常save/load | 予測値、モデル内外の来歴、schema、version-idが一致 |
| 本体書込み前・書込み途中・metadata書込み後・CURRENT切替直前で失敗 | 旧完成版だけをロード。旧本体＋新metadataという組合せは起きない |
| readerがCURRENTを読んだ直後に新CURRENTを公開 | readerは固定した旧版を読み切る。次のreaderは新版を読む |
| 本体1byte変更、metadataのtrain_end/schema/version-id変更 | digestまたは内外一致検査で明確に拒否 |
| train_end当日と翌取引日 | 当日は拒否。翌取引日は他の条件を満たせば利用可能 |
| 同じartifactルートでactive版を更新 | VaR cache keyが変わる。無関係なstaging作成だけでは採用版を変えない |
| legacy artifact、必要キーnull/NaT | エラー理由を伴って拒否。MLなし通常運転へ密かに切り替えない |

writer/reader競合はthreadのsleep頼みではなく、読み終えた位置・公開前の位置へイベントやモックを挿入して再現します。LightGBMの再学習は単体テストに不要で、既存のdummy modelで版と予測の違いを表せます。**完了条件は、readerが旧完成版か新完成版のどちらかを読み、混在版を決して採用しないこと、および適用日制約とcache識別が同じ版に対して働くことです。**

**S05の実装仕様：JP休場日の空行だけを計算前に取り除く**

変更対象は`preprocessor.py::preprocess_data`です。今回の必須再現はJP休場日の全NaN行です。既存の`core/market_calendar.py::is_market_closed`は日本市場用で、USにそのまま使用できません。`pyproject.toml`にはUS取引所calendar用の依存は現在ありません。JPの修正を、根拠のないUS祝日表の追加へ広げないでください。

1. indexを日付へ正規化した後、TOPIXを別Seriesへ分離する**前**に、JP open/closeの休場日空行を正規化します。関数を分けるなら`_remove_closed_jp_padding_rows`等を新設し、元の`data`を書き換えずコピーを返すようにします。
2. 除去条件は「JP calendarで休場と確認できる日」かつ「その日のJP open/closeに有効な観測値がない」です。TOPIXも含め、片方のframeに値があれば単なる全NaNのpaddingと扱いません。片方のindexにだけ空行がある場合も含め、両frameへ同じ除去日集合を適用します。calendarで確認できない営業日の全NaN行は残し、既存strict検証の対象とします。
3. 正規化後の同じJPセッションindexを、`jp_valid_dates`、`jp_all_dates`、JP close-to-close、gap、TOPIX overnight、betaの入力へ使います。TOPIXだけ元indexを残すとgap/betaの不整合が再発します。`pct_change(fill_method=None)`のように暗黙の欠損補間をしない意図もコードへ明示します。
4. `jp_o / jp_c.shift(1) - 1`の「前行」が前の実セッションcloseになることを確認します。`dropna(thresh=...)`した日だけで前日比を計算すると、本来の営業日のデータ欠損をまたぐ複数日リターンを正常な1日分として扱うため、この方法へ置き換えません。
5. 同日のUS終値をJP当日へ使わないprovisional制約、先頭warm-upの扱い、当日未確定closeの保持を維持します。全NaNだからという理由で最新の営業日placeholderを消さないでください。JPだけ休場のときのUSセッションはUS側に残します。

US側に全NaNの休場paddingを扱う拡張を行う場合は、US取引所セッションの根拠を持つ独立のcalendarが必要です。US営業日の欠損を「日本が休みだから」と除去しません。S05のJP修正を閉じる際は、USのpaddingまで検証したかどうかを報告で区別します。

| 回帰ケース（追加先：tests/unit/test_preprocessor.py） | 修正後の必須期待値 |
|---|---|
| 2026-08-11のJP全NaN行あり/なし、価格一定100 | strict/non-strictとも8月12日・13日を保持し、両入力のdf_execが一致 |
| 休場前close=100、休場明けopen=102・close=103 | 明け日のgap=0.02、当日のOC=103/102−1。JP close-to-closeも実セッション間で0.03になる対応行を検査 |
| TOPIXも同じ休場padding、open/closeの片方だけ空行あり | TOPIX系列とJP系列の対応が揃い、indexずれによる余分なNaNが発生しない |
| 既知のJP営業日を全NaNに変更 | strictでエラー。営業日の欠損を消して前々日からのリターンへ変えない |
| 最新営業日のclose未確定、最初のwarm-up、同日US closeを摂動 | 既存provisional/warm-upの回帰を維持。当日JP判断へ同日US終値を混ぜない |
| paddingのない既存の市場別入力 | 正規化による値変更なし。正常系列・列順を維持 |

`pd.bdate_range`は祝日を除外しないため、テスト入力のcalendarを意図して作ってください。回帰の期待日は実行日のtodayに依存させません。**完了条件は、休場行の有無だけが異なる二つの入力から同じ結果が得られ、営業日の真の欠損は引き続き検出されることです。**

**既存ML artifactと段階Cを閉じるための別作業**

S04は保存の正しさの修正です。`models/ml_order_overlay/phase2_8`のprovenance不足はS04の単体テストが通っても解消しません。移行を依頼された担当者は、検証可能なdf_exec・gap bundle・解決済み設定を固定し、学習期間と評価期間を先に決め、既存学習CLIで別の候補出力先へ生成します。実口座や本番出力先をテストに使いません。

候補artifactについて、全評価日の`trade_date > train_end`、必要なラベルの確定時点、入力データ範囲を検査します。一つのartifactで評価期間を賄えなければfold別artifactが必要です。適用できない日をflatとして記録する実装を利用しても、それを「同一ML戦略を全期間評価済み」とは報告しません。全評価日数・artifact適用可能日数・実際に適用した日数・flat理由を分けて残してください。

本番とBTの一致検査は、同じ候補artifact、解決済み設定、df_exec/snapshot、gap bundle、PIT履歴で行います。比較するのは同じ段階の`w_final`で、ML前のweightsとML後のweightsを比べません。モデルnet/grossとside_leverage後の実効値も分けます。値の差があるなら、入力→分布→score→RuleD→MLのどこで変わったかまで記録します。候補でこの検査が完了することと、本番configへ反映することは別の作業です。未準備なら「コード修正済み・artifact移行未了」と報告してください。

**統合検証と担当者の完了報告**

各S項目の失敗再現→回帰テストを通した後、統合したworking treeで次を実行します。以下は将来の修正担当向けコマンドであり、今回の文書追記で全テストを再実行したという記録ではありません。新設テストも`tests/`以下へ配置し、全体実行に含めてください。

```sh
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n 4 --tb=short -q
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python -m compileall -q src/leadlag tests tools scripts src/research
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python -m ruff check src/leadlag tests
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/python -m mypy src/leadlag
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 .venv/bin/lint-imports
git diff --check
```

変更した研究コード・toolsもRuffの対象へ追加します。依存が不足する検証は未実行と記録し、全体PASSへ含めません。監査閾値の緩和、テストskip、全fixtureのflat化、未来日付を過去へ書換える方法で通してはいけません。正常weightsのfixtureが変わったら、どの計算契約の修正による差かを説明してから更新します。

設計文書は、安全境界ADRと`docs/ARCHITECTURE.md`へ結果の流れ・例外/戻り値・artifact版の契約を、`docs/モデル技術仕様書.md`へ入力日付・休場処理・ML適用期間を反映します。設計案の記述と実装済みの記述を分け、未実装の将来計画を現在の保証として書かないでください。

完了報告はS01〜S05ごとに「変更ファイルと関数／追加テスト名／修正前と修正後の観測値／正常経路を維持した証拠／残る制約」を並べます。全テストの件数・警告・ログ・終了コード、検証したHEADと未commit差分の識別、artifact移行の成否も残します。**全S項目の修正完了と、段階A〜C全体の完了は別判定です。** 後者には元の完了条件、特に本番/BTの同一入力でのweights一致と利用可能artifactの確認が必要です。

---

## 修正実装結果（2026-09-13）

上記S01〜S05の実装を行い、修正後の回帰を別証跡へ保存した。対象コードはHEAD `cf97bfaf4577a20c95ce931afd10ac72a6f08285` 上の未commit作業ツリーにある。既存のユーザー変更を含む作業ツリー全体を保持している。

### S01 — MH分布の来歴を監査へ接続

- `distribution_source.py` に `DistributionResult.metadata`、`df_exec`実行行からの来歴生成、`sig_date`/`signal_date`の別名一致、`trade_date`・`horizon`一致、NaT・空値・未来日拒否を実装した。
- `overlay_applier.py::_multi_horizon_scores_with_metadata` は本番MH分岐でh=1/3/5ごとに `FallbackPolicy.resolve` を使い、μ・Ωと同じ取得結果のsource/metadataを混合診断へ保存する。来歴不正cacheは採用せず、許可されたon-demandへ同じhorizonで切り替える。全horizonが解決できない場合はh=1だけへ縮退せずflatにする。
- `decision_engine.py` はMH失敗時に `audit_failure=true`、`gap_data_missing=false`（来歴不正の場合）、`diagnostics.distribution_provenance.status=rejected`を保持する。`audit_comparator.py` は寄与した全signal日をリーク監査へ渡す。
- 回帰 `test_multihorizon_future_provenance_flats_in_real_decide_path` はh=3の未来 `sig_date=2026-08-17`（trade date 2026-08-14）で `w_final=0`、`audit_failure=true`を確認した。`test_multihorizon_rejects_bad_cache_provenance_and_recomputes_on_demand` はh=3の不正cacheからon-demandへ切り替わり、source=`on_demand`・signal日=`2026-08-13`を確認した。`test_single_horizon_future_provenance_does_not_reach_audit` は単一horizonにも同じ拒否境界があることを確認した。修正後probeでもMHケースはgross=0、fallback audit failureとなった。

### S02 — 部分失敗後の照合と取消済み約定

- `broker_ops.py` に `OrderExecutionIncomplete`（summary/log_path保持）を追加し、最小APIログ保存後に同型を送出する。
- `post_decision.py` はこの例外を一度受けて、約定詳細、position、wallet、daily journalを独立に試行してから元例外を再送出する。収集失敗は `reconciliation_errors` とAPIログへ保存する。
- `output_ops.py` は照会失敗を `raise_on_error=True` で上位へ通知でき、照会成功・建玉0件は空のposition snapshotとして保存する。`pricing.py::fetch_fill_prices` は実order_idを持つCANCELLED/EXPIREDも照会し、注文数量と累積 `fill_quantity`を分離する。
- `test_partial_submission_reconciles_before_propagating` は不完了例外を維持したまま fills/positions/wallet/journalの全呼出しを確認し、`test_cancelled_order_still_queries_accumulated_fill` はCANCELLED・注文100株・累積約定30株を確認した。`test_fill_query_failure_preserves_previously_confirmed_quantity` は再照会失敗で確定数量を消さないこと、`test_empty_position_snapshot_is_distinct_from_query_failure` は空snapshotを確認した。

### S03 — close結果の終了コードと記録

- `close.py::run_close_positions_mode` はclose summaryを返し、照会・snapshot・journal失敗を個別収集してログへ残す。建玉なしでも `close_execution_log.json` と一貫したゼロ件summaryを生成する。client解放は `finally` で維持する。
- `cli.py::_handle_close` は `close_incomplete=true` を終了コード2へ変換する。全件拒否の `close_all_positions` を実行した `test_close_cli_returns_nonzero_for_all_rejected_orders` は、休日早期returnを無効化し、close呼出し1回・終了コード2・client.closeを確認した。
- 修正後probeの観測値は `filled=0 / pending=0 / failed=1 / close_incomplete=true / cli_return_code=2`。

### S04 — ML artifactの一組公開

- `ml_order_overlay.py::save_overlay_model` はmodel/metadataを一つのversion directoryへfsyncしてから、atomicな `CURRENT` pointerを切り替える。ロードはCURRENTを一度解決し、artifact version、model SHA-256、model/metadataの学習期間・data/config hash一致を検証する。
- `verify_overlay_models.py` はCURRENT配下を検証し、`fix_overlay_metadata.py` はimmutable versionを直接書き換えず再学習・新version公開を促す。学習CLIの出力説明もversioned artifactへ更新した。
- `test_overlay_publish_failure_keeps_previous_active_version` はCURRENT公開失敗後も旧versionをロードできること、`test_overlay_loader_rejects_active_model_digest_mismatch`・`test_overlay_loader_rejects_external_provenance_mismatch` は本体・外部metadataの改変を拒否すること、`test_overlay_loader_rejects_path_traversal_current_pointer` はCURRENTの逸脱を拒否することを確認した。`test_overlay_save_does_not_accept_conflicting_structural_metadata` はcaller metadataによるschema上書きを拒否する。修正後probeは公開失敗後もmarker=`trained-through-2026`、train_end=`2026-08-14`、active version unchangedを確認した。
- 既存 `models/ml_order_overlay/phase2_8` は検証可能なtemporal/data provenance metadataを持たないため、ProductionRunnerが従来どおり拒否する。根拠のないmetadata補完や実artifact移行・再学習は今回実施していない。

### S05 — JP休場日のNaN padding

- `preprocessor.py` はJP calendar上の非取引日で、JP close/openの全対象列がともにNaNの場合だけ計算前に除去する。JP営業日の全NaNは残し、strict validationでデータ欠損として扱う。US側の休場行をJP理由で除去する処理は追加していない。
- `test_jp_holiday_nan_padding_does_not_remove_following_sessions` と修正後probeで、2026-08-11のNaN行を除去した後も2026-08-12/13の有効行が残り、strict/non-strict両方で復元することを確認した。

### 検証結果と残る判定

- 全テスト: **605 passed / 17 warnings / 707.31s**（`.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python -m pytest tests/ -n 4 --tb=short -q`）。警告は既存の定数入力相関・ゼロ除算に関するもの。
- Ruff（変更対象）、mypy（`src/leadlag` 120 files）、compileall、import-linter（154 files / 380 dependencies、4 contracts kept / 0 broken）、`git diff --check` は全てPASS。リポジトリ全体を `src/research` まで含めたRuffは、今回の変更対象外にある既存66件で失敗するため、その結果を全体PASSとは扱っていない。
- 修正後のオフライン証跡は [implementation_reproductions.json](./implementation_reproductions.json)、stderrは [implementation_reproductions.stderr.log](./implementation_reproductions.stderr.log) に保存した。
- 下位モデルが集積しやすい機械可読の判定一覧は [implementation_summary.json](./implementation_summary.json) に保存した。
- 既存の `review_manifest.json` と `validation.json` は修正前レビューの履歴証跡であり、今回の実装結果を上書きしていない。今回の修正後判定は本節と `implementation_reproductions.json` に分けて記録した。
- S01〜S05の修正実装と回帰検証は完了した。ただし、段階A〜C全体の完了は別判定であり、利用可能な本番ML artifactの移行、実データでの本番/BT同一入力weights一致、収益性・OOS検証、実brokerとの照合はこの依頼では実施していない。
