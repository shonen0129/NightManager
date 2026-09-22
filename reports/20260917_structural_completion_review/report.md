# S0–S8 実装確認・追加修正

実施: 2026-09-17〜18 / 対象: [構造改善計画](../20260915_structural_improvement/plan.md)

## 1. 結論

**S0–S8全体は部分完了。完了報告だけでは確認できなかった不具合・移行漏れを修正した。**
とくに、注文の照合前完了、日付をまたぐ未解決run、期限後に残る子プロセス、旧h=3/5 bundleの拒否、
研究overlayがBTへ反映されない経路を修正した。15:40の既存ジョブへ読取専用の復旧照合も接続した。

S2には、旧compatibility経路の全履歴アクセスと一部の暗黙の入力取得が残る。as-of計算viewと
strict typed pathの入力受け渡しは追加したが、実データでの入口間比較、verified本番ML
artifact、実scheduler・実口座の証跡も不足する。これらを完了扱いにしない。
実装済みのサブ範囲と全体の受入条件を分け、計画・ADR・Architecture・運用手順・roadmapを更新した。

## 2. 比較基準と変更範囲

- 開始時HEAD: `cf97bfaf4577a20c95ce931afd10ac72a6f08285`。開始時点に多数の未commit変更があった。
- HEADへ戻さず、修正前の作業ツリー492ファイルを[hash一覧](before_manifest.json)と
  [ソースarchive](before_source.tar.gz)に固定した。[開始時status](before_status.txt)も保存した。
- 本レビューの差分は[review_changes.patch](review_changes.patch)。既存のユーザー変更からの
  追加差分であり、HEADとの差分全体ではない。文書は開始時archive対象外のため、このpatchに含めていない。
- [修正後hash](after_manifest.json)と[検証用digest・数値比較一覧](evidence.json)を保存した。
  binary fixtureは別途hashで識別し、既存の数値期待値を更新して差を隠す変更はしていない。
- 実口座への発注・決済・取消・再送、schedulerの登録変更、メール送信、本番ML学習・artifact更新は実行していない。
  broker経路の検証はstubと一時SQLiteで実施した。

開始時の全`tests/`は**722 passed / 18 warnings / 373.14秒**。
警告は研究テストの定数相関・ゼロ除算等であり、警告を抑制して成功にしていない。

## 3. 修正した不具合・移行漏れ

### 3.1 注文終端と照合完了の混同（S4/S5/S6）

**問題:** 発注結果がFILLEDになるとrunがcompletedになり、約定詳細・建玉・余力・journal保存の
失敗後も復旧候補から外れ得た。再照合失敗の更新や、reconciliation_requiredからの完了も不足していた。

**修正:**

- broker送信処理はpending checkpointまでにとどめ、後処理が確認済み約定と開始在庫から
  期待建玉を求め、保存snapshotの実在庫と照合する。
- `mark_completed`は成功したcomplete checkpointを必須とする。記録失敗もCLI結果へ伝える。
- 新規と決済の両経路で、照合終了までleaseを保持する。short開始在庫は符号付きで集計する。
- 同一runの異なる注文計画を無視して使い続けず、数量等の相違をtransaction内で拒否する。
- 分割注文と最終summaryの並び替えでもbroker IDからintentを維持し、分割ごとのrequested quantityを記録する。
  取得後のfill価格・feeも永続化し、未取得費用をゼロ確定にしない。
- SQLite connectionをtransaction終了時に閉じる。

実装: [state_store](../../src/leadlag/execution/state_store.py)、
[broker_ops](../../src/leadlag/execution/broker_ops.py)、
[post_decision](../../src/leadlag/execution/post_decision.py)、[close](../../src/leadlag/execution/close.py)。
再現と回帰: [初期再現](reproductions.log)、
[execution recovery tests](../../tests/unit/test_execution_recovery.py)、
[structural completion tests](../../tests/unit/test_structural_completion.py)。

### 3.2 未解決run・再起動後の復旧入口（S5）

**問題:** 同じ日・jobだけを止める制約では、送信成否不明のまま翌日や別jobで新規注文できる。
復旧候補の列挙だけでは、確認済みrunを安全に完了へ進められない。

**修正:** 同じ口座・戦略の未解決runは日付・jobをまたいで送信を止める。
`execution.reconcile`は保存済みbroker IDについて注文状態・約定・建玉・余力を読み、
開始在庫と照合したjournalを保存してから完了する。ID不明・数量不足・残高不一致・記録失敗は
未解決のまま残す。送信・取消・再送を行う処理は持たない。

既存の15:40 `run_pnl_report.sh`へ`reconcile --pending`を接続した。
設定済みbrokerの`production_v2`だけが対象で、simulationや別口座は除外する。
候補なしの場合はbrokerへ接続しない。照合失敗でも独立したPnL資料の収集を試みるが、
batchは非ゼロで終了する。手動の候補表示・特定run照合も使える。

実装: [reconcile](../../src/leadlag/execution/reconcile.py)、
[引け後batch](../../scripts/batch/run_pnl_report.sh)。
手順: [SCHEDULER_SETUP](../../docs/SCHEDULER_SETUP.md)。
回帰は、送信メソッドを持たないbroker stubで正常完了・不明ID・建玉不一致・別口座除外・
候補なし・checkpoint保存失敗を確認した。

### 3.3 子プロセスの残存・排他の迂回・batch連携（S5）

**問題:** process leaderがTERMで終了すると、生存中の子孫をKILLせずに終了し得た。
環境変数の真偽だけでlease取得を省略でき、外部TERMも子孫へ適切に反映されなかった。
decisionとgap batchのguard判定の相違により、子のgap生成がlease競合で拒否される経路もあった。

**修正:**

- leaderの終了後もprocess groupを追跡し、猶予後に残存子孫を停止する。SIGTERM/SIGINTと正常終了後の
  background workerにも対応する。
- 親がowner/scope/DBを渡し、発注入口がSQLite上の有効leaseを検証する。
  decision・gap・close・引け後reportのshell入口を同じownerトークンへ統一した。
- decision内のgapは親の全体期限を共有する。引け後の照合・reportも全体期限付きで実行する。
- 未使用の`wait_and_auto_close`を撤去し、durable state/leaseを迂回する別決済経路を残さない。
- dry-runのaccount namespaceを実口座と分ける。decision batchの`--gap-dir latest`上書きを撤去し、
  本番設定のSQLite参照を維持する。gap失敗時のログも実装どおりのfallback順へ訂正した。

回帰: [job guard tests](../../tests/unit/test_job_guard.py)。実processで期限・外部TERM・子孫停止を確認した。
4つの実shell入口のguard部分を、単独/親からの起動で実行した。引け後batchは照合を失敗させても
reportを試行し、終了コードを非ゼロにすることを、通信しないinterpreter stubで確認した。

### 3.4 入力の所有権・時刻・旧経路（S2）

**問題:** `HistoricalInputs.calculation_frame()`が内部DataFrameを返し、入力版の中身を変更できた。
timezone変換、当日09:10のsnapshot取得、要求日とsnapshotの不一致に不足があった。
予測契約に評価targetを渡せる設計や、複数の旧入力入口も残っていた。

**修正:**

- calculation frameはpandas 3のCopy-on-Writeを使う分離viewにし、毎日の全履歴deep copyを避ける。
  旧pandasではdeep copyへfallbackする。
- KnownMarketInputsの配列をbytes-backedにし、`setflags(write=True)`でも変更できなくする。
  as_ofをJSTへ変換し、未来のobserved_at・日付不一致を拒否する。
- EvaluationInputsをDecisionInputsから除外する。
- `RunnerInputs`を撤去。Runnerと内部engineはDecisionInputsを使い、公開`decide`の旧引数だけを
  外側で一度正規化する。df_exec/lakeの同時指定を拒否する。
- `data/cache.py` shimを撤去し、研究・tools・testsの利用者とpatch先を正本へ移行する。
- targetの純粋計算を`core.target_returns`へ移し、取得は`data.intraday_inputs`に集約する。
  h=1の既存演算と欠損処理を保持し、preprocessorの中継を撤去する。

**この修正だけでS2は完了しない。** `calculation_frame(as_of)`で通常のtyped decisionの
可視性は閉じたが、旧compatibility経路、horizon別の終了時刻、モデル途中の一部暗黙I/Oは6章の残件である。
copy保護とstrictly historicalは別の条件である。

### 3.5 分布取得と旧NPY manifest（S3）

- モデルの可変な`_current_gap_input_dir`を撤去し、source/policyへpathを明示する。
- 公開`compute_distribution`をsingle/MHと同じFallbackPolicyへ接続し、重複resolver moduleを撤去する。
  cache-only取得でも検証済みmetadataを保持する。
- `generate_v2_production_portfolio`の別構築wrapperを撤去し、利用者をtyped configと
  `ProductionV2Model`へ移行する。学習・特徴量生成のML無効条件は明示して維持する。
- 旧format_version=1のh=3/5 manifestをh=1と解釈して実fixtureを拒否する不具合を修正した。
  horizon省略時は**SHA-256検証済みmetadata**から復元する。新schemaの明示horizon、payload digest、
  trade date、provenance、shapeの検査は維持する。

旧fixture2件の失敗は[修正前再現](legacy_bundle_reproduction.log)、修正後は
[bundle tests](bundle_tests.log)と最終全テストで確認した。NPY readerは具体的な固定fixtureの
利用者がある対応形式として維持し、無条件に古いbundleを許可する変更はしていない。

### 3.6 VaRの設定とsnapshot/worker（S1/S3）

- 本番設定の継承とCLI相当のslippage上書きを一度だけ解決し、その有効設定をcache keyとBTの両方へ渡す。
  broker認証情報をdigestのために保存しない。選択済みoverlay objectも同じものを渡す。
- fingerprintとcopyを`var_inputs`、計算worker・遅延保存・snapshot寿命を`var_worker`へ分離する。
- 実際のPIT loaderが読む隣接`full_history_diagnostics.csv`をgap snapshotとkeyへ含める。
  原本が訂正されても実行中snapshotは変わらず、次runではkeyが変わる。
- 全体の絶対期限、遅延結果の不採用、worker終了前に入力を消さない条件を維持する。

回帰: [VaR/deadline tests](../../tests/unit/test_stage_abc_followup_fixes.py)と
[PIT履歴snapshot回帰](../../tests/unit/test_structural_completion.py)。
macro/ADR/09:10等の入力固定はまだ含まれず、全入力を捕捉したkeyとは扱わない。

### 3.7 研究overlayとwheel（S7/S8）

**問題:** phase 1/2 driverは旧generatorをmonkeypatchしていたが、現行BTはモデルを直接呼ぶため、
研究overlayが実際のweightsへ反映されなかった。

**修正:** `run_v2_backtest(decision_transform=...)`へ明示的に接続する。
gap/audit fallbackは変換せず、変換後にも数値監査を適用する。監査失敗はflat化する。
phase 2の状態付きEMAは従来どおり逐次処理に限定する。
[BT回帰](../../tests/unit/test_backtest_transform.py)で、実weightsへの反映と監査違反のflatを確認した。

production wheelを新しい一時source treeで実ビルドし、削除済みmoduleやresearchが混入しないことを確認した。
一時installationからrepository sourceを除いた状態でCLI helpとversioned合成ML artifactの保存・読込・推論を実行した。
pickle class pathは`leadlag.models.ml_order_overlay.MLOrderOverlayModel`のまま保持する。
CIにもこのsmokeとJUnit保存を追加した。hosted CIを起動したという意味ではない。

### 3.8 CIテストのローカルcache・ネットワーク依存（S8）

Sprint診断テストはローカルdf_exec/5分足cacheとyfinanceの価格・出来高に依存しており、
この環境で通ることを空のCI環境の再現性と扱えなかった。
[研究テスト用fixture](../../tests/research/conftest.py)で、取得境界だけを固定df_exec・
価格/出来高・09:10 bars・macro・一時診断CSVへ置き換えた。モデル・診断・損益計算は実物を実行する。
09:10実データ相当と欠損時proxyの両方を入力し、校正結果が空の場合にassertionを省略する
テストを、非空のrolling/full-sample校正を検査するテストへ強化した。

### 3.9 時点履歴とrun所有入力（R1/R2の追加実装）

R1/R2のうち、コードで閉じられる入力境界を追加した。

- `HistoricalInputs.calculation_frame(as_of)`が予測日までに切り詰め、当日・未来の
  close/target列をマスクする。h=1/3/5の未来target摂動を同じ可視viewに対して比較する回帰を追加した。
  内部の完全履歴は評価・再現用に保持し、予測計算へそのまま渡さない。
- `HistoricalInputs`に`open_910_returns`、`macro_prices`、`adr_features`をrun所有のコピーとして追加し、
  fingerprintへ含めた。BTは09:10入力をadapter境界で一度取得し、BLPX on-demandへ明示的に渡す。
- typed decisionでは、intraday target、PIT IR、macro、ADRを不足したままモデル内部で再取得しない。
  explicit frame/historyがない場合は警告・安全側のfallbackとなる。旧公開adapterと`pit_lake` compatibility sourceは
  既存のoffline挙動を保つため、adapter境界の暗黙取得を許容する。
- 対象回帰20件とBLPX/V2回帰89件を実行した。loaderを呼ぶと失敗するstubで、strict typed fallbackが
  macro/PIT sourceを再度開かないことも確認した。

これはR1/R2全体の完了ではない。rank-reversalの入力snapshot化、主要adapterでの実データ・観測時刻の充填、
全入力を含むVaR key、live/BT/gap入口の実データ比較は残る。

## 4. 検証結果

最終版の検証結果は次のとおり。途中の失敗・中断は下記に別記する。

| 検証 | 結果・証拠 |
|---|---|
| 修正前の全テスト | 722 passed / 18 warnings。[baseline_tests.log](baseline_tests.log) |
| 執行復旧・4 batchの対象回帰 | 24 passed。[recovery_batch_tests.log](recovery_batch_tests.log) |
| 固定入力でのSprint診断 | 6 passed / 15 warnings。[sprint_fixture_tests.log](sprint_fixture_tests.log) |
| 修正後の全`tests/`（R1/R2追加後） | **764 passed / 16 warnings / 104.79秒**。[final_tests_after_r1_final4.log](final_tests_after_r1_final4.log)、[JUnit](tests_after_r1_final4.xml)、[全体期限の実行記録](test_guard_after_r1_final4.json) |
| Ruff（今回変更したproduction/data・対象unit） | PASS。なおresearch全体を含む一括実行には既存のimport/未使用警告が残る。[ruff_after_r1.log](ruff_after_r1.log) |
| mypy production | 143 source files、PASS。[mypy_after_r1.log](mypy_after_r1.log) |
| import-linter | 7 kept / 0 broken。[imports.log](imports.log) |
| compileall | PASS。[compile_after_r1.log](compile_after_r1.log) |
| lock固定 | offline `uv lock --check` PASS。[lock.log](lock.log) |
| 4 batchの構文 | `bash -n` PASS |
| wheel実ビルド・隔離インストール | PASS。[wheel_build.log](wheel_build.log)、[wheel_smoke.log](wheel_smoke.log) |
| 固定入力の変更前後比較 | 8ケースで厳密一致。R1/R2後replayも同一。[evidence.json](evidence.json)、[replay_after_r1.json](replay_after_r1.json) |
| 文書リンク・差分空白 | 138リンク検査・`git diff --check` PASS。[docs_after_r1.log](docs_after_r1.log) |

初期再現ログは意図的な失敗を含む。途中の全体実行（750件の収集）は引け後job追加前の版で
開始していたため中断し、[途中ログ](interrupted_tests_before_recovery_wiring.log)を保存した。
続く全体実行でも研究テストのデータ取得依存を確認したため、[fixture導入前の途中ログ](interrupted_tests_before_ci_fixture.log)
を保存して中断した。いずれもPASSに数えず、最終版を改めて全体検証する。
固定fixture導入後の初回全体実行は753 passed / 3 failedだった。外側の検証用job_guardが
渡すlease環境変数とテスト用SQLiteが不一致になり、productionの検証が正しく拒否したためである。
[失敗ログ](validation_environment_failure.log)を残し、検証子プロセスだけから当該環境変数を除いて再実行した。
発注側のlease検証やテストのassertionは緩めていない。

最終の16 warningsは研究計算の定数相関・ゼロ除算15件と、Python 3.12のmultithread状態での
forkに関するDeprecationWarning 1件である。警告を抑制していない。実行時間は研究テストの
外部取得を固定fixtureへ変えた後の値であり、本番計算の速度改善とは扱わない。

主な再実行コマンド（repository root、既存`.venv`）:

```bash
.venv/bin/python -m leadlag.execution.job_guard \
  --scope validation:s0_s8_final --timeout 1800 --grace 10 \
  --state-db /private/tmp/leadlag-completion-final-validation.sqlite \
  --guard-log reports/20260917_structural_completion_review/test_guard.json \
  -- env -u LEADLAG_LEASE_OWNER -u LEADLAG_LEASE_SCOPE -u LEADLAG_LEASE_DB \
  .venv/bin/python -m pytest tests/ -q \
  --junitxml=reports/20260917_structural_completion_review/tests.xml

.venv/bin/python reports/20260912_workspace_audit/watchdog.py 240 \
  .venv/bin/python reports/20260917_structural_completion_review/verify_distribution.py
```

## 5. 数値比較の範囲

[replay.py](replay.py)を別々の修正前/修正後source treeで実行した。
[before](replay_before.json)と[after](replay_after.json)は厳密一致した。

- h=1/3/5 × cache/on-demandの6ケース: μ・Ωを比較。
- single/MHの2ケース: scores・weights・PIT・fallbackを比較。非ゼロのweightsも必須にした。
- 固定df_exec・h=1/3/5の行列は既存regression fixtureを使用する。旧manifest読込の修正と数値比較を
  分けるため、一時storeへ同じ行列を現行manifestでpublishする。
- **ML/macro/cs overlayは無効、09:10調整はゼロ固定**。本番ML artifact・ADRの比較は含めない。
- cacheとon-demandは、それぞれの**変更前後**を比較した。両source同士の同値性を証明したものではない。
  実際のgap pipelineとon-demand、liveとBTの全機能比較でもない。

このレビューは構造・安全な失敗処理の検証であり、収益性能の改善実験ではない。
Sharpe・DD等の新たな採用判断、モデル・費用パラメータの最適化、本番昇格は行っていない。

## 6. 全体完了に必要な残件

| ID / 対象 | 未完了の具体的な内容 | 閉じるための作業・検証 |
|---|---|---|
| R1 / S2 | `HistoricalInputs.calculation_frame(as_of)`で予測日までの行と当日・未来targetの可視性をAPI化し、h=1/3/5の未来target摂動回帰を追加した。 | horizon別のラベル終了時刻（9:10/引け）をデータ仕様として全target列へ付与し、既存BT/gapの全入口で同じcutoffを検証する |
| R2 / S2・S3 | `open_910_returns`・macro・ADR・PIT・rank-reversalをrun所有`HistoricalInputs`へ渡す欄と、strict typed pathでの暗黙再取得禁止を実装した。旧compatibility adapterと主要adapterの観測時刻充填は残る。 | 全取得済み入力をsnapshotへ渡し、原本訂正後もrunが不変・次runが更新を認識することを全有効機能で確認。利用可能時刻の証拠がない過去データは完全PITと扱わない |
| R3 / S0・S1・S3 | Step 1の共分散とon-demand、live/BT/gap/VaRを揃えた実データの全機能比較が不足 | prior・residualization・target・9:10価格・共分散版・gap係数を固定し、raw μ/Ω→gap μ/Ω→scores→weightsの差を分解。許容誤差拡大やfixture更新で吸収しない |
| R4 / S5・S6 | 復旧・15:40接続・会計のコードとローカル回帰はあるが、実schedulerの登録状態と実口座の約定・費用・残高突合は未確認 | 実環境で停止・部分約定・翌日復旧を照合し、注文ID不明時の運用判断と会計残差を記録。新たな送信を伴う確認は対象操作を明示する |
| R5 / S7 | 現行本番MLのverified versioned artifact再生成と本番/BT weights照合が未実施 | 学習期間・data/config hash・特徴量契約を持つartifactを生成し、同じartifact・入力で両経路を比較。synthetic artifactのwheel smokeを代用しない |
| R6 / S8 | ローカル検査は確認したが、remote hosted CIの実行結果はない | PR/CI上で同じlockとworkflowを実行し、JUnitとwheel artifactを確認 |

R1/R2には実装済みのサブ範囲と未実装の構造境界が混在しており、単なる運用証跡待ちではない。
例: [fallbackのmacro/PIT取得](../../src/leadlag/models/v2/fallback.py)、
[MLのADR取得](../../src/leadlag/models/ml_overlay_inference.py)、
[BLPXのintraday target取得](../../src/leadlag/models/blp_base.py)。
VaRの現在のkeyはこれら全ての外部データ内容を捕捉しておらず、全入力再生の受入条件は未達である。
一方、全履歴が見えるという構造欠落だけを、実際の計算リークを再現した証拠とは扱わない。

## 7. 更新した正本

- [計画](../20260915_structural_improvement/plan.md): 冒頭に段階別の最新判定を追加し、過去PASSとの混同を解消。
- [ADR](../../docs/decisions/2026-09-15-structural-improvement-boundaries.md): 境界の削除・状態遷移・復旧・入力所有権の理由を記録。
- [Architecture](../../docs/ARCHITECTURE.md): 実際のmodule/API・台帳・batch・wheel検査へ更新。
- [scheduler手順](../../docs/SCHEDULER_SETUP.md): 期限・排他・候補表示・read-only復旧・15:40連携を記載。
- [CI](../../docs/CI.md) / [研究環境](../../docs/RESEARCH_ENV.md) / [roadmap](../../docs/refactor_roadmap.md): 最新の受入範囲と残件を揃えた。

**判定:** 今回の不具合修正と移行漏れの是正をもって、S0–S8全体完了や本番昇格可とは宣言しない。

## 8. 2026-09-18 追補

追加レビューで判明した入力境界の2点を修正した。

- `HistoricalInputs` の `open_910_returns`・macro・ADR は内部所有フィールドへ分離し、公開プロパティは深いコピーを返すようにした。取得後の呼出元のDataFrame変更で `InputVersion` が変わらないことを回帰で確認した。
- Backtest と V2 live bridge は、09:10入力・有効時のmacro/ADR・PIT履歴をadapter境界で明示的に組み立てて `DecisionInputs` へ渡すようにした。BacktestのPIT履歴は日付別に固定し、各日の計算が将来日の履歴を参照しない。strict typed pathではloaderを再度開かない。

追加・関連回帰はPASS、対象ファイルのRuffもPASS。追補後の全`tests/`は **770 passed / 16 warnings**、compileall・mypy・import-linterもPASS。

## 9. 2026-09-18 レビュー指摘の修正

追加レビューで確認した2件を修正した。

- PITのIR配列と履歴取引日を `HistoricalInputs` に対で保持し、Backtest/live bridgeから `run_leakage_audit` まで伝搬した。strict typed pathで日付が欠落・件数不一致の場合は、監査を省略せず失敗側へ倒す。
- ADR artifactはBT/liveともrun所有の全体フレームを受け取り、overlay適用時に同じtrade date・鮮度判定を行うよう統一した。欠損日・期限切れでは両経路ともoverlayを適用しない。

回帰は対象38件、全`tests/`は **770 passed / 16 warnings**。対象ファイルのRuffと`git diff --check`もPASS。

## 10. 2026-09-19 レビュー指摘の修正

前節後の全体レビューで判明した、h=3/5 on-demandの9:10入力欠落とPIT境界の2点を修正した。

- typed経路が保持する`open_910_returns`を、h>1 target計算の開始日9:10価格へ再構成するようにした。明示`p_910_df`がある場合はそちらを優先し、欠損時だけopen-to-09:10入力、さらに欠損する場合だけ始値へフォールバックする。h=3のadapter入力回帰を追加した。
- `load_pit_ir_history`は時刻付き`trade_date`をJSTの日付へ正規化してから履歴を切るようにした。同日09:10のIRを履歴へ含めない回帰を追加した。

対象回帰64件・関連回帰66件、全`tests/`は **773 passed / 16 warnings / 105.54秒**。Ruff、対象2ファイルのmypy、全体compileall、`git diff --check`もPASSした。R1〜R6の残件は引き続き未完了であり、S0〜S8全体完了や本番昇格可とは判定しない。

## 11. 2026-09-19 追加レビュー指摘の修正

追加レビューで、h>1のon-demand再構成に使う`open_910_returns`の分母が5分足始値、
`df_exec`の終値再構成が`jp_open_trade`の日次始値であり、両者が異なる日にcache経路と
on-demand経路の9:10価格がずれることを確認した。

- `build_open_910_returns`は、`jp_open_trade_*`が有効な場合は同じ値を分母に使うよう修正した。
  実行フレームに列がない互換経路だけは5分足始値へフォールバックする。
- 5分足始値と日次始値を意図的に異ならせ、h=1/3/5で直接`p_910`経路と
  `open_910_returns`再構成経路が一致する回帰を追加した。

対象回帰は **67 passed**。追加の関連回帰63件も通過し、全`tests/`は **776 passed / 16 warnings / 104.12秒**。
Ruff、対象ファイルのmypy、全体compileall、`git diff --check`もPASSした。この修正で数値経路の不一致は閉じたが、
R1〜R6の残件は別の受入条件として残る。

## 12. 2026-09-20 R1〜R6実施結果

実装・検証結果は [20260920_structural_completion/report.md](../20260920_structural_completion/report.md) に記録した。R1/R2は主要typed経路とVaRのrun-owned入力snapshotまで実施し、R3は同一入力fingerprintでcache/on-demandの2日×h=1/3/5一致を確認した。R4 scheduler、R5一時artifact、R6 local CIも確認した。gap生成全入口のlabel availability証跡、R4実口座照合、R5本番artifact昇格、R6 Hosted CIは外部証跡または再生成条件が必要なため残件として明示している。

## 13. 2026-09-20 追加レビュー指摘の修正

R3の実経路証跡を再確認し、h=1がファイルcacheではなくon-demandへ落ちていた点と、生成bundleの来歴4項目が空だった点を修正した。

- signal dateのStep 1 `Omega_struct` が該当しない場合、古い日付のfallback行列をh=1出力へ流用せず、BLPX共分散を使う。該当日ファイルとlegacy trade-date fixtureだけは明示overrideとして扱う。
- gap bundle metadata / manifestへ `input_version`、`model_version`、`config_version`、`ticker_order` を保存し、実行時にas-of入力・有効設定・銘柄順と照合する。入力が変わったbundleは拒否する。
- 2026-08-14/17のh=1/3/5を再生成し、全て `file_cache` 対 `on_demand` の `ready`、最大差分はμ 2.22e-16、Ω 1.73e-18だった（[再検証証跡](../20260920_structural_completion/evidence/r3_real_paths.json)）。全`tests/`は **789 passed / 16 warnings**。
- stale fallbackの回帰、bundle来歴必須・入力変更拒否の回帰を追加した。対象テストとcompileallはPASS。

この追補でR3の実データcache/on-demand比較とbundle来歴の指摘は完了と判定する。R1/R2/R4/R5/R6の外部受入条件は前節までの残件を維持する。

## 14. 2026-09-20 追加修正（今回）

前節までの再確認で見つかった2件を修正した。

- strict typed経路でrank-reversalの当日snapshot行が欠けた場合、overlayが日付ファイルを暗黙に再読込せず、元のscoresを維持して警告付きでスキップする。従来の互換経路は暗黙読込を許可したままにした。
- PIT診断CSVのキャッシュ判定をmtime/サイズから内容SHA-256へ変更した。同じmtime・サイズで原本が置換されても、次の呼出しで新しいIR履歴を読み込む。

回帰テスト2件を追加し、対象回帰は57件、全`tests/`は **791 passed / 17 warnings / 105.27秒**。Ruff、compileall、`git diff --check`もPASSした。R1/R2/R4/R5/R6の外部受入条件と、S0–S8全体完了の判定は変更しない。

## 15. 2026-09-20 ラベル可用時刻の修正と残件再確認

`HistoricalInputs.calculation_frame()` の明示 `label_available_at` が、利用可能時刻以降の履歴を全て隠していたため修正した。現在は、指定された可用時刻より前の実行時点では当日行だけを隠し、可用時刻到達後は当日行を利用できる。過去行を継続的に削除しない回帰テストを追加した。

関連テスト78件、全`tests/`は **792 passed / 17 warnings / 105.63秒**。Ruff、mypy、compileall、`git diff --check`もPASSした。R1/R2のgap生成全入口・外部入力のavailable_at証跡、R4実口座照合、R5本番期間artifact昇格、R6 Hosted CIはこの環境だけでは完了できないため、未完了のまま明記する。

## 16. 2026-09-20 live bridge as-of修正

V2 live bridgeがtrade dateを00:00のままsnapshotへ渡しながら、09:00/09:10の`observed_at`を設定していたため、`KnownMarketInputs`の時刻検証で`ValueError`になる不整合を修正した。日付キーは従来どおり正規化したまま、snapshotと`DecisionInputs`には09:10 JSTのdecision cutoffを渡すよう統一した。

`tests/unit/test_s2b_input_contracts.py`にlive bridgeの09:10契約回帰を追加した。対象テスト22件、全`tests/`は **793 passed / 17 warnings / 105.29秒**、compileallもPASSした。

## 17. 2026-09-20 指摘3件の修正

追加レビューで確認した、09:10入力のcache来歴、h=3/5 snapshot、BT評価targetの3件を修正した。

- gap bundleのidentityへas-of `open_910_returns` の指紋を追加し、生成側と利用側で同じ入力版を照合するようにした。09:10入力だけが訂正された場合も旧cacheを拒否する。
- h=3/5 on-demandは、直近行をPIT snapshotのgap・beta・TOPIX夜間値へ反映し、過去h-1行と累積する。これによりliveの当日gapを失わず、既存の累積窓も維持する。
- V2 backtestのrealized targetは、decisionと同じrun-owned `HistoricalInputs.open_910_returns`を明示的に使用する。評価時に別cacheを再取得しない。

回帰を3件追加し、関連回帰と全`tests/`は **795 passed / 16 warnings / 100.4秒**（watchdog 300秒）、RuffとcompileallもPASSした。詳細と証跡は[20260920実施結果](../20260920_structural_completion/report.md)と[r6_local_ci_current.json](../20260920_structural_completion/evidence/r6_local_ci_current.json)に更新した。
R1/R2の全gap生成入口・available_at証跡、R4実口座照合、R5本番artifact昇格、R6 Hosted CIは未完了であり、S0–S8全体完了や本番昇格可とは判定しない。

## 18. 2026-09-20 PIT timezone修正

PIT診断CSVの`trade_date`にtimezone付き時刻が含まれる場合、UTC日付のまま比較され、
JSTの同日09:10時点で除外すべき行が履歴へ混入する不具合を修正した。
読み込み時に各行をJSTへ変換してから日付キーへ正規化し、naiveな既存JST日付との混在も扱う。

回帰テストで`2025-06-01T15:00:00+00:00`（JST 6/2 00:00）が6/2のPIT履歴へ入らず、
前日の`2025-06-01T14:59:59+00:00`だけが残ることを確認した。
対象テスト7件、全`tests/`は **796 passed / 17 warnings / 105.15秒**、Ruff、compileall、
`git diff --check`もPASSした。R1〜R6の外部受入条件とS0〜S8全体完了の判定は変更しない。

## 19. 2026-09-20 provenance・ADR境界のtimezone修正

来歴指紋（`input_version` / `open_910_version`）とADR検証が、timezone付き入力をUTCの暦日や
timezone-aware indexのまま扱う不整合を修正した。共通のJST正規化を使い、ADRの返却フレームも
JSTのnaiveな日付indexへ揃える。ML overlayの入口も同じ日付変換を通す。

`2025-06-01T15:00:00+00:00`（JST 6/2 00:00）の来歴指紋とADR行を6/2として扱う回帰を追加した。
今回追加した回帰を含むserial全体テストは **803 passed / 17 warnings**。既存のxdist証跡は追加回帰前の
**798 passed / 16 warnings**である（[今回のserial実行証跡](../20260920_structural_completion/evidence/r6_local_ci_timezone_fix.json)）。
R1〜R6の外部受入条件とS0〜S8全体完了の判定は変更しない。

## 20. 2026-09-20 空ADR artifact修正

空のADR DataFrameがML overlayへ渡ると、全銘柄のADR値を0.0としてoverlayが適用される境界を修正した。
`validate_adr_features()` は空artifactを`None`として返し、既存の安全なoverlayスキップ経路へ統一した。
空snapshotの回帰テストを追加し、`tests/` 全体は **803 passed / 17 warnings**、対象Ruff、compileall、
`git diff --check`もPASSした。

## 21. 2026-09-20 PIT snapshot境界の修正

同一日でも、09:10の意思決定へ15:31のsnapshotを渡せる契約漏れを修正した。明示された時刻はsnapshotと完全一致させ、日付だけの互換呼出しは09:10 JSTまでに制限した。空DataFrame、NaT、JST正規化後の重複trade dateもPITDataLake初期化時に拒否する。

時刻不一致・空入力・NaT・重複入力の回帰を追加し、全`tests/`は **803 passed / 17 warnings**。対象RuffとcompileallもPASSした。R1/R2/R4/R5/R6の外部受入条件とS0–S8全体完了の判定は変更しない。

## 22. 2026-09-20 再レビュー指摘の修正

再レビューで見つかった2件を修正した。

- `label_available_at` の明示値を標準JP close（15:30）の既定マスクより優先し、09:10以前に利用可能な標準ラベルを正しく表示できるようにした。
- `PITDataLake.build_decision_inputs()` のdate-only互換呼出しは09:10 JSTでsnapshotを作成し、明示された別時刻のsnapshotは拒否するようにした。これにより、09:10価格を00:00時点の既知値として扱わない。

回帰2件を追加し、対象テストは **57 passed**、全`tests/`は **805 passed / 17 warnings / 104.76秒**。compileall、対象Ruff、`git diff --check`もPASSした。

## 23. 2026-09-21 再レビュー指摘の修正

timezone付き入力をJSTのカレンダー日へ揃える境界を追加修正した。

- `HistoricalInputs.pit_history_trade_dates` は要素ごとにJSTへ変換してから不変配列へ保存する。UTCの暦日をそのまま比較して、当日PIT行を履歴として通すことを防ぐ。
- 公開`ProductionV2Model.decide()`と内部`v2.decision_engine._decide()`の`trade_date`照合に共通のJST日付正規化を使う。同じJST日を表すoffset-aware値を不一致として拒否しない。
- 上記2境界の回帰テストを3件追加し、timezone変換後の日付を`run_leakage_audit()`へ渡して当日履歴がFAILになることも確認した。

対象テスト51件、全`tests/`は **808 passed / 17 warnings / 104.48秒**（watchdog 300秒）。対象Ruff、compileall、`git diff --check`もPASSした。R1/R2/R4/R5/R6の外部受入条件とS0–S8全体完了の判定は変更しない。

## 24. 2026-09-21 intraday・rank-reversal timezone修正

再レビューで確認したtimezone付き入力の正規化漏れを修正した。

- 09:10価格・open-to-09:10リターンのadapterは、`df_exec`と5分足をJSTへ正規化して日付照合する。元のindexは保持するため、target計算のreindex契約を壊さない。
- rank-reversalのdate loaderも共通のJST日付正規化を使い、UTC時刻が前日のファイルへずれる経路を閉じた。
- timezone付き`df_exec`/5分足とUTCのrank-reversal日付を使う回帰を追加した。

対象回帰は **28 passed**。現行ツリーの全`tests/`は **820 passed / 17 warnings / 104.61秒**。Ruff、対象ファイルのcompileall、`git diff --check`もPASSした。R1/R2/R4/R5/R6の外部受入条件とS0–S8全体完了の判定は変更しない。

## 25. 2026-09-21 再レビュー指摘（signal provenance・09:10欠損）の修正

再レビューで残っていた、timezone付きsignal dateの誤判定と、全NaNの09:10入力を明示入力として受け入れる2件を修正した。

- `distribution_provenance`、on-demand metadata、V2 decision helperは、offset-awareの日付を先にJSTへ変換してから日付へ正規化する。`2026-09-15T15:00:00Z`を`2026-09-16`として扱い、trade dateと同日になる値はprovenance rejectへ倒す。
- strict typedのtarget計算・file cache・on-demand sourceは、`open_910_returns`の型・列・実行日・全値の有限性を検証する。全NaN、部分欠損、日付欠落は`inputs_missing`としてBLPX計算へ渡さず、FallbackPolicyのflat経路へ進む。strictでない互換経路の既存セル単位fallbackは維持した。
- 上記境界の回帰を追加し、既存のoffset-aware metadata fixtureもJST上で同日となる値へ更新した。

対象回帰は **59 passed**、全`tests/`は **825 passed / 17 warnings / 106.21秒**（watchdog 300秒）。対象Ruff、対象4ファイルのmypy、全体compileall、`git diff --check`もPASSした。R1/R2/R4/R5/R6の外部受入条件とS0–S8全体完了の判定は変更しない。

## 26. 2026-09-21 実装追加（R1/R2/R4/R5）

残っていたローカル実装境界を閉じた。

- gap生成のh=1/3/5はrun所有の`HistoricalInputs`を作り、各日09:10の`calculation_frame()`を検証する。モデルへ渡す残差行列は、その日15:30より後のJP close派生ラベルをマスクする。実現targetは評価配列として分離し、bundleへ計算時点・label availability・履歴fingerprintを保存する。
- `HistoricalInputs`へ共通および日付別の`observed_at`を追加し、backtest/VaR/live bridgeがopen_910、macro、ADR、rank-reversal、PIT履歴の時刻を渡す。live bridgeのmacro/ADRは当日より後の行を保持しない。
- `reconcile --account-snapshot`で、brokerのwallet/positionsを発注・取消・再送なしでJSON保存できる。既知order IDのfill照合は従来どおり`--run-id`で行う。
- ML overlay学習の開始日を2015-01-05以降へ固定し、明示したopen_910入力とADR/data fingerprintをartifact metadataへ記録する。production configのCURRENT自動更新は行わない。

全`tests/`は **843 passed / 17 warnings / 106.1秒**（watchdog 360秒）、Ruff、mypy（147 files）、compileall、import-linter、文書84リンク、`git diff --check`、`uv lock --check`がPASS。wheel再ビルドはPyPI DNS解決不可で未実行（既存clean wheel証跡は維持）。

R1/R2のfresh外部データavailable_at、R4実口座wallet/positions/fills、R5本番期間artifact生成・OOS評価・production昇格、R6 Hosted CI URL/job結果は外部実行が必要なため未完了。S0–S8全体完了・本番昇格可とは判定しない。
