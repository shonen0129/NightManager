# 構造改善案：再現可能な計算境界と執行状態の分離

- Date: 2026-09-15
- Status: proposed（全体計画。2026-09-18に実装を再確認。S2の履歴可視性・入力固定、経路間比較、運用証拠は未完了）
- 詳細計画: [対象、移行順序、受入条件](../../reports/20260915_structural_improvement/plan.md)

## 背景

最新の判定は[実装確認と修正報告](../../reports/20260917_structural_completion_review/report.md)を参照する。
以下の段階別実施記録のPASSは、各サブ範囲の受入を示す。ADR全体の実装完了を意味しない。

[9月12日の監査](../../reports/20260912_workspace_audit/report.md)に対する
修正を経て、[第8回レビュー](../../reports/20260915_stage_abc_round8_review/review.md)は
指摘修正の受入条件を満たした。本番ML artifact再生成と本番・BTのweights照合は
引き続き運用の残件である。

現在も、本番・BT・gap生成の入力取得、分布計算の経路が分かれている。
`HistoricalInputs`のas-of cutoffとrun-owned 09:10/macro/ADR/PIT/rank-reversal欄、strict typed pathの
暗黙I/O禁止は実装済みである。旧compatibility adapter・主要adapterの観測時刻充填と全入力の版固定は
残っており、型とキャッシュの仕組みだけで完全PITとは扱わない。
注文結果の保存・照合も実装済みである一方、実行途中からの復旧を判断する永続状態とは
責務が異なる。修正した契約を維持しながら、これらの境界を明確にする。

## 提案する設計判断

1. **V2同期経路を維持する。** [ADR-P35](2026-08-17-p35-pipeline-canon.md)を継承し、
   `ProductionRunner`、`BacktestEngine`、`ProductionV2Model`を既存の入口として活かす。
   runnerには共通のモデル構築と一日の計算組立を置き、発注や損益集計を集めない。
2. **設定の正規化を入口に集約する。** YAML継承・旧キー・環境変数・CLI上書きの
   現行優先順位を保存し、内部は用途別の型へ渡す。`AppConfig`全体を一律配布せず、
   数理、費用、リスク、実行環境、認証の境界を保つ。
3. **取得済み入力と計算を分ける。** 当日既知の入力、確定済み学習ラベル、評価専用の
   実現値を別契約にする。`PITDataLake`を入口adapterとして移行し、`PITMatrixView`と
   既存監査を維持する。データのavailable_atが不明な場合、過去に利用可能だったと補完しない。
4. **分布取得の結果に理由と来歴を持たせる。** `FallbackPolicy`、bundle transaction、
   共通provenance validatorを再利用する。μ・Ω・horizon・版を一組として扱い、
   source失敗、PIT減額、監査失敗を個別に追跡する。
5. **計算結果、注文計画、注文観測、照合結果を分ける。** `PortfolioDecision`と
   `OrderRequest`/`OrderResult`の既存型を起点にする。計算を再生しても発注は行われない。
   永続注文台帳と再起動後の復旧は、挙動追加として独立した変更にする。
6. **損益の計算規則を共有する。** まず現在のweightベースBTを出力同値で抽出する。
   次に実約定・在庫・費用の台帳を整備する。実測約定とシミュレーションの差は保存し、
   実約定価格に含まれるslippageを二重控除しない。
7. **研究・配布を段階的に分ける。** ML学習をresearchへ移し、推論型・特徴量契約・
   artifact読込は本番側に残す。pickleのクラス参照先、LightGBM推論依存、
   `CURRENT`によるimmutable公開を維持する。
8. **修正前後の比較を実装より先に定める。** 現在の未commit変更も含めて比較版を固定する。
   別コード版・別インスタンス・独立設定で数値、分岐、監査、コストを比較する。
9. **旧経路の撤去までを各段階に含める。** 同時に利用者を移せる場合は互換wrapperを
   作らない。分割移行で一時導入する場合は、追加時に利用者・削除対象・撤去段階を定める。
   旧経路が残る間は移行中と記録し、S1は旧設定入口と重複構築を撤去するS1dで閉じる。

## 採用を勧めない案と理由

- 全処理を巨大な共通Runnerに集める案：学習、数理、broker、帳票の依存を再び結合する。
- `PITDataLake`の導入だけで非リークを保証する案：全履歴へのアクセス、ラベルの確定時刻、
  ndarray/dictの変更可能性が残る。
- cacheとon-demandを無条件に同じ実装へ置換する案：現行の共分散入力と生成手順の
  同値性が未確認。差があれば原因と仕様を別途判断する必要がある。
- ファイル行数だけを基準に分割する案：呼出関係や責務の数を減らす保証がない。
- 先にasync基盤、全面的なイベントソーシング、新optimizerへ移行する案：今回必要な
  同期処理の状態管理と再現性に対して変更範囲が大きく、数値・運用挙動の比較も難しくなる。

## 影響と移行条件

互換adapterが必要な分割移行では、一時的にファイル数が増える。各段階の完了には
本番CLI、BT、gapバッチ、VaR、対応する研究入口、固定fixtureの切替と旧経路の撤去を含む。
旧moduleをmockするテストの都合だけでは互換層を残さず、同じ検証を正本へ移す。

正規の公開API、AGENTS.mdで指定された研究registry窓口、固定fixture等が使う`.npy`
reader、versioned artifactのclass参照は、具体的なサポート対象を持つ境界として区別する。
責務のある境界は維持し、内部の中継・再変換・二重正本を撤去する。外部契約を廃止する
場合は利用者とデータの移行を含める。未知の利用者の可能性だけで旧入口を無期限に残さない。

[Stage A–Cの安全境界](2026-09-13-stage-abc-safety-boundaries.md)を維持する。
新しいavailability制約、schema強制、永続化失敗時の発注停止、再送判断などは、
機械的なリファクタリングの同値性では承認できないため、個別の仕様と受入条件を持つ。

S1c/S1dの実装結果と残るS2以降の境界は、[実行記録](../../reports/20260915_structural_improvement/s1c_s1d_execution.md)に記録する。
S1の再確認は[完了確認](../../reports/20260915_structural_improvement/s1_completion.md)、
S2aのmacro/ADR/9:10入力抽出は[実行記録](../../reports/20260915_structural_improvement/s2a_execution.md)
に記録する。S2aでは既存の欠損処理を保ち、available_atやread-only入力の新契約を先取りしない。

本案は過去Phaseの完了宣言や本番昇格判断を変更しない。
各段階の実装時に`docs/ARCHITECTURE.md`、対応ADR、運用手順、ロードマップを更新する。

## 追加の設計判断（2026-09-18）

- `ProductionV2Model.decide`は既存の公開引数を入口で一度だけ`DecisionInputs`へ変換する。
  内部engineと`ProductionRunner.run`はこの型を受ける。`RunnerInputs`、`data.cache`、
  `distribution_resolver`、`generate_v2_production_portfolio`、preprocessorのtarget中継は、
  研究・testsを含む利用者を移して撤去する。別名を増やして中継を維持しない。
- `HistoricalInputs.calculation_frame`はpandas 3のCopy-on-Writeで所有データを保護する。
  `EvaluationInputs`は予測入力に含めない。ただし全履歴の返却をやめるには、h=3/5の
  ラベル終了時刻と各モデルの学習窓を明示する移行が必要であり、このコピー保護だけでは完了しない。
- `FallbackPolicy`へgap参照を明示し、モデルの一時的なpath属性を廃止する。
  VaRの有効設定は一度解決したobjectをkeyと実計算で共用し、snapshotとworkerの寿命は
  別moduleが所有する。旧NPYは対応する固定fixtureがあるためdigest検証付きreaderを維持する。
- 注文終端とrun完了を区別する。完了には約定・建玉・余力・journalの照合記録が必要である。
  未解決runは日付やjobが変わっても同じ口座・戦略の送信を止める。復旧CLIは保存済み注文IDを
  読取専用で照合し、送信の成否が不明な数量を推測して再送しない。
  既存15:40レポートジョブから`--pending`を呼び、候補がなければbroker接続を省く。
  照合失敗時もレポート収集を試みるが、ジョブ全体は非ゼロを返す。
- batch内の子処理は、親guardが発行したowner/scope/DBを共有する。発注入口でSQLite上の
  leaseを検証する。単純な環境変数の真偽だけで排他を省略しない。親終了後の子孫processも停止する。
- 研究overlayは明示的な`decision_transform`としてBTへ渡し、変換後に数値監査を再実行する。
  本番factoryの旧関数をmonkeypatchする方式は、実際のBT経路と乖離するため採用しない。
- wheelはrepository sourceを除いたimport経路でCLIと合成artifact推論を検証する。
  これは配布・pickle参照の契約を確認する検査であり、本番学習済みartifactの妥当性の代替にはしない。

## 段階別の実装履歴（2026-09-17時点）

S2bの入力契約・PIT分離・read-only所有権を実装した。`KnownMarketInputs`、
`HistoricalInputs`、`EvaluationInputs`、`DecisionInputs`、`InputVersion`を追加し、
ProductionRunner、V2 decision、BacktestEngine、日次bridgeを新契約へ切り替えた。
詳細な検証結果と残存互換境界は[S2b実行記録](../../reports/20260915_structural_improvement/s2b_execution.md)に記録する。
S2全体の完了とADR全体のaccepted判定は、残存shim/wrapperの利用者切替・撤去と
available_atの証拠確認後に行う。

S3aでは、分布結果・source理由・試行順を`leadlag.domain.distribution`の型付き契約へ移した。
`FallbackPolicy`とsingle/MH decisionは`DistributionStatus`、`DistributionReason`、
`DistributionAttempt`を使い、alert文字列の検索で監査失敗を推定しない。既存の外部互換fieldは
S3完了までの移行用に残す。詳細な検証結果とS3b/S3cへの残件は[S3a実行記録](../../reports/20260915_structural_improvement/s3a_execution.md)に記録する。

S3bでは、gap生成のh=1/h=3/h=5で共有できる純粋なgap補正計算を
`leadlag.pipeline.gap_distribution`へ抽出し、Step 1 `Omega_struct`を明示的な共分散入力として扱う。
研究診断CSVの組立・保存は`leadlag.pipeline.gap_reporting`へ分離し、研究固有の入力組立、
portfolio評価、PIT比較・plot・report生成は`research/diagnostics/{gap_inputs,gap_portfolio,gap_outputs}.py`
へ移した。研究入口はこれらの境界を呼ぶだけで、production codeは研究専用依存を持たない。
S3b全体の実施結果は[S3b報告](../../reports/20260916_structural_improvement_s3b/report.md)に記録する。

S3cでは、分布を構成するμ・Ω・metadata・trade date・horizonを
`leadlag.domain.gap_bundle.GapBundleRef`で一つのmanifest契約へ束ねる。SQLiteの`GapStore`は
行列、metadata、manifestを一つのtransactionで保存し、`load_horizon_bundle`で同一snapshotから
読み出す。NPY互換経路はmanifestをcommit markerとして扱い、payload digestと版情報を検証する。
既存の三値`load_horizon`と旧format_version=1 manifestは互換のため残す。

VaR/ESのreturn cacheでは、`VaRCacheIdentity`がeffective config、df_exec、code、overlay、gap入力の
版を既存のcache key形式へ安定して反映する。`DeadlineBudget`はmonotonic clockの絶対期限を一つだけ
所有し、履歴準備・copy・worker計算へ残時間を渡す。S3c実施範囲はbundle/cache契約と期限部品の抽出で、
VaRのsnapshot/worker lifecycle全体の分割、legacy readerの撤去、実データ同値性は後続の残件とする。
実施結果は[S3c報告](../../reports/20260916_structural_improvement_s3c/report.md)に記録する。

### h=1 cache/on-demand 共分散とbundle来歴（2026-09-20）

- signal dateに一致するStep 1 `Omega_struct`が存在する場合だけ、研究生成のh=1で明示的な構造共分散を使う。
  日付の古いfallback行列を出力へ流用するとon-demandのBLPX共分散とずれるため、fallbackは診断へ記録し、
  計算はBLPX共分散へ戻す。trade-date名のlegacy fixtureは互換境界として明示overrideに限る。
- 新規gap bundleは、as-of入力版、model版、effective config版、固定ticker順をmetadataとmanifestへ保存する。
  df_execを持つproduction cache readerは4項目を現在の入力と比較し、不一致・欠落をcache不採用へ変換する。
  低レベルの旧fixture readerは移行期間のため残すが、来歴なしartifactをdecision経路で採用しない。

### strict rank-reversal と診断キャッシュの追補（2026-09-20）

- typed decisionがsnapshotからrank-reversal signalを受け取れない場合、overlayは元のscoresを維持して
  警告付きで終了する。strict経路からdate別ファイルを再読込せず、暗黙I/Oは互換経路だけに限定する。
- PIT診断CSVのキャッシュidentityはmtime・サイズではなく内容SHA-256で判定する。同一メタデータの原本置換でも
  履歴を再解析し、古いIR配列を再利用しない。

### 明示ラベル可用時刻の追補（2026-09-20）

`HistoricalInputs.label_available_at` は、列ごとの現在行ラベルが利用可能になる時刻として扱う。
可用時刻より前の `as_of` では現在行だけをマスクし、可用時刻後に既知の履歴行を削除しない。
この境界を before/after の回帰テストで固定した。gap生成全入口へ可用時刻の証跡を接続する作業は残件である。

S4a/S4bでは、`ExecutionPlan`、`OrderObservation`、`ExecutionReport`を追加し、
発注計画・broker観測・照合結果を分離した。新規注文と引け決済のstatus pollは
`execution.order_lifecycle.poll_order_statuses`へ統合した。S4後半では
`PortfolioDecision`のmapping protocolと`from_dict`/`to_dict`を撤去し、モデル・執行・
backtest・研究overlayは属性参照へ移行した。legacy mappingはreporting writerの
`_coerce_decision`だけで受ける。対象回帰100件、全体698件、静的検査、import-linterを
確認済みである。

S5a/S5bでは、`ExecutionStateStore`にrun/order intent/order observation/
reconciliation checkpointとleaseを追加し、broker呼出前のintent commit、送信後の
復旧候補、同一jobの再送拒否、decision/closeの共通排他を実装した。`job_guard`は
process group deadlineとTERM→KILLを担当し、batch入口は`run_decision_v2.sh`へ統一した。
詳細な検証結果と実scheduler・実口座での確認範囲は[S5実施報告](../../reports/20260916_structural_improvement_s5/report.md)
に記録する。S6では、weight-based BTの純粋な損益計算を`core.pnl`へ抽出し、
確認済み実約定と残存建玉を`Fill`/`InventoryLedger`へ接続した。単位ラベル、
overnight保存、`MetricsSpec`も追加した。実scheduler登録と実口座での残高・約定突合は
引き続き運用環境で確認する。

S7a/S7b（2026-09-17）では、ML overlayの特徴量、artifact保存・provenance検証、
本番推論、研究学習を分離した。pickle互換のため`MLOrderOverlayModel`のclass pathは
元moduleに残し、旧importは薄いcompatibility exportに限定した。setuptoolsのproduction
package discoveryは`leadlag*`へ限定し、学習入口は`research` extraと
`research.experiments.ml_overlay_training`へ移した。成功・中断の学習イベントは
canonical `ExperimentRegistry`へ記録する。wheel実ビルドは環境のbuild dependency不足と
ネットワーク制限で未実行だが、package discoveryと隔離importは検証済みである。
実artifact再生成、本番/BT weights照合、実データ学習の性能記録は運用残件として保留する。

S8（2026-09-17）では、[CI実施報告](../../reports/20260917_structural_improvement_s8/report.md)のとおり、
Python 3.12と`uv.lock`固定のworkflowへcompileall、Ruff、mypy、import-linter、文書リンク、
production wheelの`research`除外、全テストを組み込んだ。production→research、純粋計算→I/O adapter、
domain→application configのimport契約を追加し、7契約すべてをKEPTとした。重複していた
`src/leadlag/models/blpx.py`は利用者・動的load・配布参照を確認して撤去し、`models/blpx/`を正本とした。
ローカルではlock/compileall/Ruff/mypy/import-linter/文書リンクをPASSしたが、wheel buildは環境の
build依存・ネットワーク制約によりCI実行へ委譲した。S8はCIと文書の統合完了を示すもので、S2c/S3cの
lifecycle、実scheduler・実口座突合、S7の実artifact再生成・weights照合を完了扱いにしない。
