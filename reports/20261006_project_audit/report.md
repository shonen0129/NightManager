# プロジェクト全体監査 — 2026-10-06

対象: `/Users/shonen/leadlag`、HEAD `b5b901e6d467fb000263f9178722e3954d652720`。
本番設定は `configs/production/production.yaml` と継承先を正規loaderで解決した。

**結論: actual-liveの新規建て・現行MLの収益性を根拠とする本番受入は BLOCK。オフライン研究・修正作業の継続は可能。**

安全停止・注文照合・入力来歴・本番/研究分離は大幅に整備されている。全984テスト、本番lint、型検査、7つのimport契約は成功した。しかし、損益の時間区間、指標の周辺経路、機能の実際の適用状態には再現できる欠陥が残る。実市場の09:10入力取得は失敗し、照合済み実口座PnL producerも未整備。長期の高Sharpeは主に始値→大引けの代替targetによる回顧値であり、現行09:10戦略の実現収益を証明しない。

本監査はレビュー依頼として実施した。ソース・本番設定・モデルartifact・入力cacheを修正せず、監査用スクリプト、ログ、報告だけをこのディレクトリへ追加した。brokerへの接続、発注、決済、再送、schedulerの登録・変更は実施していない。GitHubは状態とCIの読み取りだけを行った。

## 1. 判定の読み方

P0=即時の重大障害、P1=本番・主要機能の障害/受入阻害、P2=限定条件の不具合または重要な検証不足、P3=具体的な保守上の問題。P0は確認していない。

「確定不具合」は現行コードとオフライン再現で確認したもの。「現況/既知未完了」は既にissueやADRで認識されている障害・未実装。「証拠不足」は成果を認定できない理由であり、その機能に必ず損失が出るという意味ではない。「構造負債」は修正の増幅や責務の混乱が具体的に存在するもの。31項目を同じ種類のバグとして数えていない。

| ID | 優先度 | 種類 | 課題 |
|---|---|---|---|
| F01 | P1 | 確定不具合 | 持越し建玉の翌日寄付→09:10損益が欠落 |
| F02 | P1 | 確定不具合 | 設定保存・HTTP例外に認証情報が残る |
| F03 | P1 | 現況/既知未完了 | 09:10 capture失敗が継続し、前向き観測を作れない |
| F04 | P1 | 現況/既知未完了 | 実口座risk snapshot producerが未整備 |
| F05 | P1 | 現況＋品質欠陥 | ADR更新停止、ML skipが構造化されていない |
| F06 | P2 | 確定規約違反 | 一部の長時間バッチに全体期限がない |
| F07 | P1 | 証拠不足 | 09:10戦略の精度・収益を混合targetから認定できない |
| F08 | P1 | 証拠不足/既知未完了 | 現行MLの付加価値・前向きgateが未成立 |
| F09 | P2 | 証拠不足 | 実費、容量、貸株、約定後β中立性が未認定 |
| F10 | P2 | リスク評価の限界 | ES停止までの余裕が2.1bp、尾部標本3件 |
| F11 | P2 | 確定契約欠陥 | 評価開始日と2010–2014 prior期間を入口で分離しない |
| F12 | P2 | 確定不具合 | データ開始前の終了日を先頭データへ丸める |
| F13 | P2 | 確定不具合 | 月次指標の省略形が245倍で年率化される |
| F14 | P2 | 確定不具合 | 要約JSONの最大DDが初期損失を落とす |
| F15 | P2 | 確定不具合 | MinVar有効時にshort_countが無視される |
| F16 | P2 | 確定契約欠陥 | DSRが不整合観測数・invalid指標を拒否しない |
| F17 | P2 | 確定契約欠陥 | 欠損labelでflat日のPnLもNaN、指標は日を削除 |
| F18 | P2 | 損益境界の欠陥 | 最終持越し在庫・artifact切替時の連続性が未定義 |
| F19 | P2 | 指標定義の欠陥 | turnoverが実際の売買フローと一致しない |
| F20 | P2 | 潜在API不具合 | 未接続providerが時刻を無視、任意Volume欠損で全行削除 |
| F21 | P2 | 潜在配布不具合 | wheel配置で実行rootがsite-packages上位へずれる |
| F22 | P2 | 構造境界の欠陥 | 運用バッチがresearchへ直接依存、CI契約の外側 |
| F23 | P3 | 構造負債 | 本番と研究でBLPX計算本体が14組重複 |
| F24 | P3 | 構造負債 | 計算より境界の調停に大きな関数が集中 |
| F25 | P3 | 構造負債 | 未接続層・互換wrapper・reports内の実行依存が残る |
| F26 | P2 | 運用文書の欠陥 | 全建玉squareと現行の持越し設定が矛盾 |
| F27 | P3 | ハーネス負債 | Skill/IDE手順の古い契約・死んだ参照が残る |
| F28 | P2 | 統計governanceの不足 | study横断の探索履歴を十分に集約できない |
| F29 | P1 | 現況/既知未完了 | mainのCI必須化・保護が無効 |
| F30 | P2 | 確定品質欠陥 | 上場後のUS欠損もpre-inception proxyで隠される |
| F31 | P2 | 確定境界不具合 | static範囲外の年末年始休場をtradingと返す |

## 2. 確認範囲と検証

| 対象 | 規模・確認 |
|---|---|
| `src/leadlag` | Python152ファイル、36,806行。全AST走査、import契約、型検査、本番入口から主要計算/保存/執行経路を手動追跡 |
| `src/research` | 199ファイル、59,107行。全AST/重複/lint走査、学習・registry・主要比較実験を追跡 |
| `tests` | Python111ファイル、18,841行。全984テスト実行、重要な不変条件と未検出ケースを照合 |
| `scripts` / `tools` | Python11/26ファイル、816/11,578行。batch、plist、CI、検証・研究入口の責務/期限を確認 |
| ハーネス | AGENTS、10 Skillと参照、Devin Skill、Windsurf plan/workflows、CI設定。計22 Markdown文書 |
| 設定・文書 | tracked configs55ファイル、docs62ファイル。正本設定、architecture、技術仕様、運用手順、現行roadmapと関連ADRを照合 |
| 証拠 | tracked reports1,066ファイルの所在をinventory化し、直近の再学習・paired比較・コスト・感応度・実約定・運用受入を重点照合 |
| 運用状態 | 保存済みcapture記録、ADR artifact、読み取り専用SQLite、launchd状態。外部broker照会なし |
| GitHub | HEAD一致、tracker #22、#18、main属性/rulesets、当該HEADのHosted CIと各stepを読み取り確認 |

全ファイルの全行を形式的に証明した監査ではない。機械走査は全対象、手動レビューは正本・安全境界・統計/会計境界・運用producerと関連実験を中心に行った。すべての歴史レポートの数値を新たに再計算したという意味ではない。

| 検証 | 結果 | 証跡 |
|---|---|---|
| 固定regression baseline | 1 passed | [baseline log](pytest_baseline.log) |
| それ以外の `tests/` 全体、xdist4 workers | 983 passed、3 warnings | [full log](pytest_full.log)、[JUnit](tests.xml) |
| 内訳 | unit852、integration90、research14、features27、regression1 | [coverage inventory](audit_details.json) |
| compileall 必須5tree | PASS | [checks](checks_static.json) |
| CI指定Ruff | PASS | [log](ruff_ci.log) |
| mypy `src/leadlag` | PASS | [log](mypy.log) |
| import-linter | 7 contracts PASS | [log](import_linter.log) |
| CI指定9文書のreference検査 | PASS | [log](docs_ci.log) |
| 全Python tree Ruff | 122指摘、F821=0 | [全lint](ruff_all.log)、[集計](evidence.json) |
| `uv lock --check --offline` | PASS | [package checks](checks_package.json) |
| ローカルwheel再build | 未実行: 既存venvに `build` がなく終了。依存追加なし | [build log](package_build.log) |
| 当該HEADのHosted CI | 全step成功。wheel build/manifest/隔離smoke/全testsを含む | [GitHub run](https://github.com/shonen0129/NightManager/actions/runs/37349218940)、[取得証跡](external_checks.json) |
| 合成・オフライン再現 | 18観点を保存 | [probe code](probes.py)、[results](probes.json) |

Ruff122件は研究/一部script等の既知backlogを含む。CI対象の本番lint失敗とは扱わない。内訳はI00164、UP01525、F40121、F8415、UP0172、F5413、F8112。今回のテストwarningsは研究診断の定数系列等で、984件に失敗/skipはない。CI成功は実市場取得・実費照合・モデル収益性の成功を意味しない。

## 3. 損益・情報保存・運用の優先課題

### F01 [P1/確定] 持越し建玉の寄付→09:10リターンが消える

場所: `src/leadlag/execution/backtester.py:91`、`src/leadlag/core/pnl.py:497`、同`:508`。

当日の新規ウェイトには09:10→close targetを掛け、前日持越しには翌日のopen/前日close gapだけを掛けている。翌日09:10まで保有してからリバランスする在庫のopen→09:10区間が、どちらにも入らない。現行のlong carry=.75/short=.5で実際に関係する。

再現はモデルnet=0/gross=2のlong1/short−1。long銘柄は前日close100、翌日open100、09:10=110、close110、short銘柄は価格不変。翌日ウェイトはflat、費用0。実在庫のlongには+10%が発生するが、正規target/gap抽出→PnL計算は両日0を返す。[probe `carry_open_to_910`](probes.json)。

実害: net PnL、DD、VaR/ES、carry比較の基準系列が不完全。欠落区間の方向によって過大/過小の両方が起きるため、過去成績が一律に上振れしたとは断定しない。

既存不足: `tests/unit/test_backtester_pnl.py` はflat遷移の決済費用と09:10 target抽出を個別に検査するが、持越し在庫の翌朝markまで連結していない。修正案: close→翌09:10までのcarryを同一価格基準と会計日で定義し、翌朝の在庫評価と09:10 rebalanceを連結する。long/short、週末、翌日flat、gapとopen→09:10逆方向のケースを正規経路で照合する。修正後は主要研究比較とrisk replayを同じ定義で再評価する。

### F02 [P1/確定] backtest artifactとHTTP例外に認証情報が残る

場所A: `src/leadlag/data/backtest_store.py:189`、同`:353`。CLIは `src/leadlag/execution/backtest.py:230` で `AppConfig` を渡す。名前が `_safe_config` でも、そのまま `model_dump()` してSQLiteの `config_json` へ保存する。`AppConfig.kabu` のpassword/token、`tachibana` の第二password等は通常の文字列である。

場所B: `src/leadlag/broker/tachibana/api.py:207`、同`:223`、同`:386`。認証/リクエストpayloadをURL queryへ入れ、`raise_for_status()` の例外を変換せず伝播する。broker clientのhealth/order等は例外を `%s` でlogする。

再現: 合成のAppConfig password/token/第二passwordが全てJSONに残る。mock HTTP404のlogin例外には合成sAuthIdとqueryが残る。[`backtest_config_secrets` / `auth_http_exception`](probes.json)。過去captureにもqueryを含むエラーが存在した。現行capture collectorのsafe error処理は改善済みだが、APIの例外境界と別の保存経路は依然未統一。

実害: 結果DBやログの共有、support提出、後日のartifact公開によって不要な認証情報が持ち出される経路を作る。今回、実credential値の再掲載・有効性試験はしていない。

既存不足: 研究registryのredactionテストだけではCLI backtest storeやbroker HTTP例外を保護しない。修正案: artifact向けallowlistとAPI境界での安全な例外を正本化し、合成secretを全保存先/logへ通して残存しないことを検査する。過去 `.env.example` の値除去は9/29 ADRに記録済みで、実資格だった場合の失効/再発行は今回未確認。過去記録を不用意に配布しない運用と、該当資格の状態確認が残る。

### F03 [P1/現況] 09:10 captureの認証停止が継続

場所: `tools/validation/collect_0910_microstructure.py`、`scripts/batch/run_0910_microstructure_capture.sh`、保存済み `var/shadow_runs/ml_overlay_value/microstructure/capture_*.json`。

9/28、9/29、9/30、10/1、10/2、10/5の6日分が全てFAILED、各2attempt。初期は旧v4r9へのHTTPError、後続はvirtual URLがないValueError。**10/5には新しい診断で `sKinsyouhouMidokuFlg=1` を確認した。** 10/4報告時点の原因未確定から一段進んだ証拠である。frozen quote artifactは0件、forward `daily.jsonl` / `outcomes.jsonl` は双方未作成。[安全な集計](evidence.json)。capture launchdは読み取り時点でnot running、runs7、last exit code1だった。

実害: read-only Stage1の市場入力受入、同一snapshotを使うgap/shadow、MLの前向き評価が開始できない。actual-liveが不適格snapshotを拒否する挙動は正しい。

既存不足: HTTP fixtureは本番口座のログイン前提状態を証明しない。対応: broker側の開示書類確認と認証前提を正規の手順で解消した後、次の取引日でread-only preflight→認証診断→09:10 frozen→gap/shadow同一IDの受入を行う。監査のために自動で書類確認や発注へ進むことはしない。対応範囲は既存[#27](https://github.com/shonen0129/NightManager/issues/27)と関連する。

### F04 [P1/既知未完了] 実口座lossガードのproducerがない

場所: `src/leadlag/execution/account_risk.py:14`、[9/29 ADR](../../docs/decisions/2026-09-29-frozen-0910-account-risk-forward-eval.md)、[実装記録](../20260929_frozen_0910_account_risk_forward_eval/report.md)。

`account-risk-snapshot-v1` の検証・新規リスク停止はあるが、完全照合済みのcash・position・fill・feeからそれを生成するproducerは未実装。既定の `var/live/pipeline_data/account_risk/latest.json` は不存在。現行execution SQLiteもrun/intents/observations/reconciliation各table0行で、新しい運用経路の実市場受入証拠はない。この0行から過去に取引が一度もなかったとは結論しない。

これは既存のユーザー判断でfail-closedを継続している未完了であり、安全ガードの欠陥ではない。受入保証金をPnLへ置き換えて解除してはならない。

対応/受入: [#25](https://github.com/shonen0129/NightManager/issues/25)でPnLの正本・照合責任を確定し、入出金、実現/評価差分、全費用、建玉/数量を整合させるproducerを完成させる。日次/月次の境界と取りこぼしを検査し、[#27](https://github.com/shonen0129/NightManager/issues/27)のcontrolled live受入に接続する。

### F05 [P1/現況＋品質] ADRが8/17で停止し、MLの非適用理由が結果に残らない

場所: `src/leadlag/data/adr_features.py:50`、`src/leadlag/models/ml_overlay_inference.py:65`、`scripts/batch/_update_market_data.py:36`。

既定ADR artifactは最大日付2026-08-17、10/6用の検証はNoneを返した。現行artifactはadr特徴を要求するので、この既定入力で直近日を処理するとMLはskipする。market-data launchdはruns10、last exit code1。失敗原因の全てを今回特定したわけではない。

欠損ADRを古い値で流用せずskipすること自体は適切。しかしskip時は元のPortfolioDecisionをそのまま返し、成功時だけ `overlay_applied=1` を付ける。`overlay_applied=0`、skip reason、feature freshnessが標準summaryにない。ML enabled=true、numerical/leakage PASS、fallback=0から「本番MLが適用された」と誤読できる。回顧269日のskip25日と、現在のartifact更新停止は別の状態である。

既存不足: ADR欠損でskipするunit testはあるが、設定→機能適用→結果集計の状態契約と日次更新SLOを検査しない。対応: 更新失敗を独立して診断し、ADRの日付・coverageをpreflight/manifestへ載せる。機能ごとのenabled/applied/skipped/rejectedとreasonを必須にし、終端flat/on-demand/PIT multiplierと区別して監視する。

### F06 [P2/確定] 非発注のscheduled pipelineに全体期限がない

場所: `scripts/batch/update_market_data.sh:30`、`scripts/batch/run_distribution_diagnostics.sh:42`。補助の `scripts/run_full_backtest.sh:8`、`scripts/run_tests_unit_only.sh` も同様。

decision/gap/close/capture等にはjob guard/phase deadlineがある一方、8:00更新と診断pipelineはPythonを直接実行する。ADRの `run_with_timeout` はdaemon threadをキャンセルしないため、プロセス全体の終了期限にはならない。AGENTSの長時間CLI規約を満たさない。

実害: 朝のproducerが停止したまま次段の入力鮮度を失う。別ジョブや手動再実行とのcache書込競合にも共通leaseを使っていない。今回、実際のhang発生や同時書込破損までは確認していない。

既存不足: job_guard自体のテストはあっても、全scheduled入口がそれを使う契約検査がない。対応: 非発注producerにも全体deadline、phase budget、single-flight、失敗時の終了記録を適用する。stall subprocessと子process回収を使って検査する。既存注文の監査/照合機構は維持する。

## 4. モデル精度・収益・採用判断

### F07 [P1/証拠不足] 主戦略と評価targetの大部分が一致しない

場所: `src/leadlag/core/target_returns.py:15`、`src/leadlag/data/intraday_inputs.py:51`、[長期感応度監査](../20260924_sensitivity_pipeline_audit_long/report.md)、[短期監査](../20260924_sensitivity_pipeline_audit/report.md)。

仕様は09:10→closeだが、価格proxyがないセルはopen→closeを使う。長期2779営業日の価格proxyは1,480/47,243銘柄日=3.13%、全17同時に揃うのは8日。96.87%は別区間target。この長期値のnet Sharpe7.618/gross10.606を09:10戦略の予測/収益能力として扱えない。MLは感応度寄与分離のため無効化されており、現行MLを含む完全なproduction再現でもない。

同じく既知の短期103日ではproxy coverage84.52%、net Sharpe−1.9335/gross0.1602。訂正後Rank ICは現行+0.03381、JPX33案+0.03527、差CI[−0.00060,+0.00355]。長期Rank IC+0.2247と短期+0.03381はtarget coverage・期間が異なるため、精度劣化の原因を期間だけに帰せない。

High/Low midpointはbid/askの実行可能価格ではない。09:10とlabel付けされた5分足の開始/終了時刻、提供遅延、調整価格の履歴も当時の可用性として完全には証明されていない。現行コードのmacro窓はtrade date未満に切られており、今回「当日macro終値を必ず使うリーク」は認定していない。ただしhistorical inputへ付けた09:00/09:10時刻ラベルはproviderの実公表時刻の証明にはならない。

対応: measured09:10とopen fallbackの成績/coverageを明確に分け、全評価営業日を保った上で不足日を無効評価として管理する。frozen quote主体の同一target、provider available_at、修正履歴、実費基準を揃えてforwardで検証する。既知期間を新しいOOSと呼ばず、旧Rank IC訂正は訂正recordを使う。現行感応度の変更をこの監査だけで提案しない。

### F08 [P1/証拠不足] MLの限界利益が確立していない

場所: 現行 `models/ml_order_overlay/production_20260923/CURRENT`、[269日paired比較](../20260929_ml_overlay_paired_269/report.md)、[9/29実装記録](../20260929_frozen_0910_account_risk_forward_eval/report.md)。

現行versionは `20260926T192555935698Z-ee306a32f3ec`、train/label終端2025-12-30、metadata verified。これはtrain cutoffの契約を満たすが、artifact固定後の収益採用gateの成立とは別である。

269日比較（2025-07-29〜2026-09-25、旧版99日＋現行cutoff版170日、252日年率）:

| 指標 | ML on | ML off | on−off |
|---|---:|---:|---:|
| net Sharpe | 4.307 | 4.364 | −0.057 |
| 最大DD | −33.4010% | −31.9408% | 1.4602pp悪化 |
| 複利net return | 218.7577% | 215.5599% | +3.1978pp |
| 平均weight turnover | 1.3154 | 1.3036 | +0.0118 |
| fallback | 0日 | 0日 | 0 |

日次net差平均+0.004366%、95%block CI[−0.008305%,+0.016804%]で0を含む。現行cutoff版170日だけでは複利net差−0.4116pp。269日を現行artifact固定の250日gateへ数えられない。paired forwardファイルがないため、今回確認できる有効な前向き観測は0日。

後続BLPX残差targetの2020–2024診断も3候補全て不採用。既知期間のwithin-study DSR=1はraw MLに優れる確率ではない。詳細は[残差target report](../20260929_ml_overlay_blpx_residual/report.md)。

実害: MLのartifact有効化・コード完成と、実費後の価値を混同すると、更新/特徴取得/複雑性の費用を負いながら採用理由を説明できない。現行MLが必ず無価値という結論でもない。

対応: [#23](https://github.com/shonen0129/NightManager/issues/23)の事前固定artifact/入力/paired基準を守り、同日on/off、実行可能baseline、全日評価、bootstrap、DD/turnover/skip率を蓄積する。train/OOS日付の分離に加え、選択前に未閲覧だった期間を別条件として管理する。既存configのMLをこのレビュー中に勝手に無効化しない。

### F09 [P2/証拠不足] コスト・容量・βの現実性を認定できない

場所: [MLコスト感度](../20260927_ml_overlay_cost_sensitivity/report.md)、[Stage8 summary](../20260923_profitability_order_8/stage8_summary.json)、[実約定レビュー](../20260927_actual_execution_review/actual_execution_cost_capacity_neutrality.md)。

旧版368日、同じweights/grossをslippageだけ再価格付けしたML on Sharpeは5bpsで5.164、10bpsで3.115、20bpsで−0.971、20bps時の最大DD−58.01%。5bpsの基準が不適切と断定したのではなく、モデル費用の仮定に収益が強く依存する証拠である。financing/borrow/reverseは推定値据え置きで、非約定・ロット丸め・impactを含まない。

さらに `BacktestEngine.run_v2_backtest` はmodel weight→PnLの計算であり、liveの `v2_bridge.py:749` 以降にある日次/月次損失、VaR/ES、actual-account evidence、reduction-only等のpost-decision制御を各日へ再生しない。CLI backtestの事後VaR計算も、STOP時に在庫がどう変わるかのsimulationではない。**本番modelの一致と、本番実行結果の一致は別である。** risk基準用の未停止model return系列は必要なので、それを再帰的なrisk stop適用系列へ置き換えることも適切ではない。

公式CSVとの数量照合207株に対し価格まで厳密一致したのは8株、費用確認16円、その部分でも10円未照合。注文IDの一対一対応ではない。モデルの実費後成績、AUM容量、銘柄別貸株可能量はまだ確定しない。

Stage8のmodel net≃0/gross≤2はPASSだが、gapβ proxyのabs p95=.5359/max=1.0469、単一model weight最大.6774。これは実約定後βではなく旧版モデル診断。ドル中立はβ中立を保証しない。現行side1.30でも実在庫・板・貸株・資本制約の測定が必要。

対応: order/fill ID、当時のquote、未約定・部分約定、実際の全fee、数量/lot、在庫とcashを結合する。未停止model referenceと、risk/在庫/執行制御を含む実行replayを別に報告する。実費とcounterfactual費用を分け、AUM×執行volume×depth/borrowで容量を測る。最低限の銘柄集中・factorβ監視は定義できるが、閾値を増やすなら別の事前実験とOOS採否が必要。モデルweightの規約違反としてβ proxyを判定しない。

### F10 [P2/限界] レバレッジ1.30のhistorical ES余裕は2.1bp

場所: [9/30risk reduction](../20260930_var_es_risk_reduction/report.md)、`src/leadlag/core/risk.py`。

同一保存269日から末尾250日を使った1.30のVaR99=2.646%、ES99=3.979%、stopは3%/4%。ES余裕は0.021百分率点=2.1bp、尾部3件である。1.25のESは3.826%。規定を変えず1.30へ下げた判断は履歴上の条件を満たすが、日次窓の通過・実口座risk・安定性を証明しない。

実害: 閾値直下という単一標本の採択を安全余裕と読むと、少しの費用増、窓移動、F01修正で再びstopへ入る。現在liveのVaR/ESが必ず停止閾値超過だとは認定していない。

対応: 正規損益定義を修正した後、rolling窓・最悪日寄与・費用stress・尾部不確実性を報告する。日次guardは維持し、モデル停止とactual-account停止を分ける。今回新しいleverage最適化や閾値緩和は行っていない。

## 5. 計算・設定・統計の境界欠陥

### F11 [P2/確定契約欠陥] baseline期間を評価入口で拒否しない

場所: `src/leadlag/execution/backtester.py:246`、同`:59`、`src/leadlag/core/pipeline.py:347`、`src/leadlag/core/correlation.py:236`。

run_v2_backtestはmin_start_idx=0で日時解決を呼び、2010–2014のstart/endを受理する。baseline相関は全frameから固定2010–2014を使う。AGENTSは評価を2015-01-05以降に限定しているが、default値だけで強制していない。

再現: resolverへ2010/2014/2015のindexと2010–2014の要求を渡すと2010/2014を選択する。source上、priorの固定窓を避ける別guardはない。実市場の2010評価を全pipelineで新規実行したわけではない。

実害: API/CLIで早期期間を指定でき、その期間で計算が成功する日には未来baselineを使う構成になる。短いwarm-upで一部flatになっても期間分離の保証ではない。対応: 正規入口で開始日≥2015-01-05を検証し、source期間と学習/評価期間の交差を検査する。先頭1260行等の代替priorを復活させない。

### F12 [P2/確定] 終了日がデータより前でも一日実行される

場所: `src/leadlag/execution/backtester.py:75`。

`searchsorted(end,right)-1` が−1のとき `max(0,...)` で0へ戻す。開始要求9/1・終了9/30、最初のデータ10/1なら10/1が選択される。[`end_before_data`](probes.json)。休日の次営業日混入は防いでいるが、この空区間境界は残る。

実害: 指定期間の外の成績を計算する。対応: 空交差、start>end、NaT、空frameを明示拒否/空結果として処理し、要求期間と実際の期間をmanifestへ記録する。既存の非営業日end testに加え、全データの前/後を検査する。

### F13 [P2/確定] 月次frequencyの省略形が245年率になる

場所: `src/leadlag/reporting/metrics.py:105`、同`:121`。

`calculate_metrics(series,frequency="monthly")` は月次集計するがannualizationを245のまま使う。`MetricsSpec(frequency="monthly",annualization_periods=12)` は正しい。4か月入力で前者AR517.70%/Sharpe11.2788、後者AR9.328%/Sharpe2.4962。[`monthly_annualization`](probes.json)。

現行CLIのdefault daily指標はこの分岐に入らない。公開APIが同じ月次意味で2つの結果を返す潜在欠陥である。既存testはSpecの不正245を拒否するだけでfrequency引数単独を検査しない。対応: frequencyとannualizationを一契約にし、省略形を正規値へ解決するか廃止し、両入口の等価性を検査する。

### F14 [P2/確定] CLI summaryのDDだけ初期wealthを含まない

場所: `src/leadlag/execution/output_ops.py:68`、`src/leadlag/execution/backtest.py:220`。

共有metricsとbacktesterは初期wealth1をhigh-water markへ含めるが、save_summary_filesは `wealth/wealth.cummax()-1` を再計算する。入力[−10%,0]でmetrics MDD=−10%、run_summary.max_drawdown=0%。[probe `summary_mdd`](probes.json)。

実害: 同じrunのCSV/JSONでDDが矛盾し、初日に損失がある比較が楽観的になる。対応: shared compute_drawdown_series/metricsを正本にして再実装を削除し、成果物間の一致を検査する。既存shared metrics初期損失testだけではwriterを保護しない。

### F15 [P2/確定] MinVarはlong_countを両sideへ使う

場所: `src/leadlag/models/v2/decision_engine.py:168`、`src/leadlag/core/signal.py:325`、`src/leadlag/config/schemas.py:598`。

long/short idxを別々に選ぶが、MinVar builderはq=long_count/17を受けて両basketを再選択する。long5/short3、minvar=trueを合成分布で実行するとlong5/short5、数値監査PASSED。[`minvar_counts`](probes.json)。本番の5/5は現状影響しないが、正規schemaは非対称値を許す。

対応: basketを一度だけ選び明示indicesで計算する、またはサポートしない非対称設定をschemaで拒否する。銘柄数上限・side重複も入口で検査する。既存の同数basket testだけでは検出しない。

### F16 [P2/確定契約欠陥] DSRが無効/不整合標本を受理する

場所: `src/leadlag/experiment_registry.py:157`、`src/research/experiment_utils.py:195`。

return momentsは有限値だけへ削るがTをその件数へ照合しない。metric_statusも拒否しない。有限4returns、trials10、annual SR2でT4ならDSR=.087826、T1000なら.993164、invalid statusでも.993164。[`dsr_inconsistent_count`](probes.json)。extra_metricsはcomputed fieldsを無条件updateできるため、helperのinvalid検出やcomputed観測数を上書きできる。

実害: 誤ったTやinvalid recordが過学習補正済みとして見える。今回、registry内の全既存DSRがこのバグで誤っているとは判定していない。

対応: 指標schema/status、finite全評価系列、T、年率係数、trials/variance定義を一体で検証する。computed fieldは追加metricで上書きさせず、意図したoverrideも別の明示契約へ限定する。既存testsのtrials罰則と年率変換に加え、観測不整合、NaN、unknown frequency、invalid statusを拒否するケースが必要。

### F17 [P2/確定契約欠陥] 欠損ターゲットの日が指標から消える

場所: `src/leadlag/core/pnl.py:497`、`src/leadlag/reporting/metrics.py:102`、同`:108`。

flat weight0でもtarget NaNへ乗じるとPnL NaNになる。metricsはdropnaし、全営業日評価の意味を失う。2日入力、2日目flat+NaN targetでPnLは1日非有限、metricsは有限1日だけでTotal Returnを出す。[`missing_flat_target`](probes.json)。registry helperは一部invalid検出を持つが、CLI/shared metricsは同じ保護を持たない。

実害: 欠損が平坦日・損失日を評価から除外し、runのsamplesと評価件数がずれる。対応: label unavailableを計算前に判別し、欠損を成功評価から落とさない。zero exposureで価格損益0が確定する場合と、active exposureでlabel不明の場合を区別し、全日coverageとinvalid statusを保存する。値を恣意的に0補間して通さない。

### F18 [P2/損益境界] 最終持越し在庫とsegment間の連続性が未定義

場所: `src/leadlag/core/pnl.py:508`、同`:513`、同`:545`、[9/30replayのsegment記述](../20260930_var_es_risk_reduction/report.md)。

最後の行にもcarry alphaを適用してclose費用を減らすが、次day gapは計上せず、残在庫のmark/費用期間/最終cashを結果へ出さない。一日weight1/alpha1/slip10bpsでslippageは10bpsのみ。全解消を終端とするなら20bps、open inventoryを終端とするなら残高/評価時刻を報告すべきだが、現在はどちらの契約も十分に表現しない。

269日replayはartifact segment別に状態resetを再現している。再現のための既存resetと、実ポートフォリオがartifact切替日に持越し在庫を維持することは別である。回顧結果にその境界効果を含めている限界が残る。

対応: terminal policyとinitial holdingsを明示し、artifact切替を在庫resetとして扱わない連続replayを主評価にする。terminal flat日の追加は観測/費用の意味を決めてから行う。既存の「次行でflatにしたときの費用test」は配列最終行そのものを保護しない。

### F19 [P2/定義欠陥] turnoverはinventory flowを表さない

場所: `src/leadlag/core/pnl.py:510`、同`:512`、`src/leadlag/execution/backtester.py:573`。

費用はopening+close tradeの正しいフローを用いるが、turnoverは当日/前日target weight差のL1/2。alpha0、連日weight1ならreported turnover=[.5,0]、実際の一方向notional volume=[2,2]、slip=[.002,.002]。[`turnover`](probes.json)。

実害: 主指標turnoverや容量/コスト削減の比較が、保有target変更しか測らない。実効side leverageも含まない。対応: target-weight turnoverを補助名へ明示し、主たる執行turnover/volumeはopening/closing inventory flowから同一定義で出す。gross2やside1.30を含む単位・1/2係数を固定する。既存レポートの値を新定義へ無印で置き換えない。

### F20 [P2/潜在API] providerの時刻・欠損契約が破れている

場所: `src/leadlag/data/providers/yfinance_provider.py:86`、同`:102`、`src/leadlag/data/providers/tachibana_provider.py:55`。

YFinance intradayはatを受けるが最新Closeを返す。mockで09:10価格100、15:00価格200、at=09:10でも200。daily OHLCは任意volumeをNaNで作ってから全column dropnaするため、正常OHLC1行/Volumeなしで0行。Tachibana providerもrequested atではなくbroker opening_priceを返す。

これらは現在の本番fetcher/broker経路から参照されていない。現行actual-liveのfrozen quoteにこの未来価格が流れたとは指摘しない。未接続で見た目だけ正規化された層が、将来利用した瞬間に時刻契約を破る具体例である。対応: 利用意図がなければ本番packageから撤去/研究へ移し、維持するならat以前の観測とOHLC必須列だけの検査へ統一する。既存provider testはこれらの境界を検査しない。

## 6. ディレクトリ・依存・過剰な複雑性

### F21 [P2/潜在配布不具合] source位置をruntime rootと同一視する

場所: `src/leadlag/config/paths.py:17`、`src/leadlag/data/adr_features.py:18`、`src/leadlag/data/macro.py:22`、`src/leadlag/runner/model_factory.py:70`。

project_rootは `__file__.parents[3]`。checkoutでは正しいが、一般的なwheel配置 `.../lib/python3.12/site-packages/leadlag/config/paths.py` では `.../lib/python3.12` を返す。変更せずinstalled位置でmoduleを評価するprobeで再現した。これは新規wheelでのlive実行検証ではなく、配置に対するpath契約の直接再現である。

相対overlay path、ADR、macro cache、varの正本がcheckout側から離れる。現在のschedulerはcheckout/PYTHONPATHを使うため現行ホストの失敗原因とは断定しない。wheel smokeはhelp/合成artifactで成功し、標準runtime rootの使用を確認しない。

対応: package code位置とdeployment data rootを別の契約にする。rootを明示的に受け、config/model/data/outputsを同じrun rootへ解決する。旧rootへの探索fallbackを増やさない。installed packageでtmp deployment rootを指定したdry/read-only offline実行を受入条件へ加える。

### F22 [P2/構造境界] scheduled updaterがresearchに直結

場所: `scripts/batch/_update_market_data.py:20`、`scripts/batch/run_distribution_diagnostics.sh:42`、`pyproject.toml`のimport-linter契約。

運用updaterは `research.scripts.experiments.build_adr_features` を直接importする。scheduled diagnosticsもtools/researchの入口を使う。leadlag→research禁止とwheel除外は通っているが、scriptsはimport契約の外側なので、運用全体としてresearch分離が成立するわけではない。

実害: checkoutにだけある研究スクリプトのrename/dependency変更が日次運用へ影響する。研究の実験関数にproducerのSLO/atomic publish責任が付いてしまう。ADR pickle/CSVも直書きで、GapStoreのatomic bundle publishほど明確なpublication契約がない。

対応: 安定したADR生成の必要部分をdata/operational producerへ置き、実験固有処理はresearchに残す。scheduled pathを依存検査へ含め、2artifactのpublicationを一つのversion/manifestで扱う。研究全体をwheelへ入れることは解決にならない。

### F23 [P3/構造負債] BLPXの本番/研究間で計算本体が重複

場所例: `src/leadlag/models/blpx/blp_solver.py:125` ↔ `src/research/models/sector_relative_ensemble_blp_enhanced.py:514`、`signal_computer.py:41` ↔ 同`:729`、`prior_builder.py:17` ↔ 同`:224`。

docstring/位置情報を除いたAST bodyが同一の本番↔研究関数14組を確認。Tikhonov、PCA prior、confidence weighting、asymmetric solve、sector prior、144行のcompute_blp_signalまで含む。単純stubや抽象methodを根拠に数を増やしていない。[clone inventory](inventory.json)。研究実験間にもDSR、simulation、bootstrapの複製がある。

実害: 数値安定性やリーク修正が片側にしか届かず、研究採用→production組込み時に別実装になる。body一致だけで全class挙動同一とは主張しない。

対応: productionで使用する純粋な数学の正本を共有し、研究側はモデル構成/変種だけを所有する。抽象classや多層pluginを追加するより、明示入力の小さな関数へ寄せる。共通化のbefore/afterを同じfinite/nonfinite/PSD/window入力で数値比較し、研究成果物は保持する。

### F24 [P3/構造負債] 大きい関数が責務境界の調停を抱えている

場所: `src/leadlag/execution/v2_bridge.py:206`（580行、AST branch65）、`execution/close.py:88`（458行/34）、`data/preprocessor.py:136`（438行/64）、`execution/var_history.py:239`（407行/43）。本番100行以上のfunctionは54個。

このbranch値はIf/loop/except等のAST数でありcyclomatic complexityではない。行数だけでbad codeと判定したわけでもない。問題はquote/artifact/risk/state/output/例外の調停が長い関数で混ざり、F01/F05/F14/F17のような境界の整合を局所testだけでは追いにくいこと。

対応: 価格入力の確定、計算、判断plan、broker副作用、reconciliationのrun-owned contractsを維持して分割する。特にquoteとaccount-riskのpreflight結果を型付きで渡し、文字列alertsや任意dictから後段が再推論しないようにする。SQLite/監査/復旧を「複雑だから」と削除しない。巨大汎用frameworkへの置換も必要ない。

### F25 [P3/構造負債] 未接続層と旧wrapper、reports内の実行依存

場所: `src/leadlag/execution/cost_calculator.py:88`、`src/leadlag/core/convex_optimizer.py:120`、`src/leadlag/utils/gap_matrix_io.py:452`、`src/leadlag/models/ml_overlay_inference.py:173`、`src/leadlag/execution/backtester.py:116`、`scripts/run_tests_parallel.sh:19`。

CostCalculatorは「unified」でも本番/研究の実処理から呼ばれずtestだけ。convex optimizerは研究とtestのみ。provider層も同様。load_gap_matricesはload_gap_bundleの互換wrapperでtest側だけが使う。backtesterのPnL wrapperも互換adapterと明記され、研究用ML wrapperはcache-only modelを作り直してから別にoverlayを掛ける。AGENTS/10/4ADRの「互換層を維持しない」と未解消の部分がある。

またtest runnerの実行に `reports/20260912_workspace_audit/watchdog.py` が必須。報告を整理/別checkout化すると通常検証が壊れる。artifact evidenceの保存と常用utilityの配置が混ざっている。

対応: 呼び出し元ごと正規APIへ更新し、productionで不要な計算はresearchへ移すか撤去する。必要なwatchdogはscripts/toolsの正規入口へ置く。単なる別名再公開を増やさない。注文状態・入力型・GapStore transactionのように不変条件を担う分割は、この整理対象に含めない。

## 7. ハーネス・文書・探索履歴・CI

### F26 [P2/確定] 運用手順と現行保有/監査の意味が違う

場所: `docs/日次運用手順書.md:78`、同`:86`、`docs/モデル技術仕様書.md:360`。

日次手順は15:30まで全square、overnight原則禁止とする。一方、正規cost設定とclose経路はlong75%/short50%持越しを扱う。事故時の「正しい残在庫」を運用者が違って解釈し得る。

監査説明も旧汎用auditorのensemble/P3/P4項目、単一all_passed、V1 variantを現行日次自動監査のように並べる。現行V2のleakage/numerical/fallback/実行gateの実契約に揃っていない。技術仕様の実装mapは撤去済みproductionのenhanced model/_BLPBase/BaseModel/旧configを参照する。

対応: 保有方針と手順は継承解決後のconfig/close実装に揃える。仕様・監査表は「実施項目/保証できない項目/停止結果/残在庫/復旧」を現行コードへ対応付ける。過去の数理説明は歴史節へ残せるが、現行mapと混ぜない。operational docもCIで参照・契約照合の対象にする。

### F27 [P3/ハーネス負債] SkillとIDE workflowの意味が古い

場所/確認例:

- `.agents/skills/leak-audit/SKILL.md:30` はleakage FAILEDの同条件flatを仮定しないと記述。現行 `audit_comparator.py:151` は同flagで明示flatにする。別経路を追跡する注意は有益だが、現行挙動の説明は更新が必要。
- `.agents/skills/experiment-design/SKILL.md:30` はhelperがtrialsを同名件数で上書きすると説明。現行 `experiment_utils.py:202` は明示trialsを保持し、study_idも扱う。
- debugging reference`:14` は存在しない `data/decision_cache.py` を指す。正規はmarket_data_cache側。
- `.devin/skills/prod-backtest-consistency/SKILL.md:20`等は旧 `scripts/experiments/` を多用。Windsurf planにも旧config/experiment参照がある。単純literal抽出だけで25個の不存在path参照を検出した。[一覧](audit_details.json)。code block内などはこの25に含まれない。
- `.windsurf/workflows/run-tests.md:19` は7process、実runnerは10。存在しないtest_backtester_910.pyや `_check_syntax.py` を指す。
- `.windsurf/workflows/walkforward-validation.md:35` は不採用結果をSKILLへ追記するよう求め、AGENTSのreports/graveyard分離と矛盾する。ただし同workflowは事後分割を真のOOSと呼ばない注意を既に持つので、そこを虚偽OOS手順とは認定しない。
- AGENTSのtest runner説明はunit/integration/researchのみと古い。**現在はfeaturesと唯一のregression testもrunnerに含む。未収録regressionがあるという指摘はしない。** 将来増えた場合に手動の列挙が再び漏れる余地はある。

実害: 別agent/IDEが正しい修正を古いAPIへ戻す、探索履歴がharnessへ蓄積する、不要な直接registry書込を選ぶ。CIのdocs検査は9文書だけで、AGENTS/Skill/operational docsを含まない。validatorはrelative linkとPath見出しtableを検査するため、通常のbacktick code pathや意味の不一致も捕まえない。

対応: AGENTSを不変条件の正本に保ち、Skillには作業手順だけを残す。IDE workflowから正本Skillへ短く参照し、動作の複製を減らす。current文書だけのliteral参照/公開symbol検査を追加し、歴史レポートへ現行API適合を強制しない。日時・保有・fallback等の意味はcontract test/対応表で検査する。

### F28 [P2/統計governance不足] 探索全体を1つの試行数として追えない

場所: `src/research/experiment_utils.py:202`、`src/leadlag/experiment_registry.py`、`var/experiments/registry.jsonl`。

raw registry101行は9/19以降。study_idなし87、trials欠落/1が70、DSR非null20。relatedなML/感応度試行が日付・候補別nameへ分散し、nameベースcounterでは探索family全体を数えられない。過去レポートはこのregistry開始以前にも多い。

81行に元record_idがないが、現行readerは決定的なlegacy IDを生成する。これは履歴を失うバグとは扱わない。長期感応度10候補のtarget訂正も追記recordとして保存されており、元行を削除すべきではない。現在のDSR=1を採用根拠にしない限界も近年レポートで明記されている。

残る実害: 名目DSR、試行間variance、未登録探索を含む実際の選択回数を一意に説明できない。コード変更後も同じreport名で再現できるか、helperの既定記録だけではcode/data/period hashが必須になっていない。

対応: 仮説familyのstudy ID、全候補/棄却/中断、選択時点、code/config/data/target/cost schema、IS/OOSとpurgeを事前記録する。過去試行は裏付けるreportから索引化し、不明はunknown/lower boundと明記する。見かけの件数を埋めるために架空試行を作らない。観測できた成功だけの探索履歴にしない。

### F29 [P1/現況] mainの必須CIをGitHub側で強制していない

場所: [main API](https://api.github.com/repos/shonen0129/NightManager/branches/main)、[issue #18](https://github.com/shonen0129/NightManager/issues/18)、[取得証跡](external_checks.json)。

確認時点でprotected=false、required checks enforcement=off/contexts=[]、repository rulesets=[]. 当該HEADのHosted CIは全step成功している。問題はCIを新設することではなく、失敗commit/未実行commitの通常main更新を防ぐ設定がないこと。

対応: #18の既定方針どおりrequired `quality-and-tests`、PR、bypass/direct pushの扱いをGitHub側へ設定し、その実効対象を確認する。今回はread-only監査なので保護設定を変更していない。

### F30 [P2/確定品質欠陥] US欠損proxyが上場前に限定されない

場所: `src/leadlag/data/preprocessor.py:243`。

XLC/XLRE/MTUM/VLUE/USMVのNaNを他ETFで埋めるが、pre-inceptionの条件を付けていない。2026-09-02のXLC closeを1個欠損させた合成raw入力をstrict_validation=Trueで処理すると、翌9/3にus_cc_XLC=.01の正常値として受理され、proxy provenance columnもない。[`post_inception_proxy`](probes.json)。

実害: データ取得障害を検証済みのreal XLCとして隠す。historical baselineに必要なpre-inception proxyと、現在の実銘柄の欠損は意味が異なる。窓/ベータ/予測/availabilityの監査では元データ欠損を復元できない。

対応: inceptionより前の明示研究proxyと、上場後の欠損を分離する。実銘柄欠損は品質statusへ残し、当日入力の欠損規則で拒否/flat等へ進む。proxyを使う期間・ticker・作成法・targetとの関係をprovenanceに保存する。既存US/JP非対称休日の整合修正は維持する。

### F31 [P2/確定境界] 年表外のJPX年末年始休場をtradingと返す

場所: `src/leadlag/core/market_calendar.py:127`、同`:140`。

static tableは2025–2027だけ。jpholidayは国民の祝日を検査するため、JPX固有の1/2・1/3・12/31閉場を補わない。追加probeは2024-12-31/2028-01-03をtrading=trueと返した。[JPX FAQ](https://www.jpx.co.jp/faq/others_general.html)の年末年始閉場規則と一致しない。

現行2026のstatic table内にこの欠落はない。歴史日時でのprevious/next session検査や2028以降のscheduler/risk照合で、不存在セッションを要求する境界不具合である。対応: 恒常的な取引所閉場規則を年表から分離し、未知年の祝日データ不足を明示する。年境界のprevious/next trading dayを検査し、現在の祝日tableの更新だけで全期間を保証しない。

## 8. 維持すべき整備済みの部分

旧issue/旧reportだけを根拠に現在も同じ欠陥があるとは判定しなかった。以下は現行コード・テスト・10/4ADR・trackerとの照合で維持されている。

- 正規ProductionV2Modelとtyped/versioned DecisionInputs、live/backtestの共通model factory。
- 当日bundle identity、config/input/ticker/horizon fingerprint、GapStoreのatomic bundle取得。前日行列を当日扱いするfallbackは本番正規経路にない。
- on-demandが失敗/無効ならflat。PIT不足multiplierと終端flatを区別する契約。
- leakage FAILEDとnumerical FAILEDの両方にfail-closed処理がある。既存の厳しい監査閾値は維持されている。
- risk STOPで新規リスク増加を防ぎ、照合済みreduction/closeを別に扱う。order intent、status polling、durable state/reconciliation、復旧の境界がある。
- 当日の未知targetを計算窓から除く既存property tests、fixed2010–2014のcomplete finite baseline検査。これを市場provider可用性の完全証明とは扱わない。
- US/JP休日の非対称alignment、仮行の同日US close混入防止、fracdiff warm-upの厳格化。
- versioned ML artifact/CURRENT/HISTORY、train cutoff/metadata検証、本番wheelのresearch除外。
- shared daily245指標、初期wealthを含むshared MDD、全評価営業日を主評価とする方針。F13/F14/F17は周辺入口に残る欠陥である。
- 現行leverage1.30、モデルgross上限2.0と実効gross上限2.6、risk上限3.0を分けている。gross≤2.0と実効≤2.6の混同で違反扱いしない。
- 研究結果をPENDING/REJECTEDとし、retrospectiveをforwardや実費成績と呼ばない直近レポートの姿勢。

監査・注文の状態機械・lease・transaction・入力の不変性は、実害を防ぐための複雑性である。削除して単純化する対象ではない。

## 9. 対応順と完了条件

| 順序 | 作業 | 完了の証拠 |
|---|---|---|
| 1 | F02のsecret保存/例外境界、F01のcarry損益、F13–F19の指標/会計契約を修正 | 合成最小例、正規入口の回帰、全tests/静的検査、成果物間一致。損益変更は主要baseline再評価 |
| 2 | F03/F05/F06/F22の市場/ADR producerを回復・受入 | 次の取引日のread-only preflight、凍結quote IDの同一伝播、ADR日付/coverage、deadline/失敗状態の記録 |
| 3 | F04のauthoritative account producerを完成 | cash/position/fill/feeの完全照合、日次/月次基準、欠損/不一致の停止、controlled live/scheduler受入 |
| 並行 | F29のmain保護、F26/F27の正本文書/ハーネス更新 | GitHub effective required checks、保有/監査対応表、現行symbol/path整合 |
| 4 | F07–F10/F28の実測forwardと収益採否 | 固定artifact、同日paired入力、全評価日、実費/反実仮想を区別、未閲覧期間、試行数補正と不確実性 |
| 5 | F21/F23–F25/F30の構造整理 | 呼び出し元を同時更新、research/runtime境界検査、pure math数値一致、wheel runtime-root受入 |

quote/shadowの前向き蓄積はaccount producer完成を待たず進められる。actual-liveの新規建てにはaccount producerも必要。研究の新しいパラメータ追加を先行させるより、正規の測定と会計を作り、どの機能が付加価値を生んでいるか識別することが優先される。

## 10. 未確認事項・追加調査

以下は今回未実施・未証明の追加調査事項である。

1. API認証前提を解消した後の実通信、取引日の新規frozen quote、約定・cash・feeの完全照合。今回brokerは呼んでいない。
2. F01等を直した後の2779日/269日/368日成績の変化量。保存レポートを照合したが新しい長期BT/再学習は行っていない。
3. 当時のYahoo/ADR/macro提供時刻、5分足barの時刻規約と9:10時点の取得可能性、historical revisionを含む可用性証明。
4. provider側の不調、休日、銘柄別貸株停止/逆日歩急増、partial fill、資本制約下での長期容量。旧AUMシナリオは容量実績ではない。
5. current artifactへ固定した250日forward gate。その経過時間と有効な観測を架空の回顧日数で埋められない。
6. すべての過去実験のtrial family復元、複数探索の有効試行数、report→registryの完全逆引き。
7. 新規ローカルwheel build（build dependencyなし）。同一HEADのHosted build/smoke成功は取得済みだが、実運用rootの受入は別である。
8. 部分障害・プロセス強制停止・SQLite復旧の実運用受入。mock/property testの成功から市場障害時の全動作を保証しない。

## 11. 再現資料

- [正本/設定/AST/registry inventory](inventory.json)、[source hash](source_manifest.json)、[関連tracked681ファイルのmanifest](audit_manifest.json)。実credentialを含む.env等は走査/転載対象にしていない。
- [安全化した運用/ML metadata evidence](evidence.json)、[追加harness/coverage統計](audit_details.json)、[GitHub read-only evidence](external_checks.json)。
- [合成probe](probes.py) / [結果](probes.json)。credentialは全て合成marker、ネットワークはmock。
- [全チェックdriver](run_checks.py)、[test checks](checks_tests.json)、[static checks](checks_static.json)、[package checks](checks_package.json)。
- [31項目の機械可読索引](findings.json)、[最終検証](verification.json)。確認したtracked681ファイルはhash一致、source参照の行は存在、報告書リンクは検証済み。

再現例（checkout rootから）:

```bash
timeout -k 10s 120s .venv/bin/python reports/20261006_project_audit/probes.py
timeout -k 10s 120s .venv/bin/python reports/20261006_project_audit/audit_details.py
timeout -k 10s 2700s .venv/bin/python reports/20261006_project_audit/run_checks.py tests
timeout -k 10s 1200s .venv/bin/python reports/20261006_project_audit/run_checks.py static
```

reports内のscriptsは本監査の再現用であり、日次運用の新しい正本にはしない。公開/共有前も既存raw broker log、取引CSV、credentialを含み得るDBをそのまま添付しない。
