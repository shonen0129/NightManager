# 監査issueの修正結果（2026-10-06）

監査HEAD `b5b901e6d467fb000263f9178722e3954d652720` に対するローカル修正。既存の監査資料を保持した。GitHubへのcomment/close、push、本番発注・決済・認証・scheduler変更、live入力の更新は行っていない。

**8件のissueでローカル実装上の主不具合を修正し、6件を部分対応した。actual-live受入は引き続きBLOCK。** issue全体の外部受入・履歴修正まで完了したとは扱わない。

## issue別の到達点

| issue | 到達点 | 実装・残件 |
|---|---|---|
| #33 (F01/F18/F19) | 部分対応 | carryの寄付→09:10を補完。実効売買flowのturnoverと旧target-weight turnoverを分離し、model/effective gross、volume、会計versionを成果物へ保存。終端在庫・initial holdings・cash/数量会計・artifact間の連続replay、主要長期成績とriskの再評価は残る |
| #34 (F02) | 部分対応 | SQLite設定を非broker allowlist＋config hashへ変更。HTTP/login/order/retry/JSON errorのURL・raw textを安全な例外へ変換。合成secretのDB/traceback回帰を追加。既存DB/logのscrub・実資格の失効/再発行状態確認は未実施 |
| #35 (F05/F22) | 部分対応 | ML applied/skipped/rejectedと固定reasonをsummaryへ明示。skipでweight・監査状態を保持。ADRの実更新復旧、鮮度manifest、producerのresearch依存解消・複数artifactのatomic publishは残る |
| #36 (F06) | ローカル修正済み | market-dataとdiagnosticsに共通lease・全体deadline・phase deadline。full-backtest/unit-testsにも全体deadline。test runnerのreports内watchdog依存を撤去。実shell guard prefixのstandalone/nested leaseを追加検証。scheduler実運用受入は未実施 |
| #37 (F07/F09/F10) | 未解決 | 同一09:10 target/実費/数量replay/容量/rolling tail stressの評価は未実施。モデル参照PnLとactual-account riskを引き続き区別 |
| #38 (F11/F12) | ローカル修正済み | V2開始日≥2015-01-05、要求期間の非空交差、start>end/NaT/空・未整列・重複indexの拒否。要求/実評価/source期間を `evaluation_period` に保存 |
| #39 (F13/F14/F17) | ローカル修正済み | 月次12年率、summary DDを共有正本へ統一、flat欠損price PnLは0、active欠損は無効。shared metricsは欠損評価日を除外せず拒否。registryにはinvalid coverageを保持 |
| #40 (F15) | ローカル修正済み | MinVarへ選択済みindicesを渡しlong5/short3を保持。schema銘柄数上限、basket重複/範囲/covariance次元を検証。全本番/研究callerを更新 |
| #41 (F16/F28) | 部分対応 | invalid status/schema/frequency/finite/T/trials/varianceをDSRで検証し、computed metricsの追加metric上書きを拒否。study横断の既存探索履歴復元・trial family集約は残る |
| #42 (F20) | ローカル修正済み | Yahoo OHLCでVolume欠損を許可。requested atまでに完了した同日1分barのみを使用。Tachibana adapterはtimestamp指定をサポートできないため明示拒否し、opening_priceを代用しない。未接続providerをactual-live受入として扱わない |
| #43 (F21) | ローカル修正済み | package位置とdeployment rootを分離し `LEADLAG_RUNTIME_ROOT` を必須化。相対model/ADR/macro/varとVaR code fingerprintを修正。ローカルbuild wheelを隔離して実import/CLI/ML inference/指定root検査に成功 |
| #44 (F23) | 未解決 | BLPX本番/研究純粋計算の重複統合は未実施 |
| #45 (F24) | 未解決 | 日次/close/前処理/VaRの大規模責務分割は未実施 |
| #46 (F25) | 部分対応 | reports内watchdog依存を既存phase_deadlineへ変更。未使用層と互換wrapperの全撤去は未実施 |
| #47 (F26/F27) | 部分対応 | 通常closeの持越し、現行監査/gate、仕様の実装map、Skillのfallback/trials/cache契約、IDE test/walkforward手順とAGENTSのrunner説明を更新。歴史plan/全公開symbolを網羅する意味検査の新設は未実施 |
| #48 (F30) | ローカル修正済み | fund inception前だけproxyを使用しcell別provenanceを保存。後のUS欠損はstrict拒否/非strict record拒否。旧前処理cacheのcontract version不一致はstale fallbackでも拒否。既存運用cacheは書き換えず、次回の正規strict再構築が必要 |
| #49 (F31) | ローカル修正済み | static年表外の1/1–1/3/12/31もJPX休業日。年越しprevious/next sessionを回帰検証。未知年の祝日データ不足は従来のwarningが残る |
| #27 (F03) | 未解決 | broker開示書類確認・認証前提を実口座側で解消し、次の取引日でread-only captureを受入する必要がある |
| #25 (F04) | 未解決 | authoritative cash/position/fill/feeの照合済みaccount-risk producerと日次/月次基準が必要。既存fail-closedを維持 |
| #23 (F08) | 未解決 | fixed artifactのforward gate、有効paired観測と実費比較が必要。回顧成績を前向き観測に置換しない |
| #18 (F29) | 未解決 | GitHub mainのrequired checks/ruleset設定と実効保護確認が必要。本作業ではGitHub設定を変更していない |

冒頭の8件は #36/#38/#39/#40/#42/#43/#48/#49。部分対応は #33/#34/#35/#41/#46/#47 の6件である（issue完了とローカル実装修正を区別する）。

## 検証

- 対象回帰・関連既存テスト171件PASS、cache/registry追加確認13件PASS。最終のfrequency/timezone境界56件もPASS。
- 固定regression baseline: **1 passed**。
- その他 `tests/` 全体、xdist4 workers: **1,041 passed、3 warnings、93.17秒**。合計 **1,042 passed**。warningsは既存のDataFrame fragmentationと研究constant series診断。
- Ruff本番/tests/maintained tools: PASS。mypy本番152ファイル: PASS。import契約7件: PASS。
- compileall必須5treeと変更した5shellの `bash -n`: PASS。
- wheel: cached setuptools/wheelを読み取りで再利用し、temporary project copyからbuild。157fileのsource manifest/research除外PASS。wheel展開先のみから `python -I` でimport、CLI help、ML artifact roundtrip inference、temporary deployment root指定PASS。global/project環境へ依存を追加していない。
- 各長時間検証に `timeout -k` を使用。全testsはbaseline300秒、その他1800秒、対象回帰300秒、静的120/240秒、wheel120秒。注文・networkを使わない合成fixtureで検証した。

証跡は [検証概要](verification.json)、[全tests JUnit](tests.xml)、[全testsログ](pytest_full.log)、[対象回帰](pytest_targeted.log)、[固定baseline](pytest_baseline.log)、[Ruff](ruff.log)、[mypy](mypy.log)、[import契約](import_linter.log)、[wheel build](wheel_build.log)、[隔離wheel smoke](wheel_smoke.log)。

## 適用範囲と残る制約

1. carry価格区間とturnoverの定義が変わったため、旧2779/269/368日成績の再認定はしていない。数量/cash/終端inventory、費用日数、artifact切替時の連続性は #33 に残る。現行weight型PnLの売買量はsimulated notionalであり、実約定量・価格変動による保有数量のdriftを保証しない。
2. proxy拒否の影響で既存raw dataの欠損が表面化し、正規cache再構築が失敗する場合がある。欠損を補間してPASSにせず、来歴と原因を確認する。運用cacheをこの修正で再保存していない。
3. installed wheelはdeployment data rootをimport前に指定する必要がある。config/model/dataはそのrootへ明示配置する。checkoutからinstalled code位置への探索fallbackは増やさない。
4. 未接続Yahoo providerのminute bar completionはtimestampに基づく境界制御であり、providerの真のavailable_at/遅延を証明しない。actual-liveは既存frozen quote経路の受入が必要。
5. secretの過去記録scrub・失効確認、ADRの実市場更新、main保護、口座PnL producer、forward250日観測は未完了。これらのため既存の新規リスク停止を解除していない。

設計理由・日付一次資料は [ADR](../../docs/decisions/2026-10-06-audit-boundaries.md)、対応一覧は [元のissue対応表](../20261006_project_audit/issue_mapping.md)。
