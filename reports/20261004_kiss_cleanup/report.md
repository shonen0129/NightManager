# KISS整理・正本の一本化

2026-10-04。依頼されたローカル実装・文書整理の判定は **PASS**。
本番パッケージを精査し、CLI、日次判断、V2バックテスト、研究・運用ツール、テストの利用者を
正規APIへ更新した。整理前の作業ツリーを基準に、本番Python実装は **1,748行減**
（38,438 → 36,690行）、4 moduleを撤去。既存のユーザー変更を含むHEADとの差分とは区別する。
将来の作業も後方互換層を持たない方針を [AGENTS.md](../../AGENTS.md) に明記した。

## 修正した問題

| 重要度 | 発生条件・実害 | 修正・根拠 |
|---|---|---|
| P1 | `AppConfig(risk=RiskConfig(...))`で閾値を指定しても、実行側が`StrategyConfig`の重複値を再構築し、指定した損失停止条件が届かなかった | リスク閾値を`RiskConfig`だけに置き、日次経路から直接渡す。日次損失0.5%に対し停止閾値0.2%で出力・発注へ進まず、2%なら進む回帰を追加 |
| P2 | typed inputの`source`文字列が暗黙I/Oの許可を決め、同じ入力契約でも取得経路が変わった | `source`は来歴のみ。型付き判断は明示入力を要求し、旧source名でもmacro等の読込を許可しない回帰を追加 |
| P2 | PIT履歴で基準費用IR列が欠けると、計算式の異なるex-ante IRを自動代用した | `pred_ir_gap_baseline_cost`のみを使用。欠損は設定済みの履歴不足処理へ進み、異なる式の履歴を拒否する回帰へ変更 |
| P2 | cache roundtripテストが未使用のpath関数をpatchし、実際のworkspace cacheへ書いた | 正規ストアのパスを`tmp_path`へ隔離し、現行schemaのDataFrameで検証 |
| P3 | 二重の市場履歴ストア、ML/モデル再公開、旧引数wrapper、状態booleanとenum、旧名、無効CLI引数が併存した | 利用者を同時に移し、不要な経路・別名・推論を撤去。package/APIの正本を一つにした |
| P3 | architectureが未接続の板執行ルールや旧モデル階層を現行実装として記載した | 現行module・責務・判断契約へ更新。roadmapはissueへの入口、過去ADRは当時の記録として扱う |

リスク設定の回帰は修正前に失敗することを確認した。既存の戦略数式や監査閾値を緩めて
通過させていない。旧API自体を対象にしたテストは撤去・現行契約へ変更し、現行挙動の
回帰・境界テストを追加した。

## 最終的な構造

- 判断は `ProductionV2Model.decide(inputs: DecisionInputs)`。
  日付・価格・履歴・cache利用設定を一つの入力契約が所有する。
  artifactだけの再生は `decide_from_cache(trade_date, gap_input_dir)`。
- 分布状態は `DistributionResult.status` / `reason`。booleanとの二重保持やmetadataからの
  horizon推論を廃止した。fallbackはtypedな状態と理由を使う。
- ML container、特徴量、artifact、推論は本番の各module、学習はresearch。
  学習入口を `tools/research/train_ml_order_overlay.py`へ移した。writerは`PortfolioDecision`のみを受ける。
- リスクは `AppConfig.risk`、ML設定と費用は`AppConfig.v2`。
  YAML sectionの解決は入口で一度だけ行い、未使用のML設定コピーを保持しない。
- 市場cacheは `market_data_cache.py` / `SqliteCacheStore`。
  二重書込・旧ストアへの切替・raw pickle読込・DataFrame pickle fallbackを撤去した。
  保存形式はJSON envelope / Parquet。市場パスは `var/market_data`。
- gapは `GapStore.load_horizon_bundle`で行列・metadata・manifestを同時に読む。
  manifestの欠けたkeyや旧versionを補完しない。相関cacheは呼出側が明示して所有する。
- 未使用low-rankモデル、V1型、dispersion/gap-tolerance関数、SessionState、未接続板執行規則、
  gap診断中継、余分なpure helperを撤去。研究shadowの無効なmonkeypatch・履歴制限optionも撤去した。
- 依存extraは推論/CI用`ml-overlay`、学習用`research`。重複extraをなくし、lockを更新した。
  必須pandas 3とPyArrowに合わせて旧versionの分岐を撤去した。

判断の正本は [設計ADR](../../docs/decisions/2026-10-04-kiss-canonical-apis.md)。
module配置・運用境界は [ARCHITECTURE](../../docs/ARCHITECTURE.md)、
研究環境は [RESEARCH_ENV](../../docs/RESEARCH_ENV.md) に反映した。
NPY回帰bundle、SQLite transaction、監査、注文照合、欠損価格の処理、研究成果・過去の判断記録には
現行の責務があり、保持した。

## 数値・不変条件の検証

整理前と整理後を別のコード版・別process・独立configで比較した。
4 seed × minvar有/無 × kernel/ML overlay/型付きcache判断/artifact再生 = **32ケース**で、
`w_final`、scores、mu、sigma、Omega、fallback、PIT、leakage/numerical監査、現行summaryが
**完全一致**した（NaN同士は同値）。撤去した`p_trade_*`別名のみ比較から除外した。
ML overlayは明示ADR入力を渡し、実際に適用された8ケースをassertした。

| 指標 | 合成比較の最大値 | 継承解決後の制約 |
|---|---:|---:|
| モデルgross | 2.0000000000000004 | 2.0（浮動小数点丸めを除く） |
| モデルabs(net) | 3.06e-16 | 0.05 |
| 実効gross (`side_leverage=1.30`) | 2.6000000000000005 | risk max gross 3.0 |
| 実効abs(net) | 3.89e-16 | risk max net 0.05 |

本番設定は`strict=True`で継承解決した。N_U=15 / N_J=17、
`ondemand_fallback_enabled=true`、`fallback_on_audit_failure=true`を確認。
片道slippage 5bps、金利・貸株・逆日歩の解決値は [effective_config.json](effective_config.json) に保存した。
PIT計算窓、当日ターゲット遮断、baseline期間、当日gapと来歴、数値/リーク監査の失敗処理、
cache→on-demand→flatを対象の回帰と全テストで確認した。合成比較だけを非リークの証明にしていない。

証拠は [数値比較](numerical_comparison.json)、[整理前出力](synthetic_before.json)、
[整理後出力](synthetic_after.json)、[source manifest](source_manifest.json)。
整理前のsource snapshotは`/tmp/leadlag_kiss_before/src`、基準HEADは
`5bfb55bf6347f8e9c2014e7827f1c02804448f58`。基準は着手時のユーザー変更を含む作業ツリーであり、
HEAD単体を整理前として扱っていない。snapshotは一時領域なので、永続的な証拠には出力とsource digestを残した。

再実行するcapture入口は [verify_kiss_refactor_20261004.py](../../src/research/scripts/experiments/verify_kiss_refactor_20261004.py)。
`--old`は比較専用scriptが保存済み旧sourceの入口を選ぶためのoptionで、本番に互換層を設けない。

```sh
timeout -k 10s 120s env PYTHONPATH=src .venv/bin/python \
  src/research/scripts/experiments/verify_kiss_refactor_20261004.py --output /tmp/kiss_after.json
timeout -k 10s 120s env PYTHONPATH=/tmp/leadlag_kiss_before/src .venv/bin/python \
  src/research/scripts/experiments/verify_kiss_refactor_20261004.py --old --output /tmp/kiss_before.json
cmp /tmp/kiss_before.json /tmp/kiss_after.json
```

## 品質検証と範囲

最終結果は [validation.json](validation.json)、全体pytest結果は [tests.xml](tests.xml) に保存する。
長時間コマンドにはプロセス全体の停止期限を付け、既存`.venv`を使った。
warningは既存fixtureのDataFrame fragmentation、fork、定数相関で、失敗を成功扱いしていない。

| 検証 | 結果 |
|---|---|
| 整理前`tests/ -n 4` | 967 passed |
| 整理後の全`tests/`（固定回帰を直列で分離） | **972 passed**（971並列 + 1直列）、3 warnings |
| 重点契約回帰 | 127 passed |
| Ruff（本番、tests、maintained tools、学習入口、比較script） | PASS |
| Mypy本番 | 152 source files、エラーなし |
| compileall（src/leadlag、tests、tools、scripts、src/research） | PASS |
| import-linter | 7 contracts kept、0 broken |
| lock整合、文書リンク、`git diff --check` | PASS、文書参照157件確認 |
| clean wheel、research除外、インストール後CLI・ML artifact推論 | PASS |

初回の既存テストは旧`var/market_data/decision_cache.sqlite`へ合成DataFrameを書き込む問題があった。
その旧ストアは今回の一本化で読まれなくなり、現在のテストは一時領域に隔離している。
この既存ファイルを本番履歴に昇格・コピー・自動移行していない。

この判定は依頼されたコード・文書整理の完了に関するもの。本番運用全体のリリース判断は
**CONDITIONAL PASS**で、実取引日の運用受入、actual-account producer、ML forward評価等の
既存backlogは [roadmap](../../docs/refactor_roadmap.md) が参照するissueで管理する。
本作業では実発注・決済・本番昇格や性能改善実験を実施していない。
そのためSharpe・DDやライブ価格の妥当性を新たに評価したとは主張しない。
研究tree全体の既存lint backlogも、現行CI対象外まで解消したとは扱わない。
