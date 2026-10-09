# 日次実行・close・前処理・VaRの責務境界

## 状態

2026-10-08採択。GitHub Issue [#45](https://github.com/shonen0129/NightManager/issues/45) の実装境界を定める。

## 決定

大きな実行入口は、行数ではなく、入力を選ぶ責務、純粋計算、計画、外部副作用、永続観測、照合の境界で分ける。各runが使う入力選択とpreflight結果は型付き契約で渡し、下流moduleが文字列や暗黙I/Oから要件を再推定しない。

- `execution.v2_bridge.run_v2_decision` はCLI/config/date解決の後、選択価格を `CurrentPricePreflight`、PITとの組合せと検証済みmarket snapshotを `QuotePreflight`、口座リスク証跡の要否・結果を `AccountRiskPreflight` に束ねる。日次入力の取得とDecisionInputs組立は独立helperへ置き、後続decision、risk gate、発注・manifest出力の順序を保つ。
- `execution.close.close_all_positions` は `CloseOrderPlan` で純粋な数量・注文計画を作り、送信・status観測と永続保存を一括ごとに順番どおり進め、最後にfill取得、照合、出力を行う。最初のbatchのstatus確認が失敗した場合に次batchを送らない既存規則も維持する。
- `data.preprocessor.preprocess_data` は `PreprocessorInputs` に整列済みsource、リターン・proxy・対象日・beta計算結果を束ねる。execution record検証とTOPIX/beta診断出力を別helperに分け、proxy、暦、欠損、厳格検証の意味を変えない。
- `execution.var_history.get_hist_returns_for_risk` は `VaRHistorySource` がdate-bounded入力・cache/database scope・deadline・snapshot所有権を持ち、`VaRHistoryReplayPlan` がcache identityとreplayに共通する入力を固定する。cache判定、worker実行、監査、期限内cache保存、snapshot解放を明示し、timeout時にworkerが読むsnapshotの寿命を保つ。

## 保存する挙動

リファクタリングは戦略または運用仕様の変更ではない。固定fixtureでdecision・manifest・end reasonを保つ。数値監査・PIT・artifactの失敗はfail-closedのままにする。closeのSTOP、部分約定、拒否、照合・再起動復旧の永続化順序を変えない。VaRの同一絶対deadline、cache key/鮮度判定、snapshot cleanup ownershipを維持する。過去固定期間、価格source、account-risk gate、flat fallback、broker発注の保護条件も弱めない。

レビュー基準は、各境界の入力と出力を局所的に検証できること、外部副作用の所有者と順序が読めること、既存fixture・fault testで結果と失敗理由が比較できることとする。行数やASTのサイズは完了条件にしない。

## 上位計画との整合

この変更は[構造改善ADR](2026-09-15-structural-improvement-boundaries.md)の「取得済み入力と計算の分離」「計算結果・注文計画・観測・照合の分離」「修正前後の比較を先に定める」を、残る日次・close・前処理・VaR lifecycleへ適用する。P35の同期V2 canonical pathと、[audit境界ADR](2026-10-06-audit-boundaries.md)のfail-closed方針を維持する。新しいavailability・数値・発注判断は追加しない。

`docs/refactor_roadmap.md`はGitHub tracker #22をbacklogの正本とするため、Issue #45へのリンクと題目のみ登録し、進捗状態は複製しない。対象moduleの責務は`docs/ARCHITECTURE.md`にも反映する。

## 実施・検証記録

実装境界、変更対象、検証結果は[Issue #45実施記録](../../reports/20261008_issue45_refactor/report.md)を参照する。
