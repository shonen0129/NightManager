# S0–S8 全体再レビュー

実施日: 2026-09-21 / 対象: 現行作業ツリー、[構造改善計画](../20260915_structural_improvement/plan.md)、[修正報告](../20260917_structural_completion_review/report.md)。

## 判定

**BLOCK。P1が4件、P2が4件。** 前回指摘したPIT履歴日付の監査伝搬とADR欠損日のBT/live共通判定は修正されている。
ただし入力・cache・執行復旧・会計に下記の問題が残り、現行のCIと同じmypyコマンドも失敗する。
全825テストのPASSだけではS0–S8の受入完了とは判断できない。

レビューとしてsource・設定・既存テストは変更していない。本ディレクトリへレビュー用の再現スクリプト、ログ、wheel、報告を追加した。
broker照会・発注・取消・再送、scheduler変更、学習・本番artifact変更は実行していない。

## 指摘

### F1 [P1] 過去の09:10欠損で、通常の実行入力が分布解決できない

- **箇所:** [intraday_inputs.py](../../src/leadlag/data/intraday_inputs.py) 183–192行、[distribution_source.py](../../src/leadlag/models/v2/distribution_source.py) 212–227行・373行以降。
- **条件:** 当日17銘柄の09:10値が揃っていても、`df_exec`の過去行に1セルでも欠損がある。
- **問題:** strict typed pathのfinite検査は当日の観測だけでなく全履歴を対象にする。長期の日次履歴と5分足の収録範囲が異なる通常入力でfile cache/on-demandの双方が`inputs_missing`になり、既定policyはflatへ進む。BTとVaRの計算にも同じ条件が適用される。
- **再現:** ローカルSQLiteをread-onlyで取得した4,135行（2009-01-07〜2026-08-17）では、17銘柄が揃う日は8行、欠損セルは68,815。直近の完全観測日2026-07-24でも、過去行を含めると両sourceが`inputs_missing`。データを補間・更新せず確認した。
- **修正案:** 当日の必須観測と学習履歴の入力契約を分ける。必要な履歴データを整備するか、仕様上許容する過去の価格代替を取得境界で来歴付き入力として固定する。現在の09:10欠損拒否を単純に無効化する修正にはしない。
- **不足テスト:** 数行の全有限/全NaNケースだけでなく、長期履歴に欠損があり当日は完全な現実の入力で、正常計算できることと欠損方針を検証する。
- **証拠:** [probe_inputs_and_state.py](probe_inputs_and_state.py)、[実行ログ](probe_inputs_and_state.log)。

### F2 [P1] 準備済みrunの再開が、別runの未解決状態を迂回する

- **箇所:** [state_store.py](../../src/leadlag/execution/state_store.py) 330–351行。
- **条件:** 同一口座・戦略にPREPAREDまたはFAILED_BEFORE_SUBMISSIONのrun Aがあり、別run BがEXECUTING/RECONCILIATION_REQUIREDになった後、Aを再開する。
- **問題:** 既存job keyの分岐が351行でreturnし、357行以降の未解決run検査へ到達しない。Aの`mark_submission_started()`も別runを検査しないため、Bの約定・在庫が未確定のまま追加送信へ進める。leaseは同時実行を防ぐが、順番に起動するこのケースは防がない。
- **再現:** 一時SQLiteでcloseを準備→decisionを照合待ち→同日のcloseを再開し、未解決runが2件（executing / reconciliation_required）になった。
- **修正案:** 新規作成と既存run再開の両方で、自runを除いた未解決runを同一transaction内で検査する。送信開始への遷移でもこの制約を維持する。
- **不足テスト:** 既存の「翌日/別jobの新規作成を拒否」に、作成済みrunの再開を追加する。
- **証拠:** [probe_inputs_and_state.py](probe_inputs_and_state.py)、[実行ログ](probe_inputs_and_state.log)。

### F3 [P1] 当日snapshotの訂正が分布cacheの来歴検査に入らない

- **箇所:** [distribution_source.py](../../src/leadlag/models/v2/distribution_source.py) 229–238行、[gap_provenance.py](../../src/leadlag/utils/gap_provenance.py) の`bundle_identity()`。
- **条件:** `df_exec`と`open_910_returns`は同じだが、`KnownMarketInputs`/`MarketSnapshot`の当日gap・beta・TOPIX夜間値が異なる。live bridgeはAPI価格からsnapshotのgapを組み立てるため、履歴フレームと異なり得る。
- **問題:** cache側はsnapshotをidentityに含めず、古いgapで計算したμ・ΩをREADYとして返す。一方on-demand側はsnapshotを使うため、cache有無で同じ意思決定入力に対する出力が変わる。
- **再現:** 2026-08-17のフレームと本番設定のBLPXで、1銘柄のgapだけを+0.01訂正。cache/on-demandともREADYだが、μ最大差0.0059504714、Ω最大差1.1074764e-6。cacheは訂正前から変化0。独立したF1の影響を避けるため、この比較の09:10リターンは明示的なゼロ合成入力で固定した。
- **修正案:** 実際に分布計算へ使う当日既知入力も生成時・読込時のidentityへ含める。合わないcacheはon-demandへ送り、snapshotと履歴のどちらを用いたかを証跡に残す。
- **不足テスト:** 履歴/09:10フレームの変更だけでなく、snapshotだけを変えるcache/on-demand比較。
- **証拠:** [probe_cache_snapshot.py](probe_cache_snapshot.py)、[実行ログ](probe_cache_snapshot.log)。

### F4 [P1] 約定価格・費用未確認でもrunを完了してしまう

- **箇所:** [post_decision.py](../../src/leadlag/execution/post_decision.py) 327–356行、[close.py](../../src/leadlag/execution/close.py) 637–658行、[reconcile.py](../../src/leadlag/execution/reconcile.py) の`verify_position_checkpoint()`。
- **条件:** FILLED数量と建玉は一致するが、約定詳細の価格または費用が欠ける。broker応答にその項目がない場合も取得関数が例外を出すとは限らない。
- **問題:** 通常終了時のcheckpointは数量だけを検査し、missing price/feeを検査する復旧経路と契約が異なる。completedにすると復旧候補から消え、`list_observed_fills()`は不完全な約定を会計用Fillから除外する。
- **再現:** post-decisionのI/O境界をstub化し、価格欠損・費用欠損それぞれで`completed`、復旧候補0件、会計Fill0件を確認した。実brokerは呼んでいない。
- **修正案:** 通常終了と再起動後の照合に、数量・有限かつ正の価格・明示された有限費用の共通検査を使う。未確認ならreconciliation_requiredを維持する。
- **不足テスト:** 保存例外だけでなく、正常応答に必要項目がないケース。
- **証拠:** [probe_reconciliation.py](probe_reconciliation.py)、[実行ログ](probe_reconciliation.log)。

### F5 [P2] 部分約定後の取消が実現損益から消える

- **箇所:** [core/pnl.py](../../src/leadlag/core/pnl.py) 211–213行。
- **条件:** 注文の残数量が取り消され、status=CANCELLEDだが`fill_quantity > 0`で確定済みの価格・費用がある。
- **問題:** `fill_from_record()`がstatusだけでreturn Noneするため、日次PnLから実約定分も脱落する。
- **再現:** 100円で保有した株のうち4株を110円で売却、費用2円、残6株取消という入力で、正しい実現損益38円に対して帳票0件。同じ約定値のFILLED対照ケースでは38円。
- **修正案:** 注文の終端状態と約定の有無を分け、取消後も確認済み数量・価格・費用の分を計上する。
- **不足テスト:** 確定した部分約定を持つCANCELLEDの会計回帰。
- **証拠:** [probe_accounting_and_ml.py](probe_accounting_and_ml.py)、[実行ログ](probe_accounting_and_ml.log)。

### F6 [P2] 日次PnLが未取得の費用を0円として確定する

- **箇所:** [core/pnl.py](../../src/leadlag/core/pnl.py) 231–244行、[daily_pnl_report.py](../../src/leadlag/reporting/daily_pnl_report.py) 141行。
- **条件:** 約定価格・数量はあるが費用がNone、または費用項目自体がない。
- **問題:** 費用キーの存在だけで確認済みと判定し、`raw_fee or 0.0`でNoneを0にする。さらに実帳票は`source="observed_fill"`なので、`source == "observed"`の欠落チェック自体を通らない。SQLite側の「未確認費用をFillから除外」とも不一致。
- **再現:** 費用None・項目なしの両方で、報告レコード1件・確定費用0円となった。
- **修正案:** 実測sourceの表記と未確定値の判定を統一し、明示的な0円と未取得を区別する。未取得費用を含む損益は未確定として扱う。
- **不足テスト:** helperのデフォルトsourceだけでなく、日次帳票から呼ぶ経路とNone値を検証する。
- **証拠:** [probe_accounting_and_ml.py](probe_accounting_and_ml.py)、[実行ログ](probe_accounting_and_ml.log)。

### F7 [P2] ML学習終了日のtimezone変換漏れで当日の適用を許す

- **箇所:** [ml_overlay_artifact.py](../../src/leadlag/models/ml_overlay_artifact.py) 79–84行。
- **条件:** artifactのtrain_endがoffset-aware timestampで、JST変換すると翌日になる。
- **問題:** trade_dateはJST正規化されるが、train_endは元timezoneの暦日に切られるため、同じJST営業日なのに学習期間外と判定する。
- **再現:** `train_end=2026-09-17T15:00:00+00:00`（JST 9/18）が9/17へ正規化され、9/18の`apply_overlay()`が`overlay_applied=1`となった。実artifactの学習データ混入を確認したという意味ではなく、provenance検査の抜けの再現である。
- **修正案:** 学習期間の各日付も共通JST正規化を使うか、offset付き値を契約上拒否する。
- **不足テスト:** 学習終了日と取引日がtimezone変換後に同日となるartifactの拒否。
- **証拠:** [probe_accounting_and_ml.py](probe_accounting_and_ml.py)、[実行ログ](probe_accounting_and_ml.log)。

### F8 [P2] 現行CIのmypyゲートが失敗する

- **箇所:** [v2_bridge.py](../../src/leadlag/execution/v2_bridge.py) 110行。
- **再現:** `.venv/bin/mypy --config-file pyproject.toml src/leadlag`で`Returning Any from function declared to return "str" [no-any-return]`。147ファイル中1エラー。
- **実害:** Hosted CIの同じ型検査stepが通らない。過去の対象ファイルだけのmypy PASSを、現行production全体のPASSと扱えない。
- **修正案:** `_assert_not_future()`の戻り値を既存方針に沿って明確なstr型にし、production全体を再検証する。型検査設定を緩めない。
- **証拠:** [mypy.log](mypy.log)。

## 観点別の確認状況

| 対象 | 今回の確認・判定 |
|---|---|
| S0 比較基準 | 既存のbaseline/replay・R3/R5証跡と受入条件を照合。過去の2日比較を現在の全機能・全入口の保証にはしない。F1/F3により現行版の再比較が必要。 |
| S1 設定・factory | 本番YAMLの継承先baseと解決済み設定、model factory、BT/live/VaRの呼出を確認。モデルgross 2.0、side leverage 1.5、実効gross上限3.0・net上限0.05を確認。今回、運用の実効exposureは計測していない。 |
| S2 入力/PIT | 所有コピー、JST/as-of、targetマスク、PIT日付伝搬、ADR共通判定を確認。F1。gap全入口とavailable_at証拠の既知残件は未解消。 |
| S3 分布/cache/VaR | file→on-demand→flat、bundle来歴、h=3/5 snapshot、VaR snapshot/key/workerを確認。F1/F3。 |
| S4 型・執行結果 | decision/plan/observation/report境界とpost-decision/closeを確認。F4。 |
| S5 台帳・復旧・期限 | intent保存、状態遷移、lease、guard、4 batch、read-only recoveryのコードと回帰を確認。F2/F4。実scheduler・実brokerは再検証していない。 |
| S6 会計 | 共通PnL、日次報告、Fill保存・照合を確認。F4/F5/F6。実口座残差の検証は未実施。 |
| S7 ML・配布 | factory、artifact読込/保存、学習終了日チェック、research分離、隔離wheelを確認。F7。本番artifact再生成・昇格の既知残件は未解消。 |
| S8 CI・文書 | 下表を実行。F8。文書は全体部分完了を維持しているが、計画冒頭の808件など過去の検証件数が現行825件と混在している。Hosted CIは未実行。 |

## 実行検証

| 検証 | 結果 |
|---|---|
| 全`tests/`、外側watchdog 360秒 | **825 passed / 17 warnings / 105.19秒**。[tests.log](tests.log) |
| 上記8指摘の補助確認 | read-onlyの実データ、合成入力、一時SQLite、stubで再現。各節のスクリプトとログを参照。 |
| Ruff（CIと同じproduction/tests/tools範囲） | PASS。[ruff.log](ruff.log) |
| mypy production | **FAIL、1 error / 147 files**。[mypy.log](mypy.log) |
| import-linter | 7 kept / 0 broken。[imports.log](imports.log) |
| compileall | PASS。[compile.log](compile.log) |
| architecture/CI/scheduler/ADR/planのリンク | 84リンクPASS。[docs.log](docs.log) |
| uv lock offline | PASS。[lock.log](lock.log) |
| wheel build・research除外・隔離CLI/合成ML推論 | PASS。[wheel.log](wheel.log)、[build](wheel_build.log)、[smoke](wheel_smoke.log) |
| 4 batchのbash構文、git diff --check | PASS。 |

全テストは既存の期待値を変更せず実行した。補助再現は既存テストが通過する一方で残る境界問題を確認するもので、既存テストのFAILとは区別する。
F3の初回補助実行では報告用config属性の参照ミスで出力時に停止したため、補助スクリプトのみ修正して再実行した。production codeは変更していない。

全体監査は主要な実行経路・契約・異常系を対象とした。全分岐の形式検証、全期間・全機能の数値同値性、実口座照合、外部データの観測時刻、Hosted CIを証明したものではない。

## 2026-09-21 再レビュー指摘の修正

残っていたsnapshot provenanceと09:10価格欠損の2件を修正した。

- `MarketSnapshot`を伴うcache読込は、run-owned `df_exec`との`gap_inputs_version`照合を必須化した。旧形式bundleはsnapshotなしの互換読込に限定し、snapshotありで来歴項目が欠ける場合は`PROVENANCE_REJECTED`からon-demandまたはflatへ進む。
- 本番17銘柄の`DecisionInputs`は、ProductionV2Model/ProductionRunnerの境界で全`current_prices`の存在・有限性・正値を検証する。欠損を0 gapとしてモデルへ渡さない。
- 研究用multi-horizon blendも共有`load_gap_bundle`を使い、bundle identityと`gap_inputs_version`を必須化した。生成側のrun-owned identityを渡し、旧形式matrixを黙って混ぜない。
- 旧baseline bundleへ、同じPIT入力から計算した`gap_inputs_version`とmanifest digestを追加した。

回帰テストを3件追加し、対象テストは **37 passed**、全`tests/`は **836 passed / 16 warnings / 96.65秒**（pytest-xdist、watchdog 300秒）。compileall、production mypy（147 files）、Ruff F821、`git diff --check`もPASSした。R1/R2/R4/R5/R6の外部受入条件とS0–S8全体完了の判定は変更しない。

## 2026-09-21 指摘事項の修正結果

上記 F1–F8 を修正し、回帰テストを追加した。

- F1: strict 09:10入力の有限性検査を当日必須行へ限定し、過去行の欠損は既存の明示的なtarget fallbackで扱うようにした。
- F2: PREPARED/FAILED_BEFORE_SUBMISSION の再開と EXECUTING への遷移で、同一口座・戦略の別未解決runを同一transaction内で拒否するようにした。
- F3: gap/beta/TOPIX夜間値から `gap_inputs_version` を生成し、研究生成・cache読込の新形式bundleでsnapshot訂正を検出するようにした。旧bundleは既存の来歴項目を検証したうえで互換読込する。
- F4: 通常終了と復旧照合のcheckpointで、確認済み数量に対する有限・正の約定価格と、明示された有限・非負の費用を必須化した。
- F5/F6: CANCELLED後の確認済み部分約定を会計へ残し、`observed`/`observed_fill` の未取得費用（None・NaNを含む）を0円として確定しないようにした。
- F7: overlay artifactの日付正規化をJST基準へ統一した。
- F8: `v2_bridge.py` のmypyエラーを戻り値型の明示で解消した。

追加回帰を含む対象テスト、全`tests/`（regressionを含む）は **833 passed / 17 warnings / 96.56秒**（pytest-xdist）で完了した。`compileall`、production mypy（147 files）、Ruff F821、並列テストスクリプトもPASSした。Hosted CI、実口座照合、外部データの最新取得はこの修正検証には含めていない。

## 2026-09-21 修正後の全体再レビュー

**判定: BLOCK。P2が2件。** 前回F1–F8の失敗条件は修正されている。F1については、過去の09:10欠損時に寄付→引けを使う既存仕様を不具合と誤認していたため、下記のとおり指摘を撤回する。

### F1再確認: 指摘撤回

過去の09:10値が欠損する場合に`jp_oc_*`（寄付→引け）へフォールバックする挙動は既存仕様である。構造改善計画はh=1の現行target計算と従来の欠損処理を同値に維持すると定め、S2a実行記録も欠損フォールバックを受入条件としてPASSにしている。`target_returns.py`の公開契約と回帰テストも同じ挙動を明示している。

更新後のローカルcacheをread-onlyで確認した結果、欠損セルが`jp_oc`へ置換されることも確認できたが、これは仕様どおりの結果であり不具合の証拠ではない。当日必須行だけをstrict検査する今回の修正は、過去欠損の互換フォールバックと当日入力の必須性を両立している。再現は[probe_910_after_fix.py](probe_910_after_fix.py)。したがってF1は修正済みと判定する。

### F9 [P2] Hosted CIと同じRuff範囲が失敗する

`tests/unit/test_s2a_input_boundaries.py`のimport順1件（I001）で、`.github/workflows/ci.yml`のRuff stepと同じコマンドが失敗する。全pytestのPASSだけでは現在のCIを通過しない。

### F10 [P2] 反復wheelビルドが削除済みmoduleを同梱する

現作業ツリーから作成したwheelに、sourceでは削除済みの`leadlag/models/v2/distribution_resolver.py`が残った。無視対象の`build/lib`にある旧出力をsetuptoolsが再利用している。`scripts/ci/verify_wheel.py`はresearch packageの有無しか検査しないため、このwheelをPASSとした。wheel自体のimport/CLI/ML smokeはPASSしているが、削除した旧経路を配布物から除いた保証にはならない。ビルド前の隔離・clean化と、許可manifestまたは削除済みmoduleの否定検査が必要である。Hosted CIのfresh runnerでは通常再現しないが、ローカル反復ビルドとrelease artifactの再現性が不足する。

### 検証

- 全`tests/`: **836 passed / 17 warnings / 105.30秒**。
- production mypy: 147 files PASS。import-linter: 7 contracts PASS。compileall、文書84リンク、lockfile、batch構文、`git diff --check`: PASS。
- Ruff: **FAIL 1件**（F9）。
- wheel: build・research除外検査・隔離smokeはPASSしたが、削除済みmodule混入を確認（F10）。

R1/R2の全生成入口・available_at証跡、R4実口座照合、R5本番期間artifact昇格、R6 Hosted CIも既存記録どおり未完了であり、S0–S8全体完了・本番昇格可とは判定しない。

## 2026-09-21 残件修正

F9/F10を修正した。

- `tests/unit/test_s2a_input_boundaries.py`のimport順を修正し、CIと同じRuff範囲をPASSさせた。
- `scripts/ci/build_wheel_clean.py`を追加し、既存の`build/`と`src/leadlag.egg-info/`をビルド中だけ隔離・復元するようにした。CIのwheel生成もこの入口へ変更した。
- `scripts/ci/verify_wheel.py`はsource配下のPython module manifestとwheelを照合し、削除済みmoduleの混入とproduction moduleの欠落を検出する。
- clean wheelは152 filesで検証・隔離CLI/ML smokeをPASSし、旧wheelに残っていた`distribution_resolver.py`は検出・排除できた。

修正後のRuff、mypy（147 files）、import-linter（7 contracts）、compileall、文書84リンク、全`tests/` **836 passed / 17 warnings**、`git diff --check`はPASSした。ローカル実装上のF9/F10は完了したが、R1/R2/R4/R5/R6の外部受入条件は残るため、S0–S8全体完了・本番昇格可とはまだ判定しない。

## 2026-09-21 実装追加（R1/R2/R4/R5）

残っていたローカル実装境界を閉じた。

- gap生成のh=1/3/5は`build_gap_historical_inputs()`でrun所有の`HistoricalInputs`を作り、各日09:10の`calculation_frame()`を検証する。モデルへ渡す残差行列は、その日15:30より後のJP close派生ラベルをマスクする。実現targetは評価配列として分離し、bundleへ`calculation_as_of`、`label_available_at`、履歴fingerprintを保存する。
- `HistoricalInputs`へ共通および日付別の`observed_at`を追加した。backtest/VaR/live bridgeがopen_910、macro、ADR、rank-reversal、PIT履歴の取得時刻をrun契約へ渡す。live bridgeのmacro/ADRは当日より後の行を保持しない。
- `reconcile --account-snapshot`を追加した。brokerのwallet/positionsを発注・取消・再送なしでJSONへ保存できる。既知order IDのfill照合は従来どおり`--run-id`で行う。
- ML overlay学習は開始日を2015-01-05以降へ固定し、明示したopen_910入力とADR/data fingerprintをartifact metadataへ記録する。production configのCURRENTを自動更新する操作は追加していない。

検証結果: 全`tests/` **843 passed / 17 warnings / 106.1秒**（watchdog 360秒）、Ruff、mypy（147 files）、compileall、import-linter（7 kept/0 broken）、文書84リンク、`git diff --check`、`uv lock --check`はPASS。wheel再ビルドは今回の環境でPyPI DNS解決ができず未実行（既存clean wheel検証は前節の証跡を維持）。

R1/R2のfresh外部データavailable_at証跡、R4の実口座wallet/positions/fills結果、R5の2015年以降artifact生成・OOS評価・production昇格、R6のHosted CI URL/job結果は外部実行が必要なため未完了。S0–S8全体完了・本番昇格可とは判定しない。
