# 構造改善 S0–S8 全体レビュー

レビュー日: 2026-09-21〜22。対象は現在の作業ツリー。2026-09-22にP2の修正と回帰テストを追加した。

## 結論

**判定: BLOCK — P1が1件残る。P2は修正済みだが、S0–S8全体完了・本番リリース可とは判定しない。**

- **P1:** バックテストとML学習が、利用可能な明示9:10入力を判断用の価格・ギャップへ反映していない。本番と異なるスコア・ウェイトを生成する。9:10欠損時の寄付代替はこの指摘に含めない。
- **P2（修正済み）:** VaR履歴の入力指紋計算を全体タイムアウトの`timed_call`へ含めた。

全テストは **853 passed / 17 warnings**。CI対象の静的検査、依存境界、lock、文書参照、fresh wheel・隔離推論も通過した。新しい最小再現では、既存テストが検証していない入口間の差と期限漏れを確認し、P2修正後の回帰も通過した。

共通factory、入力の所有権、分布取得policy、注文台帳と復旧、損益・MLの責務分離は実装されている。一方、現行の[計画](../20260915_structural_improvement/plan.md)、[ADR](../../docs/decisions/2026-09-15-structural-improvement-boundaries.md)、[最新対応報告](../20260922_structural_completion/report.md)も全体を部分完了としている。P2修正後も、P1と既存の実データ・運用受入条件は残る。

## 1. P1 — 9:10入力がある場合もBT・学習の判断入力が寄付価格のままになる

**主箇所:** [backtester.py](../../src/leadlag/execution/backtester.py) 484–490行。関連: [pit_lake.py](../../src/leadlag/data/pit_lake.py) 267–299行、[ml_overlay_training.py](../../src/research/experiments/ml_overlay_training.py) 132–138行・182行。

### 発生条件と原因

**仕様の確認（2026-09-22追記）:** 9:10価格が欠損する場合の寄付価格による代替は仕様であり、代替自体を不具合とは判定しない。実際に[リターン計算](../../src/leadlag/core/target_returns.py)のh=1では非有限の9:10リターンに対して`jp_oc`を保持し、h>1では開始日の9:10価格が得られなければ寄付価格を使う。今回の再現は当日の17銘柄すべてに有限の9:10リターンを明示しており、欠損時の代替条件には該当しない。

発生条件は、利用可能な9:10入力があり、日次データの寄付価格と9:10価格が異なる日。BTは`open_910_returns`を`HistoricalInputs`に保持するが、判断入力には`lake.get_snapshot(decision_as_of)`の値をそのまま使う。

このsnapshotの`current_prices`は`jp_open_trade_*`、`jp_gap_returns`は日次の`jp_gap_*`である。`preprocessor.py` 385行の価格は日次Openであり、9:10への変換は行われない。`as_of`を09:10にするだけでは値は更新されない。

本番bridgeは[現在価格からgapを再計算](../../src/leadlag/execution/v2_bridge.py)する（370–392行）。そのため、同じ日次履歴と9:10観測を用意しても、BTと本番の入力が一致しない。BTの実現targetだけは9:10入力を使用するため、判断と評価の価格基準も揃わない。

ML collectorにも同じsnapshot生成があり、特徴量の`gap`をさらに日次行から直接取得する。したがって、BTと学習の一致だけを確認しても、本番との不一致を見落とす。

### 再現と実害

[再現コード](probe_boundaries.py)は固定の2026-08-14回帰データを使い、当日の寄付→9:10リターンを17銘柄で−1%〜+1%に設定した。実際のBT判断入口を経由して入力・出力を捕捉し、同じ入力契約の価格・gapだけを次の値へ置き換えた実モデル出力と比較した。

```text
p_0910 = jp_open_trade × (1 + open_910_returns)
gap_0910 = p_0910 / jp_close_sig − 1
```

| 比較 | 9:10リターンが全て0の対照 | 当日−1%〜+1% |
|---|---:|---:|
| gapの最大絶対差 | 9.89e-17 | 0.0100203 |
| μの最大絶対差 | 2.22e-16 | 0.00706954 |
| scoreの最大絶対差 | 6.55e-14 | 1.76108 |
| モデルweightの最大絶対差 | 9.83e-15 | **0.182209** |

両経路とも終端flatではなく、数値監査はPASSED。モデルgrossは約2、netは丸め誤差内の0であり、市場中立制約の監査ではこの入力不一致を検出できない。`side_leverage=1.5`を適用した実効grossは約3で、設定の`risk.max_gross_exposure=3.0`内に収まる。

実際のML collectorでも17行を取得し、scoreはBTと一致したが、9:10価格を反映した判断とは最大1.76108異なった。学習特徴量gapの差も最大0.0100203だった。

**検証の範囲:** 継承解決した本番設定から、検証中だけ`ml_overlay_enabled=False`とした。BLPX・h=1/3/5のモデル計算は実装を使用し、macro/ADR/rank入力の欠損条件は両者で揃えた。PIT履歴と当日の価格変化は制御した検証入力であり、実取引の損失額や性能評価、本番ML有効時の完全再現を示す数値ではない。モデル・artifactの学習や公開は行っていない。

証拠: [数値結果](probe_results.json)、[成功した再現ログ](boundary_probes_verified.log)。

### 既存テストの不足

- `test_backtest_builds_run_owned_typed_inputs`は9:10の履歴frameが渡ることを検証するが、判断用`current_prices`・`jp_gap_returns`の値は検証していない。モデル出力もstubである。
- `test_backtest_target_uses_run_owned_open_910_returns`は実現targetを検証するが、判断入力との一致は対象外。
- ML collectorの型付き入力テストは9:10リターンが0のため、この差が現れない。

### 修正案と完了条件

日次Open・前日終値・当日の明示9:10観測から判断snapshotを作る処理を入力adapterへ集約し、BTと学習で使う。価格とgapを同時に更新し、学習のgap関連特徴量も同じsnapshotを参照する。現在の`PITDataLake.build_decision_inputs(current_prices=...)`が行うgap再計算を活用できる。学習特徴量の二重実装も、本番と共通の特徴量構築へ寄せる。

回帰は非ゼロの寄付→9:10変化を必須にし、次を確認する。

1. 本番相当snapshot・BT・学習の価格、gap、score、特徴量が一致する。
2. h=1/3/5のcacheとon-demandが同じ観測を使用する。正しい9:10 bundleが異なるgap identityとして拒否されない。
3. BTの判断価格と実現targetが同じ9:10観測を使う。
4. 9:10欠損時の寄付代替を検証し、使用価格の出所を区別する。来歴不正・監査失敗による拒否／flatは別ケースとして検証する。欠損を一律にflatとする仕様変更は今回の修正要件に含めない。

## 2. P2 — VaR入力指紋が全体期限の外で計算される（修正済み）

**2026-09-22 修正結果:** [検討記録](../../docs/decisions/2026-09-22-var-input-fingerprint-deadline.md)では、固定入力の指紋計算は約4〜26ms、本番経路の期限は既定300秒と確認した。緊急性は低いが、期限契約を揃える小さな修正として対応した。メモリ上の候補検証に加え、実装後の対象回帰・全テストを通過した。

**箇所:** [var_history.py](../../src/leadlag/execution/var_history.py) **272–275行**。

### 修正内容

`historical_input_snapshot.fingerprint`を既存の残時間付き`timed_call`へ含めた。期限超過時は既存の準備失敗処理へ入り、空のrisk履歴を返して後続のcache読込・BT・cache書込を開始しない。通常のcache key、返却系列、snapshot解放、例外伝播は維持する。

**修正前の挙動:** `_build_var_historical_inputs()`は残時間付きの`timed_call`で呼ぶ一方、直後の`historical_input_snapshot.fingerprint`は同期で直接計算していた。[fingerprint本体](../../src/leadlag/domain/inputs.py) 540–562行は、日次履歴・PIT配列・9:10・macro・ADR・rankを再度ハッシュする。保存済み文字列の参照ではない。

**修正前の影響:** 入力が大きい場合や、先行処理で残時間が少なくなった場合、この処理の終了まで期限を超えて待った。修正後は同じ絶対期限で打ち切る。

### 再現

実際の`HistoricalInputs.fingerprint`に有限の0.5秒遅延を注入し、VaR全体期限を0.1秒にした（修正前後の挙動比較）。データ読込とsnapshot準備は固定入力へ置換し、実ローカルSQLite cacheを使用した。

- 修正前の呼出し終了まで **約0.510秒**。修正後は約0.107秒で返った。指紋計算はどちらも1回実行された。
- 次のcode hash計算には到達せず、終了時には空のrisk履歴を返す。
- brokerやBT workerは呼んでいない。

空履歴でリスク判定を止める性質は保たれる。修正前は期限内に呼出し元へ制御を返す契約を満たさなかった。外側batchの停止期限は独立した最後の保護であり、修正前のAPI期限漏れを解消するものではなかった。

証拠: [再現コードのdeadline_probe](probe_boundaries.py)、[数値結果のvar_deadline](probe_results.json)。実際に大容量入力で0.5秒を要したという測定ではなく、遅い準備処理への期限適用を検証した障害注入である。

**既存テストの不足（修正前）:** `test_stage_abc_followup_fixes.py`はcache読込・書込、SQLite snapshot待ち、workerの期限を検証するが、入力指紋の遅延ケースがなかった。今回、同テストへ遅延回帰を追加した。

**修正結果:** propertyの評価を同じ残時間の`timed_call`に含めた。回帰では指紋計算を遅延させ、設定期限内に空履歴を返し、後続workerを起動しないことを確認した。

## 3. 構造全体の確認

| 範囲 | 今回確認した内容と評価 |
|---|---|
| S0 比較基準 | 固定入力・feature matrix・過去レビューを照合。全有効機能の実データによる入口間比較は未充足。ゼロ変化や同じ日次gapを用いる比較だけではP1を検出できない。 |
| S1 設定・構築 | 継承解決したAppConfig、共通factory、runner/BT/VaRのoverlay選択を確認。MH 1/3/5、重み0.8/0.1/0.1、監査失敗fallbackを維持。既存artifactの構築拒否は後述。 |
| S2 入力・時点 | 所有コピー、readonly配列、JST日付、観測時刻検証、当日未確定labelの計算view、型付き経路の暗黙I/O制限を確認。P1により入口の値の統一は未完了。 |
| S3 分布・cache | cache→許可されたon-demand→flat、horizon・入力版・gap identityの照合、snapshot/worker分離を確認。P1は分布入力に影響する。P2のVaR期限は修正済み。 |
| S4 注文契約 | 計画・観測・結果型、共通poll、accepted/filled/partial/failedの区別をコード・既存回帰で確認。今回追加の確定不具合なし。 |
| S5 永続化・復旧 | 送信前のplan保存、poll前の観測保存、照合checkpoint後の完了、日付をまたぐ未解決runの遮断、leaseと子プロセス停止を確認。実口座・実schedulerの受入は未確認。 |
| S6 損益 | 純粋なBT損益計算、共通FillとFIFO在庫、取消後の部分約定、未観測feeの扱い、journal/positions照合を確認。実約定と費用の突合は未確認。 |
| S7 ML・配布 | 学習分離、pickle class path、artifact来歴検証、timezone/NaN target修正と回帰を確認。P1の学習側対応、本番artifact再生成・OOSが残る。 |
| S8 CI・文書 | ローカルCI相当の検査、fresh wheelの147 Pythonファイルと現sourceのバイト一致、research非同梱、隔離CLI・合成artifact推論を確認。Hosted CIの実行証跡は未取得。 |

## 4. 実行した検証

| 検証 | 結果・証拠 |
|---|---|
| `pytest tests/ -q --tb=short` | **853 passed / 17 warnings / 105.75秒**。[ログ](pytest_after_p2_final.log)、[外側600秒watchdog](pytest_after_p2_final.json) |
| CI対象Ruff＋学習入口 | PASS。[ログ](ruff.log) |
| `mypy --config-file pyproject.toml src/leadlag` | 147ファイル、PASS。[ログ](mypy.log) |
| `compileall src/leadlag tests tools scripts src/research` | PASS。[ログ](compileall.log) |
| import-linter | 7契約すべて維持。[ログ](imports.log) |
| architecture/CI/scheduler/ADR/planの文書参照 | PASS。[ログ](docs.log) |
| `uv lock --check --offline` | 書込可能な一時cacheを指定してPASS。[ログ](uv_lock_local.log) |
| fresh wheel build・隔離install・CLI・artifact推論 | PASS。古いbuild出力の排除・復元も検証。[コード](verify_distribution.py)、[ログ](wheel_fresh.log) |
| 新規の境界再現 | P1を再現し、P2修正後の遅延回帰を確認。[コード](probe_boundaries.py)、[P1ログ](boundary_probes_verified.log)、[P1結果](probe_results.json)、[P2対象テスト](p2_file.log) |
| P2対象検証 | 対象11件、Ruff、compileall、mypy、import-linterがPASS。[対象ログ](p2_file.log)、[Ruff](p2_ruff.log)、[mypy](p2_mypy.log)、[import](p2_imports.log) |

長時間検証は[review_tools.py](review_tools.py)のプロセスグループwatchdogで上限を設定した。RuffはCIで維持対象とする範囲であり、過去の研究コード全体がlint済みという意味ではない。

wheelの最初の試行は`.venv`に`build`がなく失敗した。その後、一時sourceと既存cacheのbuild依存を使い、プロジェクト環境に依存を追加せず完了した。uvの最初の試行も既定cacheへの権限制限で失敗し、一時cacheで再検証した。これらの初回失敗ログは残している。再現スクリプトも初回の本番artifact拒否、拡張時の引数誤りを修正してから上記成功ログを取得した。

## 5. 既存の受入残件

新規のコード指摘とは分けて、次を引き継ぐ。

1. **本番MLの構築は現在も停止する。** 本番設定は`models/ml_order_overlay/phase2_8`を参照し、共通factoryは`Legacy root overlay artifact is not accepted`で終了した（[今回の確認結果](probe_results.json)）。これはS0の基準にも記載済みの拒否で、今回の新規回帰ではない。検証用ML無効化を本番復旧扱いにしない。verified artifact、必要期間の固定入力、同一artifactのwalk-forward OOSと本番/BT照合が必要。
2. **providerの実available_atと全入口比較。** 最新のgap再生成証跡は`session_boundary_contract`であり、実取得時刻の証明ではない。gap/live/BT/VaRの全入力の固定と照合は未完了。
3. **実口座・運用。** wallet/positions/fills、partial-fill recovery journal、実約定費用の照合は今回未実施。schedulerについても実登録の変更・再確認は行っていない。前回保存証跡には`update-market-data`の旧workspace参照・終了127が残るため、テンプレート検証だけで運用完了にはできない。
4. **Hosted CI。** ローカル検査は成功したが、GitHub Actionsのrun URL・全job成功結果は今回取得していない。

今回のレビューは現行構造と関連経路を読み、全`tests/`と境界再現を実行したものである。新規の市場データ取得、実口座操作、scheduler登録、本番artifact昇格は行っていない。

## 6. 推奨する修正順

**P1の入力adapterと学習特徴量を先に揃える。** P2の指紋期限修正は回帰済みなので、既存の実データ・artifact・運用受入へ進む前にP1を修正する。入力値が揃う前のartifact再学習は、今回の不一致を引き継ぐため受入完了にはできない。

対象HEAD: `cf97bfaf4577a20c95ce931afd10ac72a6f08285`。HEADとの差分には以前の修正も混在するため、今回のレビュー対象は[開始時manifest](before_manifest.json)で固定した作業ツリーである。今回のP2修正は[var_history.py](../../src/leadlag/execution/var_history.py)と回帰テストへ適用した。
