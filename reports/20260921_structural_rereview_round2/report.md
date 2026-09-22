# S0–S8 全体再レビュー（修正後・第2回）

実施日: 2026-09-21。対象: 現行作業ツリー、[計画](../20260915_structural_improvement/plan.md)、[修正報告](../20260917_structural_completion_review/report.md)、[前回レビューと修正追補](../20260921_structural_rereview/report.md)。

## 結論

**BLOCK。新規指摘はP1が1件、P2が1件。** 前回F1–F10の修正は確認できた。全843テストとCI相当のローカル検査は通るが、学習と推論の計算経路、および補助入力の観測時刻検査に未完了がある。S0–S8全体完了・本番昇格可とは判定しない。

レビューのみを実施し、source・設定・既存テストは変更していない。このディレクトリに再現スクリプト・検証ログ・wheel・報告を追加した。brokerへの照会や発注、scheduler変更、本番artifactの学習・昇格は実行していない。

## 指摘

### N1 [P1] ML学習が本番のmulti-horizon経路を通らず、特徴量と教師ラベルがずれる

- **箇所:** [ml_overlay_training.py](../../src/research/experiments/ml_overlay_training.py) 71–89行、特に74行。分岐条件は[decision_engine.py](../../src/leadlag/models/v2/decision_engine.py) 313–320行。
- **条件:** 本番設定の`mh_blend_enabled=true`、h=1/3/5、重み0.8/0.1/0.1でML overlayを学習する。標準学習CLIは解決済み本番設定をそのまま渡す。
- **問題:** 学習は`ProductionV2Model(run_cfg)`をBLPXなしで作り、`decide()`へ日付とcacheパスしか渡さない。multi-horizon分岐にはBLPX・履歴・価格が必要なため、h=1だけのcache経路になる。本番推論はh=1/3/5をブレンドして標準化したscoreをMLへ渡す。学習側の`score`・交互作用特徴量が本番と異なり、`side * realized - cost`のsideも変わる。open_910やADRのhashを追加した今回の修正では、この経路差は閉じない。
- **再現:** 同一の来歴付きh=1/3/5 bundleと本番のhorizon/weight設定で比較。macroは無効にして差を分離し、rank入力は両経路とも欠損時の既定処理とした。学習とtyped推論のscore最大差は**1.5262197386**、17銘柄中6銘柄で符号（ゼロを含む）が不一致。分布欠損・監査失敗によるflatではない。BLPXにはcache分岐を有効にするmarkerを渡し、全分布は保存済みbundleから解決した。合成入力による契約差の再現であり、実モデルの成績比較ではない。
- **不足テスト:** S7の既存テストは研究側への移設、artifact保存、開始日制約等を検証しているが、実際の学習行生成と本番推論の特徴量一致を検証していない。
- **修正案:** 学習も同じfactory・run所有入力・分布解決経路から、ML適用直前の特徴量を取得する。ML無効条件を維持し、h=1/3/5のscore、特徴量、教師ラベルのsideまで比較する。設定hashに記録した経路を実際に実行し、gap/PIT/rank/macroの入力版も固定する。
- **証拠:** [再現スクリプト](probe_training_parity.py)、[実行ログ](probe_training_parity.log)。

### N2 [P2] HistoricalInputsの未来の観測時刻が検査されず、補助入力が予測へ入る

- **箇所:** [decision_engine.py](../../src/leadlag/models/v2/decision_engine.py) 288–295行、[inputs.py](../../src/leadlag/domain/inputs.py) の`HistoricalInputs.observed_at_for()`と`DecisionInputs.__post_init__()`。
- **条件:** 当日のrank等の補助入力に、判断時刻より後の`historical_observed_at_by_date`が明示される。
- **問題:** 追加された観測時刻は保存・fingerprint化されるが、モデルへ補助入力を渡す際には参照されない。`KnownMarketInputs`には未来時刻の拒否がある一方、`HistoricalInputs`の補助入力には同じ制約がない。主フレームの`calculation_frame(as_of)`による当日targetマスクでは、別フィールドのrank/ADR/macro等は保護されない。
- **再現:** `as_of=2026-09-11 09:10`に対してrankの観測時刻を同日**09:20**と明示したtyped入力を受け付け、監査失敗なしにscoreと最終ウェイトが変化した。rankなしとの最大差はscore **0.0826814397**、`w_final` **0.0167522218**。実運用データのリークを確認したという意味ではなく、明示された未来入力を契約が拒否しない再現である。
- **不足テスト:** 既存回帰は`observed_at_for()`の日付別取得と`KnownMarketInputs`の拒否を確認しているが、補助入力の時刻を意思決定境界で検査するケースがない。
- **修正案:** 選択した日付の補助入力の観測時刻を`known.as_of`と照合し、未来と判明している入力を拒否または所定の欠損処理へ送る。共通・日付別metadataの両方に適用する。原本の取得時刻が不明なケースは別に扱い、固定の09:00/09:10を実測時刻の証拠とはしない。
- **証拠:** [同じ再現スクリプト後半](probe_training_parity.py)、[実行ログ末尾](probe_training_parity.log)。

## 前回指摘の確認

| 前回ID | 今回確認した修正 |
|---|---|
| F1 | strictな09:10検査を当日の必須行へ限定。過去欠損時の既存target代替と当日欠損拒否を区別している。 |
| F2 | 準備済みrun再開と送信開始への遷移で、別runの未解決状態を同一transaction内で検査する。 |
| F3 | snapshotのgap/beta/TOPIXを`gap_inputs_version`へ含め、horizon別に生成・照合する。live snapshot付きで旧metadataを受理しない。 |
| F4 | 通常終了・復旧で、正の約定数量に対応する有限の価格と明示された費用を検査する。 |
| F5・F6 | CANCELLEDの確認済み部分約定を取り込み、`observed_fill`の費用欠損・nullをゼロ円にしない。 |
| F7 | ML artifactの学習期間を共通JST正規化へ変更。 |
| F8・F9 | CIと同じmypyおよびRuff範囲がPASS。 |
| F10 | clean wrapperとsource manifest照合を確認。一時的なsourceコピーに削除済みmoduleと古いegg-infoを置き、wrapperによる排除・元出力の復元・wheel検証・隔離smokeを実行してPASS。 |

前回の指摘と同じ条件の修正確認であり、N1/N2や下記の受入残件まで解消したという意味ではない。

## S0–S8の確認範囲

| 領域 | 確認内容・残る限界 |
|---|---|
| S0 比較基準 | 固定baseline、過去比較の範囲、現行回帰を照合。ML学習との一致にはN1が残る。実データ全機能・全入口の比較を今回新たに完了したとは扱わない。 |
| S1 設定・構築 | production.yaml→base.yamlの継承、factory、live/BT/VaRの構築経路を確認。学習側にN1の経路差が残る。 |
| S2 入力・PIT | run所有コピー、JST、target可視性、09:10必須行、観測時刻追加を確認。補助入力時刻にはN2が残る。 |
| S3 分布・cache・VaR | cache→on-demand→flat、snapshot来歴、h=1/3/5、gap生成のJP未来行マスク、VaR入力固定とworker境界を確認。実データavailable_at証拠は未確認。 |
| S4 型・結果 | PortfolioDecision/ExecutionPlan/約定結果の呼出境界と会計へ渡す確認項目を確認。 |
| S5 執行・復旧 | 再開/送信開始の未解決run検査、lease/期限、read-only account snapshot追加、関連回帰を確認。実scheduler・実broker照合は未実施。 |
| S6 会計 | 価格/費用未確認、部分約定後取消、日次帳票との接続に関する前回指摘を再確認。実口座残差は未検証。 |
| S7 ML・配布 | artifact期間検査、学習入力のhash、研究分離、実wheel生成・隔離推論を確認。N1。本番artifact再生成・OOS・昇格は未実施。 |
| S8 CI・文書 | 下記ローカル検査を実行。roadmapの部分完了表記、対象ADRと計画を照合。Hosted CIの成功URLは未確認。 |

## 実行した検証

| 検証 | 結果・証拠 |
|---|---|
| 全`tests/`（外側watchdog 360秒） | **843 passed / 17 warnings / 105.41秒**。[tests.log](tests.log) |
| 追加の契約差確認（watchdog 60秒） | N1/N2を再現。[probe_training_parity.log](probe_training_parity.log) |
| Ruff | CI対象production/tests/tools + ML学習module、PASS。[ruff.log](ruff.log) |
| mypy | 147 source files、PASS。[mypy.log](mypy.log) |
| import-linter | 7 kept / 0 broken。[imports.log](imports.log) |
| compileall | src/leadlag・tests・tools・scripts・src/research、PASS。[compile.log](compile.log) |
| 文書リンク | architecture/CI/scheduler/ADR/planの84リンク、PASS。[docs.log](docs.log) |
| lockfile | `uv lock --check --offline`、PASS。[lock.log](lock.log) |
| wheel | fresh treeのoffline build後、さらにCIのclean wrapperに古いbuild出力を与えて再検証。152 files、manifest一致・research非同梱・CLI/合成ML artifact推論PASS。[検証スクリプト](verify_distribution.py)、[wheel.log](wheel.log)、[build](wheel_build.log)、[smoke](wheel_smoke.log) |
| batch構文 / diff空白検査 | 4 batchの`bash -n`、`git diff --check`、PASS。 |

wheel検証では既存uv cache内のbuild 1.6.1を一時プロセスから再利用し、ネットワーク取得やプロジェクト環境への追加インストールを避けた。wheelの合成ML smokeは実際の本番学習・性能・本番/BT一致の代替ではない。

## 受入の残件

N1/N2はローカル実装と回帰テストで修正できる。修正報告の「残るのは外部実行の証跡」という整理には、この2点を追加する必要がある。

その後も、外部データの真のavailable_at、実scheduler/口座の照合、本番期間でのverified ML artifact生成・OOS評価・同一入力での本番/BT比較、Hosted CI結果は既存の受入残件として残る。収益性能・実効exposure・口座残高について、今回新たな実測結果は出していない。
