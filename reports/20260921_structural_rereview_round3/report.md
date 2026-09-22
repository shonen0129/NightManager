# S0–S8 全体再レビュー（修正後・第3回）

実施日: 2026-09-21。対象は現行作業ツリー、[構造改善計画](../20260915_structural_improvement/plan.md)、[実装確認報告](../20260917_structural_completion_review/report.md)、[第2回レビュー](../20260921_structural_rereview_round2/report.md)と[その修正報告](../20260921_structural_rereview_round2/fix_report.md)。

## 結論

**全体完了の受入判定はBLOCK。残る指摘はP2が2件。** 前回N1のmulti-horizon学習経路は修正を確認した。N2の未来観測時刻チェックも単独の共通・日付別metadataでは機能するが、両者の併用時に検査が漏れる。

全845テスト、CI相当の静的検査、wheelの実ビルド・隔離実行はPASS。これらで未収録の境界条件を下記probeで再現した。レビューのみを行い、実装・設定・既存テストは変更していない。今回の追加物はこのディレクトリの報告・再現コード・検証成果物である。

## 指摘

### M1 [P2] 日付別metadataが共通metadataを丸ごと置き換え、未来の補助入力を受理する

- **箇所:** [inputs.py](../../src/leadlag/domain/inputs.py) 419–439行、特に422–425行の`observed_at_for()`。
- **発生条件:** 共通`observed_at`にrankの観測時刻09:20があり、同じ判断日の`observed_at_by_date`に別の特徴量（macro）の09:00がある。判断時刻は09:10。二つのmappingに重複キーはない。
- **問題・実害:** 日付別mappingが見つかると、共通mappingのrank時刻を捨てた値だけを`validate_observed_at()`へ渡す。一方、rankの値自体はモデルへ渡るため、未来と明示されたrankを拒否せず意思決定に使用する。前回N2が組合せ条件で残っている。
- **再現結果:** 共通rank時刻のみならValueError。無関係な日付別macro時刻を足すと受理され、監査失敗によるflatにならず、rankなしに比べscore最大差 **0.0826814397**、`w_final`最大差 **0.0167522218**。合成入力で契約違反を再現したもので、実運用データでのリークを観測したという意味ではない。
- **修正案:** 共通mappingを基礎にして、日付別mappingで明示されたキーだけを上書きする等、別特徴量の共通時刻を失わない解決規則にする。解決後の全時刻を判断時刻と比較する。
- **不足テスト:** 共通のみ・日付別のみの拒否に加え、異なるキーを持つ両mappingの併用と、同一キーの上書き規則を検証する。
- **証拠:** [probe_boundaries.py](probe_boundaries.py)、[実行ログのM1](probe_boundaries.log)。

### M2 [P2] UTC付きADR入力では、推論が受理できても学習が日時比較で停止する

- **箇所:** [ml_overlay_training.py](../../src/research/experiments/ml_overlay_training.py) 282–284行。
- **発生条件:** ADR artifactのindexがtimezone-aware（例: JSTの取引日00:00をUTCの前日15:00で保存）で、学習CLIが通常の`YYYY-MM-DD`を`train_end`に指定する。
- **問題・実害:** `load_adr_features()`は`trade_date`なしでは元indexを返す。学習はJST正規化前にnaiveな`train_end_ts`と比較するため、`Invalid comparison between dtype=datetime64[us, UTC] and Timestamp`で停止する。後段の`HistoricalInputs`へ到達しない。同じartifactは推論の`validate_adr_features()`では正常にJST日付へ正規化されるため、学習・推論の入力契約が一致しない。
- **再現結果:** 一時pickleを実際のADR loaderで読んだUTC版はTypeErrorでcollector未到達。同じ値・同じJST日付へ正規化した版はcollectorまで到達した。probeでは学習前処理を分離し、09:10入力を固定、collectorを空結果に置換している。LightGBMの学習・artifact公開は行っていない。
- **修正案:** ADRのindexと学習期間を共通のJST規則で正規化してから期間抽出し、その同じframeを履歴入力・特徴量生成へ渡す。重複するJST日付などの既存拒否規則も保持する。
- **不足テスト:** ADR単体loader/推論のtimezone回帰だけでなく、実学習前処理にnaive/JST/UTCの同値入力を渡し、選択日付と特徴量が一致することを確認する。
- **証拠:** [probe_boundaries.py](probe_boundaries.py)、[実行ログのM2](probe_boundaries.log)。

## 前回修正とS0–S8の確認状況

前々回F1–F10について、当日09:10必須行、未解決run排他、snapshotのgap/beta/TOPIX来歴、約定価格・費用の確認、取消後の部分約定、費用nullの拒否、ML期間のJST正規化、静的検査、古いbuild出力の排除に関する修正と回帰を再確認した。同じ条件での再発は確認していない。

| 領域 | 今回確認した範囲・残件 |
|---|---|
| S0 比較基準 | baseline・既存比較の範囲と現行回帰を照合。新たなprobeでは学習/推論のscore差0・符号差0、通常17特徴量とticker交互作用あり63特徴量の厳密一致を確認。ただしmacro無効・ADRなし・保存済みh=1/3/5の合成ケースであり、本番全機能・全入口の実データ同値性は未確認。 |
| S1 設定・構築 | 継承解決後の本番設定、共通BLPX factory、live/BT/VaRと学習の構築経路を確認。前回N1のh=1のみになる分岐差は解消。 |
| S2 入力・PIT | run所有コピー、JST、targetのas-ofマスク、補助入力時刻検査を確認。M1が残る。取得元の真のavailable_atの証拠は別途必要。 |
| S3 分布・cache・VaR | 当日cache→許可されたon-demand→flat、h=1/3/5、snapshotを含む来歴、gap生成のtarget遮断、VaRの入力固定・期限の境界を確認。実データのStep 1/on-demand比較を今回追加完了したとは扱わない。 |
| S4 型・結果 | DecisionInputs、PortfolioDecision、ExecutionPlanと約定・照合結果の接続を確認。今回の範囲で新規指摘なし。 |
| S5 執行・復旧 | prepared再開と送信開始の別run検査、transaction、完了前の照合、read-only account snapshot入口、batch構文と関連回帰を確認。実scheduler登録・実brokerへの照会は未実施。 |
| S6 会計 | 正の約定数量に対する価格・費用確認、取消後の確認済み部分約定、費用欠損時の扱いを確認。実口座の建玉・費用・残高との突合は未実施。 |
| S7 ML・配布 | typed入力からのMH学習、特徴量一致、artifact期間・hash、研究分離、wheelの隔離推論を確認。M2が残る。verified本番artifact再生成・OOS・同一入力での本番/BT照合は未実施。 |
| S8 CI・文書 | 下記ローカル検証はPASS。roadmap・対象ADR・計画は全体未完了として扱う。Hosted CI結果は未確認。計画冒頭の808件と最新修正報告の845件が混在するため、完了履歴を更新する際は最新報告への参照も揃える。 |

## 実行した検証

| 検証 | 結果・証拠 |
|---|---|
| 全`tests/`、watchdog 360秒 | **845 passed / 17 warnings / 105.17秒**。[tests.log](tests.log) |
| 前回N1/N2の修正確認 | 更新済み第2回probeを再実行してPASS。[parity.log](parity.log) |
| 特徴量一致・追加境界probe、watchdog 60秒 | 正規化済み合成入力の一致を確認し、M1/M2を再現。[probe_boundaries.log](probe_boundaries.log) |
| Ruff | production/tests/toolsのCI対象とML学習module、PASS。[ruff.log](ruff.log) |
| mypy | 147 source files、PASS。[mypy.log](mypy.log) |
| import-linter | 7 kept / 0 broken。[imports.log](imports.log) |
| compileall | src/leadlag・tests・tools・scripts・src/research、PASS。[compile.log](compile.log) |
| 文書リンク | CI対象5文書、84リンクPASS。[docs.log](docs.log) |
| lockfile | `uv lock --check --offline`、PASS。[lock.log](lock.log) |
| wheel | sourceコピーに古いbuild/egg-infoを置き、clean wrapperの排除と復元、manifest照合、research非同梱、隔離install・CLI・合成ML推論を検証してPASS。[コード](verify_distribution.py)、[結果](wheel.log)、[build](wheel_build.log)、[smoke](wheel_smoke.log) |
| batch / diff | 4 batchの`bash -n`、`git diff --check`がPASS。[bash.log](bash.log)、[diff_check.log](diff_check.log) |

wheelは既存のuv cacheにあるbuild toolを一時プロセスから利用し、ネットワーク取得やプロジェクト環境への追加インストールは行っていない。

M1/M2はローカル実装・回帰テストで修正できる残件。それらの修正後も、実際の情報取得時刻、実scheduler/口座の照合、verified本番ML artifactとOOS・本番/BT比較、Hosted CIの実行証跡という従来の受入残件は別に残る。
