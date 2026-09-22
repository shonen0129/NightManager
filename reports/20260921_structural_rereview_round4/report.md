# S0–S8 全体再レビュー（修正後・第4回）

実施日: 2026-09-21。対象は現行作業ツリー、[構造改善計画](../20260915_structural_improvement/plan.md)、[実装確認報告](../20260917_structural_completion_review/report.md)、[前回レビュー](../20260921_structural_rereview_round3/report.md)と[修正報告](../20260921_structural_rereview_round3/fix_report.md)。確認したソース等569ファイルのhashと継承解決後の設定は[source_manifest.json](source_manifest.json)に保存した。

## 結論

**S0–S8全体の受入判定はBLOCK。新たに確認した残件はP2が2件で、いずれもML学習入口にある。** 前回M1（観測時刻mappingの併用）・M2（ADRのtimezone）は修正を確認した。

全848テスト、静的検査、wheelの実ビルド・隔離実行はPASS。以下の2件は既存テストに含まれない条件で再現した。今回新たに発見した問題であり、直近修正が導入原因だと断定するものではない。

レビューのみを行い、実装・設定・既存テストは変更していない。追加物はこのディレクトリの報告・再現コード・検証成果物である。

## 指摘（影響順）

### Q2 [P2] 欠損した実現targetを0に置き換えて学習へ渡す

- **箇所:** [ml_overlay_training.py](../../src/research/experiments/ml_overlay_training.py) 150–151、203–212、229–242行。特に232行のtargetに対する`_safe()`。
- **発生条件:** 学習期間のある日・銘柄で、09:10入力とV2予測は正常だが、実現リターン`jp_oc_*`がNaN。未確定の大引けや欠損した過去データを含めた場合に該当する。
- **問題・実害:** collectorはその行を除外せずNaNのtargetを保持する。回帰ではfit直前に`_safe()`が0へ変換し、分類ではNaNが負例の0になる。未観測の結果を観測済みの教師ラベルとして学習し、学習標本数・target分散・予測を歪める。
- **再現:** 17銘柄のうち1銘柄の当日`jp_oc`だけをNaNにした。実際のtarget計算はNaNを返すが、実際のV2 cache経路を通るcollectorは17行すべてを採用した。LightGBMのfit呼出に渡される当該targetは回帰で`0.0`、分類targetは`0`だった。
- **修正案:** targetの残差化・二値化の前に、実現値が有限でない日・銘柄を除外または明示的に拒否する。fit入口でも不正な教師ラベルを検証し、特徴量用の欠損補完をtargetへ適用しない。除外数を学習記録に残す。
- **不足テスト:** finite/NaN/Infを混ぜた教師ラベルについて、raw・residual・classification等の変換前後で未観測行がfitへ入らないこと。既存collector回帰は有限のゼロtargetのみを使っている。
- **証拠:** [probe_boundaries.py](probe_boundaries.py)、[実行ログのQ2](probe_boundaries.log)。実現target計算・V2予測・収集は実装を使用し、fit先のみ置換して渡された教師配列を確認した。モデル学習・artifact公開は行っていない。

### Q1 [P2] timezone付きdf_execでは学習期間の抽出前に停止する

- **箇所:** [ml_overlay_training.py](../../src/research/experiments/ml_overlay_training.py) 276–283、298–299行。
- **発生条件:** `df_exec.index`がtimezone-aware（JSTまたは同じ瞬間のUTC表現）で、通常の`YYYY-MM-DD`を学習期間として指定する。
- **問題・実害:** 学習期間はJST-naiveに正規化されるが、`df_exec`はそのまま。`build_open_910_returns()`も元indexを保持するため、283行でawareとnaiveの大小比較がTypeErrorになる。283行だけを直しても299行に同じ比較が残る。ADRの修正ではこの経路は解消されない。
- **再現:** 同じ25営業日をnaive/JST/UTCで表現し、同じ合成5分足から実際の`build_open_910_returns()`で入力を作った。naive版はcollectorまで到達したが、JST/UTC版は双方`Invalid comparison between dtype=datetime64[...] and Timestamp`でcollector未到達だった。
- **修正案:** 学習入口で所有コピーした`df_exec`を共通のJST日付規則へ正規化し、そこから09:10入力、market volatility、target、学習日付、履歴契約を構築する。JST日付への変換で生じる重複も既存入力契約に合わせて拒否する。
- **不足テスト:** ADRだけでなく`df_exec`自体のnaive/JST/UTC同値ケースで、選択される日付・教師ラベル・特徴量を比較する。追加済みADR回帰の`df_exec`はnaive固定。
- **証拠:** [probe_boundaries.py](probe_boundaries.py)、[実行ログのQ1](probe_boundaries.log)。5分足の取得元を合成frameへ置換し、collectorを空結果にして学習前処理の到達点を比較した。naive側の`No training samples collected`はこの置換による意図した終端である。

## 前回修正の確認

| 指摘 | 今回の結果 |
|---|---|
| M1: 日付別metadataにより別キーの共通時刻を失う | **解消。** 共通rankの09:20と日付別macroの09:00を併用しても、09:10の判断ではValueErrorになる。共通のみ・日付別のみの拒否も確認。 |
| M2: UTC付きADRで学習停止 | **解消。** 実pickle loader経由のUTC artifact、UTC frameの直接注入、JST-naive frameがいずれもcollectorへ到達し、渡されるADR frameが一致。 |
| N1: 学習がh=1だけになる | **再発なし。** h=1/3/5のcacheから、学習と本番のscore差0・符号差0。通常17特徴量、交互作用あり63特徴量が厳密一致。 |
| それ以前のF1–F10、N2 | 当日09:10検証、cache来歴、未解決run排他、約定価格/費用確認、取消後部分約定、費用null拒否、期間日付、静的検査、配布の関連コード・回帰を再確認。同じ条件での再発は確認していない。 |

特徴量の比較はmacro無効・ADRなし・保存済みh=1/3/5の合成ケース。全有効機能の実データ比較や本番artifactの妥当性を証明するものではない。

## S0–S8の確認状況

| 領域 | 確認した範囲と限界 |
|---|---|
| S0 比較基準 | 既存baseline・比較範囲を照合し、今回の作業ツリーhashと追加probeを保存。原S0の全機能・全入口の比較証拠不足を解消したとは扱わない。 |
| S1 設定・構築 | 本番configの継承先、共通factory、live/BT/VaR/gap/学習の構築境界を確認。MH 1/3/5、重み0.8/0.1/0.1、ML・macro・CS有効、監査失敗時fallback有効という設定を維持。 |
| S2 入力・PIT | 所有コピー、JST、当日/future target遮断、補助入力の時刻検査と修正を確認。Q1は学習入口の正規化漏れ。取得元の真のavailable_atは今回も未確認。 |
| S3 分布・cache・VaR | 当日cache→許可されたon-demand→flat、horizon別gap snapshotの来歴、gap生成のJP labelマスク、VaR入力の版固定・期限・所有権を確認。新たな長期実データ比較は未実施。 |
| S4 型・結果 | typed入力・decision・注文計画・約定・照合結果の接続と関連回帰を確認。今回の範囲で新規指摘なし。 |
| S5 執行・復旧 | prepared再開/送信時の未解決run検査、transaction、完了時checkpoint、read-only復旧入口、batch構文を確認。実口座照会・実scheduler状態確認は未実施。 |
| S6 会計 | 正の約定数量に対する価格/費用検証、取消後の部分約定、費用欠損と明示0の区別を確認。実口座の費用・建玉・残高との突合は未実施。 |
| S7 ML・配布 | 前回修正、特徴量一致、研究分離、artifactの隔離推論を確認。Q1/Q2が残る。verified本番artifactの再生成・OOS・同一入力での本番/BT比較は今回未実施。 |
| S8 CI・文書 | 下記ローカル検証はPASS。roadmap・ADR・計画は部分完了の扱いを維持。Hosted CIは未確認。計画冒頭の808件など過去の件数は、今回の848件と区別する。 |

## 実行した検証

| 検証 | 結果・証拠 |
|---|---|
| 全`tests/`、watchdog 360秒 | **848 passed / 17 warnings / 104.62秒**。[tests.log](tests.log) |
| 修正確認・特徴量比較・追加境界probe、watchdog 60秒 | M1/M2解消、特徴量一致、Q1/Q2再現。[probe_boundaries.log](probe_boundaries.log) |
| Ruff | CI対象production/tests/toolsとML学習module、PASS。[ruff.log](ruff.log) |
| mypy | 147 source files、PASS。[mypy.log](mypy.log) |
| import-linter | 7 kept / 0 broken。[imports.log](imports.log) |
| compileall | src/leadlag・tests・tools・scripts・src/research、PASS。[compile.log](compile.log) |
| 文書リンク | CI対象5文書、84リンクPASS。[docs.log](docs.log) |
| lockfile | `uv lock --check --offline`、PASS。[lock.log](lock.log) |
| wheel | 古いbuild/egg-infoを置いた一時sourceでclean buildと復元、manifest、research非同梱、隔離install・CLI・合成artifact推論がPASS。[検証コード](verify_distribution.py)、[結果](wheel.log)、[build](wheel_build.log)、[smoke](wheel_smoke.log) |
| batch / diff | 4 batchの`bash -n`、`git diff --check`がPASS。[bash.log](bash.log)、[diff_check.log](diff_check.log) |

wheelは既存uv cacheのbuild toolを一時プロセスから使用した。ネットワーク取得や既存プロジェクト環境への依存追加は行っていない。

Q1/Q2の修正・回帰追加後も、原計画の比較範囲、実available_at、実口座/運用証跡、本番ML artifactとOOS・本番/BT照合、Hosted CIという既存の受入残件は別途残る。全テストPASSをS0–S8全体完了・本番昇格可とは判定しない。
