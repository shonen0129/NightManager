# R1〜R6 構造改善 実施結果

実施日: 2026-09-20（追加修正・再検証）

## 結論

ローカルで完了できる契約・実行経路・artifact・CI検証を実施した。R3は、同一の実行入力snapshotを使うcache/on-demand比較で数値一致を確認した。R1/R2は主要なtyped経路とVaRのrun-owned入力snapshotまで実装したが、全入口のavailable_at証跡と、R4の実ブローカー照合、R6のHosted CIは外部環境の証跡が必要なため保留とした。

| 項目 | 状態 | 実施内容 |
|---|---|---|
| R1 | 部分完了 | 09:10 cutoff、JP close 15:30 label availability、h=1/3/5のtyped calculation viewと共有horizon変換を実装。バックテストは09:10 JSTへ統一。gap生成全入口のlabel availability検証は残る。 |
| R2 | 部分完了 | rank-reversal、09:10、macro、ADR、PITをrun-owned `HistoricalInputs`へ束ね、VaR cache keyへsnapshot fingerprintを追加。observed_atの実データ証跡と外部取得失敗時の完全な来歴固定は残る。 |
| R3 | 完了 | 同一`df_exec` fingerprintの実行snapshotで、2日×h=1/3/5の**file cache/on-demand** μ・Ωを比較。最大差分はμ 2.22e-16、Ω 1.73e-18。h=1の古いStep 1 fallbackは使わずBLPX共分散へ統一し、bundle来歴も検証した。 |
| R4 | 部分完了 | LaunchAgentを現workspaceへ再登録し`launchctl print gui/501/com.leadlag.*`で4件を確認、state DBのrecovery candidate 0件、read-only reconcile PASS。実口座のaccount/fill/partial-fill照合は未実施。 |
| R5 | 部分完了 | verified versioned artifactを一時領域へ再生成。2026-08-14の同一入力でproduction形式とbacktestのウェイト差分は0。production configへの昇格は、十分な履歴と外部macro/ADR入力を揃えるまで保留。 |
| R6 | 部分完了 | lock/compile/ruff/mypy/import/docs/wheel/smoke/全pytestをローカルでPASS。Hosted CIのrun URL/resultは未取得。 |

## 変更内容

- `HistoricalInputs.calculation_frame()` にJP close（15:30 JST）の標準label availabilityを追加し、09:10時点では当日close/targetをマスク、close後の評価では表示可能にした。
- 明示 `label_available_at` は、利用可能時刻より前の実行時点で当日行だけをマスクし、可用時刻後に履歴行を誤って隠さないよう修正した。
- バックテストのsnapshot時刻を09:10 JSTへ統一し、US/gap/beta/TOPIX/current price/previous closeの`observed_at`を契約へ記録した。
- `src/leadlag/data/rank_reversal.py` を追加。rank-reversal signalをrun開始時に読み込み、`HistoricalInputs`のfingerprintへ含めるようにした。
- rank-reversal overlayは明示的signalを受け取れるようにし、strict typed decision pathではsnapshot欠損時もdate別ファイルを直接開かないようにした。互換経路の暗黙読込は維持した。
- PIT診断CSVのキャッシュ判定を内容SHA-256へ変更し、mtime・サイズを維持した原本置換も検出する回帰を追加した。
- provenance不備時のmulti-horizon on-demand retryにも`open_910_returns`と`allow_implicit_io`を伝播した。
- strict typed h=1/3/5 on-demandでは`open_910_returns`が欠損している場合に5分足cacheを再取得せず失敗させ、cache側も全horizonで同じ入力欠損を受け付けないようにした。

## 検証証跡

- R3比較: [r3_real_paths.json](evidence/r3_real_paths.json)
  - 同一入力fingerprintの2026-08-14/17、h=1/3/5でcache sourceが全て`file_cache`、on-demand sourceが全て`on_demand`、両方`ready`。
  - 最大差分はμ 2.22e-16、Ω 1.73e-18。bundle metadataには`input_version`、as-of `open_910_version`、`model_version`、`config_version`、17銘柄の`ticker_order`が入り、入力変更時はcacheを拒否する回帰も追加した。
  - 再生成元: `var/results/20260920_structural_completion/r3_capture_gap/20260920_221007/`。
  - 再生成時のraw market downloadはネットワーク解決不可だったため、固定済みlocal cacheを使用した。fresh external dataの証跡はR1/R2の残件として扱う。
- R3差分分解: [r3_stage_decomposition.json](evidence/r3_stage_decomposition.json)、[r3_source_frame.json](evidence/r3_source_frame.json)
  - 旧差分は、研究生成側のh=3/5累積gap・overnight列とon-demandの1日値、および生成側とローカルcacheの入力欠損差によるものだった。共有変換と同一snapshot比較へ修正した。
- R4 reconcile: [r4_reconcile.json](evidence/r4_reconcile.json)
  - `{"runs": [], "complete": true}`。state DBのexecution_runs/order_intents/order_observations/reconciliationsは全て0件。
  - [r4_scheduler.txt](evidence/r4_scheduler.txt)で4つのLaunchAgent登録と現workspace pathを確認。
- R5 artifact: `var/results/20260920_structural_completion/ml_overlay_retrained/`
  - `metadata_status=verified`、`CURRENT`あり、train_end=2026-08-13。
  - [r5_weights.json](evidence/r5_weights.json)で`max_abs_weight_diff=0.0`、`l2_weight_diff=0.0`。
- R6 local CI: [r6_local_ci_current.json](evidence/r6_local_ci_current.json)
  - 今回のローカル全テストは **798 cases / 0 failure / 0 error / 16 warnings / 99.08秒**。証跡JSONを正本とする。

## 2026-09-20 追加修正（09:10・horizon・BT評価・strict I/Oの整合）

前回レビューで残っていた3件を修正した。

- gap bundleの来歴へas-of `open_910_returns` の指紋を追加し、研究生成側とcache利用側を同じ指紋で照合するようにした。`df_exec` が同じでも09:10入力が訂正された場合は旧cacheを拒否する。
- h=3/5のon-demand gap入力は、直近行だけPIT snapshotのgap・beta・TOPIX夜間値へ差し替え、過去h-1行は履歴から累積するようにした。従来の累積窓とlive snapshotを同時に保持する。
- V2 backtestの実現targetは、decision生成と同じrun-owned `HistoricalInputs.open_910_returns`を明示的に受け取るようにした。別の5分足cacheを評価時に再読込しない。

追加回帰を含む全`tests/`は **798 passed / 0 failure / 0 error / 16 warnings / 99.08秒**（watchdog 300秒）で完了した。対象RuffとcompileallもPASSした。
R1/R2の全gap生成入口・available_at証跡、R4実口座照合、R5本番artifact昇格、R6 Hosted CIは引き続き外部受入条件として未完了であり、S0–S8全体完了や本番昇格可とは判定しない。

## 2026-09-20 timezone境界の追加修正

来歴指紋、ADR検証、ML overlay入口のtimezone付きtrade dateをJST日付へ統一した。ADRの返却フレームも
日付indexへ正規化し、UTCの暦日ずれによる欠損判定を防ぐ。回帰を2件追加し、serial実行で**803 passed / 17 warnings**を確認した
（[実行証跡](evidence/r6_local_ci_timezone_fix.json)）。
既存のxdist証跡（追加回帰前の798 passed / 16 warnings）は履歴として残している。

## 2026-09-20 PIT snapshot境界の修正

同一日でも、09:10の意思決定へ15:31のsnapshotを渡せる契約漏れを修正した。明示された時刻はsnapshotと完全一致させ、日付だけの互換呼出しは09:10 JSTまでに制限した。空DataFrame、NaT、JST正規化後の重複trade dateもPITDataLake初期化時に拒否する。

時刻不一致・空入力・NaT・重複入力の回帰を追加し、全`tests/`は **803 passed / 17 warnings**。対象RuffとcompileallもPASSした。R1/R2/R4/R5/R6の外部受入条件とS0–S8全体完了の判定は変更しない。

## 2026-09-21 timezone境界の再修正

再レビューで確認した2つのtimezone境界を修正した。PIT履歴日付は要素ごとにJSTへ変換してから保存し、UTCの暦日ずれで当日行が履歴へ混入しないようにした。公開モデルと内部decision engineの`trade_date`照合も共通のJST日付正規化へ揃え、同じJST日を表すoffset-aware入力を拒否しないようにした。

3件の回帰テストを追加し、対象テスト51件、全`tests/`は **808 passed / 17 warnings / 104.48秒**（watchdog 300秒）。対象Ruff、compileall、`git diff --check`もPASSした。R1/R2/R4/R5/R6の外部受入条件とS0–S8全体完了の判定は変更しない。

## 外部で必要な残作業

1. R1/R2: gap生成入口にも`HistoricalInputs.calculation_frame(as_of)`とlabel availabilityを接続し、macro/ADRのavailable_at証跡を固定する。
2. R4: 実口座のread-only account/positions/fills照合を実行し、partial-fill recovery journalまで確認する。注文送信・取消・再送は含めない。
3. R5: 2015年以降の本番学習期間、macro/ADRの固定snapshot、OOS評価を揃えたartifactを生成し、production configへ昇格する。
4. R6: GitHub ActionsのHosted CIを実行し、run URLと全job PASSを記録する。現環境には`gh` CLIがないためローカル実行では代替できない。

## 2026-09-21 再レビュー指摘の修正

再レビューで残っていた3件を修正した。

- strict typed pathのcache/on-demandは、h=1だけでなくh=3/5でも`open_910_returns`を必須化した。欠損時は暗黙の5分足取得や開始日openへのフォールバックを行わず、FallbackPolicyのflat終端へ進む。
- `PITDataLake.build_decision_inputs`と`KnownMarketInputs`は、09:10 JST前のintraday `as_of`を拒否する。日付だけの互換呼出しは従来どおり09:10へ解決する。
- `BacktestEngine.run_v2_backtest`は、渡された`HistoricalInputs`の正規化済みframeと`df_exec`の内容fingerprintが一致しない場合に開始前に拒否する。

追加回帰を含む対象テスト60件、全`tests/`は **815 passed / 17 warnings / 104.48秒**（watchdog 360秒）。compileall、対象Ruff、mypy、`git diff --check`もPASSした。
