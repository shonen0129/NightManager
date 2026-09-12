# 日米リードラグ・ファンド改善ガイド

## このファイルと Skill の役割

- 本ファイルは共通の実行規約・不変条件の正本。`.agents/skills/` は作業別手順を補完する。食い違いは本ファイルに合わせる。ユーザーの明示的な依頼は Skill の一般指針に優先する。
- 修正依頼は、依頼範囲の実装・検証・必要な文書更新まで進める。既存のユーザー変更を保持し、関連 Phase の未了を理由に範囲を広げない。
- 可逆なローカル修正・検証と、その変更が原因の失敗修正は依頼の範囲で進める。確認は、結果を左右する未確定事項や依頼で認められていない操作がある場合に限り、独立して進められる作業は続ける。
- 設計の理由は `docs/decisions/`、実験結果は `reports/`、不採用索引は `docs/experiment_graveyard.md` に記録する。AGENTS.md / Skill に実験成績や完了履歴を蓄積しない。
- 以下のパスはリポジトリルート基準。現行挙動はコード・継承解決後の設定・テストで確認する。実装が不変条件に違反する場合、規約を緩めて整合させず不具合として報告する。

## 実行規約

- Python は既存のプロジェクト環境を優先する。`.venv` がある場合は有効化して `python3` を使うか、`.venv/bin/python` を指定する。文書検証のためだけにグローバル環境へ依存を追加しない。
- **`python3 -c "..."` のインライン実行は禁止**。Python コードはスクリプトファイルに保存して実行する。既存の `python3 -m ...` エントリーポイントは使用可。
- **長時間コマンドにはプロセス全体のタイムアウトを設定する**。ツールの出力待ち時間・`yield_time_ms` は停止期限ではない。`timeout` / `gtimeout` があれば終了猶予付きで使用し、なければプロセスグループを期限付きで停止できる watchdog スクリプトを使う。詳細は `hang-prevention` と `docs/スタック再発防止策.md`。
- 本番発注・決済・再送は、その操作が依頼で認められた範囲でのみ行う。調査・テストのために `--api-enable` を追加しない。タイムアウト後の再送前には注文・約定・建玉を照合する。

## 戦略と正本

米国セクターETF（SPDR 11 + Style 4）のクローズ情報から、次の対応する日本営業日の TOPIX-17 ETF の **9:10→大引けリターン** を予測する市場中立戦略。

| 対象 | 正本・参照先 |
|---|---|
| 本番モデル | `src/leadlag/models/production_v2.py::ProductionV2Model`（処理本体は `src/leadlag/models/v2/`） |
| 本番設定 | `configs/production/production.yaml`。`__base__` の継承先も読み、`src/leadlag/execution/config.py::load_config_from_yaml` で解決する |
| V2 バックテスト | `src/leadlag/execution/backtester.py::BacktestEngine.run_v2_backtest()`、CLI `backtest` |
| 日次実行 | CLI `decision`、V2 同期パイプライン（ADR-P35） |
| ティッカー・感応度ラベル | `src/leadlag/data/tickers.py` の US/JP 定義・`SENSITIVITY_LABELS` |
| リスク・グロス調整 | `src/leadlag/core/risk.py`、`src/leadlag/core/portfolio.py::adjust_gross_exposure()` |
| ランタイム出力パス | `src/leadlag/config/paths.py`。運用データ `var/live/pipeline_data/`、結果 `var/results/`、shadow `var/shadow_runs/` |
| 設計・数理仕様 | 構造・責務の変更は `docs/ARCHITECTURE.md`、数式・統計は `docs/モデル技術仕様書.md`、設計判断は `docs/decisions/` の対象文書 |

`production_v2_primary_ruleD.yaml` は旧設定。現行機能の有効・無効やパラメータ値は本番設定で確認する。V1 のコード・バックテストを本番経路へ戻さない。

## データ整合規約

`df_exec` の行 t は次の対応を保持する（`src/leadlag/data/preprocessor.py`）。D_t / D_{t+1} は暦日の単純加算ではなく日米営業日の対応を表す。

- **US列 `us_cc_*`**: 米国営業日 D_t のクローズ・トゥ・クローズリターン（JST 翌朝に確定）。
- **JPターゲット**: 対応する取引日 D_{t+1} の 9:10→大引けリターン（`compute_jp_target_returns`）。予測時点では未確定。
- **`jp_gap_*`**: 取引日の寄付ギャップ。9:10 の判定時点で既知ならシグナルに利用可。
- 相関窓は `all_returns[window_start:current_index]` のように**当日行を除外**する（`src/leadlag/core/signal.py`）。当日既知の入力と、学習に使えない当日ターゲットを区別する。

## 不変条件

1. **ルックアヘッド禁止**: ローリング統計・ベータ・PIT ビニングは strictly historical。`src/leadlag/compliance/auditor.py` と `v2_auditor.py` の監査を無効化・緩和しない。日付チェックの PASS だけで計算窓の非リークを保証したことにしない。
2. **ベースライン期間を分離**: 事前分布・基準相関 `c_full` は 2010–2014 固定。バックテスト開始日は 2015-01-05 以降。`_prepare_residual_prior` の先頭1260行フォールバックに依存する構成を作らない。
3. **テストを弱めない**: 実行コード・挙動に関わる設定の変更後は全テストを通す。文書・Skill だけの変更は構造・参照先・意味の整合を検証する。失敗・未実行を成功扱いしない。実行方法は下記。
4. **市場中立制約**: RuleD 適用後のモデルウェイト `w_final` は net exposure ±0.05、gross ≤ 2.0。`side_leverage` 適用後の実効エクスポージャーとは区別し、両者を報告する。実効値は継承解決後の `risk` 設定にも照合する。既存監査のより厳しい閾値を緩めない。
5. **ティッカー定義を一元化**: 本番は N_U=15、N_J=17、計32次元。銘柄・感応度ラベルの変更は `tickers.py` を起点にし、配列順序・共分散・設定・キャッシュ・テストへの影響も確認する。「その1ファイルの変更だけで完了」と仮定しない。研究ユニバースは研究設定で分離する。
6. **前日 gap 行列の流用禁止**: 当日の日付に一致するファイル / SQLite 行だけを使用する。前日行列をコピー・改名して当日データとして使わない。
7. **フォールバックを維持**: 標準経路は当日 cache → on-demand BLPX（`ondemand_fallback_enabled=true` かつ必要なモデル・入力あり）→ 失敗・無効時にフラット `w_final=0`。V1 フォールバックは廃止済み。`src/leadlag/models/v2/fallback_policy.py` / `distribution_source.py` が実装先。PIT 履歴不足時の `fallback_multiplier` はフラット化と別に扱う。
8. **監査失敗を本番へ通さない**: 本番では `fallback_on_audit_failure=true` を維持し、失敗時はフラット化または発注停止を確認する。現行の数値監査のフラット化とリーク監査の結果処理は別経路なので両方追跡する。`false` でウェイトを保持する実装は本番推奨ではない。

## 作業別の Skill

作業に必要な Skill と、その中で対象条件に合う参照だけを読む。文書内の言及やモデルファイルへの接触だけで Skill を連鎖適用しない。短い文書修正に戦略調査・実験を追加しない。

| 作業 | Skill（`.agents/skills/` 配下） |
|---|---|
| 戦略改善・V2 バックテスト・本番昇格 | `leadlag-fund-improvement` |
| シグナル・モデル・パラメータの性能比較実験 | `experiment-design` |
| 時系列・監査・フォールバックの変更や検証 | `leak-audit` |
| 発生した不具合・異常出力・テスト失敗の調査と修正 | `debugging` |
| 境界値・異常系の分析 | `edge-case-finder` |
| 振る舞いを変えないコード整理 | `refactor` |
| コードレビュー・リリース監査 | `code-review` |
| 回帰テストの設計・追加 | `test-gen` |
| バックテスト結果の報告（実行後に適用） | `backtest-report` |
| 長時間 CLI 実行、ネットワーク・ロック待ちの診断や変更 | `hang-prevention` |

## 改善ワークフロー

1. 仮説と評価基準を実験前に定め、`docs/experiment_graveyard.md`・関連レポート・実験 registry で既存検証を確認する。
2. 実験スクリプトは `src/research/scripts/experiments/`、実験用モジュールは `src/research/experiments/`、設定は `configs/research/` に置く。本番パスへ直接実験コードを入れない。
3. 比較時はデータ・期間・コスト・執行条件を揃える。設定コピーは `leadlag.config.safe_config_copy` を優先し、dict には `copy.deepcopy` も可。ネストした設定の shallow copy は禁止。
4. 新パラメータ追加は原則避ける。追加時は **±摂動の感度分析と Deflated Sharpe（試行回数補正）を必ず報告**する。採用判断はウォークフォワード OOS 検証を経て行う。区間・purge・embargo・閾値は対象データとラベル期間から事前に設計する。
5. 試した設定・結果・判定を `ExperimentRegistry`（`src/leadlag/experiment_registry.py`、研究側の窓口は `src/research/experiment_registry.py`）へ記録する。標準保存先は `var/experiments/registry.jsonl`。過去試行の未登録を無視して試行回数を過小評価しない。
6. 本番昇格前に `tools/validation/monitor_residual_blpx_shadow_performance.py` 等でライブ整合を検証する。昇格が依頼範囲にある場合に本番 config を更新し、`docs/ARCHITECTURE.md` に記録する。研究上の採用判断と本番反映を区別する。
7. 採用・不採用・保留を `reports/<作業名>/` に残し、不採用索引は `docs/experiment_graveyard.md` に追記する。再現用コード・設定・データ参照を不採用だけを理由に自動削除しない。

## 検証と完了報告

- コード変更時は対象の回帰テストを実行後、`tests/` 全体を検証する。例: `python3 -m pytest tests/ -n auto`（pytest-xdist が必要）。直列なら `-n auto` を外す。pytest-timeout が利用可能なら `--timeout=300` 等を追加できるが、いずれも外側にプロセス全体の停止期限を設ける。
- `bash scripts/run_tests_parallel.sh` は既存の分割実行手段（ログ `/tmp/pytest_parallel/`）。現在は unit / integration / research を対象とするため、全テスト扱いする前に `tests/regression/` 等の未収録分も確認・実行する。所要時間やワーカー数は環境に合わせる。
- コード変更時の構文確認: `python3 -m compileall src/leadlag tests tools scripts src/research`。lint / 型チェックは `pyproject.toml` と既存 CI の設定に従い、新規エラーを残さない。文書・Skill のみなら構造・参照先・意味の整合を検証する。必要な検証が通った後の再実行は、新しい変更・失敗・未解決の懸念がある場合に限る。
- リファクタリングや Phase の進捗・完了を扱う場合は `docs/refactor_roadmap.md` と対象 ADR を照合する。`rg -n '\[ \]|未完了|部分完了' docs/refactor_roadmap.md` で確認し、チェックボックスがないことを全項目完了と解釈しない。ADR の accepted も実装完了を意味しない。
- 依頼範囲の実装・文書・設定・運用手順を揃えて完了を報告する。Phase 全体の完了は、その Phase の要件を全て検証できた場合のみ宣言する。範囲外の未了は別記する。

## 評価指標と注意点

- 主指標は **net Sharpe、最大DD、ターンオーバー、フォールバック発動率**。gross/net とコスト内訳（slippage / financing / borrow / reverse）を報告する。本番の片道 slippage 基準は5bps、金利・貸株・逆日歩も含める。実際に解決された設定・単位・集計期間を残す。
- 主評価はフラット日を含む全評価営業日を使う。稼働日だけの指標は補助値として分ける。on-demand 利用、終端フラット、PIT multiplier の率を混同しない。
- 9:10 執行価格の5分足 High/Low 中値近似は楽観側になり得る。コスト検証時は実約定ログと突合し、未取得なら限界を明記する。金利等の保有日数には週末を含む暦日を使う。
- 250日窓 VaR99 の尾部標本は約2.5個。stop 判定変更時は標本不足に注意する。
- 欠損補間・キャッシュ更新前に原因と下流への影響を確認する。入力の書き換えで異常を隠したり、`var/live/pipeline_data/` を実験結果の整理に巻き込んだりしない。
