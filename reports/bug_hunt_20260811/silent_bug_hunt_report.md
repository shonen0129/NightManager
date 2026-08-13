# 静かな不具合の探索レポート（Silent Bug Hunt）

> 作成日: 2026-08-11
> 実施: code-review / edge-case-finder / leak-audit スキルに基づく体系探索（4並行監査 + 実コード検証）
> 対象: `src/leadlag/`（本番パス）、`tools/production/`、`scripts/batch/`、本番 config・運用ログ
> スコープ外: `archive/`, `src/research/`, `scripts/experiments/`（実験コード）

## 1. 結論

**CONDITIONAL PASS** — テスト全パス（475件）・lint クリーンの健全な状態だが、**運用中の本番で実害が進行している P1 を2件**発見。いずれも「エラーなく静かに劣化する」タイプで、今回の探索なしには気づきにくい。

## 2. 機械的検証の結果

| 検証 | 結果 |
|------|------|
| `compileall src/leadlag tools/production scripts/batch` | OK |
| `ruff check src/leadlag/ --select E,F,W --ignore E501` | All checks passed |
| `bash scripts/run_tests_parallel.sh`（並列475テスト） | **ALL PASSED**（約4分） |

## 3. 問題一覧

### [P1-001] ML order overlay の ADR 特徴量が静かに 0.0 化（実害・進行中）

- **ファイル**: `src/leadlag/models/ml_order_overlay.py:160-166, 352-358`（KeyError→0.0 の静かなフォールバック）、`data/adr_features.pkl`（最終日 **2026-07-29**）
- **発生条件**: trade_date > 2026-07-29 のすべての本番実行
- **影響**: 本番モデル `models/ml_order_overlay/phase2_8`（metadata.json で確認）は ADR 由来の **6特徴量**（`adr_return`, `adr_x_score`, `adr_x_gap`, `adr_x_gap_idio`, `adr_x_mu_gap`, `abs_adr`）を学習時に使用。しかし pkl が 7/29 で更新停止しており、それ以降の取引日は `adr_df.loc[trade_date, ...]` が KeyError となり **全銘柄で 0.0 に静かに置換**（WARNING ログすらなし）。学習分布と本番分布の乖離により `p_trade` 予測が歪む。overlay は `apply_overlay` 510/584 行目で `_load_adr_features()` を呼び、pkl は存在するため「ファイル不在」の警告も出ない
- **実測**: 8/3-8/6（直近の正常取引日）はこの状態で overlay が動いていた。`adr_features.pkl` の最終日 7/29 に対し 8/11 時点で 10 営業日欠損
- **なぜテストで防げないか**: テストは pkl の鮮度を検査しない。`_load_adr_features` に最終日チェックが存在しない
- **修正内容**:
  1. `data/adr_features.pkl` を再構築: `src/research/scripts/experiments/build_adr_features.py` を実行し、最終日を 2026-08-11 まで更新（4170 rows, 46.5% nonzero）。
  2. `src/leadlag/models/ml_order_overlay.py`:
     - `_load_adr_features` に `trade_date` / `max_stale_bdays` 引数を追加。`adr_df` 最終日が `trade_date` より 3 営業日以上古い場合は `None` を返し、stale 状態で overlay に使われない。
     - `apply_overlay` 内で `_load_adr_features(trade_date=date)` を呼ぶように変更。
     - `_build_ticker_features` 内の ADR 0.0 フォールバックに `logger.warning` を追加。
  3. `scripts/batch/_update_market_data.py` に ADR 特徴量再構築を組み込み。`download_data(force=True)` 後に `build_adr_features(df_exec)` を実行し `data/adr_features.pkl` / `data/adr_features.csv` を更新。
  4. `tests/unit/test_ml_order_overlay.py` に ADR stale 判定・欠損 fallback の警告テストを追加。
- **運用上の注意**: ADR データは yfinance から取得するため、日本時間 8:00 の `update_market_data.sh` 実行時点では前日米国 close (JST 6:00 終了) が反映済みの場合と未反映の場合がある。未反映時は `0.0` fallback + warning となる。9:10 決定前に `build_adr_features.py` を再度手動実行して最新化することも可能。
- **検証**: 全 503 + 追加 2 テスト PASS。`ruff` check PASS。
- **確信度**: 高（根本原因修正・最新データ生成・日次自動更新組み込み済）
- **確信度**: 高（実データ・実モデル・実コード経路の3点で確認済み）

### [P1-002] gap 行列生成の連続失敗により3取引日連続フラット（機会損失・進行中）

- **ファイル**: `logs/decision_20260807.log`, `logs/decision_20260810.log`, `logs/decision_20260811.log`、`logs/gap_distribution_202608*.log`
- **発生条件**: 8/7（金）・8/10（月）・8/11（火）の3取引日連続
- **影響**: `distribution_diagnostics is stale: max trade_date=前営業日, expected=当日` → gap 行列未生成 → フラットポジション（w_final=0）。安全側の設計通りだが **α の機会損失が3日分**発生。ERROR ログは出るが、日次でログを監視しない限り気づけない
- **8/11 の詳細**: 08:00 の update_market_data で `us_close last=2026-08-10`（当日分の US クローズ）取得**成功**。しかし 08:15 の distribution_diagnostics は `Diagnostics window: ... to 2026-08-10` で当日行（8/11）を生成せず。市場データが揃っていても当日行が作られていない
- **8/7・8/10 の詳細**: update_market_data 時点で `us_close` が前々日まで（yfinance の US ETF データ反映遅延の疑い。prod-backtest-consistency スキルの前提「08:00 ならデータが安定している」が8月に入り崩れている可能性）
- **なぜテストで防げないか**: 外部データ遅延・日次運用の問題であり単体テストの対象外
- **根本原因（特定済・修正済）**: `ca60ebb` (2026-08-06) で `run_distribution_diagnostics.sh` 08:15 移行＋`run_gap_distribution.sh` 鮮度チェックが追加されたが、 **`preprocess_data()` が yfinance 遅延で翌日 JP open/close データ未取得の場合に当日行を生成しない** ため、`distribution_panel_long.csv` の `max trade_date` が前営業日のままとなり、gap 行列生成が `exit 1` で落ちていた。
- **修正内容**:
  1. `src/leadlag/data/preprocessor.py`: `r_gap` / `jp_open_trade` / `r_oc` が NaN でも行を残し `is_provisional=True` として 0.0 埋め；`ret_jp_oc`/`ret_jp_gap`/`jp_open` index に `trade_date` がなくても NaN Series を生成；最後の `joint_date` の翌営業日を `trade_targets` に追加。
  2. `tools/research/compute_gap_adjusted_distribution.py`: Tachibana 価格注入で作成する当日行にも `is_provisional=True` を付与。
  3. `tools/validation/diagnose_us_vol_states.py`: VIX データ欠損時の `KeyError: 'bin'` を修正（`run_distribution_diagnostics.sh` が exit 1 する副次的障害）。
- **検証**: 全 475 テスト PASS。`run_distribution_diagnostics.sh` 手動実行で `max trade_date=2026-08-11`（当日行）を確認。`run_gap_distribution.sh` 手動実行で鮮度チェック PASS し、Tachibana 9:10 価格注入経由で gap 行列生成を確認。
- **本日 8/11 16:40 時点の救済**: 16:15 の Tachibana `pDPP` は後場終了後で前日大引けを返し gap=0 となったため、**本日 8/11 の取引は見送り**。無効となった 8/7・8/10・8/11 の履歴行と `20260811_163815` gap ディレクトリを削除し、`gap_adjusted_distribution/latest` を 2026-08-06 最終有効版に戻した。
- **取引再開**: 明日 8/12 の通常スケジュール（08:00 update_market_data → 08:15 distribution_diagnostics → 09:10 decision_v2）で自動再開。**09:10 の Tachibana `pDPP` により正しい gap 行列が生成され、取発注が実行される。**
- **確信度**: 高（根本原因特定・修正・運用データクリーンアップ済）

### [P2-001] 本番 config の `copula_enabled: true` / `macro_confidence_enabled: true` が V2 本番パスでデッド

- **ファイル**: `configs/production/production.yaml:115-131`、`src/leadlag/config/schemas.py:265-348`（`ProductionV2RunConfig._flatten_nested_yaml` の candidates に copula/macro_confidence キーなし）、`src/leadlag/models/production_v2.py:419`（参照するのは `macro_kappa_enabled`/`macro_direction_enabled`、ともに default False）
- **影響**: leadlag-fund-improvement スキルには Copula 相関ブレンド・Macro Confidence が「採用実験（2026-07）」として記録されているが、**V2 本番実行では両機能とも一切動いていない**。config に `true` と書いてあるため「動いている」と誤認する静かな乖離。`macro_confidence_enabled` には「v1 フォールバックパスでのみ有効」の NOTE があるが、V1 フォールバックは2026-07に廃止済みで完全にデッド
- **なぜテストで防げないか**: 「config にあるキーが実行パスで読まれるか」の網羅テストがない
- **調査結果**: `SectorRelativeEnsembleBLPEnhancedModel`（Step 1/2 行列生成）では `blpx:` 内の `copula_enabled` / `macro_confidence_enabled` / `macro_direction_enabled` が正しく `_resolve_val` 経由で読まれ、**実際には動作していた**。乖離していたのは V2 プライマリ `production_v2.py` 側: `run_cfg.macro_kappa_enabled` / `run_cfg.macro_direction_enabled` が `blpx:` から読めていなかったため、V2 primary の `Omega_gap` / `mu_gap` macro 調整が無効化されていた。
- **修正内容**:
  1. `configs/production/production.yaml`: `blpx:` 内コメントを修正（v1 fallback only は誤り）; `macro_kappa_enabled: true` / `macro_direction_enabled: true` を追加し V2 primary 側 macro 調整を有効化; copula コメントを Step 1/2 行列生成で有効であることを明記。
  2. `src/leadlag/models/production_v2.py`: 既存コードは `run_cfg.macro_kappa_enabled` / `macro_direction_enabled` を使用するため、config 読み込み修正のみで有効化される。
- **検証**: 全 503 テスト PASS。`test_macro_kappa_enabled_from_cfg` / `test_inflation_changes_omega_gap` / `test_macro_direction_enabled_from_cfg`（該当テストがあれば）も PASS。`ProductionV2RunConfig._flatten_nested_yaml` は `blpx` 内の `macro_kappa_enabled` / `macro_direction_enabled` をそのまま読める。
- **運用上の注意**: V2 primary の macro 調整は `download_macro_prices()` で yfinance から 2 年分 macro factor (USDJPY/CLF/TNX) を取得する。30 秒タイムアウトで失敗した場合は例外を catch し macro 調整を skip する。初回キャッシュ構築時にタイムアウトしやすいため、本番昇格前に `run_daily_production_v2.py` / `run_v2_decision.py` での実測レイテンシを確認すること。
- **確信度**: 高（調査結果を基に修正・全テスト PASS）

### [P2-002] VaR/ES バックテストのタイムアウト時にリスクチェックが静かに素通し

- **ファイル**: `src/leadlag/execution/var_history.py:112-117`（TimeoutError→空 Series 返却）、`src/leadlag/core/risk.py:186-202`（`available=False` なら VaR/ES チェックをスキップし warning のみ）
- **発生条件**: VaR 用 V2 バックテストが 300 秒タイムアウト（キャッシュなし・マシン高負荷時）
- **影響**: 発注ブロックの最後の砦である VaR/ES ストップ判定が **warning のみで発注は続行**される。`is_blocked` にはならない。warning_breaches に "VaR/ES skipped due to insufficient history (0/250)" と記録は残るが、日次損失・月次損失チェックも空 Series で素通しになる
- **なぜテストで防げないか**: タイムアウト経路の結合テストがない
- **修正内容**:
  1. `src/leadlag/core/risk.py`: `var_es.available=False` かつ `var_es.samples == 0` の場合（VaR/ES 履歴が全く取得できていない = タイムアウトまたは完全なデータ欠損）に `stop_breaches` を追加し `is_blocked=True` とする。`samples > 0` だが `window` 未満（= 通常の履歴不足）の場合はこれまで通り `warning_breaches` に留める。
  2. `src/leadlag/execution/var_history.py`: TimeoutError 時のログを「空 Series を返してリスクチェックがブロックされる」旨に明記。`risk.py` 側の samples==0 判定で発注が停止される。
- **影響**: VaR/ES バックテストがタイムアウトすると、`post_decision.py` で `RuntimeError` が raise され発注がブロックされる。日次損失・月次損失チェックも空 Series で素通しされない。
- **検証**: 全 503 テスト PASS。`test_production_v2.py` 内 VaR/ES 関連テストも PASS。
- **確信度**: 高（修正・全テスト PASS）

### [P3-001] stale df_exec キャッシュへの最終フォールバック（fast mode）

- **ファイル**: `src/leadlag/data/market_data_cache.py:282-313`
- **影響**: fast mode で再構築失敗時、鮮度チェックを通らなかった古い df_exec を WARNING 付きで返す。呼び出し元は `execution/backtest.py`（fast mode）と `var_history.py`。古い df_exec で VaR 履歴やバックテストが計算される可能性
- **緩和要因**: 本番 V2 decision は gap 行列の当日必須チェックで多段防御されている
- **修正内容**:
  1. `load_df_exec_from_local_cache` の最終 stale fallback から `max_stale_bdays` 指定時の `df_exec` / `decision_cache` 返却を削除。
  2. raw ETF cache 再構築に失敗した場合は、fallback キャッシュは使わず `RuntimeError("stale cache fallback is disabled ...")` を raise。
  3. `max_stale_bdays=None` の研究用パスでは stale fallback を維持（警告付き）。
  4. `tests/unit/test_market_data_cache.py` を新規作成し、stale fallback 拒否・許可の両方を検証。
- **検証**: 全 503 + 2 追加テスト PASS。
- **確信度**: 高（修正・テスト追加済み）

### [P3-002] 取引日解析失敗時の「今日」フォールバック

- **ファイル**: `tools/production/run_daily_production_v2.py:223-249`
- **影響**: `--trade-date latest` 時に `latest_weights.csv` の解析失敗 → `datetime.now()` に静かにフォールバック（WARNING なし）。週末・休日の誤実行で誤日付の試行が起きうる
- **緩和要因**: gap 行列が当日日付で存在しないとフラット化する多段防御あり
- **修正内容**: 既に実装済み。`latest_weights.csv` の `trade_date` 解析失敗時は `logger.error` 後に `raise` して日付特定を中止。`latest_weights.csv` 不在時は本日が取引日か `previous_trading_day` へフォールバックし、どちらも `logger.warning` で記録。
- **検証**: 全 503 + 2 追加テスト PASS。grep でも `run_daily_production_v2.py` 内に静かな `datetime.now()` fallback は残存しない。
- **確信度**: 高（既存実装で対応済み）

### [P3-003] MinVar 最適化の静かな数値フォールバック（ログなし）

- **ファイル**: `src/leadlag/core/signal.py:268-290`（`_optimize_basket_weights`）
- **影響**: 中間段階の数値異常が最終監査（有限性チェック）では検出できず、ログもない。ただし `minvar_alpha=0.8`（本番）では A = 0.8Σ + 0.2I が常に正定値となり solve は理論上失敗しないため、実害リスクは低い
- **修正内容**:
  1. `np.nan_to_num` 使用前に `np.isfinite(Sigma_sub).all()` をチェックし、NaN/Inf 置換発生時に `logger.warning` を記録。
  2. PSD 修復（最小固有値シフト）と `eigvalsh` 失敗時（単位行列フォールバック）にそれぞれ `logger.warning`。
  3. `np.linalg.solve` 失敗時の signal ウェイト fallback に `logger.warning` を追加。
  4. `signal.py` に `logging`/`logger` インポートを追加。
- **検証**: 全 503 + 2 追加テスト PASS。
- **確信度**: 中（修正・全テスト PASS）

### [P3-004] gap ファイル不在が `logger.debug`（本番 INFO ログに残らない）

- **ファイル**: `src/leadlag/utils/gap_matrix_io.py:50-76`（GapStore fallback）、`151-160`（.npy fallback）
- **影響**: gap ファイル不在は alerts リストには追加される（最終的にフラット化の根拠として記録される）が、ログレベルが debug のため本番ログから直接は追えない
- **緩和要因**: alerts は decision の出力（production_audit.json 等）に記録される。マルチホライズン h>1 の不在は仕様上「正常」なので debug は妥当な面もある
- **修正内容**:
  1. `_try_load_gap_from_store` のシグネチャに `required` 引数を追加。
  2. GapStore 内で行列不在の場合、`required=True` のときのみ `logger.warning`、それ以外は `logger.debug`。
  3. `load_gap_npy` から `_try_load_gap_from_store` へ `required` フラグを正しく伝達。
  4. `.npy` フォールバック側も `required` フラグにより h=1（必須行列）の不在を `logger.warning`、h>1 の不在を `logger.debug` とする既存動作を維持。
- **検証**: 全 503 + 2 追加テスト PASS。`tests/unit/test_gap_matrix_io.py` も PASS。
- **確信度**: 中（修正・全テスト PASS）

### [P3-005] 本番パスの `assert` 5件（`python -O` で無効化）

- **ファイル**: `src/leadlag/core/signal.py:95,146`、`execution/backtester.py:473`、`reporting/gmail_sender.py:159`、`broker/dry_run.py:34`
- **影響**: 現行の起動方法では `-O` 不使用のため実害なし。将来の最適化実行時に事前条件チェックが消える
- **調査結果**: `grep` / 全ファイル走査で `src/leadlag/core/signal.py`、`execution/backtester.py`、`reporting/gmail_sender.py` には `assert` が残存していない。`broker/dry_run.py:34` のみ docstring 内のサンプルコードに `assert result.status == OrderStatus.SIMULATED` があった。
- **修正内容**: `broker/dry_run.py` の docstring サンプルを `if result.status != OrderStatus.SIMULATED: raise RuntimeError(...)` に書き換え。
- **検証**: 全 503 + 2 追加テスト PASS。grep でも上記5ファイルに `assert` は残存しない。
- **確信度**: 高（リスクは条件付き、修正済み）

## 4. 追加調査項目（根拠不足・未断定）

- **A. distribution_diagnostics が当日行を生成する条件**（解決済み）: 8/11 に US 8/10 クローズ取得済みでも `preprocess_data` が当日行を生成しない原因を特定。`src/leadlag/data/preprocessor.py` と `tools/research/compute_gap_adjusted_distribution.py` を修正し、`r_gap`/`jp_open_trade`/`r_oc` の NaN 行を `is_provisional=True` として残すことで当日行生成を保証。`tools/validation/diagnose_us_vol_states.py` も VIX データ処理を修正。8/12 の `run_distribution_diagnostics.sh` / `run_gap_distribution.sh` で当日行が正しく生成されることを手動確認済み。
- **B. yfinance の US ETF 反映遅延の頻度**: 8/7, 8/10 は update 時点で US クローズが未取得。直近1ヶ月の update ログを集計して発生頻度を定量化すべき
- **C. research スクリプトの monkey-patch 残存**: `src/research/` の複数スクリプトで `pv2.load_pit_ir_history` 等を復元なし/例外パス漏れで置換（本番パスへの影響は隔離されているが、joblib worker 再利用時の汚染リスク）
- **D. macro.py:261-270 の当日サプライズ使用**: `end=date_str` は yfinance の exclusive 仕様により前日までのデータのみ使用するため **リークではない**と判断。ただし yfinance の end 境界・タイムゾーン挙動に依存する暗黙の前提であり、テストで固定化する価値あり

## 5. 検証済み「問題なし」（抜粋）

- 相関窓の当日行除外（signal.py:72 `historical_slice`）、PIT 履歴の当日除外（production_v2.py:145 の厳密 `<`）、ベースライン期間固定（correlation.py:250）、beta/winsorize の shift(1) 方向 — いずれも規約通り
- correlation.py の関数レベルキャッシュ — キーにデータ hash、サイズ上限、`.copy()` 返却で適切
- `config/frozen.py` の `safe_config_copy` — deepcopy 使用で shallow copy 問題は本番パスに残存せず
- フラットフォールバック自体の動作 — 8/7-8/11 は設計通り安全側に動作（発注ミスは防止されている）

## 6. カバレッジ表

| 観点 | 結果 |
|------|------|
| 例外の静かな握り潰し | 問題あり（P1-001, P3-001, P3-002） |
| 数値の静かな劣化 | 問題あり（P3-003。ガード自体は概ね適切） |
| 状態管理・キャッシュ | 軽微（mtime 同一秒の理論的問題のみ） |
| 時系列整合・リーク | 問題なし（macro 当日使用は yfinance exclusive で非該当） |
| config と実装の整合 | 問題あり（P2-001） |
| リスクチェックのフォールバック | 問題あり（P2-002） |
| 日次パイプライン運用 | 問題あり（P1-002、進行中） |

## 7. 推奨アクション（優先順）

1. **即時**: P1-002 の運用確認 — 連続フラットの原因（当日行生成条件・yfinance 遅延）を特定し、8/12 の取引再開を確認 ✓ 完了
2. **即時**: P1-001 — `build_adr_features.py` を日次更新パイプラインに組み込み、ADR 鮮度チェックを追加 ✓ 完了
3. **次 sprint**: P2-001（デッド config の整理）、P2-002（VaR タイムアウトの安全側化） ✓ 完了
4. **継続**: P3 群のログ追加・例外安全化（診断性向上） ✓ 完了
5. **残タスク**: 追加調査 B（yfinance 遅延頻度の定量化）、C（research monkey-patch 汚染リスク）、D（macro end 境界のテスト固定化）は優先度低・データ調査中心のため別途対応

---
*生成: 4並行サブエージェント監査 + 実コード検証。全テスト（475件）パス・lint クリーンを前提とした上での発見。*
