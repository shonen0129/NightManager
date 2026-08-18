# 日米リードラグ・ファンド改善ガイド

## ⚠️ 実行規約（必須・違反禁止）

- **`python3 -c "..."` のインライン実行は禁止** — ハング・スタックの主要原因。必ずスクリプトファイル（`src/research/scripts/experiments/` 配下等）を作成して `python3 scripts/...` で実行すること
- 長時間実行コマンドには必ずタイムアウトを設定すること
- 詳細は `docs/スタック再発防止策.md` 参照

## 戦略概要

米国セクターETF（15銘柄: SPDR 11 + Style 4）の当日リターンから、翌営業日の日本 TOPIX-17 セクターETF（17銘柄）の **9:10→大引けリターン** を予測する日次マーケットニュートラル戦略。

- **本番モデル**: `ProductionV2Model` (Residual-BLPX-RA v2) — `src/leadlag/models/production_v2.py`
  - BLPX 構造化投影シグナル + gap調整予測分布 + `mu_over_sigma` ランキング + RuleD 動的グロス（PIT三分位: Low→0.75x, Mid/High→1.00x）
- **フォールバック**: 
  - 監査失敗時は **フラットポジション（w_final=0）** を返す。
  - gap ファイル欠損時は、`ondemand_fallback_enabled=true` の場合 on-demand BLPX 計算を試みる。失敗・無効時は **フラットポジション（w_final=0）** を返す（V1フォールバックは2026-07に廃止）。
- **本番config**: `configs/production/production.yaml`（正本）。`production_v2_primary_ruleD.yaml` は旧版（overnight holding・multi-horizon blend・rank reversal overlay 未含む）
- **アーキテクチャ詳細**: `docs/ARCHITECTURE.md`、数理仕様: `docs/モデル技術仕様書.md`

## データ整合規約（最重要・変更禁止）

`df_exec` の行 t の意味:
- **US列 (`us_cc_*`)**: 米国営業日 D_t のクローズ・トゥ・クローズリターン（JST 翌朝に確定）
- **JP列（ターゲット）**: 取引日 D_{t+1} の 9:10→大引けリターン（`compute_jp_target_returns` in `src/leadlag/data/preprocessor.py`）
- **`jp_gap_*`**: 取引日の寄付ギャップ（9:10 判定時点で既知 → シグナルに使用可）
- 相関窓は `all_returns[window_start:current_index]` で **当日行を除外**（`src/leadlag/core/signal.py`）。この規約を崩すと即リークになる

## 不変条件（改善時に絶対に守ること）

1. **ルックアヘッド禁止**: すべてのローリング統計・ベータ・PITビニングは strictly historical。`ComplianceAuditor`（`src/leadlag/compliance/auditor.py`, `v2_auditor.py`）の監査項目（`check_pit_binning_lookahead`, `check_residualization_leakage` 等）を無効化しない
2. **ベースライン期間の分離**: 事前分布・基準相関 `c_full` は 2010–2014 固定（`compute_baseline_correlation`）。バックテスト `start_date` は 2015-01-05 以降を維持。`_prepare_residual_prior` のフォールバック（先頭1260行）が発動する構成は作らない
3. **テストを弱めない**: 変更後必ず全テストを通す。推奨は並列実行 `bash scripts/run_tests_parallel.sh`（約8分、ログは `/tmp/pytest_parallel/`）。直列 `python3 -m pytest tests/ -v` は約32分。unit + integration（`test_leakage_audit.py`, `test_production_residual_blpx.py` 等）
4. **市場中立制約**: net exposure ±0.05、gross ≤ 2.0（RuleD 適用後）。リスク正本は `src/leadlag/core/risk.py`、グロス調整正本は `src/leadlag/core/portfolio.py::adjust_gross_exposure()`
5. **ティッカー定義**: `src/leadlag/data/tickers.py` が単一正本（N_U=15, N_J=17, 計32次元）で、感応度ラベル `w3`–`w6` も `SENSITIVITY_LABELS` として同ファイルに保持。`core/correlation.py` はレジストリから生成するため、ユニバース変更時は `tickers.py` のみ更新すればよい
6. **前日gap行列の使用禁止**: 前日行列をコピーして当日日付で使用してはならない（誤ったポジションで発注するリスクがある）。`load_gap_matrices`（`production_v2.py`）は当日日付のファイルのみを検索し、シェルスクリプト側のフォールバックコピー（2026-07-14に廃止）に依存しない。当日のgap行列が存在しない場合の挙動は `fallback.ondemand_fallback_enabled` で制御する：
   - `true`（本番推奨）: まず on-demand BLPX 計算を試みる。失敗/無効時は **フラットポジション（w_final=0）** を返す。
   - `false`: **フラットポジション（w_final=0）** を返す。
   - さらに `shadow_ondemand_validation=true` とすると、file cache 読み込み時に on-demand 計算と shadow 比較を行い、差異が >1% の場合に警告を発する。

## 改善ワークフロー

1. **仮説→実験**: 実験スクリプトは `src/research/scripts/experiments/` に作成（本番パス `src/leadlag/` に直接実験コードを入れない）。実験用モジュールは `src/research/experiments/` へ
2. **完了報告前のロードマップ確認**: `docs/refactor_roadmap.md` および未完了 ADR を開き、対象 Phase の未チェックタスク（`[ ]`）が残っていないか `grep -n "\[ \]" docs/refactor_roadmap.md` で確認。テスト通過だけでなく、ドキュメント・設定・運用手順の整合をもって初めて「完了」と報告する
3. **バックテスト・日次決済**: 本番 V2 は `BacktestEngine.run_v2_backtest()`（`src/leadlag/execution/backtester.py`）を使用。CLI `backtest` / `decision` は V2 一本化。V1 SRE モデル `SectorRelativeEnsembleModel` は `archive/legacy_src/models/sre.py` に移設。レガシー V1 汎用 `BaseModel` バックテストは `research.backtest_v1.run_v1_backtest()` に移設。コストは片道5bps + 金利・貸株・逆日歩を含む **net** で評価
3. **過学習ガード（必須）**:
   - このリポジトリには過去の実験config・スクリプトが大量にあり（`archive/experiments/` 約30本）、同一ヒストリー上での反復選択が既に多い。**新パラメータ追加は原則避け、追加時はパラメータ±摂動の感度分析と Deflated Sharpe（試行回数補正）を必ずレポートに含める**
   - ウォークフォワード検証（先例: `reports/phase3_walkforward_validation_report.md`）で OOS 確認
4. **シャドー運用**: 昇格前に `tools/validation/monitor_residual_blpx_shadow_performance.py` / `shadow_runs/` でライブ整合を確認
5. **本番昇格**: `configs/production/` の config 更新 + `docs/ARCHITECTURE.md` のリファクタリング履歴へ追記
6. **レポート**: `reports/<sprint名>/` に markdown で結果を残す（既存 sprint0–3b の形式に倣う）

## 既知の落とし穴（コードレビュー指摘済み）

- **`sre.py` のインスタンス状態一時書き換え（解決済み）**: `build_c0_from_v0` グローバル差し替えは `c0_override` 引数経由に修正済み。`self.k` の一時書き換えも `k_override` 引数経由に修正済み（2026-07-13）
- **グローバルキャッシュ（解決済み）**: `_PRODUCTION_SIGNAL_CACHE` 等のモジュールレベルdictはインスタンス属性 (`self._production_signal_cache` 等) に移行済み（2026-07-13）。`predict_signals` 開始時にインスタンスキャッシュは clear される。`core/correlation.py` の `_ROLLING_CORR_CACHE` / `_BASELINE_CORR_CACHE` は関数レベルで管理されサイズ上限あり
- **9:10 価格近似**: 5分足 09:10 バーの (High+Low)/2 を執行価格としており楽観側。コスト検証時は実約定ログと突合すること
- **金利コスト日割り（解決済み）**: `backtester.py` は `calendar_days` で暦日数を計算し、`financing_daily * days_held` で課金。週末（金曜→月曜=3日）も正しく加算される
- **VaR99 の不安定性**: 250日窓の99%は尾部標本 ~2.5個。stop 判定の変更時は注意
- **ハング既知パターン**（CLI実行時）: yfinance ダウンロード、`cache.py` の fcntl ファイルロック、`close.py` の auto-close 無限待機、API再試行バックオフ。詳細は `docs/スタック再発防止策.md`。長時間実行はタイムアウト付きで
- **yfinanceのティッカー別NaN欠損（修正済み）**: yfinanceダウンロード時に特定ティッカー（IJR等）のデータが欠損することがある。`preprocess_data()` は US/JP それぞれで 50% 以上のティッカーに有効値がある日を `joint_dates` とし、欠損があっても side（US/JP）中央値で補間する。ただし 50% 未満の有効ティッカー数の日はスキップされる。`etf_data.pkl` の異常は `preprocess_data` 呼び出し前に検査・修正すること
- **config dictのshallow copy**: `base_cfg.copy()` はネストした dict（`cfg["blpx"]` 等）を共有参照する。比較実験で2つのモデルに異なるconfigを渡す際は `copy.deepcopy(base_cfg)` を使うこと。shallow copy だと一方の変更が他方に伝播し、両モデルが同一設定になる（実例: Robust PCA 比較実験で両モデルが Robust PCA 有効化されシグナルが完全一致した）

## よく使うコマンド

```bash
# テスト（並列・推奨、約8分）
bash scripts/run_tests_parallel.sh

# テスト（直列、約32分）
python3 -m pytest tests/ -v

# 日次本番実行（v2）
python3 -m leadlag.cli decision --trade-date latest --api-enable

# gap調整分布の事前計算（v2 の入力）
# on-demand フォールバックがあるため必須ではなくなったが、計算時間短縮のため朝 9:10 前に推奨
python3 tools/research/compute_gap_adjusted_distribution.py

# 本番 V2 バックテスト（推奨）
python3 -m leadlag.cli backtest --config configs/production/production.yaml --start-date 2015-01-05 --gap-dir var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite

# 本番 V2 バックテスト（gap 行列が事前計算済みの場合は --gap-dir 指定）
# 注: src/research/scripts/backtest/run_production_backtest.py は 2026-08 以降 deprecated
#     2026-08 のリファクタリングで BacktestEngine.run_v2_backtest / CLI `backtest` へ一本化

# CLI経由 V2 本番決済（--config / --gap-dir / --live-dir / --api-enable 等）
python3 -m leadlag.cli decision --config configs/production/production.yaml --gap-dir var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite --api-enable --capital-from-wallet

# 構文チェック（CLIスタック防止: python3 -c は使わずスクリプト経由で）
python3 -m compileall src/leadlag tests tools scripts src/research
```

## 評価指標の約束事

- 主指標: **net Sharpe**（コスト後）、最大DD、ターンオーバー、フォールバック発動率
- gross/net 両方を報告し、コスト内訳（slippage / financing / borrow / reverse）を分解
- 「Sharpe改善なし」の結論も価値がある（例: Health Score によるサイズ調整は検証の結果不採用、`docs/ARCHITECTURE.md` Phase 9 参照）。不採用の実験も必ずレポート化して二重検証を防ぐ
- 不採用実験の記録は `docs/experiment_graveyard.md` を参照（過去の検証済み未採用案は同ファイルに分離済み）。
