# 日米リードラグ・ファンド改善ガイド

## 戦略概要

米国セクターETF（15銘柄: SPDR 11 + Style 4）の当日リターンから、翌営業日の日本 TOPIX-17 セクターETF（17銘柄）の **9:10→大引けリターン** を予測する日次マーケットニュートラル戦略。

- **本番モデル**: `ProductionV2Model` (Residual-BLPX-RA v2) — `src/leadlag/models/production_v2.py`
  - BLPX 構造化投影シグナル + gap調整予測分布 + `mu_over_sigma` ランキング + RuleD 動的グロス（PIT三分位: Low→0.75x, Mid/High→1.00x）
- **フォールバック**: gapデータ欠損時・監査失敗時は **フラットポジション（w_final=0）** を返す（V1フォールバックは2026-07に廃止）
- **本番config**: `configs/production/production.yaml`（正本）。`production_v2_primary_ruleD.yaml` は旧版（overnight holding・multi-horizon blend・rank reversal overlay 未含む）
- **アーキテクチャ詳細**: `docs/ARCHITECTURE.md`、数理仕様: `docs/モデル技術仕様書.md`

## データ整合規約（最重要・変更禁止）

`df_exec` の行 t の意味:
- **US列 (`us_cc_*`)**: 米国営業日 D_t のクローズ・トゥ・クローズリターン（JST 翌朝に確定）
- **JP列（ターゲット）**: 取引日 D_{t+1} の 9:10→大引けリターン（`compute_jp_target_returns` in `src/leadlag/models/sre.py`）
- **`jp_gap_*`**: 取引日の寄付ギャップ（9:10 判定時点で既知 → シグナルに使用可）
- 相関窓は `all_returns[window_start:current_index]` で **当日行を除外**（`src/leadlag/core/signal.py`）。この規約を崩すと即リークになる

## 不変条件（改善時に絶対に守ること）

1. **ルックアヘッド禁止**: すべてのローリング統計・ベータ・PITビニングは strictly historical。`ComplianceAuditor`（`src/leadlag/compliance/auditor.py`, `v2_auditor.py`）の監査項目（`check_pit_binning_lookahead`, `check_residualization_leakage` 等）を無効化しない
2. **ベースライン期間の分離**: 事前分布・基準相関 `c_full` は 2010–2014 固定（`compute_baseline_correlation`）。バックテスト `start_date` は 2015-01-05 以降を維持。`_prepare_residual_prior` のフォールバック（先頭1260行）が発動する構成は作らない
3. **テストを弱めない**: 変更後必ず全テストを通す。推奨は並列実行 `bash scripts/run_tests_parallel.sh`（約8分、ログは `/tmp/pytest_parallel/`）。直列 `python3 -m pytest tests/ -v` は約32分。unit + integration（`test_leakage_audit.py`, `test_production_residual_blpx.py` 等）
4. **市場中立制約**: net exposure ±0.05、gross ≤ 2.0（RuleD 適用後）。リスク正本は `src/leadlag/core/risk.py`、グロス調整正本は `src/leadlag/core/portfolio.py::adjust_gross_exposure()`
5. **ティッカー定義**: `src/leadlag/data/tickers.py` が単一正本（N_U=15, N_J=17, 計32次元）。`core/correlation.py` の感応度ラベル `w3`–`w6` は32次元ハードコードなので、ユニバース変更時は必ず同時更新

## 改善ワークフロー

1. **仮説→実験**: 実験スクリプトは `scripts/experiments/` に作成（本番パス `src/leadlag/` に直接実験コードを入れない）。実験用モジュールは `src/experiments/` へ
2. **バックテスト**: `BacktestEngine.run_backtest()`（`src/leadlag/execution/backtester.py`）を使用。CLI: `src/leadlag/cli.py`（subcommands: decision / backtest / close）。コストは片道5bps + 金利・貸株・逆日歩を含む **net** で評価
3. **過学習ガード（必須）**:
   - このリポジトリには過去の実験config・スクリプトが大量にあり（`archive/experiments/` 約30本）、同一ヒストリー上での反復選択が既に多い。**新パラメータ追加は原則避け、追加時はパラメータ±摂動の感度分析と Deflated Sharpe（試行回数補正）を必ずレポートに含める**
   - ウォークフォワード検証（先例: `reports/phase3_walkforward_validation_report.md`）で OOS 確認
4. **シャドー運用**: 昇格前に `tools/validation/monitor_residual_blpx_shadow_performance.py` / `shadow_runs/` でライブ整合を確認
5. **本番昇格**: `configs/production/` の config 更新 + `docs/ARCHITECTURE.md` のリファクタリング履歴へ追記
6. **レポート**: `reports/<sprint名>/` に markdown で結果を残す（既存 sprint0–3b の形式に倣う）

## 既知の落とし穴（コードレビュー指摘済み）

- **`sre.py` のモンキーパッチ**: `predict_signals` 内で `signals.build_c0_from_v0` をグローバル差し替えしている（P4シグナル計算）。スレッド非安全。触る場合は `compute_signal` に `c0_override` 引数を追加して除去する方向で
- **グローバルキャッシュ**: `_PRODUCTION_SIGNAL_CACHE` 等のキーにデータ識別子が無い。`predict_signals` 経由以外で `compute_production_signal` を呼ぶと汚染リスク。`_COMMON_INPUTS_CACHE` は clear されないためメモリ単調増加
- **9:10 価格近似**: 5分足 09:10 バーの (High+Low)/2 を執行価格としており楽観側。コスト検証時は実約定ログと突合すること
- **金利コスト日割り**: `backtester.py` は `annual/365` を営業日課金 → 週末分が過小。オーバーナイト保有の実験では暦日補正を検討
- **VaR99 の不安定性**: 250日窓の99%は尾部標本 ~2.5個。stop 判定の変更時は注意
- **ハング既知パターン**（CLI実行時）: yfinance ダウンロード、`cache.py` の fcntl ファイルロック、`close.py` の auto-close 無限待機、API再試行バックオフ。詳細は `docs/スタック再発防止策.md`。長時間実行はタイムアウト付きで
- **yfinanceのティッカー別NaN欠損**: yfinanceダウンロード時に特定ティッカー（IJR等）のデータが日付以降全てNaNになることがある。`preprocess_data()` のNaNチェック（`preprocessor.py:264-272`）で1ティッカーでもNaNがあると該当日の全レコードがスキップされ、df_execが途中で切断される。`etf_data.pkl` の異常は `preprocess_data` 呼び出し前に検査・修正すること
- **config dictのshallow copy**: `base_cfg.copy()` はネストした dict（`cfg["blpx"]` 等）を共有参照する。比較実験で2つのモデルに異なるconfigを渡す際は `copy.deepcopy(base_cfg)` を使うこと。shallow copy だと一方の変更が他方に伝播し、両モデルが同一設定になる（実例: Robust PCA 比較実験で両モデルが Robust PCA 有効化されシグナルが完全一致した）

## よく使うコマンド

```bash
# テスト（並列・推奨、約8分）
bash scripts/run_tests_parallel.sh

# テスト（直列、約32分）
python3 -m pytest tests/ -v

# 日次本番実行（v2）
python3 tools/production/run_daily_production_v2.py

# gap調整分布の事前計算（v2 の入力）
python3 tools/production/compute_gap_adjusted_distribution.py

# 本番バックテスト（対応引数: --config / --start-date / --output-dir のみ）
python3 src/research/scripts/backtest/run_production_backtest.py

# CLI経由バックテスト（--slippage-bps 等の追加引数はこちら）
python3 -m leadlag.cli backtest --start-date 2015-01-05

# 構文チェック（CLIスタック防止: python3 -c は使わずスクリプト経由で）
python3 _check_syntax.py
```

## 評価指標の約束事

- 主指標: **net Sharpe**（コスト後）、最大DD、ターンオーバー、フォールバック発動率
- gross/net 両方を報告し、コスト内訳（slippage / financing / borrow / reverse）を分解
- 「Sharpe改善なし」の結論も価値がある（例: Health Score によるサイズ調整は検証の結果不採用、`docs/ARCHITECTURE.md` Phase 9 参照）。不採用の実験も必ずレポート化して二重検証を防ぐ
- **不採用実験の記録**（再検証防止用）:
  - **Robust PCA伝播行列**（2026-07）: B_struct を低ランク+スパース分解（L+S）で置換する方針を検証。セクター事前知識（M_sector）とPCA事前分布（B_pca）の統合が失われ、confidence weighting の inv_A_tikh も単位行列フォールバックになった結果、Sharpe -35%、IC -32%と大幅劣化。チューニングでは埋められない構造的欠陥が原因。コードは全て破棄済み
  - **V1フォールバック廃止**（2026-07）: gapデータ欠損時のV1ウェイトフォールバックを廃止しフラットポジション化。理由: (1) `production_v2_writer.py` がV2実行のたびに `v1_baseline_weights.csv` を `w_v1` で上書きする循環参照があり、V1ウェイトが新規計算されず凍結化していた (2) 一度ゼロになると永久にゼロになる（実例あり） (3) データパイプライン障害時に古いシグナルで取引するより取引を見送る方がリスク管理として健全
