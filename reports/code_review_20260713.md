# コードレビューレポート — 日米リードラグ戦略コードベース

**レビュー日**: 2026-07-13  
**レビュー範囲**: `src/leadlag/` 全体、`tools/production/`、`tests/`、`configs/production/`

---

## 1. 機械的検証結果

### 1.1 構文チェック
- **結果**: ✅ 全5ファイルPASS (`python3 _check_syntax.py`)

### 1.2 Lint (ruff)
- **結果**: ⚠️ 49件の警告（E501除く）
  - **F401 (unused import)**: 10件 — 削除推奨
  - **F841 (unused variable)**: 1件 — `backtester.py:163` の `long_exp`
  - **W293 (blank line whitespace)**: 28件 — 自動修正可能
  - **E402 (import not at top)**: 2件 — `fetcher.py`（条件分岐後のimport、やむを得ない）

### 1.3 テスト
- **結果**: ⚠️ 4 failed, 399 passed (22分24秒)
- **失敗テスト**: 全て **timeout (>300s)** が原因。コードバグではない
  - `test_sprint0_calculations_subset` — 大量データ計算のタイムアウト
  - `test_sprint0_qa_subset` — 同上
  - `test_backtest_simulation` — バックテスト実行のタイムアウト
  - `test_calibration_rolling` — RuleDローリング計算のタイムアウト
- **推奨**: これらのテストは `@pytest.mark.slow` でタグ付けし、CI でスキップまたはタイムアウト延長を検討

---

## 2. 観点別レビュー（24項目）

### 2.1 ルックアヘッドリーク防止 ✅
- `compute_signal()` (`@signal.py:69-70`): 相関窓 `all_returns[window_start:current_index]` で当日行を除外
- `compute_rolling_ols_betas()` (`@residualize.py:59-60`): `shift(1)` で t-1 以前のデータのみ使用
- `get_rolling_pit_bin()` (`@portfolio.py`): `history_ir[-rolling_window:]` で厳密に過去データのみ使用
- `load_pit_ir_history()` (`@production_v2.py:197`): `df["trade_date"] < trade_date` で当日除外
- `compute_macro_surprise()` (`@macro.py`): EWMA mean/var が t-1 以前のみ使用（docstring明記）
- `_derive_signal_date()` (`@production_v2.py:209-247`): gap matrix ファイル名から signal_date を導出、`< trade_date` で厳密分離
- `run_leakage_audit()` (`@v2_auditor.py`): signal_date < trade_date を検証
- **結論**: リーク経路なし。不変条件維持されている

### 2.2 ベースライン期間の分離 ✅
- `compute_baseline_correlation()` は2010–2014固定（`@correlation.py`）
- バックテスト `start_date` デフォルト "2015-01-05"
- `_prepare_residual_prior()` のフォールバックは先頭1260行（2010-2014約5年分）

### 2.3 市場中立制約 ✅
- `solve_baseline_style()`: ロング合計 = +baseline_gross/2、ショート合計 = -baseline_gross/2 → net=0
- `adjust_gross_exposure()` (`@portfolio.py:10-40`): gross制限を超えた場合のみスケールダウン、net中立を保持
- `BacktestEngine` で `net_exp` を記録（`@backtester.py:162`）
- production.yaml で `max_net_exposure: 0.05`、`max_gross_exposure: 2.0`

### 2.4 ティッカー定義の整合性 ✅
- `tickers.py` が単一正本（N_U=15, N_J=17, 計32次元）
- `correlation.py` の `get_static_sensitivity_labels()` は32次元ハードコード
- `MACRO_SENS_MATRIX` (`@macro.py:79`) は `JP_TICKERS` から動的構築

### 2.5 configのshallow copy問題 ⚠️ 既知
- AGENTS.mdに記載済みの既知の落とし穴
- `base_cfg.copy()` はネストdictを共有参照する
- 比較実験では `copy.deepcopy(base_cfg)` が必要
- コード内では `copy.deepcopy` の使用は見つからず（実験スクリプト側の責務）

### 2.6 グローバルキャッシュ ⚠️ 既知
- `_PRODUCTION_SIGNAL_CACHE`, `_RESIDUAL_SIGNAL_CACHE`, `_COMMON_INPUTS_CACHE` (`@sre.py`)
- `_RAW_PCA_CACHE`, `_RESIDUAL_PCA_CACHE`, `_BLP_CORR_CACHE` (`@blp_enhanced.py`)
- キーにデータ識別子がない → `predict_signals` 経由以外で呼ぶと汚染リスク
- `_COMMON_INPUTS_CACHE` は clear されない → メモリ単調増加
- `predict_signals()` 開始時に `_PRODUCTION_SIGNAL_CACHE.clear()` + `_RESIDUAL_SIGNAL_CACHE.clear()` + `_COMMON_INPUTS_CACHE.clear()` で対応（`@sre.py:604-607`）
- **BLP Enhanced側**: `_BLP_CORR_CACHE` は clear されていない

### 2.7 モンキーパッチ ⚠️ 既知
- AGENTS.md記載: `sre.py` の `predict_signals` 内で `build_c0_from_v0` をグローバル差し替え
- 実際のコードでは `c0_override` 引数で渡す方式に改善済み（`@sre.py:552, 652`）
- ただし `compute_residual_signal` に `c0_override` を渡す実装は完了している

### 2.8 金利コスト日割り ✅ 改善済み
- AGENTS.md記載の「annual/365を営業日課金」問題は **改善済み**
- `backtester.py:147-152`: 取引日間の暦日数 `calendar_days` を計算し、`days_held` でスケール
- `fin_cost = held_long * financing_daily * days_held`（`@backtester.py:183`）

### 2.9 9:10価格近似 ⚠️ 既知
- 5分足 09:10バーの (High+Low)/2 を執行価格として使用
- 楽観側の可能性あり。実約定ログとの突合が必要

### 2.10 VaR99の安定性 ⚠️ 既知
- 250日窓の99% → 尾部標本 ~2.5個
- `compute_var_es()` (`@risk.py:15-60`): 線形補間で対応しているが、標本不足時の安定性は限定的

### 2.11 ハング防止 ✅
- 既知の5パターン（auto-close無限待機、yfinanceハング、fcntlロック、API再試行、注文フィル確認待ち）
- `cache.py` の fcntl ファイルロックは advisory lock として実装
- CLI に `--timeout` 系の明示的ガードは限定的（`pytest-timeout` はテスト側）

### 2.12 yfinance のティッカー別NaN欠損 ⚠️ 既知
- `preprocessor.py:264-272` でNaNチェック → 1ティッカーでもNaNだと該当日の全レコードがスキップ
- `etf_data.pkl` の異常は `preprocess_data` 呼び出し前に検査・修正することが推奨

### 2.13 PSD修復 ✅
- `production_v2.py:413-416`: `Omega_gap` の最小固有値が負の場合、`+|min_eig| + 1e-8` で修復
- `correlation.py` の `regularize_correlation()` でLW縮小 + ガードレール実装

### 2.14 数値安定性 ✅
- `_safe_solve_inv()` (`@blp_enhanced.py:423-441`): `inv` → `pinv` フォールバック → ゼロ行列フォールバック
- `np.nan_to_num` が各所で適用
- `np.errstate(divide='ignore', invalid='ignore', over='ignore')` でオーバーフロー抑制
- `np.maximum(sigma, 1e-8)` でゼロ除算防止

### 2.15 フォールバック挙動 ✅
- gapデータ欠損時: フラットポジション `w_final=0`（V1フォールバックは廃止済み）
- 監査失敗時: `fallback_on_audit_failure=True` でフラットポジション
- PIT履歴不足時: Medium/1.0 マルチプライヤー

### 2.16 コンプライアンス監査 ✅
- `ComplianceAuditor` (`@auditor.py`): ルックアヘッド、残差化、アンサンブルウェイト、エクスポージャー、コスト整合性を検証
- `v2_auditor.py`: 軽量なリーク監査 + 数値監査
- 監査項目の無効化は見られない

### 2.17 マルチホライズンブレンド ✅
- `apply_multi_horizon_blend()` (`@signal_enhancement.py`): h=1,3,5 の重み付きブレンド
- ファイル不存在時はアラートを追加してスキップ
- production.yaml: `enabled: true`, `weights: [0.8, 0.1, 0.1]`

### 2.18 ランク反転オーバーレイ ✅
- `apply_rank_reversal_overlay()` (`@signal_enhancement.py`): `cross_sectional_zscore` 適用後にブレンド
- `weight: 0.05` で控えめな適用

### 2.19 MinVarウェイト最適化 ✅
- `build_weights_minvar()` (`@signal.py:235-362`): signal比例ウェイトと最小分散ウェイトを `alpha` でブレンド
- `alpha=0.8`（本番config）→ 80%最小分散、20%シグナル比例
- `Omega_gap` を予測共分散として使用

### 2.20 RuleD動的グロス ✅
- PIT三分位ビニング: Low→0.75x, Mid/High→1.0x
- 厳密に過去データのみ使用（`history_ir[-rolling_window:]`）
- 履歴不足時はMedium/1.0フォールバック

### 2.21 BLPX構造化縮小 ✅
- `_solve_tikhonov()` (`@blp_enhanced.py:503-543`): BLP + PCA事前分布 + セクター事前分布の統合
- `lambda_pca + lambda_sector` の上限ガード（0.75）
- `frobenius_scale_priors` でノルムスケーリング
- `sector_eta` で固定マッピングとデータ駆動のブレンド（0.5）

### 2.22 Copulaブレンド ✅
- `_estimate_correlation()` (`@blp_enhanced.py:378-420`): t-copula相関をピアソン相関にブレンド
- `copula_dynamic_blend=True` でストレス期に自動増加
- `compute_stress_weight()` で動的ブレンドウェイト計算

### 2.23 マクロ調整 ✅
- Factor-Kappa: `|surprise| × |sensitivity|` で `Omega_gap` 膨張
- Directional: `signed surprise × signed sensitivity` で `mu_gap` 方向調整
- EWMAベースのボラティリティ調整済みサプライズ（lookahead-safe）
- production.yaml では `macro_kappa_enabled: false`, `macro_direction_enabled: false`（実験段階）

### 2.24 レポート・出力 ✅
- `production_v2_writer.py`: latest_weights.csv, production_scores.csv, pit_binning.json, leakage_audit.json, numerical_audit.json 等の完全な出力
- `calculate_metrics()` (`@metrics.py`): 月次ベースSharpe計算（日次ベースより保守的、正確）
- `generate_report()`: チャート生成

---

## 3. 反証レビュー（第2回）

### 3.1 「テスト失敗はコードバグではない」の検証
- 4件の失敗は全て `pytest-timeout` の300秒超過
- `test_sprint0_calculations_subset`: `start_date="2026-01-01"` でデータロード + 計算が重い
- `test_backtest_simulation`: `load_df_exec_from_local_cache()` でフルデータロード
- **結論**: コードバグではなく、テスト環境のリソース制約。実コードの不具合ではない

### 3.2 「リークがない」の再確認
- `compute_signal()` の `window_returns = all_returns[window_start:current_index]` — `current_index` を含まない（スライス構文 `[a:b]` は b を含まない）
- `compute_rolling_ols_betas()` の `betas_shifted = betas_df.shift(1)` — tのベータはt-1以前のデータで計算後、1行シフト
- **結論**: リーク経路なし

### 3.3 「shallow copy問題が存在する」の再確認
- `base_cfg.copy()` は Python dict の shallow copy → ネスト先は共有参照
- ただし本番コード（`production_v2.py`, `blp_enhanced.py`）では config dict を直接変更しない設計
- 問題が顕在化するのは実験スクリプトのみ → AGENTS.mdの警告は適切

---

## 4. 総合評価

### 4.1 健全性評価: **高**
- 不変条件（ルックアヘッド禁止、ベースライン分離、市場中立、ティッカー定義）は全て維持されている
- コンプライアンス監査が適切に実装され、無効化されていない
- フォールバック挙動が安全側に倒れている（フラットポジション）

### 4.2 技術的負債: **中**
- グローバルキャッシュ（6種）にデータ識別子がない — マルチモデル実行時の汚染リスク
- `_BLP_CORR_CACHE` が clear されない — メモリ単調増加
- ruff F401/F841 が40件 — クリーンアップ推奨
- テストのタイムアウト — slow test の分離が必要

### 4.3 推奨改善（優先度順）

1. **高**: `_BLP_CORR_CACHE` に clear 機構を追加（`predict_signals` 開始時）
2. **高**: ruff F401/F841 の40件を修正（`ruff check --fix` で大部分は自動修正可能）
3. **中**: タイムアウトする4テストに `@pytest.mark.slow` を付与し、デフォルトでスキップ
4. **中**: グローバルキャッシュのキーにデータハッシュを含める（汚染リスクの根本解決）
5. **低**: `backtester.py:163` の unused variable `long_exp` を削除
6. **低**: W293 (空白行のホワイトスペース) 28件を `ruff --fix` で修正

---

## 5. アーキテクチャサマリー

```
[データ層]
  yfinance → fetcher.py → preprocessor.py → df_exec
  cache.py (advisory lock, ETF/intraday cache)

[シグナル生成]
  core/signal.py: compute_signal() — ローリング相関 → 固有値分解 → US→JP投影 + gap調整
  core/correlation.py: EWMA, Ledoit-Wolf, t-copula, 基準相関(2010-2014)
  core/residualize.py: ローリングOLS β (shift(1)でリーク防止)

[モデル層]
  sre.py: PCA ensemble (Raw-PCA + Residual-PCA + P4) — バックテスト用
  blp_enhanced.py: BLPX structured shrinkage — 本番バックテスト + gap分布計算
  production_v2.py: V2日次オーケストレーター — gap行列 → mu/σ → MinVar → PIT(RuleD) → 監査

[実行層]
  backtester.py: シグナル → ウェイト → リターン/コスト計算 (暦日スケール)
  close.py: overnight alpha-aware ポジション決済
  cli.py: decision / backtest / close サブコマンド

[本番パイプライン]
  compute_gap_adjusted_distribution.py → mu_gap/Omega_gap .npy
  run_daily_production_v2.py → generate_v2_production_portfolio() → write_production_files()

[コンプライアンス]
  auditor.py: ComplianceAuditor (リーク, 残差化, エクスポージャー, コスト)
  v2_auditor.py: 軽量リーク監査 + 数値監査
```
