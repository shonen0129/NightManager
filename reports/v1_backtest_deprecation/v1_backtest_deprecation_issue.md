# [Refactor] V1 バックテスト（`BacktestEngine.run_backtest`）の段階的廃止・V2 一本化

## 概要

本番運用は V2（`run_v2_backtest` / `generate_v2_production_portfolio`）に完全移行済みですが、`BacktestEngine.run_backtest`（V1 汎用型バックテスト）がまだ CLI backtest、VaR/ES 履歴、多数の研究スクリプト、テストに張り付いています。これにより以下の問題が生じています。

- CLI `backtest` と本番 V2 の結果が一致しない（`build_strategy()` が `SectorRelativeEnsembleModel` を返すため、BLPX-V2 ではなく PCA-Ensemble V1 を実行している）。
- `fast.py` / `data/cache.py` / `v2_bridge.py` の VaR/ES 履歴再構築が V1 リターン列を使っており、本番 V2 リスクと乖離する可能性がある。
- Phase 22 のリファクタリングで V1/V2 のコスト計算は共通化されたものの、V1 専用のシグナル/ウェイト生成・結果アセンブリが残っており、保守コストが続いている。

## ゴール

`BacktestEngine.run_backtest` を段階的に廃止し、**本番・研究・テストすべてのバックテストを V2 一本化**する。V1 固有の `BaseModel` 汎用バックテストインターフェースは、明確な後方互換性期間を設けた上で削除する。

## 背景

### V1 と V2 の違い

| 項目 | V1 `BacktestEngine.run_backtest` | V2 `BacktestEngine.run_v2_backtest` |
|---|---|---|
| 入口 | `src/leadlag/execution/backtester.py` 内 `run_backtest` | `src/leadlag/execution/backtester.py` 内 `run_v2_backtest` |
| モデル | `BaseModel` サブクラス（`SectorRelativeEnsembleModel` / `SectorRelativeEnsembleBLPEnhancedModel`）を受け取り `predict_signals()` → `build_weights()` | V2 config + 当日 `mu_gap` / `omega_gap` 行列を受け取り `generate_v2_production_portfolio()` |
| 機能 | 汎用シグナル生成 + 9:10→大引けバックテスト | gap 調整分布、PIT/RuleD 動的グロス、ML overlay、fallback、side_leverage=1.5 |
| 追加出力 | `raw_pca_signals`, `residual_pca_signals`, `p4_signals`, `signals`, `normalized_signals`, `daily_returns_*_oc` | `daily_fallback`, `v2_summaries`, `side_leverage` |
| コスト計算 | `_simulate_daily_pnl()` 共有 | `_simulate_daily_pnl()` 共有 |

### 廃止に向けた前提条件

- `ARCHITECTURE.md` Phase 20 では「SRE/PCA 汎用パス（`run_backtest`）の legacy フォールバックは既存テストとの互換性のため維持」と明記済み。Phase 22 では `run_backtest` / `run_v2_backtest` をヘルパーに分割し、共通化を進めている。
- `AGENTS.md` では「V1 フォールバックは 2026-07 に廃止」とされているが、バックテストとして `BacktestEngine.run_backtest()` が依然として記載されている。

## 依存箇所

### 1. 本番周辺

- `src/leadlag/execution/backtest.py::run_production` — CLI `backtest` サブコマンド。`build_strategy()` 経由で `SectorRelativeEnsembleModel` を使っている。
- `src/leadlag/execution/fast.py` — fast モード時の VaR/ES 履歴再構築。
- `src/leadlag/data/cache.py::get_hist_returns_for_risk` — VaR/ES 用履歴リターン取得。
- `src/leadlag/execution/v2_bridge.py` — V2 発注フロー内の VaR/ES 履歴再構築。
- `src/leadlag/execution/decision.py::run_decision` — 標準 decision（非 fast）も V1 モデル経由。

### 2. 研究・実験スクリプト

- `src/research/backtest_common.py::run_baseline_backtest` / `run_backtest_with_costs` — 多数の研究スクリプトの基盤。
- `src/research/scripts/backtest/run_production_backtest.py`
- `src/research/scripts/blpx/compare_sensitivity_matrix.py`
- `src/research/scripts/macro/sensitivity_factor_kappa.py`
- `tools/research/backtest_sector_relative_ensemble.py`
- `scripts/experiments/` 内 `experiment_vix_regime_overlay.py` など 10 本以上。

### 3. テスト

- `tests/unit/test_backtester_910.py` — 5分足 09:10 価格調整の振る舞い確認。
- `tests/integration/test_production_residual_blpx.py::test_cost_consistency`
- `tests/unit/test_sector_relative_ensemble.py::test_pipeline_completeness_and_daily_run`

### 4. アーカイブ（優先度低）

- `git tag archive-2026-08` の `archive/tools/compute_structured_prediction_covariance.py`
- `git tag archive-2026-08` の `archive/tools/backtest_blp_enhanced.py`
- `git tag archive-2026-08` の `archive/tools/backtest_blp_projection.py`
- `git tag archive-2026-08` の `archive/tools/backtest_rrr_projection.py`
- `git tag archive-2026-08` の `archive/experiments/tests/integration/test_sector_relative_ensemble_`*.py`

## 段階的移行計画

### Phase A: 本番周辺の V2 一本化（優先度：高）

1. **CLI `backtest` の V2 化**
   - `src/leadlag/execution/backtest.py::run_production` を `BacktestEngine.run_v2_backtest` 呼び出しに変更。
   - `src/leadlag/cli.py::_handle_backtest` の引数を `run_v2_backtest` に対応（`--gap-dir` 等）。
   - 副作用：CLI backtest と本番 V2 のバックテスト結果が一致する。

2. **VaR/ES 履歴の V2 化**
   - `src/leadlag/execution/fast.py`
   - `src/leadlag/data/cache.py::get_hist_returns_for_risk`
   - `src/leadlag/execution/v2_bridge.py`
   - これらを `run_v2_backtest` で作成した履歴 CSV キャッシュを使うように変更。V1 リターン列からの切り替えにより VaR/ES 値が変化する可能性があるため、本番 shadow 環境で 1-2 週間の整合確認を行う。

3. **標準 decision の V2 対応 or 非推奨化**
   - `src/leadlag/execution/decision.py::run_decision` は `build_strategy`（V1）を使っている。本番は `tools/production/run_daily_production_v2.py` を使用しているため、CLI `decision` サブコマンドを V2 化するか、非推奨化して `run_daily_production_v2.py` への移行を促す。

### Phase B: 研究・実験スクリプトの移行

1. **`src/research/backtest_common.py` の V2 化**
   - `run_baseline_backtest` / `run_backtest_with_costs` を `BacktestEngine.run_v2_backtest` ベースに変更するか、V2 専用の新しい util（例：`run_baseline_v2_backtest`）を追加。
   - 既存研究スクリプトは段階的に移行 or `archive-2026-08` へ移動。

2. **BLPX/マクロ感度スクリプトの移行**
   - `src/research/scripts/blpx/compare_sensitivity_matrix.py`
   - `src/research/scripts/macro/sensitivity_factor_kappa.py`
   - これらは `SectorRelativeEnsembleBLPEnhancedModel` + `run_backtest` を使っている。V2 化する場合は gap 行列が必要なため、実験スクリプト内で `compute_gap_adjusted_distribution` を事前実行 or 既存 gap ディレクトリを指定するように変更。

3. **`scripts/experiments/` 内の整理**
   - `grep -n "BacktestEngine.run_backtest" scripts/experiments/*.py` で呼び出しを列挙し、まだ有用なものは V2 化、不要なものは `git tag archive-2026-08` の `archive/experiments/` へ移動。

### Phase C: テストの V2 化

1. **`tests/unit/test_backtester_910.py`**
   - 5分足 09:10 価格調整のテストは `models/sre.py::compute_jp_target_returns()` を直接テストする形に書き換える。`run_backtest` を経由する必要はない。

2. **`tests/integration/test_production_residual_blpx.py::test_cost_consistency`**
   - `_simulate_daily_pnl()` を合成ウェイト/リターンで直接呼び出す unit test に変更。または V2 バックテスト + モック `mu_gap`/`omega_gap` 行列を使う integration test に変更。

3. **`tests/unit/test_sector_relative_ensemble.py::test_pipeline_completeness_and_daily_run`**
   - `SectorRelativeEnsembleModel`（PCA-Ensemble）のスモークテストを V2 スモークテストに置換 or、旧 SRE モデルが `BaseModel` として動作することを確認する軽量テストに変更。

### Phase D: V1 の正式廃止

- `src/leadlag/execution/backtester.py` から以下を削除：
  - `run_backtest`
  - `_predict_signals_and_weights_run_backtest`
  - `_assemble_run_backtest_results`
  - `_resolve_run_backtest_cost_params`（V2 側と重複しないか確認）
- 同時に `BaseModel.predict_signals` / `build_weights` 利用箇所がなくなった場合、`src/leadlag/models/base.py` 等の整理も検討。
- `ARCHITECTURE.md` / `AGENTS.md` を更新し、V1 バックテストを非推奨・削除した旨を追記。

## 受け入れ条件（Acceptance Criteria）

- [ ] CLI `backtest` が V2 を実行し、本番 `run_v2_backtest` と同じ設定・結果になる。
- [ ] VaR/ES 履歴再構築が V2 リターンを使用するようになる（shadow 運用で 1-2 週間の整合確認済み）。
- [ ] `src/leadlag/` 配下で `BacktestEngine.run_backtest` の呼び出しがゼロになる。
- [ ] `tests/` 配下で V1 バックテストに依存するテストが V2 化 or 代替テストに置き換わる。
- [ ] `run_tests_parallel.sh` または `python3 -m pytest tests/ -v` がすべてパスする。
- [ ] `ARCHITECTURE.md` / `AGENTS.md` が更新されている。

## リスク

- **VaR/ES 履歴の変化**：V1 から V2 へ切り替えると過去リターン列が変わり、VaR/ES 閾値・stop/warning 判定に影響。本番 shadow で整合を確認する必要あり。
- **CLI backtest の互換性**：`--slippage-bps` などの引数は引き続き使えるようにする必要あり。`--gap-dir` 引数の追加が必要になる可能性。
- **研究スクリプトの破損**：V2 は gap 行列が必要なため、実験スクリプトの多くに追加の前処理が必要 or 既存 gap ディレクトリを前提とする運用が必要。
- **旧 SRE/PCA モデルのテスト廃止**：`SectorRelativeEnsembleModel` のバックテストスモークテストを失う。必要なら model unit test として残す。

## 備考

- Phase 22（2026-08-07）のリファクタリングにより、V1/V2 のコスト計算（`_simulate_daily_pnl`）とターゲット/gap リターン計算、シミュレーション期間決定などのヘルパーは既に共通化されている。残っている V1 固有部分はシグナル/ウェイト生成と結果アセンブルに限られるため、削除作業は比較的集中化されている。
- ただし、V1 は **任意の `BaseModel` サブクラスをバックテストする唯一の汎用インターフェース**でもある。BLPX 以外のモデル（旧 SRE/PCA 等）のバックテストを維持する場合、V2 化ではなく V1 を廃止せずに分離して `archive-2026-08` 化する案も検討。

## 関連ファイル

- `src/leadlag/execution/backtester.py`（`BacktestEngine.run_backtest`, `run_v2_backtest`, `_simulate_daily_pnl`）
- `src/leadlag/execution/backtest.py`（`run_production`）
- `src/leadlag/cli.py`（`_handle_backtest`）
- `src/leadlag/execution/fast.py`
- `src/leadlag/data/cache.py`（`get_hist_returns_for_risk`）
- `src/leadlag/execution/v2_bridge.py`
- `src/leadlag/execution/decision.py`（`run_decision`）
- `src/research/backtest_common.py`
- `src/research/scripts/backtest/run_production_backtest.py`
- `src/research/scripts/blpx/compare_sensitivity_matrix.py`
- `src/research/scripts/macro/sensitivity_factor_kappa.py`
- `tools/research/backtest_sector_relative_ensemble.py`
- `tests/unit/test_backtester_910.py`
- `tests/integration/test_production_residual_blpx.py`
- `tests/unit/test_sector_relative_ensemble.py`
- `docs/ARCHITECTURE.md`
- `AGENTS.md`
