# Lead-Lag Market-Neutral Strategy — Architecture (v3.0)

> **最終更新**: 2026-08-10

## Overview

US ETF と TOPIX-17 セクター ETF のリードラグ相関を利用した、
日次マーケットニュートラル戦略のプロダクションシステム。

本番モデルは **Production Residual-BLPX-RA v2** （予測期待値を予測標準偏差で割ったリスク調整スコア $\mu_{\text{gap}} / \sigma_{\text{gap}}$ による銘柄選択と、予測 ex-ante IR の過去履歴に基づく動的グロス調整 RuleD を採用した、ギャップ調整予測分布ベースの最適化モデル）。
旧本番の **Sector Relative Ensemble (PCA-Ensemble)** はベンチマーク用として維持される。

**注意**: v1 fallback (Residual-BLPX) は2026-07-09に廃止されました。gap data欠損時はflat position (w_final=0) を返します。廃止理由は、v2でエラーが出る場合v1でも同様にエラーが出るため、循環依存の問題があったためです。v1 fallback関連コードは `git tag archive-2026-08` の `archive/deprecated_v1_fallback/` にアーカイブされています。

### リファクタリング履歴

- **Phase 1**: ブローカー抽象化レイヤー (`broker/`) 導入
- **Phase 2**: データ層の分割 (`data/` パッケージ + `pyproject.toml`)
- **Phase 3**: `production.py` の分解 (`runner/` サブパッケージ)
- **Phase 4**: ユニットテストスイート (`tests/`) + 本ドキュメント更新
- **Phase 5**: 設定定義のPydantic移行、モデル層・実行層・安全監査の完全デカップリング（BaseModel導入による純粋化、BacktestEngine / ComplianceAuditor への役割分担）
- **Phase 6**: 計算ボトルネックの高速化最適化（`compute_us_residualized_returns` のベクトル化、基準相関行列 `c_full` 及び残差空間事前分布 `_prepare_residual_prior` のメモリキャッシュ化による高速化）
- **Phase 7 (2026-06-15)**: 本番 v2 モデル（Residual-BLPX-RA v2）への昇格に伴い、ギャップ調整予測分布計算、リスク調整ランキング（`mu_over_sigma`）、PIT ビニングに基づく動的グロス制御（`RuleD`）の導入、および v2 → v1 → PCA-Ensemble の多段階自動フォールバック機能の実装。
- **Phase 15 (2026-07-09)**: v1 fallback (Residual-BLPX) の廃止。循環依存問題によりv2でエラーが出る場合v1でも同様にエラーが出るため、gap data欠損時はflat position (w_final=0) を返すように変更。v1 fallback関連コードを `git tag archive-2026-08` の `archive/deprecated_v1_fallback/` にアーカイブ。
- **Phase 8 (2026-06-17)**: AIMA/IOSCO モデルリスクガイドラインに準拠するため、文書体系を再編。運用方針書から詳細なアルゴリズム数理・システムパラメータ・日次実行コマンド等を分離し、別冊の《モデル技術仕様書》および《日次運用手順書》へ移行。
- **Phase 9 (2026-06-25)**: `monitoring/` 層（HealthScoreCalculator）をアーキテクチャ文書に正式反映。Health Score によるポジションサイズ動的調整のバックテスト検証結果（Sharpe改善なし）を踏まえ、常にフルポジションでの運用を決定。Health Score は記録・監視用のみ。
- **Phase 10 (2026-06-27)**: モデル層の継承階層をリファクタリング。`BaseModel` に共通ユーティリティメソッド（`_resolve_val`, `_resolve_nested`, `normalize_signals`, `build_weights`, `_resolve_slippage_bps`）を集約し、新規中間クラス `_BLPBase`（`blp_base.py`）に BLP 系モデル共通メソッド（`_prepare_common_inputs`, `compute_production_signal`, `compute_residual_signal`, `_denormalize_signal`, `_apply_gap_adjustment`）を集約。`SectorRelativeEnsembleBLPEnhancedModel`, `SectorRelativeEnsembleBLPModel`, `SectorRelativeEnsembleRRRModel` は `_BLPBase` を継承するよう変更。`compute_blp_signal` を7つのヘルパーメソッドに分割し、実験スクリプトの重複モデル定義を `scripts/experiment_models.py` に共通化。
- **Phase 11 (2026-07-01)**: アーキテクチャ文書を実態に合わせて全面更新。未記載だった `src/features/`、`src/models/`、`src/reports/` 実験パッケージ、`leadlag/cost/`、`leadlag/diagnostics/` サブパッケージ、`execution/` のLOB/スリッページ関連5ファイル、`compliance/v2_auditor.py`、`core/market_calendar.py`、`models/signal_enhancement.py`、`models/production_v2.py`、`models/net_score_ranking_lob.py`、`reporting/production_v2_writer.py`、`reporting/sprint2c_lob_report.py` を文書に反映。Repository Root に `Papers/`、`artifacts/`、`reports/`、`kabu_auto_login/`、`scratch/`、`archive-2026-08`、`live/`、`logs/`、`shadow_runs/`、`data/`、`creds/` 等の未記載ディレクトリを追加。
- **Phase 12 (2026-07-01)**: ディレクトリ構造リファクタリング実施。実験パッケージ(`features/`, `models/`, `reports/`, `diagnostics/`)を `src/experiments/` に統合。`cost/cost_calculator.py` を `execution/` に移動。LOB/スリッページ関連5ファイルを `execution/microstructure/` サブパッケージに整理。`scripts/` を `experiments/`, `sprint/`, `backtest/`, `batch/`, `test/` に分割。`tools/` を `production/`, `validation/`, `research/` に分離。`configs/` を `production/`, `research/` に分離。`scratch/` を `archive-2026-08` に移動し `.gitignore` に追加。
- **Phase 13 (2026-07-06)**: Macro Confidence（Factor-Specific Kappa）を本番モデルに統合。`core/macro.py` 新設 — マクロ因子（USDJPY, CLF, TNX）のEWMAベース・ボラティリティ調整サプライズ計算、感度行列を用いた銘柄別リスクスケーリング。`SectorRelativeEnsembleBLPEnhancedModel.predict_signals` 内でアンサンブル結合シグナルに対して `s_ens / scale_j` を適用。シグナル方向はBLPXのまま維持し、ポジションサイズのみマクロ環境に適応。YAML設定に `macro_confidence_enabled`, `macro_kappas`, `macro_surprise_halflife_mean`, `macro_surprise_halflife_vol` を追加。
- **Phase 14 (2026-07-08)**: ディレクトリ構造の完全統合リファクタリング実施。`src/experiments/` を `src/research/` にリネーム。`scripts/experiments/`（マクロ因子・BLPX実験）を `src/research/scripts/macro/` および `src/research/scripts/blpx/` に移動。`scripts/sprint/` を `src/research/scripts/sprint/` に移動。`scripts/backtest/` を `src/research/scripts/backtest/` に移動。全てのインポートパスを `from experiments.` から `from research.` に更新。研究関連コードを `src/research/` パッケージに完全統合し、Pythonパッケージとしての一貫性を確保。
- **Phase 16 (2026-07-12)**: JP Residual-PCA 残差化で推定対象の TOPIX リターンを close-to-close (`topix_cc_trade`) から open-to-close (`topix_oc_return`) に変更。`y_jp_target`（9:10→大引け）と同一時間窓のリターンを用いることで、overnight gap 成分が含まれる `topix_cc_trade` による β 推定に起因する系統的暴露の残存を解消。`blp_base.py` と `sre.py` 両方の `_prepare_common_inputs` に統一的に反映し、`beta_regressor` 等の設定不要な実装にハードコード化。比較バックテストで全期間 net Sharpe 8.355 → 8.670、MDD -6.89% → -6.06% の改善を確認。`compute_gap_adjusted_distribution.py` による gap 分布再計算が必要。
- **Phase 17 (2026-07-14)**: 前日gap行列フォールバックの廃止。`run_gap_distribution.sh` の前日行列コピー機能（`mu_gap_{PREV_DATE}.npy` → `mu_gap_{TODAY}.npy`）を削除。前日のgap行列で発注すると誤ったポジションとなるリスクがあるため、当日のgap行列が存在しない場合は flat position (w_final=0) を返すのが正しい挙動。`preprocess_data`（`preprocessor.py`）を修正し、`r_oc`（target return）がNaNの行も0埋めで残すことで、大引け前に当日のgap行列を計算可能に。AGENTS.md 不変条件 #6 として「前日gap行列の使用禁止」を明文化。
- **Phase 18 (2026-07-21)**: Fractional Differentiation（分数階差分）の本番適用。US ETFリターン列に対し、López de Prado (2018) のbinomial expansionベース分数階差分フィルター（d=0.1）を適用し、長期記憶を保持しつつ定常性を確保。`features/fractional_diff.py` 新設 — 重み計算・変換・ADF検定・Hurst指数推定・最適d探索。`core/pipeline.py::build_common_inputs` に `frac_diff_enabled`/`frac_diff_d`/`frac_diff_threshold`/`frac_diff_window` パラメータを追加し、USリターン列に 分数差分を適用。`blp_base.py::_prepare_common_inputs` からconfig経由でパラメータを読み取り。ウォークフォワード検証（2015-2026、12年次ウィンドウ、d=0.1/0.5/1.0）でd=0.1が12/12ウィンドウでd=1.0ベースラインを上回る（平均Sharpe 8.87 vs 7.75、ターンオーバー3%低減）。ComplianceAuditor・リーク監査全項目PASS。検証レポート: `reports/fractional_diff_walkforward_audit_report.md`。実験スクリプトは `git tag archive-2026-08` の `archive/experiments/` にアーカイブ。
- **Phase 19 (2026-07-30)**: BLPX の `mu_gap` 生成を改善。`SectorRelativeEnsembleBLPEnhancedModel._build_blp_diagnostics` で `mu`/`sigma` の X/Y 分割を `len(mu)//2` から `len(US_TICKERS)`/`len(JP_TICKERS)` に修正し、`vol_adjusted_target=false` パスを正しく動作させる。`blpx.vol_adjusted_target` を `true`（0平均 + 20日実現ボラ）から `false`（インサンプル平均 + インサンプル標準偏差）に変更。これにより `mu_raw = mu_Y + sigma_Y * z_hat_j` となり、2015-2026 全期間バックテストで net total 768.90% → 1006.06%、net Sharpe 6.0571 → 7.3882、max DD -8.39% → -6.99%、turnover 1.3725 → 1.3082、fallback 11.33% → 3.63% に改善。本番 `configs/production/production.yaml` に `vol_adjusted_target: false` を追加し、`compute_gap_adjusted_distribution.py` による gap 分布再計算が必要。
- **Phase 20 (2026-08-06)**: 本番config正本化（Refactoring C）。`start_date` のデフォルトを `2015-01-01` から `2015-01-05` に統一（`cli.py`、`config/schemas.py`、`execution/config.py`）。`side_leverage`（信用取引のロング+ショート合計レバレッジ倍率 = 1.5）を `configs/production/production.yaml` の `execution:` セクションに新設し、`StrategyConfig` / `load_config_from_yaml` / `execution/helpers.py` / `BacktestEngine.run_v2_backtest` で一貫して解決・伝播するよう接続。`BacktestEngine.run_v2_backtest` の fallback 値を本番コスト値（`overnight_alpha_long=0.75`、`overnight_alpha_short=0.5`）に揃え、V2 本番パスの設定再現性を向上。SRE/PCA 汎用パス（`run_backtest`）の legacy フォールバックは既存テストとの互換性のため維持。並列テストで関連 84 件 pass、既存の pandas 3.0 API / seaborn 未インストール / `test_daily_pnl_report` 環境依存以外は pass。レビュー後のフォローアップとして、`execution.side_leverage` を実験スクリプト 4 本でも優先参照するよう修正、`fast.py` の未使用 `start_date` デフォルトを 2015-01-05 に揃え、`load_config_from_yaml` の `side_leverage` 読み出しに `float()` 変換、`execute_post_decision_flow` の `config.side_leverage` に `getattr` フォールバックを追加。
- **Phase 21 (2026-08-07)**: Refactoring A — バックテスト日次コスト計算ループ（A-1）と gap 行列ローダー（A-2）を共通化。A-1: `BacktestEngine.run_backtest` と `run_v2_backtest` に重複していたコストパラメータ解決・暦日数計算・日次コスト/オーバーナイト/リターン計算ループを `_simulate_daily_pnl()` プライベートクラスメソッドに抽出。差分は `side_leverage` 倍率と OC（Open-to-Close）補助系列の有無のみ。A-2: `production_v2.py::load_gap_matrices` と `signal_enhancement.py::load_horizon_gap_matrices` / `load_rank_reversal_signal` の重複した日付フォーマット・存在確認・`np.load` ロジックを `utils.gap_matrix_io` に集約。`load_gap_matrices` は `mu_pattern` / `omega_pattern` / `pattern_kwargs`（`{h}` 等）を受け取る汎用ローダーとなり、`signal_enhancement` の multi-horizon blend・rank reversal overlay でも共有。いずれも合成データで振る舞い一致を検証し、関連テスト 114 件 pass。
- **Phase 22 (2026-08-07)**: Refactoring B-1 — `production_v2.py::generate_v2_production_portfolio` の 333 行を、docstring で定義済みの 10 ステージに沿って 5 つのプライベートヘルパーに分割。`_load_gap_or_flat`（gap 行列読み込み / flat fallback）、`_repair_and_adjust`（PSD 修復・マクロ調整）、`_compute_scores_and_weights`（mu_over_sigma、マルチホライズン・rank reversal オーバーレイ、long/short 選択、pre-gross ウェイト）、`_apply_pit_ruleD`（PIT ビニングと RuleD 乗数）、`_run_safety_audits`（リーク・数値監査・サマリー生成）。`generate_v2_production_portfolio` はそれらを呼び出すオーケストレータに整理。合成データ 7 シナリオでリファクタ前後の `w_final` / `scores` / `summary` 主要項目が一致し、関連テスト 114 件 pass。
- **Phase 22 (2026-08-07)**: Refactoring B-2 — `BacktestEngine.run_backtest` / `run_v2_backtest` を、コストパラメータ解決・シミュレーション期間決定・ターゲット/gap リターン計算・シグナル/ウェイト生成・V2 ウェイト生成・結果アセンブリの 8 つのプライベートヘルパーに分割。`run_backtest` / `run_v2_backtest` はこれらを順番に呼び出すオーケストレータに整理し、各ステージを単体テスト可能に。コストモデル、日付範囲、ウェイト生成、出力キーは変更せず、8 シナリオでリファクタ前後の出力が一致（1e-10）。ruff / mypy パス、対象テスト 16 件 pass。
- **Phase 22 (2026-08-07)**: Refactoring B-3 — `src/leadlag/cli.py::main` を 3 つのプライベートハンドラー (`_handle_decision` / `_handle_backtest` / `_handle_close`) と短いオーケストレータに分割。`main` はロギング設定・パーサー構築・ヘルプ表示・引数解析・市場休場判定・ハンドラー呼び出し・終了コード返却のみを担当。引数名・デフォルト・lazy import・`--auto-close` 非推奨警告・`--capital-from-wallet requires --api-enable` 検証・各種 `run_*` 呼び出しをすべて保持。parser・decision fast/standard モード・backtest・close の各ヘルプ出力と、mock 化した `run_production` への backtest dispatch を含む動作確認を実施。ruff / mypy パス。
- **Phase 22 (2026-08-07)**: Refactoring B-4 — `src/leadlag/execution/helpers.py::submit_orders_via_api` / `execute_post_decision_flow` を本番発注パスの段階別ヘルパーに分割。`submit_orders_via_api` から `_build_order_deltas`（target - current の delta 計算・close/new 分離）、`_submit_close_orders`（返済一括発注）、`_submit_new_orders`（新規即時発注）、`_submit_delayed_orders`（1629.T 大口 2 分割遅延発注）、`_write_api_execution_log` を抽出。`execute_post_decision_flow` から `_prepare_decision_df`（gross 調整・資本配分・DataFrame 構築）、`_log_decision_allocations`（予算/配分/未配分 logging）、`_run_risk_check_and_print`（リスク監査）、`_write_decision_output_and_submit`（CSV 出力・API 発注・journal 記録）を抽出。発注順序、`delay_ms=250`、`SPLIT_DELAY_SECONDS` 待機、`RuntimeError` 条件・メッセージ、JSON 出力、API 呼び出し順序は変更せず。DryRunBrokerClient mock を用いた 8 シナリオで baseline と一致し、`test_split_large_orders.py` / `test_close_positions.py` / `test_runner_helpers.py` 31 件 pass。
- **Phase 22 (2026-08-07)**: Refactoring B-5 — `src/leadlag/core/pipeline.py::build_common_inputs`、`archive/legacy_src/models/sre.py::_prepare_residual_prior`、`archive/legacy_src/execution/fast.py::build_precomputed_cache` を段階別ヘルパーに分割。`build_common_inputs` から fractional differencing / US/JP returns / ベースライン相関 / TOPIX 残差化 / P4 入力 / 最終アセンブリを、`sre._prepare_residual_prior` から baseline 窓選択 / V0_resid 構築 / C_full_resid 計算 / C0_resid 構築 / 結果サマリーを、`build_precomputed_cache` から cache 入力抽出 / 静的 cache 成分計算 / cache dict 構築 / 保存を抽出。関数シグネチャ・返却キー・ローリング窓・lazy import・fallback 動作は変更せず、合成データで before/after 一致を検証。ruff pass、mypy は既存エラーのみ、関連テスト 22 件 pass。
- **Phase 22 (2026-08-07)**: Refactoring D — 未使用のレガシーモデル (`sector_relative_ensemble_blp.py`, `sector_relative_ensemble_rrr.py`, `bayesian_blpx.py`, `net_score_ranking_lob.py`) を `src/leadlag/models/` から `git tag archive-2026-08` の `archive/legacy_src/models/` へ移設。関連する参照テスト (`test_sector_relative_ensemble_blp.py`, `test_sector_relative_ensemble_rrr.py`, `test_b3_characterization.py`, `test_sprint2c_lob.py`, `test_sector_relative_ensemble_blp_enhanced.py`) と実験スクリプト (`experiment_a5_bayesian_kalman_fix.py`, `run_sprint2c_lob_slippage.py`, `experiment_bayesian_blpx.py`, `final_model_selection.py`, `analyze_blp_enhanced_refined.py`) を `git tag archive-2026-08` の `archive/experiments/` へ集約。`git tag archive-2026-08` の `archive/tools/` 内の既存バックテストスクリプトも `legacy_src.models` 経由でアーカイブモデルを参照するよう更新。本番パス (`src/leadlag/`, `tools/production/`, `scripts/experiments/`) から `from leadlag.models.<archived>` インポートを除去し、保守行数を約 7,600 行削減。`SectorRelativeEnsembleBLPEnhancedModel` (v2 基盤) は `src/research/models/` に移設。
- **Phase 23 (2026-08-08)**: V1 backtest 段階的廃止 — CLI `backtest` サブコマンド、`src/leadlag/execution/backtest.py`、VaR/ES 履歴再構成 (`data/cache.py`, `v2_bridge.py`) を `BacktestEngine.run_v2_backtest` 一本化。レガシー fast モード用 `src/leadlag/execution/fast.py` は `archive/legacy_src/execution/fast.py` へ移設。レガシー `BacktestEngine.run_backtest` を `src/leadlag/execution/backtester.py` から削除し、残る研究実験向け `BaseModel` 汎用バックテストを `src/research/backtest_v1.py::run_v1_backtest` へ隔離。アクティブな研究スクリプト・テストを `research.backtest_v1` 経由に更新。`src/research/scripts/backtest/run_production_backtest.py` は V2 対応。標準 `decision` パスは V1 SRE モデル使用のため非推奨警告を追加（V2 本番は `tools/production/run_daily_production_v2.py` / `v2_bridge.py` を使用）。その後、CLI `decision` サブコマンドも `v2_bridge.run_v2_decision` 一本化し、`--config` / `--gap-dir` / `--live-dir` / `--capital` / `--api-url` / `--api-token` 等を V2 経路で受け渡し。最終的に未使用となった `decision.py` はテストで使用される `generate_daily_decision_results` のみを残す。最終的に `SectorRelativeEnsembleModel`（V1 SRE クラス）も `src/leadlag/models/sre.py` から `git tag archive-2026-08` の `archive/legacy_src/models/sre.py` へ移設。`src/leadlag/execution/helpers.py::build_strategy`、`src/research/backtest_common.py::run_baseline_backtest`、SRE 依存の研究スクリプト・ツール・テストを archive へ整理。V2 本番で使う `compute_jp_target_returns` は `src/leadlag/data/preprocessor.py` へ移設。本番系の `BacktestEngine` は V2 のみ。
- **Phase 24 (2026-08-09)**: Refactoring B-6 — `src/leadlag/execution/helpers.py` を 5 つの責務別サブモジュールに分解。`execution/pricing.py`（寄付価格・約定価格解決）、`execution/broker_ops.py`（BrokerClient 構築・ポジション/資本取得・発注・1629.T 大口分割）、`execution/risk_capital.py`（リスク設定・リスクチェック・gross 調整・資本配分）、`execution/output_ops.py`（出力ディレクトリ・決定 CSV・バックテストサマリー・position/wallet スナップショット・日次 journal）、`execution/post_decision.py`（gross 調整→リスク→配分→発注→出力の一連フロー）を新設。`execution/helpers.py` は後方互換 re-export モジュールに縮小。続けて P1-B3 として `tests/` から `research` パッケージへの依存を整理。V1 backtester テスト (`test_backtester_910.py`)、Residual-BLPX cost consistency テスト、および sprint 診断テスト (`test_sprint0_*`, `test_sprint1`, `test_sprint3b`) を `tests/research/` へ移設。`tests/integration/test_production_residual_blpx.py` から `research` import を除去。`scripts/run_tests_parallel.sh` を 8 プロセスに更新し、並列テストが全件 PASS することを確認。
- **Phase 25 (2026-08-09)**: Refactoring C-1 — Pydantic 一本化（P2-C1）を V2 パイプラインから開始。`ProductionV2Model.__init__` は `ProductionV2RunConfig` のみ受け付けるように変更。`generate_v2_production_portfolio` は Pydantic config を第一に受け取り、研究スクリプトの raw dict は境界で `parse_run_config` により `ProductionV2RunConfig` に変換。マルチホライズンブレンド・ランク反転オーバーレイのファイルパターンを `ProductionV2RunConfig` のフィールドに移行し、ネストした dict からの読み出しを除去。`BacktestEngine._generate_v2_weights` 内部でも raw dict を V2 用 Pydantic config に変換してから `ProductionV2Model` / overlay ヘルパーへ渡すよう変更。`ml_order_overlay.py` の訓練関数と `tools/production/train_ml_order_overlay.py` も Pydantic 化。続けて `save_summary_files` を `StrategyConfig` のみ受け付けるよう変更し、`backtest.run_production` では `load_config_from_yaml` から `AppConfig` を読み込み `app_config.strategy` を渡すよう変更。
- **Phase 26 (2026-08-09)**: Refactoring C-2 — `AppConfig` を V2 バックテスト・本番実行の単一正本に昇格。`AppConfig` に `v2: ProductionV2RunConfig` を追加し、`execution.config.build_app_config_from_dict` / `load_config_from_yaml` で同時に構築。`BacktestEngine.run_v2_backtest` は `AppConfig | dict` を受け付け、内部は `AppConfig` 一本化。コストパラメータは `app_config.strategy`、V2 ウェイト生成は `app_config.v2` から取得。`execution.backtest.run_production` は `BacktestEngine.run_v2_backtest` へ `AppConfig` を渡し、残差化・リスクパラメータも Pydantic 経由に変更。`AppConfig` に `gap_distribution_dir` を追加し、`v2_bridge.run_v2_decision` は YAML raw-dict 読み出しを廃止して `AppConfig` のみを使用するように変更。
- **Phase 27 (2026-08-09)**: Refactoring C-3 — 研究スクリプトの Pydantic 化完了。`scripts/experiments/` 配下のバックテスト比較・診断・シャドーラン・ライブアライメントスクリプトを `load_config_from_yaml` 経由に変更。`build_v2_production_shadow_run.py` / `run_live_aligned_v2_backtest.py` は `AppConfig.ml_order_overlay` を使用。`experiment_pit_rolling_window_tuning.py` / `overnight_sensitivity_v2.py` / `experiment_macro_kappa_v2.py` 等のパラメータスイープも `AppConfig` / `model_copy` を使用。`compare_cumulative_method.py` は V1 SRE モデルに raw YAML を残すが、V2 部分は `ProductionV2RunConfig.model_validate` 経由に変更。`src/experiments/ml_order_decision/phase1.py` / `phase2.py` も `AppConfig` 対応。
- **Phase 28 (2026-08-10)**: P2-D2 — gap 行列・weights・PnL の SQLite DB 化。`leadlag.data.gap_store.GapStore` を新設し、`mu_gap` / `omega_gap` / マルチホライズン / ランク反転行列を 1 つの SQLite ファイル（BLOB pickle、WAL）に保存。`leadlag.data.backtest_store.BacktestResultStore` を新設し、日次 PnL（return / equity / drawdown / turnover / gross / costs）と日次 weights を RDB に永続化。`utils/gap_matrix_io` から SQLite gap store を優先読み、存在しなければ `.npy` ファイルにフォールバックする統合パスを提供。DuckDB を検討したが、現時点では `sqlite3` 標準ライブラリで十分と判断（将来分析的クエリがボトルネックになれば再検討）。
- **Phase 29 (2026-08-10)**: P2-E1 — 日次運用 CLI 一本化。`leadlag.cli` に `daily` サブコマンドを追加。朝のカットオフ時刻（デフォルト 09:15）より前は `decision`、以降は `close` を自動実行。`_add_decision_args` / `_add_close_args` ヘルパーで引数定義を共通化。市場休場判定も `daily` を対象に拡張し、cron/launchd のエントリポイントを一本化。
- **Phase 30 (2026-08-10)**: P3-F1 — テスト高速化。`tests/conftest.py` に `synthetic_df_exec` フィクスチャを追加し、全カラムを満たす小規模な合成 `df_exec` を提供。`sample_df_exec` / `synthetic_df_exec` / `residual_blpx_prod_config` を `scope="session"` に変更し、ダウンロード・前処理をテスト間で共有。`tests/unit/test_synthetic_smoke.py` を新設し、schema validation、`ExecutionFrame` アクセサ、V2 フラットフォールバックをサブ秒で検証。

---

## Repository Root

```
pyproject.toml      # ビルド設定・依存関係・ruff/mypy/pytest 設定
requirements.txt    # pip 互換依存一覧
.env / .env.example # 環境変数テンプレート (BROKER_PROVIDER, API認証情報等)
日米ラグ.code-workspace  # VS Code ワークスペース設定
.agents/            # AIエージェントスキル定義 (skills/leadlag-fund-improvement/)
archive-2026-08/    # 廃止済みコード保管庫（2026-08 時点の snapshot）
archive/            # 廃止済みコード保管庫（現行：archive/legacy_src/、archive/experiments/ 等）
docs/               # 運用方針書、モデル技術仕様書、日次運用手順書などの設計・運用ドキュメント群
Papers/             # 原論文 (日米業種リードラグ.pdf / .md)
configs/            # パラメータ設定ファイル (YAML) — configs/production/, configs/research/, configs/archive/
src/                # Pythonソースコード正本 (PYTHONPATH の起点)
tests/              # ユニットテスト・統合テスト群 (unit/, integration/, fixtures/)
scripts/            # 本番・バッチ・テストスクリプト — scripts/batch/, scripts/test/
tools/              # コマンドツール — tools/production/, tools/validation/, tools/research/
kabu_auto_login/    # kabuステーション自動ログインユーティリティ (独立要件)
var/                # 唯一の実行時出力木（results, artifacts, live, logs, shadow_runs, market_data）
reports/            # sprint/phase 実験レポート群 (sprint0〜3b, phase3_walkforward)
scratch/            # 一時分析スクリプト (gitignore対象、中身は archive-2026-08 に移動済み)
creds/              # 認証情報ディレクトリ (gitignore対象)
```

---

## scripts/ ディレクトリ構造

```
scripts/
├── batch/               # バッチ実行・スケジューラ設定
│   ├── com.leadlag.close.plist
│   ├── com.leadlag.decision.plist
│   ├── run_auto_login.bat
│   ├── run_close_positions.bat
│   ├── run_close_positions.sh
│   ├── run_decision.bat
│   ├── run_decision.sh
│   ├── run_decision_v2.sh
│   ├── run_gap_distribution.sh
│   ├── setup_scheduler.ps1
│   └── setup_scheduler_macos.sh
│
└── test/                # 立花証券API接続テスト
    ├── test_tachibana_connection.py
    ├── test_tachibana_demo_order.py
    └── test_tachibana_login.py
```

---

## src/research/ ディレクトリ構造

```
src/research/            # 研究パッケージ（本番実行パスに含まれない）
├── __init__.py
├── backtest_common.py   # バックテスト共通ユーティリティ
│
├── diagnostics/         # モデル診断・sprint実験モジュール
│   ├── __init__.py
│   ├── sprint0.py             # sprint0 診断計算ロジック
│   ├── sprint0_qa.py          # sprint0 QA診断
│   └── sprint1_experiments.py # sprint1 実験ロジック
│
├── features/            # 実験用特徴量エンジニアリング
│   ├── __init__.py
│   ├── asset_exposures.py       # 資産エクスポージャー特徴量
│   ├── feature_selection_fdr.py # FDRベース特徴量選択
│   ├── hinge_features.py        # ヒンジ特徴量生成
│   └── hinge_interactions.py    # ヒンジ交互作用特徴量生成
│
├── models/              # 実験用オーバーレイモデル
│   ├── __init__.py
│   ├── hinge_elasticnet_overlay.py       # Hinge + ElasticNet オーバーレイ
│   ├── hinge_interaction_elasticnet.py   # Hinge交互作用 + ElasticNet
│   ├── hinge_interaction_gbdt.py         # Hinge交互作用 + GBDT
│   ├── hinge_interaction_overlay.py      # Hinge交互作用オーバーレイ
│   ├── hinge_interaction_ridge.py        # Hinge交互作用 + Ridge
│   ├── hinge_overlay.py                  # Hingeオーバーレイ
│   └── hinge_ridge_overlay.py            # Hinge + Ridge オーバーレイ
│
├── reports/             # 実験レポート生成スクリプト
│   ├── __init__.py
│   ├── sprint3a_hinge_report.py        # sprint3a ヒンジ特徴量レポート
│   └── sprint3b_hinge_interaction_report.py  # sprint3b ヒンジ交互作用レポート
│
├── scripts/             # 研究スクリプト（実行可能な研究スクリプト）
    ├── macro/           # マクロ因子実験スクリプト
    │   ├── analyze_gold_correlation.py
    │   ├── analyze_steel_metal_factors.py
    │   ├── compare_gold_factor_kappa.py
    │   └── sensitivity_factor_kappa.py
    │
    ├── blpx/            # BLPX実験スクリプト
    │   ├── compare_sensitivity_matrix.py
    │   ├── compare_shrinkage_ab_backtest.py
    │   ├── diagnose_shrinkage_attenuation.py
    │   └── experiment_copula.py
    │
    ├── sprint/          # sprint実験スクリプト（sprint0-3b）
    │   ├── finalize_sprint2_report.py
    │   ├── run_sprint0_diagnostics.py
    │   ├── run_sprint0_qa.py
    │   ├── run_sprint1_aum1m_tachibana.py
    │   ├── run_sprint1_experiments.py
    │   ├── run_sprint2_cost_aware_aum1m.py
    │   ├── run_sprint2b_qa.py
    │   ├── run_sprint3a_hinge_features.py
    │   └── run_sprint3b_hinge_interactions.py
    │
    ├── backtest/        # バックテスト実行スクリプト
    │   ├── run_overnight_holding_backtest.py
    │   ├── run_overnight_robustness_analysis.py
    │   ├── run_production_backtest.py
    │   └── run_selective_overnight_backtest.py
    │
    └── experiments/     # 実験スクリプト（旧 scripts/experiments 移設）
        └── _template.py

└── experiments/         # 実験用モジュール
    └── ml_order_decision/
        ├── __init__.py
        ├── phase1.py
        └── phase2.py

```

---

## src/ ディレクトリ構造

```
src/
├── leadlag/                 # 戦略パッケージ正本
│   ├── __init__.py
│   ├── cli.py               # 統合 CLI エントリーポイント (subcommands: decision, backtest, close, daily)
│   │
│   ├── core/                # 純粋ドメインロジック (I/O-free)
│   │   ├── types.py         # 型安全なドメインモデル（dataclass/Enum）
│   │   ├── correlation.py   # 相関・縮約計算
│   │   ├── signal.py        # シグナル生成
│   │   ├── residualize.py   # TOPIX 残差化
│   │   ├── portfolio.py     # ウェイト計算、Gross Exposure 調整
│   │   ├── allocator.py     # 資金・ロット配分
│   │   ├── risk.py          # VaR/ES 計算、リスクブリーチ判定
│   │   ├── market_calendar.py  # 営業日カレンダー・日付判定
│   │   └── macro.py         # マクロ因子（USDJPY/CLF/TNX）のボラティリティ調整サプライズ・Factor-Specific Kappa スケーリング
│   │
│   ├── config/              # 設定スキーマ定義・バリデーション層
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydanticを用いた型安全な設定クラス（AppConfig, StrategyConfig等）
│   │
│   ├── compliance/          # 安全監査・法令遵守検証層
│   │   ├── auditor.py       # ComplianceAuditor — 安全監査ロジックの実行
│   │   └── v2_auditor.py    # v2モデル専用監査ロジック
│   │
│   ├── models/              # 本番モデルレイヤー（純粋なシグナル生成・ウェイト計算のみ、I/Oフリー）
│   │   ├── production_v2.py               # ProductionV2Model (Residual-BLPX-RA v2) — 本番モデル
│   │   ├── signal_enhancement.py          # マルチホライズンブレンド・ランク反転オーバーレイ
│   │   └── ml_order_overlay.py            # ML order overlay 補助モデル
│   │
│   ├── data/                # データアクセス・前処理・キャッシュ層
│   │   ├── tickers.py       # ティッカー定義・変換ユーティリティ
│   │   ├── cache.py         # pkl/npz キャッシュ I/O、Pydantic設定によるバリデーション
│   │   ├── cache_store.py   # SQLite ベース汎用キャッシュストア
│   │   ├── backtest_store.py # バックテスト結果 SQLite 永続化
│   │   ├── gap_store.py     # gap 行列（mu/omega）SQLite 永続化
│   │   ├── fetcher.py       # データダウンロード (yfinance / ETFパッチ)
│   │   ├── preprocessor.py  # データ前処理（df_exec 構築、9:10→大引け target 計算）
│   │   └── market_data.py   # 寄付価格取得、ギャップ計算、価格検証
│   │
│   ├── broker/              # ブローカー抽象化レイヤー
│   │   ├── base.py          # ABC クライアントインターフェース
│   │   ├── dry_run.py       # ドライランシミュレータクライアント
│   │   ├── factory.py       # ブローカー作成ファクトリ
│   │   ├── kabu/            # kabuステーション API 接続
│   │   │   ├── api.py       # 低レベル API クライアント
│   │   │   └── client.py    # KabuBrokerClient アダプタ
│   │   └── tachibana/       # 立花証券 e-Shiten API 接続
│   │       ├── api.py       # 低レベル API クライアント (RSA暗号化/復号、セッション管理)
│   │       └── client.py    # TachibanaBrokerClient アダプタ
│   │
│   ├── execution/           # 実行管理・ランナー層
│   │   ├── config.py        # 設定ロード・Pydanticを用いた検証呼び出し
│   │   ├── broker_ops.py    # BrokerClient 構築・ポジション/資本取得・発注
│   │   ├── pricing.py       # 寄付価格・約定価格解決
│   │   ├── risk_capital.py  # リスク設定・リスクチェック・gross 調整・資本配分
│   │   ├── output_ops.py    # 出力ディレクトリ・決定 CSV・バックテストサマリー・スナップショット
│   │   ├── post_decision.py # gross 調整→リスク→配分→発注→出力の一連フロー
│   │   ├── decision.py      # generate_daily_decision_results()
│   │   ├── close.py         # 反対売買・自動クローズランナー
│   │   ├── backtest.py      # run_production() — バックテスト実行管理（CLI経由）
│   │   ├── backtester.py    # BacktestEngine — 汎用バックテストシミュレータ本体
│   │   ├── cost_calculator.py   # CostCalculator — 実コスト・スリッページ統合計算
│   │   └── microstructure/      # LOB・スリッページ・執行制御サブパッケージ
│   │       ├── __init__.py
│   │       ├── order_book_schema.py       # OrderBookSnapshot データスキーマ・バリデーション
│   │       ├── order_book_cost.py         # 板スプレッド・LOBスリッページ推定
│   │       ├── slippage_model.py          # エントリ/エグジットコストモデル (CostSource enum)
│   │       ├── execution_constraints.py   # 板ベース執行制約・空売り代替銘柄選択
│   │       └── live_quote_logger.py       # リアルタイム板ログ記録
│   │
│   ├── monitoring/          # モデル健全性監視層（記録・監視用、ポジションサイズ制御には使用しない）
│   │   └── health_score.py  # HealthScoreCalculator — IC減衰・グロス偏差・フォールバック率・シグナルドリフトの統合スコア
│   │
│   └── reporting/           # パフォーマンスレポート・出力フォーマット
│       ├── formatter.py           # ログ・テキストフォーマット
│       ├── metrics.py             # 指標計算、チャート描画
│       ├── results_format.py      # 結果フォルダ命名・マニフェスト出力
│       ├── production_v2_writer.py  # v2本番実行結果ライター
│       └── sprint2c_lob_report.py   # sprint2c LOBスリッページ分析レポート
│
└── research/             # 研究パッケージ (本番実行パスに含まれない) — 詳細は「src/research/ ディレクトリ構造」セクション参照
```

---

## Architecture Layers

### 1. Models Layer (`models/`)
本番戦略モデルの定義。`core/` の計算ロジックを組み合わせて PCA-Ensemble モデルを構成する。I/Oや実行ループ、監査プロセスから切り離された純粋なインターフェースを提供する。

**継承階層** (Phase 10 リファクタリング後):
```
ABC (abc.ABC)
└── BaseModel (base.py)
    └── _BLPBase (blp_base.py) — BLP系モデル共通メソッド
        └── SectorRelativeEnsembleBLPEnhancedModel (sector_relative_ensemble_blp_enhanced.py)
```

本番 V2 モデルは現状 `BaseModel` インターフェースを継承していない手続き的オーケストレータ（`production_v2.py::generate_v2_production_portfolio`）として動作している。`SectorRelativeEnsembleModel` (V1) は 2026-07 に `git tag archive-2026-08` の `archive/legacy_src/models/sre.py` へ移設された。

| モジュール | 責務 |
|---|---|
| `production_v2.py` | V2 本番ポートフォリオ生成 (`generate_v2_production_portfolio()`)。ギャップ調整予測分布・`mu_over_sigma` ランキング・PITビニング (RuleD) 統合。当面は手続き的モジュールとして運用 |
| `signal_enhancement.py` | マルチホライズンブレンド (`apply_multi_horizon_blend`)・ランク反転オーバーレイ (`apply_rank_reversal_overlay`) — Phase 2A/2D 成果物 |
| `ml_order_overlay.py` | ML order overlay 補助モデル |


### 2. Core Domain Layer (`core/`)
純粋な計算ロジック。**I/O 依存なし**。任意の呼び出し元から再利用可能。

| モジュール | 責務 |
|---|---|
| `types.py` | 型安全なドメインモデル（dataclass/Enum）— Position, Order, RiskMetrics 等 |
| `correlation.py` | 相関・縮約計算 |
| `signal.py` | 相関縮約、固有値分解、シグナル生成、ウェイト構築 |
| `residualize.py` | ローリング OLS ベータ推定、TOPIX 残差化 |
| `portfolio.py` | ウェイト計算、Gross Exposure 自動調整 |
| `allocator.py` | 株数への変換（予算制約付き、1629.T 10株ロット対応） |
| `risk.py` | VaR/ES 計算、リスクブリーチ判定 |
| `market_calendar.py` | 営業日カレンダー・日付判定（米国・日本市場休場日判定） |
| `macro.py` | マクロ因子（USDJPY, CLF, TNX）のボラティリティ調整サプライズ計算、感度行列（`MACRO_SENS_MATRIX`）、Factor-Specific Kappa リスクスケーリング |
| `pit.py` | Point-in-time view — ローリング窓アクセスを `as_of` 行で制限しルックアヘッドを実行時に防止 |
| `experiment_registry.py` | 実験レジストリ — 仮説・パラメータ・指標・DSR を JSONL で記録 |
| `timeouts.py` | 集中管理されたタイムアウト定数と `with_timeout` デコレータ |

### 3. Data Layer (`data/`)
市場データのライフサイクル全体を管理。

| モジュール | 責務 |
|---|---|
| `tickers.py` | US/JP ティッカー定義・変換ユーティリティの**単一正本** |
| `cache.py` | `etf_data.pkl` + `decision_cache.npz` の全 I/O 及びファイルロック制御 |
| `fetcher.py` | yfinance ダウンロード、差分更新、1629.T NAVパッチ |
| `preprocessor.py` | `df_exec` 構築（日次リターン整列、TOPIX beta計算） |
| `market_data.py` | 寄付価格取得、ギャップ計算、価格検証 |
| `schema.py` | `df_exec` の列ファミリ・型付き `ExecutionFrame` ラッパー（ADR-0001 PIT view と連携） |
| `validation.py` | データ検証ゲート — raw data / exec record / gap 行列の構造的検証 |

### 4. Broker Layer (`broker/`)
発注経路をプラグイン可能にするブローカー抽象化レイヤー。

```
BrokerClient (ABC)
├── KabuBrokerClient → leadlag.broker.kabu.api.KabuClient のアダプタ
├── TachibanaBrokerClient → TachibanaClient のアダプタ（PKI認証対応）
├── DryRunBrokerClient → ネットワーク不要のシミュレーション
└── (将来) SBIBrokerClient, RakutenBrokerClient, ...
```

kabuステーションや立花証券からの移行・別ブローカー追加時は以下の3ステップのみ：
1. 新 `broker/sbi/client.py` に `SBIBrokerClient(BrokerClient)` を実装
2. `broker/factory.py` に `case "sbi":` を追加
3. `.env` の `BROKER_PROVIDER=sbi` を変更

**production.py・strategy.py・ドメインコードの変更は不要。**

### 5. Execution/Runner Layer (`execution/`)
実行モード別のオーケストレーション。

| モジュール | 責務 |
|---|---|
| `config.py` | YAML/env の設定パラメータロード・Pydanticスキーマによる検証 (デフォルト: `configs/production/production.yaml`) |
| `broker_ops.py` | BrokerClient 構築・ポジション/資本取得・発注・1629.T 大口分割 |
| `pricing.py` | 寄付価格・約定価格解決 |
| `risk_capital.py` | リスク設定・リスクチェック・gross 調整・資本配分 |
| `output_ops.py` | 出力ディレクトリ・決定 CSV・バックテストサマリー・position/wallet スナップショット |
| `post_decision.py` | gross 調整→リスク→配分→発注→出力の一連フロー |
| `decision.py` | `generate_daily_decision_results()` |
| `close.py` | `close_all_positions()`, `wait_and_auto_close()` |
| `backtest.py` | `run_production()` — 生産バックテスト実行管理 |
| `backtester.py` | `BacktestEngine` — 汎用的なバックテスト実行シミュレータ |
| `cost_calculator.py` | `CostCalculator` — 実コスト・スリッページ統合計算（`microstructure/slippage_model.py` に依存） |

#### 5a. Microstructure Subpackage (`execution/microstructure/`)
LOB・スリッページ・執行制御関連モジュール。

| モジュール | 責務 |
|---|---|
| `order_book_schema.py` | `OrderBookSnapshot` データスキーマ・バリデーション・APIレスポンス変換 |
| `order_book_cost.py` | 板スプレッド・LOBスリッページ推定・深度計算 |
| `slippage_model.py` | エントリ/エグジットコストモデル (`CostSource` enum, `compute_entry_cost_bps`, `compute_exit_cost_bps`) |
| `execution_constraints.py` | 板ベース執行制約・空売り代替銘柄選択 (`apply_hard_rules`, `ExecutionDecision`) |
| `live_quote_logger.py` | リアルタイム板ログ記録ユーティリティ |

### 6. Compliance Layer (`compliance/`)
安全監査・法令遵守検証。

| モジュール | 責務 |
|---|---|
| `auditor.py` | `ComplianceAuditor.run_audit()` — バックテストや実行結果に対する時系列・数式漏洩等の包括的な安全監査の実行 |
| `v2_auditor.py` | v2モデル専用監査ロジック — ProductionV2Model の出力に対する個別検証 |

### 7. Monitoring Layer (`monitoring/`)
モデル健全性の定量的監視。**記録・監視専用**であり、ポジションサイズ制御には使用しない（常にフルポジションで運用）。

| モジュール | 責務 |
|---|---|
| `health_score.py` | `HealthScoreCalculator` — IC減衰・グロス偏差・フォールバック率・シグナルドリフトの4成分を統合したモデル健全性スコア（0-100）を算出。ターンオーバー成分は日次全額決済運用のため除外。 |

> **設計決定**: Health Score によるポジションサイズ動的調整をバックテストで検証した結果、Sharpe比率の改善は見られず、常にフルポジション（グロスエクスポージャー200%）での運用が最適であることを確認済み。Health Score はモデル健全性の記録・監視用としてのみ利用する。

### 8. Reporting Layer (`reporting/`)
| モジュール | 責務 |
|---|---|
| `formatter.py` | ログ出力・テキスト注文フォーマット・リスクレポート |
| `metrics.py` | 指標計算、チャート描画 |
| `results_format.py` | 結果フォルダ命名・マニフェスト出力 |
| `production_v2_writer.py` | v2本番実行結果ライター — 日次実行結果のファイル出力 |
| `sprint2c_lob_report.py` | sprint2c LOBスリッページ分析レポート生成 |

### 9. Research Package (`src/research/`)
研究用モジュール群。本番実行パスには含まれない。`src/research/scripts/` から `from research...` として参照される。

| サブパッケージ | 内容 |
|---|---|
| `research/diagnostics/` | sprint0/sprint0_qa/sprint1_experiments — モデル診断・分布診断・AUM1億シミュレーション |
| `research/features/` | ヒンジ特徴量・交互作用特徴量・FDR特徴量選択・資産エクスポージャー |
| `research/models/` | Hinge + ElasticNet/Ridge/GBDT オーバーレイモデル（Phase 2C実験成果物） |
| `research/reports/` | sprint3a/3b ヒンジ特徴量・交互作用レポート生成 |
| `research/scripts/` | 研究スクリプト（macro/, blpx/, sprint/, backtest/） |

---

## Key Design Decisions

### 設定定義の Pydantic 移行による堅牢化
`src/leadlag/config/schemas.py` 内に `AppConfig`、`StrategyConfig` などの Pydantic スキーマモデルを定義し、設定読み込み時にすべてのフィールド値の型や有効範囲（`ge`, `le`）をバリデーションしています。また、設定オブジェクトは `model_config = {"frozen": True}` によってイミュータブル（不変）に保護されています。

### ティッカー定義の一元化
`data/tickers.py` が US_TICKERS / JP_TICKERS / TOPIX_TICKER / N_US / N_JP / N_TOTAL の**単一正本**。
`config.py` 経由で各設定オブジェクトへ伝搬されます。

> **Note:** 実装上の US_TICKERS は 15 銘柄（Select Sector SPDRs 11 + Style ETFs 4）である。
> 運用方針書（§3.1）では論文に基づき N_U = 11 と記述している。
> 追加の 4 銘柄（MTUM, VLUE, IUSG, USMV）はシグナル精度向上のために実装で追加されたものであり、
> 事前部分空間ベクトル（v_1 〜 v_6）の次元は実装上 32 次元（15 + 17）に拡張されている。

### ブローカー抽象化
`BrokerClient` ABC が発注・ポジション・残高の全 I/O インターフェースを定義。
`execution/` レイヤー（`decision.py`, `close.py` 等）は BrokerClient のみを参照し、kabu 固有コードに依存しない。

### Gross Exposure 調整
`leadlag/core/portfolio.py::adjust_gross_exposure()` が正本。
`classify_actions()` による BUY/SELL/HOLD 分類もここに統合。

### リスクロジックの一本化
VaR/ES 計算・リスクチェック評価は `leadlag/core/risk.py` が正本。
`leadlag/execution/risk_capital.py::run_risk_checks()` を呼び出す。

### 結果出力ディレクトリ方針
`var/results/YYYYMMDD_HHMMSS_<run_name>/` が実行時出力の一つの形態。
`results_format.py::create_results_output_dir()` 経由で作成。各実行に `run_manifest.json` を生成。

---

## Data Flow

```
[Market Data Sources]
  ├── yfinance → leadlag/data/fetcher.py → etf_data.pkl
  ├── Google Finance → leadlag/data/market_data.py
  ├── CSV → leadlag/data/market_data.py
  └── kabu API → BrokerClient.fetch_open_prices()
             ↓
       leadlag/data/preprocessor.py → df_exec (pandas DataFrame)
             ↓
  [Production v2 Flow]
  tools/research/compute_gap_adjusted_distribution.py → (mu_gap, omega_gap) matrices
             ↓
  tools/production/run_daily_production_v2.py 
    ├── leadlag/models/sector_relative_ensemble_blp_enhanced.py (BLPX model)
    ├── mu_over_sigma ranking & baseline_style sizing (leadlag/core/portfolio.py)
    ├── PIT binning (RuleD ex-ante IR dynamic gross scaling: 0.75x or 1.00x)
    └── Fallback checks (gap data missing → flat position)
             ↓
  [Compliance/Risk/Order Flow]
    ├── leadlag/core/risk.py → evaluate_risk_checks()
    ├── leadlag/compliance/auditor.py (ComplianceAuditor)
    └── leadlag/broker/base.py → BrokerClient.submit_orders_batch()
             ↗ leadlag/broker/kabu/client.py      (kabuステーション)
             ↗ leadlag/broker/tachibana/client.py (立花証券)
             ↗ leadlag/broker/dry_run.py          (シミュレーション)
```

---

## テスト実行

```bash
# テストスイート全体
python3 -m pytest tests/ -v

# 特定の単体テストのみ
python3 -m pytest tests/unit/test_ticker_registry.py -v
python3 -m pytest tests/unit/test_dry_run_broker.py -v
```

---

## 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| [運用方針書.md](運用方針書.md) | 投資目的・哲学、投資ユニバース、検証原則、リスク管理制限値、ガバナンス枠組み等（原則書） |
| [モデル技術仕様書.md](モデル技術仕様書.md) | シグナル構築数理、PCA・BLPXモデル定式化、パラメータ仕様、事前固有ベクトル設計等の技術仕様 |
| [日次運用手順書.md](日次運用手順書.md) | 日次のシステム実行タイムライン、自動安全監査 (Safety Audit) 項目、手動ロールバック、監視・アラート手順 |
| [MODE_USAGE_GUIDE.md](MODE_USAGE_GUIDE.md) | CLI 実行モード一覧・戦略モード・コマンド例・入出力仕様 |
| [README.md](README.md) | プロジェクト概要・セットアップ手順 |
| [model_summary_for_improvement.md](model_summary_for_improvement.md) | モデル改善履歴・サマリ |
| [研究メモ202606.md](研究メモ202606.md) | 研究メモ・実験記録 (2026年6月) |
| [SCHEDULER_SETUP.md](SCHEDULER_SETUP.md) | タスクスケジューラ設定（旧実行環境用） |
| [api/kabu_STATION_API.yaml](api/kabu_STATION_API.yaml) | kabuステーション API 仕様書 (OpenAPI/Swagger) |
| [api/立花証券API.md](api/立花証券API.md) | 立花証券 e-Shiten API 仕様書 |