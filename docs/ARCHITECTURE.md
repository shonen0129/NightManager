# Lead-Lag Market-Neutral Strategy — Architecture (v3.0)

V2 同期パス (ProductionV2Model) を本番正本とし、Next-Gen 非同期パイプライン・凸最適化は 2026-08-17 の ADR (docs/decisions/2026-08-17-p35-pipeline-canon.md) に基づき archive/legacy_src/ へ移設された。`PITDataLake` は `leadlag.data.pit_lake` に実験資産として保持。

> **最終更新**: 2026-08-13

## Overview

US ETF と TOPIX-17 セクター ETF のリードラグ相関を利用した、
日次マーケットニュートラル戦略のプロダクションシステム。

本番モデルは **Production Residual-BLPX-RA v2** （予測期待値を予測標準偏差で割ったリスク調整スコア $\mu_{\text{gap}} / \sigma_{\text{gap}}$ による銘柄選択と、予測 ex-ante IR の過去履歴に基づく動的グロス調整 RuleD を採用した、ギャップ調整予測分布ベースの最適化モデル）。
旧本番の **Sector Relative Ensemble (PCA-Ensemble)** はベンチマーク用として維持される。

**注意**: v1 fallback (Residual-BLPX) は2026-07-09に廃止されました。gap data欠損時はflat position (w_final=0) を返します。廃止理由は、v2でエラーが出る場合v1でも同様にエラーが出るため、循環依存の問題があったためです。v1 fallback関連コードは `git tag archive-2026-08` の `archive/deprecated_v1_fallback/` にアーカイブされています。

### リファクタリング履歴

過去の大規模なアーキテクチャ変更・実験の経緯は `docs/history.md` に分離しました。

## Repository Root

```
pyproject.toml      # ビルド設定・依存関係・ruff/mypy/pytest 設定
requirements.txt    # pip 互換依存一覧
.env / .env.example # 環境変数テンプレート (BROKER_PROVIDER, API認証情報等)
leadlag.code-workspace  # VS Code ワークスペース設定
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
  python3 -m leadlag.cli decision
    ├── leadlag/execution/v2_bridge.py::run_v2_decision()
    ├── leadlag/runner/production.py::ProductionRunner (BLPX model)
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