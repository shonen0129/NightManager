# Lead-Lag Market-Neutral Strategy — Architecture (v3.0)

V2 同期パス (ProductionV2Model) を本番正本とし、Next-Gen 非同期パイプライン・凸最適化は 2026-08-17 の ADR (docs/decisions/2026-08-17-p35-pipeline-canon.md) に基づき archive/legacy_src/ へ移設された。`PITDataLake` は `leadlag.data.pit_lake` の本番入力adapterとして保持し、`DecisionInputs`へ変換してモデルへ渡す。

> **最終更新**: 2026-09-17

## Overview

US ETF と TOPIX-17 セクター ETF のリードラグ相関を利用した、
日次マーケットニュートラル戦略のプロダクションシステム。

本番モデルは **Production Residual-BLPX-RA v2** （予測期待値を予測標準偏差で割ったリスク調整スコア $\mu_{\text{gap}} / \sigma_{\text{gap}}$ による銘柄選択と、予測 ex-ante IR の過去履歴に基づく動的グロス調整 RuleD を採用した、ギャップ調整予測分布ベースの最適化モデル）。
旧本番の **Sector Relative Ensemble (PCA-Ensemble)** はベンチマーク用として維持される。

### ML overlay の責務境界

ML order overlay は、pickle の互換性を持つ fitted container だけを
`leadlag.models.ml_order_overlay.MLOrderOverlayModel` に残す。特徴量計算は
`ml_overlay_features.py`、検証済みの immutable artifact の保存・読込は
`ml_overlay_artifact.py`、本番適用は `ml_overlay_inference.py` が正本である。
学習・LightGBM の fit・学習データ収集は `research.experiments.ml_overlay_training`
に分離し、本番 package から research を import しない。既存の import path は
段階移行のための薄い compatibility export としてのみ残す。研究環境と artifact
運用は [RESEARCH_ENV.md](RESEARCH_ENV.md) を参照する。
旧 module の学習関数名は production 側では fail-closed の移行案内を返し、研究 package を動的に読み込まない。

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
│   ├── com.leadlag.update-market-data.plist
│   ├── com.leadlag.distribution-diagnostics.plist
│   ├── com.leadlag.close.plist
│   ├── com.leadlag.decision.plist
│   ├── com.leadlag.pnl_report.plist
│   ├── _update_market_data.py
│   ├── run_close_positions.sh
│   ├── run_distribution_diagnostics.sh
│   ├── run_decision_v2.sh
│   ├── run_pnl_report.sh
│   ├── run_gap_distribution.sh
│   ├── install_launchd.sh
│   ├── update_market_data.sh
│   └── setup_scheduler_macos.sh
│
└── run_tests_parallel.sh # 全テストの外側期限付き分割実行
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
│   │   ├── macro.py         # マクロ因子の純粋なサプライズ・Factor-Specific Kappa 計算
│   │   └── pnl.py           # weight-based BT PnL と FIFO fill/inventory accounting
│   ├── domain/              # 実行層間で共有する型付き契約
│   │   ├── distribution.py  # 分布結果・理由・source試行トレース（S3a正本）
│   │   ├── inputs.py        # Known/Historical/Evaluation/DecisionInputs と入力版
│   │   └── gap_bundle.py    # μ/Ω/metadata の一括公開manifest（S3c）
│   │
│   ├── config/              # 設定合成・スキーマ定義・バリデーション層
│   │   ├── __init__.py
│   │   ├── loader.py        # YAML __base__ 合成・循環参照検出（S1a正本）
│   │   └── schemas.py       # Pydanticを用いた型安全な設定クラス（AppConfig, StrategyConfig等）
│   │
│   ├── compliance/          # 安全監査・法令遵守検証層
│   │   ├── auditor.py       # ComplianceAuditor — 安全監査ロジックの実行
│   │   └── v2_auditor.py    # v2モデル専用監査ロジック
│   │
│   ├── models/              # 本番モデルレイヤー（入力adapterを受け、broker/発注・実行I/Oを持たない）
│   │   ├── production_v2.py               # ProductionV2Model (Residual-BLPX-RA v2) — 本番モデル
│   │   ├── v2/                             # decision engine・distribution source・fallback・監査比較
│   │   ├── blpx/                           # ProductionBLPXModel と信号/事前分布部品
│   │   ├── signal_enhancement.py          # マルチホライズンブレンド・ランク反転オーバーレイ
│   │   └── ml_order_overlay.py            # ML order overlay 補助モデル
│   │
│   ├── runner/              # 本番・BTで共有する依存部品の組立層
│   │   ├── model_factory.py # V2のBLPX/decision/overlay構築の正本
│   │   └── production.py    # ProductionRunner — 一日分の決定組立
│   │
│   ├── pipeline/            # 入口から分離した純粋計算・診断出力境界
│   │   ├── gap_distribution.py # raw/gap μ・Ωの一日分計算（S3b）
│   │   └── gap_reporting.py # gap診断DataFrame・CSV出力adapter（S3b）
│   │
│   ├── data/                # データアクセス・前処理・キャッシュ層
│   │   ├── tickers.py       # ティッカー定義・変換ユーティリティ
│   │   ├── macro.py         # macro価格の取得・正規化・キャッシュ（S2a）
│   │   ├── adr_features.py  # ADR特徴量artifactの読込・鮮度判定（S2a）
│   │   ├── intraday_inputs.py # 5分足からの9:10入力抽出（S2a）
│   │   ├── decision_cache.py # 日次判断・価格cache
│   │   ├── market_data_cache.py # 市場履歴cacheの取得と鮮度検査
│   │   ├── cache_store.py   # SQLite ベース汎用キャッシュストア
│   │   ├── backtest_store.py # バックテスト結果 SQLite 永続化
│   │   ├── gap_store.py     # gap 行列と一括manifestのSQLite永続化（S3c）
│   │   ├── fetcher.py       # データダウンロード (yfinance / ETFパッチ)
│   │   ├── preprocessor.py  # データ前処理（df_exec 構築、日米session整列）
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
│   │   ├── var_cache.py         # VaR/ES cache identity と絶対期限budget（S3c）
│   │   ├── var_inputs.py        # VaR入力fingerprint・gap/PIT履歴snapshot
│   │   ├── var_worker.py        # worker終了までのsnapshot所有・期限付きcache保存
│   │   ├── state_store.py       # 注文intent/観測/reconciliationと実行lease（S5a）
│   │   ├── reconcile.py         # 保存済み注文IDと約定/建玉の読取専用復旧照合
│   │   ├── job_guard.py         # batch期限・process group停止・lease（S5b）
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
│       ├── daily_pnl_report.py    # 実約定 close fill と残存建玉の円建て PnL 表示
│       ├── results_format.py      # 結果フォルダ命名・マニフェスト出力
│       ├── production_v2_writer.py  # v2本番実行結果ライター
│       └── sprint2c_lob_report.py   # sprint2c LOBスリッページ分析レポート
│
└── research/             # 研究パッケージ (本番実行パスに含まれない) — 詳細は「src/research/ ディレクトリ構造」セクション参照
```

---

## Architecture Layers

### 1. Models Layer (`models/`)
本番戦略モデルの定義。`core/` の計算ロジックを組み合わせて V2 の分布・シグナル・ウェイトを構成する。
入力adapterは`data/`と`domain/inputs.py`で明示的に受け、broker・発注・実行I/Oは持たない。
`HistoricalInputs`はas-of cutoffと09:10/macro/ADR/PIT/rank-reversalのrun-owned欄を持ち、strict typed
decisionでは不足入力をモデル内部から再取得しない。close-derived labelはJP 15:30 JSTを
availability cutoffとする。旧compatibility adapter、主要adapterの観測時刻充填、gap生成を含む
全入口の同一cutoffはS2/S3の残件である。VaR keyはeffective config・df_exec・code・overlay・gap
bundleに加えて、同じVaR runへ渡す09:10/macro/ADR/PIT/rank-reversalの入力snapshot fingerprintを束ねる。

**継承階層** (Phase 10 リファクタリング後):
```
ABC (abc.ABC)
└── BaseModel (base.py)
    └── _BLPBase (blp_base.py) — BLP系モデル共通メソッド
        └── SectorRelativeEnsembleBLPEnhancedModel (sector_relative_ensemble_blp_enhanced.py)
```

本番 V2 の正本は `ProductionV2Model` で、処理本体は `models/v2/` の decision engine、分布source、fallback、監査比較へ分割されている。
`generate_v2_production_portfolio()`は利用者を正本APIへ移して撤去済みである。`SectorRelativeEnsembleModel` (V1) は
2026-07 に `git tag archive-2026-08` の `archive/legacy_src/models/sre.py` へ移設された。

| モジュール | 責務 |
|---|---|
| `production_v2.py` | `ProductionV2Model`の公開API。旧公開引数は入口で`DecisionInputs`へ一度だけ正規化し、分布・ランキング・RuleDを`models/v2/`へ委譲 |
| `signal_enhancement.py` | マルチホライズンブレンド (`apply_multi_horizon_blend`)・ランク反転オーバーレイ (`apply_rank_reversal_overlay`) — Phase 2A/2D 成果物 |
| `ml_order_overlay.py` | ML order overlay 補助モデル |
| `blpx/` | `ProductionBLPXModel`、相関・事前分布・信号計算。旧root module `models/blpx.py` は撤去済み |


### 2. Core Domain Layer (`core/`)
純粋な計算ロジック。**I/O 依存なし**。任意の呼び出し元から再利用可能。

| モジュール | 責務 |
|---|---|
| `types.py` | 型安全なドメインモデル（dataclass/Enum）— Position, Order, RiskMetrics 等 |
| `target_returns.py` | 取得済み9:10価格・寄付調整を明示入力としてJP targetを計算。cache取得は`data.intraday_inputs`が担当 |
| `correlation.py` | 相関・縮約計算 |
| `signal.py` | 相関縮約、固有値分解、シグナル生成、ウェイト構築 |
| `residualize.py` | ローリング OLS ベータ推定、TOPIX 残差化 |
| `portfolio.py` | ウェイト計算、Gross Exposure 自動調整 |
| `allocator.py` | 株数への変換（予算制約付き、1629.T 10株ロット対応） |
| `risk.py` | VaR/ES 計算、リスクブリーチ判定 |
| `market_calendar.py` | 営業日カレンダー・日付判定（米国・日本市場休場日判定） |
| `macro.py` | マクロ因子（USDJPY, CLF, TNX）のボラティリティ調整サプライズ計算、感度行列（`MACRO_SENS_MATRIX`）、Factor-Specific Kappa リスクスケーリング。ネットワーク・キャッシュI/Oは持たない |
| `pnl.py` | weight-based BTの日次損益・費用計算と、観測/仮定FillをFIFO在庫へ評価する純粋な会計プリミティブ |
| `pit.py` | Point-in-time view — ローリング窓アクセスを `as_of` 行で制限しルックアヘッドを実行時に防止 |
| `experiment_registry.py` | 実験レジストリ — 仮説・パラメータ・指標・DSR を JSONL で記録 |
| `timeouts.py` | 集中管理されたタイムアウト定数と `with_timeout` デコレータ |

### 2.1 Typed input boundary (`domain/inputs.py`)

`KnownMarketInputs`はJSTのas-of時点で既知のUS/gap/価格/betaと特徴量を保持する。
`HistoricalInputs`は学習窓・horizon・固定prior・PIT履歴を保持し、`EvaluationInputs`は
事後の実現値だけを保持する。`DecisionInputs`は既知入力と履歴を一つに束ね、
`InputVersion`でschemaと内容をSHA-256識別する。`PITDataLake.build_decision_inputs`が
旧DataFrame/snapshot入力をこの契約へ変換するため、ProductionRunner、BacktestEngine、
日次bridge、V2 decisionの内部経路は複数の候補入力を再解釈しない。snapshotとPIT viewの
配列・mappingは境界で所有コピーをread-only化し、履歴はrunごとに一度だけ所有する。

### 3. Data Layer (`data/`)
市場データのライフサイクル全体を管理。

| モジュール | 責務 |
|---|---|
| `tickers.py` | US/JP ティッカー定義・変換ユーティリティの**単一正本** |
| `decision_cache.py` / `market_data_cache.py` | 判断/価格cacheと市場履歴cacheの正本。旧`cache.py` shimは撤去済み |
| `fetcher.py` | yfinance ダウンロード、差分更新、1629.T NAVパッチ |
| `preprocessor.py` | `df_exec` 構築（日次リターン整列、TOPIX beta計算） |
| `macro.py` | macro価格の取得・列名正規化・timeout・キャッシュ。計算層へDataFrameを渡す入力adapter |
| `adr_features.py` | ADR特徴量pickleの読込、`sig_date`除去、trade date欠損・鮮度の既存fallback判定 |
| `intraday_inputs.py` | 5分足cacheから09:10 midpointと寄り→09:10 returnsを抽出し、target計算へ明示入力として渡す |
| `market_data.py` | 寄付価格取得、ギャップ計算、価格検証 |
| `gap_store.py` | gap行列と一括manifestのSQLite永続化。`save_horizon` / `load_horizon_bundle`で同一snapshotを扱う |
| `schema.py` | `df_exec` の列ファミリ・型付き `ExecutionFrame` ラッパー（ADR-0001 PIT view と連携） |
| `pit_lake.py` | as-of snapshotの抽出と`DecisionInputs`の構築。旧`df_exec`引数は入口adapterでのみ扱う |
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
| `close.py` | `close_all_positions()`とCLIの決済・照合。leaseを迂回する旧auto-close helperは撤去 |
| `backtest.py` | `run_production()` — 生産バックテスト実行管理 |
| `backtester.py` | `BacktestEngine` — 汎用的なバックテスト実行シミュレータ |
| `cost_calculator.py` | `CostCalculator` — 板/フォールバック費用見積り（bps。`core.pnl`の日次return費用とは別契約） |
| `var_cache.py` | `VaRCacheIdentity`でeffective config・入力版・overlay・gap bundleをcache keyへ束ね、`DeadlineBudget`で絶対期限を共有 |
| `var_inputs.py` / `var_worker.py` | fingerprint・gap/PIT履歴snapshotの取得と、timeout後もworker終了まで保持する所有権・期限付き保存 |
| `state_store.py` | SQLiteのrun/order intent/observation/reconciliation台帳と口座・戦略単位の実行lease |
| `reconcile.py` | 保存済みbroker IDの注文・約定・建玉・余力を照合する復旧入口。送信・取消・再送は行わない |
| `job_guard.py` | batch process groupの期限、TERM→KILL、lease、guard結果JSON |

S4の意思決定・執行境界は、`leadlag.domain.portfolio.PortfolioDecision`（属性専用）から
`execution.contracts.ExecutionPlan`、brokerの`OrderObservation`、`ExecutionReport`へ接続する。
新規注文と引け決済のstatus pollは`execution.order_lifecycle.poll_order_statuses`を共有し、
legacy mappingの受け入れはJSON/CSV/Markdown writerの`_coerce_decision`だけに限定する。
`leadlag.execution.state_store.ExecutionStateStore`は、broker呼出前の注文意図、broker観測、
reconciliation checkpointをSQLiteへ追記する。送信開始後にプロセスが停止したrunは
`executing`または`reconciliation_required`として復旧候補に残り、同じ口座・戦略では日付・jobを
またいでも新たな送信をブロックする。`completed`は約定・建玉・余力・journalの照合後にのみ記録する。
同じ注文計画の数量変更や、記録失敗を成功として扱わない。decisionとcloseは
`live:production_v2` leaseを共有し、同時実行を許可しない。brokerとの原子的transactionや
exactly-once発注は保証せず、再送前に注文・約定・建玉を照合する。
復旧候補の表示と読取専用照合の手順は[SCHEDULER_SETUP.md](SCHEDULER_SETUP.md)を参照する。
注文ID不明・開始在庫未保存の旧runは自動完了にせず、人による照合が必要となる。

S6では観測へside、累積filled quantity、fill price、fee、fill ID/sourceも保存する
（execution state schema v2）。`list_observed_fills`は累積pollを注文単位で最新化して
`core.pnl.Fill`へ変換し、同じbroker観測を複数回PnLへ加算しない。

`scripts/batch/run_decision_v2.sh`、`run_close_positions.sh`、
`run_gap_distribution.sh`、`run_pnl_report.sh`は`leadlag.execution.job_guard`でprocess group全体を監視する。
子のgap生成は親decisionのleaseを共有する。発注入口はowner/scope/DBをSQLiteへ照合する。
期限超過はTERM後に猶予を置いて子孫もKILLし、exit 124とguard JSONを残す。実schedulerへの
登録はplist適用後に別途確認する。旧macOS入口`run_decision.sh`は撤去済みである。
15:40の`run_pnl_report.sh`は`reconcile --pending`を先に呼び、引け注文の終端後の照合を行う。
未解決状態はレポートの成否にかかわらず維持し、batchへ非ゼロを返す。

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
| `daily_pnl_report.py` | 確認済み実約定Fillと残存建玉の円建て実現/未実現PnLを表示 |
| `sprint2c_lob_report.py` | sprint2c LOBスリッページ分析レポート生成 |

#### PnL accounting boundary (S6)

`core.pnl.simulate_daily_pnl` is the numerical reference for the existing
weight-based backtest. It reports decimal return fractions for intraday plus
overnight gross return, slippage, financing, borrow, reverse fee, turnover,
and exposure. `BacktestEngine` only aligns dates and assembles the result
Series; it does not carry a second copy of the cost loop. Research overnight
holding uses the same calculator with an explicit per-asset carry mask.

`core.pnl.Fill`, `InventoryLot`, and `FeeAccrual` are the shared accounting
vocabulary. `daily_pnl_report` translates confirmed close execution records
and position snapshots into these objects. Observed fill prices are consumed
as executed; backtest slippage is not applied a second time. FIFO matches
partial closes and reversals, allocates entry/exit fees once, and marks open
lots separately. Missing position prices retain the legacy reported total as
a compatibility fallback rather than inventing a mark.

The bps `execution.cost_calculator.CostBreakdown`, decimal-return
`domain.portfolio.CostBreakdown`, and currency Fill fees expose explicit unit
labels. `BacktestResultStore.daily_pnl` also persists `overnight_return` so the
day/night attribution is not lost when results are reloaded. `MetricsSpec`
fixes evaluation frequency, annualisation, flat-day treatment, and return/cost
units for reports.

### 9. Research Package (`src/research/`)
研究用モジュール群。本番実行パスには含まれない。`src/research/scripts/` から `from research...` として参照される。

Gap 分布診断は、raw/preprocessed market data・TOPIX trade return・realtime とモデル共通入力の組立を `research/diagnostics/gap_inputs.py`、
数値ポートフォリオ評価を `gap_portfolio.py`、PIT比較・可視化・レポートを
`gap_outputs.py` に分ける。h=1/h=3/h=5 の分布計算と安定したCSV frameは
`leadlag.pipeline` の共有部品を使い、production codeから研究専用モジュールを逆参照しない。

| サブパッケージ | 内容 |
|---|---|
| `research/diagnostics/` | gap_inputs（入力組立）、gap_portfolio（ポートフォリオ評価）、gap_outputs（PIT/plot/report）、sprint0/sprint0_qa/sprint1_experiments |
| `research/features/` | ヒンジ特徴量・交互作用特徴量・FDR特徴量選択・資産エクスポージャー |
| `research/models/` | Hinge + ElasticNet/Ridge/GBDT オーバーレイモデル（Phase 2C実験成果物） |
| `research/reports/` | sprint3a/3b ヒンジ特徴量・交互作用レポート生成 |
| `research/scripts/` | 研究スクリプト（macro/, blpx/, sprint/, backtest/） |

---

## Key Design Decisions

### 設定定義の Pydantic 移行による堅牢化
`src/leadlag/config/schemas.py` 内に `AppConfig`、`StrategyConfig` などの Pydantic スキーマモデルを定義し、設定読み込み時にすべてのフィールド値の型や有効範囲（`ge`, `le`）をバリデーションしています。また、設定オブジェクトは `model_config = {"frozen": True}` によってイミュータブル（不変）に保護されています。

### 設定合成と V2 モデル構築の正本（S1a〜S1d）
YAML の `__base__` 合成は `config/loader.py`、V2 の mapping 正規化は
`config/schemas.py::parse_run_config` が担当します。broker/env を含む
`execution/config.py::load_config_from_yaml` は `AppConfig` を構築する正規のアプリ境界として
残し、下流から private 合成関数や設定型の再exportを呼びません。

本番・バックテスト・gap生成の BLPX/V2/overlay 構築は
`runner/model_factory.py` に集約しています。gap生成は overlay を読み込まない
`build_blpx_model`、本番とバックテストは `build_v2_model_bundle` を使います。
有効 V2 設定は実行成果物へ SHA-256 とともに記録し、同じ設定での再生を可能にします。

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
    ├── leadlag/runner/model_factory.py::build_v2_model_bundle (BLPX/V2/overlay構築)
    ├── leadlag/runner/production.py::ProductionRunner (一日分の決定組立)
    ├── mu_over_sigma ranking & baseline_style sizing (leadlag/core/portfolio.py)
    ├── PIT binning (RuleD ex-ante IR dynamic gross scaling: 0.75x or 1.00x)
    └── Fallback checks (gap data missing → flat position)
             ↓
  [Compliance/Risk/Order Flow]
    ├── leadlag/core/risk.py → evaluate_risk_checks()
    ├── leadlag/compliance/auditor.py (ComplianceAuditor)
    ├── leadlag/execution/contracts.py → ExecutionPlan / ExecutionReport
    ├── leadlag/execution/order_lifecycle.py → 共通status poll
    └── leadlag/broker/base.py → BrokerClient.submit_orders_batch()
             ↗ leadlag/broker/kabu/client.py      (kabuステーション)
             ↗ leadlag/broker/tachibana/client.py (立花証券)
             ↗ leadlag/broker/dry_run.py          (シミュレーション)
```

### 実行時の安全境界

`decision` のマルチホライズン経路は、各horizonのμ・Ωと取得元metadataを
一組で解決する。`sig_date`（または一致する`signal_date`）が取引日より前で
あることを確認し、来歴不正のcacheは同じhorizonのon-demandへ切り替える。
結果は`leadlag.domain.distribution.DistributionResult`で返し、
`DistributionStatus`・`DistributionReason`・`DistributionAttempt`がsourceの試行順と
拒否理由の正本になる。従来の`is_available`・`is_flat`・alertは互換出力として残る。
全horizonを解決できない場合は `fallback.audit_failure=true` のflatを返し、
`diagnostics.distribution_provenance`と`distribution_resolution`へsource・来歴・拒否理由・試行を残す。

gapのμ・Ω・metadataは `leadlag.domain.gap_bundle.GapBundleRef` を一つの公開単位として扱う。
SQLiteでは `GapStore.save_horizon` が行列・metadata・manifestを同一transactionで保存し、
`load_horizon_bundle` が同一snapshotから読み出す。NPY互換経路でもmanifestをcommit markerとして
使い、payloadのSHA-256・trade date・horizon・storage formatを検証してから返す。新規bundleはさらに
as-of入力版、model版、effective config版、固定ticker順をmetadata/manifestへ保存する。df_execを
持つproduction readerは4項目を現在の入力と比較し、不一致・欠落をcache不採用としてon-demandへ送る。
VaR/ES return cacheは
`leadlag.execution.var_cache.VaRCacheIdentity`でeffective config、df_exec、code、overlay、gap入力、
run-owned入力snapshotの版を束ね、`DeadlineBudget`の絶対期限を準備と計算に共有する。これは旧cache key形式を保ったまま、
入力版の取り違えと期限切れ後の再計算を防ぐための境界である。

h=1 gap生成はsignal dateに一致するStep 1 `Omega_struct`だけを明示共分散として使う。一致するファイルがなく
古いfallbackしかない場合は、fallbackを診断へ記録し、on-demandと同じBLPX共分散へ戻す。日付の異なる構造行列を
キャッシュ出力へ流用してsource間の分布を変えない。

JPの前処理では、JP calendarで休場と確認でき、close/openの対象列が両方とも
全NaNのpadding行だけを計算前に除去する。営業日の全NaNは欠損としてstrict
validationへ渡す。発注後・決済後は不完了summaryを保持して約定、建玉、余力、
journalを順に照合し、close CLIは不完了時に終了コード2を返す。

ML overlayはartifact rootの `CURRENT` が指すimmutable versionを読む。
`versions/<version>/model.pkl` と `metadata.json` のdigest・学習期間・data/config
hashを検証し、公開途中のstagingは読まない。root直下のlegacy artifactは
modelとmetadataの組を同一versionとして証明できないため常に拒否し、再学習して
versioned artifactとして公開する。VaR/ESのreturn cacheはCURRENTが指すversionだけを
fingerprintし、inactive versionやstagingの作成では無効化しない。decision実行で
overlayを選択した場合は、その同じ検証済みobjectをVaRのcache keyとrisk backtestへ
渡し、途中のCURRENT切替で別versionの系列を同じkeyへ保存しない。

---

## テスト実行

```bash
# テストスイート全体
python3 -m pytest tests/ -v

# 特定の単体テストのみ
python3 -m pytest tests/unit/test_ticker_registry.py -v
python3 -m pytest tests/unit/test_dry_run_broker.py -v
```

CIでは`uv.lock`を固定してPython 3.12環境を構築し、compileall、Ruff、mypy、
import-linter、文書相対リンク、production wheelの`research`除外検査、`tests/`全体を順に実行する。
wheelは隔離インストール後にCLI helpと合成versioned ML artifactの推論も確認する。
詳細は[CI と構造境界の検証](CI.md)と`.github/workflows/ci.yml`を参照する。

---

## 関連ドキュメント

### 9:10価格とartifact受入（2026-09-22）

BT・ML学習・gap事前生成は`PITDataLake.get_execution_snapshot`で、所有した
`open_910_returns`から判断価格とgapを作る。観測セルは9:10価格、欠損セルは正の日次寄付を
使い、`price_sources`で出所を保持する。raw観測の欠損は保持し、入力指紋に反映する。
Inf・−100%以下の観測や使用不能な価格、来歴不正を寄付代替で隠さない。
h=3/5のgap生成も当日成分に同じsnapshotを使う。
詳細は[価格契約ADR](decisions/2026-09-22-execution-price-and-artifact-acceptance.md)、
artifact・運用の最新受入状態は[実行報告](../reports/20260922_production_acceptance/report.md)を参照する。

| ドキュメント | 内容 |
|---|---|
| [運用方針書.md](運用方針書.md) | 投資目的・哲学、投資ユニバース、検証原則、リスク管理制限値、ガバナンス枠組み等（原則書） |
| [モデル技術仕様書.md](モデル技術仕様書.md) | シグナル構築数理、PCA・BLPXモデル定式化、パラメータ仕様、事前固有ベクトル設計等の技術仕様 |
| [日次運用手順書.md](日次運用手順書.md) | 日次のシステム実行タイムライン、自動安全監査 (Safety Audit) 項目、手動ロールバック、監視・アラート手順 |
| [MODE_USAGE_GUIDE.md](MODE_USAGE_GUIDE.md) | CLI 実行モード一覧・戦略モード・コマンド例・入出力仕様 |
| [README.md](../README.md) | プロジェクト概要・セットアップ手順 |
| [model_summary_for_improvement.md](model_summary_for_improvement.md) | モデル改善履歴・サマリ |
| [研究メモ202606.md](研究メモ202606.md) | 研究メモ・実験記録 (2026年6月) |
| [SCHEDULER_SETUP.md](SCHEDULER_SETUP.md) | macOS launchdの現行batchスケジューラ設定（Windows入口はlegacy archive） |
| [CI.md](CI.md) | lock固定、静的検査、import契約、wheel分離、全体テストのCIゲート |
| [api/kabu_STATION_API.yaml](api/kabu_STATION_API.yaml) | kabuステーション API 仕様書 (OpenAPI/Swagger) |
| [api/立花証券API.md](api/立花証券API.md) | 立花証券 e-Shiten API 仕様書 |
