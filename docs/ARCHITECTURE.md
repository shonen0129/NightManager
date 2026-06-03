# Lead-Lag Market-Neutral Strategy — Architecture (v2.1 / 再構成済み)

> **最終更新**: Phase 1–4 完了 (2026-05)

## Overview

US ETF と TOPIX-17 セクター ETF のリードラグ相関を利用した、
日次マーケットニュートラル戦略のプロダクションシステム。

- **Phase 1**: ブローカー抽象化レイヤー (`broker/`) 導入
- **Phase 2**: データ層の分割 (`data/` パッケージ + `pyproject.toml`)
- **Phase 3**: `production.py` の分解 (`runner/` サブパッケージ)
- **Phase 4**: ユニットテストスイート (`tests/`) + 本ドキュメント更新

---

## Repository Root

```
ETF_1629.csv        # 1629.T NAVパッチ用 CSV
pyproject.toml      # ビルド設定・依存関係・ruff/mypy/pytest 設定
requirements.txt    # pip 互換依存一覧 (pyproject.toml で管理中に移行)
docs/               # ドキュメント
data/               # 市場データキャッシュ (etf_data.pkl 等)
logs/               # 実行ログ
results/            # 実行出力ルート
scripts/            # 補助スクリプト
src/                # Pythonソースルート (PYTHONPATH の起点)
tools/              # 分析・調査ツール
```

---

## src/ ディレクトリ構造

```
src/
│
├── production.py            # ★ CLI エントリーポイント (thin facade)
│                            #   --mode decision / backtest / close-positions
│                            #   実装は runner/ に委譲
│
├── config.py                # 中央設定管理（戦略パラメータ, API設定, アセット数定数）
├── strategy.py              # LeadLagStrategy / PrecomputedLeadLagStrategy 高レベルIF
├── data_loader.py           # ★ thin facade → data/ パッケージに委譲
├── performance.py           # パフォーマンス指標計算・チャート生成
├── results_format.py        # 結果出力ディレクトリ命名・manifest 管理
├── kabu_client.py           # kabuステーション API クライアント（低レベル正本）
│
├── runner/                  # ★ Phase 3 新設: production.py の分解先
│   ├── __init__.py
│   ├── config.py            # ProductionConfig dataclass（全ランナー共通）
│   ├── helpers.py           # 共有ユーティリティ（broker/risk/alloc/orders）
│   ├── decision.py          # run_decision() — 標準発注フロー
│   ├── fast.py              # run_decision_fast() — 高速推論（no yfinance）
│   ├── close.py             # close_all_positions(), wait_and_auto_close()
│   └── backtest.py          # run_production() — バックテスト
│
├── data/                    # ★ Phase 2 新設: データ層パッケージ
│   ├── __init__.py          # 公開 API
│   ├── ticker_registry.py   # US/JP ティッカー定義・変換ユーティリティ (単一正本)
│   ├── cache.py             # pkl/npz キャッシュ I/O
│   ├── downloader.py        # yfinance ダウンロード + 差分更新 + 1629.T NAVパッチ
│   └── preprocessor.py      # df_exec 構築（日次リターン整列・TOPIX beta計算）
│
├── broker/                  # ★ Phase 1 新設: ブローカー抽象化レイヤー
│   ├── __init__.py
│   ├── base.py              # BrokerClient ABC + WalletInfo/Position/BrokerConfig
│   ├── dry_run.py           # DryRunBrokerClient（ネットワーク不要・テスト用）
│   ├── factory.py           # create_broker_from_args() ファクトリ
│   └── kabu/
│       ├── __init__.py
│       └── client.py        # KabuBrokerClient（kabu_client.KabuClient のアダプタ）
│
├── domain/                  # Pure domain logic (I/O-free)
│   ├── models/
│   │   └── types.py         # 型安全なドメインモデル（dataclass/Enum）
│   ├── signals/
│   │   └── lead_lag.py      # シグナル生成・相関計算・ウェイト構築
│   ├── portfolio/
│   │   ├── optimizer.py     # ウェイト計算, Gross Exposure 調整
│   │   └── allocator.py     # ウェイト→株数変換（資本配分）
│   └── risk/
│       └── metrics.py       # VaR/ES 計算, リスクチェック評価
│
├── services/                # Application services
│   ├── market_data.py       # 寄付価格取得, ギャップ計算, 価格検証
│   ├── cache_service.py     # ファイルロック, 日次リターンキャッシュ, 戦略キャッシュ
│   └── formatting.py        # テキスト注文出力・リスクレポート・指標ログ
│
├── tests/                   # ★ Phase 4 新設: ユニットテストスイート
│   ├── unit/
│   │   ├── test_ticker_registry.py  # data.ticker_registry
│   │   ├── test_portfolio.py        # domain.portfolio.optimizer
│   │   ├── test_risk.py             # domain.risk.metrics
│   │   ├── test_allocator.py        # domain.portfolio.allocator
│   │   ├── test_dry_run_broker.py   # broker.dry_run + broker.factory
│   │   ├── test_runner_config.py    # runner.config
│   │   └── test_runner_helpers.py   # runner.helpers
│   ├── integration/                 # (将来拡充予定)
│   └── fixtures/                    # テスト固定データ
│
├── infrastructure/
│   ├── execution/engine.py  # 注文実行エンジン
│   └── storage/cache_repo.py
│
├── app/
│   ├── runner.py            # 代替 CLI entry point
│   └── workflow.py          # 取引ワークフロー統合
│
├── backtest/
│   └── runner.py            # バックテスト実行エンジン
│
└── test/                    # 研究・検証スクリプト（ユニットテストではない）
    ├── significance_suite.py
    ├── logic_combo_exhaustive.py
    └── ...
```

---

## Architecture Layers

### 1. Domain Layer (`domain/`)
純粋な計算ロジック。**I/O 依存なし**。任意の呼び出し元から再利用可能。

| モジュール | 責務 |
|---|---|
| `signals/lead_lag.py` | 相関縮約、固有値分解、シグナル生成、ウェイト構築 |
| `portfolio/optimizer.py` | ウェイト計算、Gross Exposure 自動調整 |
| `portfolio/allocator.py` | 株数への変換（予算制約付き、1629.T 10株ロット対応） |
| `risk/metrics.py` | VaR/ES 計算、リスクブリーチ判定 |

### 2. Data Layer (`data/`)
市場データのライフサイクル全体を管理。

| モジュール | 責務 |
|---|---|
| `ticker_registry.py` | US/JP ティッカー定義・変換ユーティリティの**単一正本** |
| `cache.py` | `etf_data.pkl` + `decision_cache.npz` の全 I/O |
| `downloader.py` | yfinance ダウンロード、差分更新、1629.T NAVパッチ |
| `preprocessor.py` | `df_exec` 構築（日次リターン整列、TOPIX beta計算） |

`data_loader.py` は後方互換のための thin facade（既存コードは変更不要）。

### 3. Broker Layer (`broker/`)
発注経路をプラグイン可能にするブローカー抽象化レイヤー。

```
BrokerClient (ABC)
├── KabuBrokerClient → kabu_client.KabuClient のアダプタ
├── DryRunBrokerClient → ネットワーク不要のシミュレーション
└── (将来) SBIBrokerClient, RakutenBrokerClient, ...
```

kabuステーション廃止・移行時は以下の3ステップのみ：
1. 新 `broker/sbi/client.py` に `SBIBrokerClient(BrokerClient)` を実装
2. `broker/factory.py` に `case "sbi":` を追加
3. `.env` の `BROKER_PROVIDER=sbi` を変更

**production.py・strategy.py・ドメインコードの変更は不要。**

### 4. Runner Layer (`runner/`)
`production.py` から抽出された実行モード別のオーケストレーション。

| モジュール | 責務 |
|---|---|
| `config.py` | `ProductionConfig` dataclass（全ランナー共通） |
| `helpers.py` | broker/risk/capital alloc/orders の共有ユーティリティ |
| `decision.py` | `run_decision()` — 標準発注フロー（yfinance 使用） |
| `fast.py` | `run_decision_fast()` — 高速推論（yfinance 不要・precomputed cache） |
| `close.py` | `close_all_positions()`, `wait_and_auto_close()` |
| `backtest.py` | `run_production()` — バックテスト |

### 5. Services Layer (`services/`)
production.py から抽出されたアプリケーションサービス。

| モジュール | 責務 |
|---|---|
| `market_data.py` | Google Finance / CSV / API からの寄付価格取得、ギャップ計算 |
| `cache_service.py` | ファイルロック付きキャッシュ IO |
| `formatting.py` | ログ出力・テキスト注文フォーマット・リスクレポート |

---

## Key Design Decisions

### ティッカー定義の一元化
`data/ticker_registry.py` が US_TICKERS / JP_TICKERS / TOPIX_TICKER / N_US / N_JP / N_TOTAL の**単一正本**。
`config.py` の `N_US_ASSETS` 等はここから参照。`data_loader.py` facade 経由でも後方互換。

### ブローカー抽象化
`BrokerClient` ABC が発注・ポジション・残高の全 I/O インターフェースを定義。
`production.py` や `runner/` は BrokerClient のみを参照し、kabu 固有コードに依存しない。

### Gross Exposure 調整
`domain/portfolio/optimizer.py::adjust_gross_exposure()` が正本。
`classify_actions()` による BUY/SELL/HOLD 分類もここに統合。

### リスクロジックの一本化
VaR/ES 計算・リスクチェック評価は `domain/risk/metrics.py` が正本。
`runner/helpers.py::run_risk_checks()` は薄いラッパーとして呼び出す。

### fast mode
precomputed cache (`strategy_cache.npz`) を使うことで、
相関行列計算・固有値分解をスキップし、yfinance ダウンロードも不要。
kabu API から US リターン + JP 寄付価格を直接取得してシグナルを生成。

### 結果出力ディレクトリ方針
`results/YYYYMMDD_HHMMSS_<run_name>/` が唯一の正本ディレクトリ。
`results_format.py::create_results_output_dir()` 経由で作成。各実行に `run_manifest.json` を生成。

---

## Data Flow (再構成後)

```
[Market Data Sources]
  ├── yfinance → data/downloader.py → etf_data.pkl
  ├── Google Finance → services/market_data.py
  ├── CSV → services/market_data.py
  └── kabu API → BrokerClient.fetch_open_prices()
             ↓
       data/preprocessor.py → df_exec (pandas DataFrame)
             ↓
  domain/signals/lead_lag.py → compute_signal()
             ↓
  domain/portfolio/optimizer.py → compute_trade_decision()
             ↓
  domain/portfolio/allocator.py → allocate_capital()
             ↓
  domain/risk/metrics.py → evaluate_risk_checks()
             ↓
  broker/base.py → BrokerClient.submit_orders_batch()
    ↗ broker/kabu/client.py  (kabuステーション)
    ↗ broker/dry_run.py      (シミュレーション)
    ↗ (将来) broker/sbi/client.py
```

---

## テスト実行

```bash
# ユニットテスト (97件)
cd src && python -m pytest tests/unit/ -v

# 特定モジュールのみ
python -m pytest tests/unit/test_ticker_registry.py -v
python -m pytest tests/unit/test_dry_run_broker.py -v
```

## CLI コマンド

```bash
# 標準発注モード
python src/production.py --mode decision --jp-opens-csv jp_opens.csv --capital 10000000

# ファストモード（yfinance 不要）
python src/production.py --mode decision --fast-mode --api-enable

# バックテスト
python src/production.py --mode backtest --start-date 2020-01-01

# 引け時決済
python src/production.py --mode close-positions --api-enable [--api-dry-run]
```