# Lead-Lag Market-Neutral Strategy — Architecture (v3.0)

> **最終更新**: 2026-06-15

## Overview

US ETF と TOPIX-17 セクター ETF のリードラグ相関を利用した、
日次マーケットニュートラル戦略のプロダクションシステム。

本番モデルは **Production P8P3-BLPX v2** （予測期待値を予測標準偏差で割ったリスク調整スコア $\mu_{\text{gap}} / \sigma_{\text{gap}}$ による銘柄選択と、予測 ex-ante IR の過去履歴に基づく動的グロス調整 RuleD を採用した、ギャップ調整予測分布ベースの最適化モデル）。
前版の **P8P3-BLPX (v1)** は第1フォールバックとして本システム内に保持され、さらに旧本番の **Sector Relative Ensemble (SRE)** は第2フォールバックおよびベンチマーク用として維持される。

### リファクタリング履歴

- **Phase 1**: ブローカー抽象化レイヤー (`broker/`) 導入
- **Phase 2**: データ層の分割 (`data/` パッケージ + `pyproject.toml`)
- **Phase 3**: `production.py` の分解 (`runner/` サブパッケージ)
- **Phase 4**: ユニットテストスイート (`tests/`) + 本ドキュメント更新
- **Phase 5**: 設定定義のPydantic移行、モデル層・実行層・安全監査の完全デカップリング（BaseModel導入による純粋化、BacktestEngine / ComplianceAuditor への役割分担）
- **Phase 6**: 計算ボトルネックの高速化最適化（`compute_us_residualized_returns` のベクトル化、基準相関行列 `c_full` 及び残差空間事前分布 `_prepare_residual_prior` のメモリキャッシュ化による高速化）
- **Phase 7 (2026-06-15)**: 本番 v2 モデル（P8P3-BLPX v2）への昇格に伴い、ギャップ調整予測分布計算、リスク調整ランキング（`mu_over_sigma`）、PIT ビニングに基づく動的グロス制御（`RuleD`）の導入、および v2 → v1 → SRE の多段階自動フォールバック機能の実装。
- **Phase 8 (2026-06-17)**: AIMA/IOSCO モデルリスクガイドラインに準拠するため、文書体系を再編。運用方針書から詳細なアルゴリズム数理・システムパラメータ・日次実行コマンド等を分離し、別冊の《モデル技術仕様書》および《日次運用手順書》へ移行。

---

## Repository Root

```
pyproject.toml      # ビルド設定・依存関係・ruff/mypy/pytest 設定
requirements.txt    # pip 互換依存一覧
docs/               # 運用方針書、モデル技術仕様書、日次運用手順書などの設計・運用ドキュメント群
market_data/        # 市場データキャッシュ 及び 1629.T NAV パッチ用 CSV
configs/            # 本番動作パラメータ設定ファイル (configs/production.yaml)
results/            # バックテストおよび日次推論の実行出力ルート (SRE本番用)
scripts/            # 定期実行・環境構築用のバッチ/シェルスクリプト
src/                # Pythonソースコード正本 (PYTHONPATH の起点)
tests/              # 本番モデル用テスト群
tools/              # 本番 v2 日次実行 (run_daily_production_v2.py)、ギャップ予測分布計算 (compute_gap_adjusted_distribution.py)、シャドウ/検証用コマンドツール
```

---

## src/ ディレクトリ構造

```
src/
└── leadlag/                 # 戦略パッケージ正本
    ├── __init__.py
    ├── cli.py               # 統合 CLI エントリーポイント (subcommands: decision, backtest, close)
    │
    ├── core/                # 純粋ドメインロジック (I/O-free)
    │   ├── types.py         # 型安全なドメインモデル（dataclass/Enum）
    │   ├── correlation.py   # 相関・縮約計算
    │   ├── signal.py        # シグナル生成
    │   ├── residualize.py   # TOPIX 残差化
    │   ├── portfolio.py     # ウェイト計算、Gross Exposure 調整
    │   ├── allocator.py     # 資金・ロット配分
    │   └── risk.py          # VaR/ES 計算、リスクブリーチ判定
    ├── config/              # 設定スキーマ定義・バリデーション層
    │   ├── __init__.py
    │   └── schemas.py       # Pydanticを用いた型安全な設定クラス（AppConfig, StrategyConfig等）
    │
    ├── compliance/          # 安全監査・法令遵守検証層
    │   └── auditor.py       # ComplianceAuditor — 安全監査ロジックの実行
    │
    ├── models/              # 本番モデルレイヤー（純粋なシグナル生成・ウェイト計算のみ、I/Oフリー）
    │   ├── base.py          # BaseModel 抽象モデルインターフェース
    │   └── sre.py           # SectorRelativeEnsembleModel 本体
    │
    ├── data/                # データアクセス・前処理・キャッシュ層
    │   ├── tickers.py       # ティッカー定義・変換ユーティリティ
    │   ├── cache.py         # pkl/npz キャッシュ I/O、Pydantic設定によるバリデーション
    │   ├── fetcher.py       # データダウンロード (yfinance / ETFパッチ)
    │   ├── preprocessor.py  # データ前処理（残差リターン計算など、df_exec の構築）
    │   └── market_data.py   # 寄付価格取得、ギャップ計算、価格検証
    │
    ├── broker/              # ブローカー抽象化レイヤー
    │   ├── base.py          # ABC クライアントインターフェース
    │   ├── dry_run.py       # ドライランシミュレータクライアント
    │   ├── factory.py       # ブローカー作成ファクトリ
    │   └── kabu/            # kabuステーション API 接続
    │       ├── api.py       # 低レベル API クライアント
    │       └── client.py    # KabuBrokerClient アダプタ
    │
    ├── execution/           # 実行管理・ランナー層
    │   ├── config.py        # 設定ロード・Pydanticを用いた検証呼び出し
    │   ├── helpers.py       # 共通ヘルパー (監査ログ・発注指示)
    │   ├── decision.py      # run_decision() — 標準判定ランナー、generate_daily_decision_results()
    │   ├── fast.py          # run_decision_fast() — 高速判定ランナー (no yfinance)
    │   ├── close.py         # 反対売買・自動クローズランナー
    │   ├── backtest.py      # run_production() — バックテスト実行管理（CLI経由）
    │   └── backtester.py    # BacktestEngine — 汎用バックテストシミュレータ本体
    │
    └── reporting/           # パフォーマンスレポート・出力フォーマット
        ├── formatter.py     # ログ・テキストフォーマット
        ├── metrics.py       # 指標計算、チャート描画
        └── results_format.py# 結果フォルダ命名・マニフェスト出力

```

---

## Architecture Layers

### 1. Models Layer (`models/`)
本番戦略モデルの定義。`core/` の計算ロジックを組み合わせて SRE モデルを構成する。I/Oや実行ループ、監査プロセスから切り離された純粋なインターフェースを提供する。

| モジュール | 責務 |
|---|---|
| `base.py` | モデルのI/Oフリー共通抽象化インターフェース (`predict_signals`, `build_weights`) |
| `sre.py` | SectorRelativeEnsembleModel ロジック（P0/P3/P4 シグナル生成、Zスコア正規化、アンサンブル、ウェイト算出）の正本 |
| `sector_relative_ensemble_blp_enhanced.py` | SectorRelativeEnsembleBLPEnhancedModel （本番 v2 基盤の BLPX 構造化投影および確信度調整モデル） |


### 2. Core Domain Layer (`core/`)
純粋な計算ロジック。**I/O 依存なし**。任意の呼び出し元から再利用可能。

| モジュール | 責務 |
|---|---|
| `signal.py` | 相関縮約、固有値分解、シグナル生成、ウェイト構築 |
| `residualize.py` | ローリング OLS ベータ推定、TOPIX 残差化 |
| `portfolio.py` | ウェイト計算、Gross Exposure 自動調整 |
| `allocator.py` | 株数への変換（予算制約付き、1629.T 10株ロット対応） |
| `risk.py` | VaR/ES 計算、リスクブリーチ判定 |

### 3. Data Layer (`data/`)
市場データのライフサイクル全体を管理。

| モジュール | 責務 |
|---|---|
| `tickers.py` | US/JP ティッカー定義・変換ユーティリティの**単一正本** |
| `cache.py` | `etf_data.pkl` + `decision_cache.npz` の全 I/O 及びファイルロック制御 |
| `fetcher.py` | yfinance ダウンロード、差分更新、1629.T NAVパッチ |
| `preprocessor.py` | `df_exec` 構築（日次リターン整列、TOPIX beta計算） |
| `market_data.py` | 寄付価格取得、ギャップ計算、価格検証 |

### 4. Broker Layer (`broker/`)
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

### 5. Execution/Runner Layer (`execution/`)
実行モード別のオーケストレーション。

| モジュール | 責務 |
|---|---|
| `config.py` | YAML/env の設定パラメータロード・Pydanticスキーマによる検証 |
| `helpers.py` | broker/risk/capital alloc/orders の共有ユーティリティ |
| `decision.py` | `run_decision()` — 標準発注フロー（yfinance 使用）、`generate_daily_decision_results()` |
| `fast.py` | `run_decision_fast()` — 高速推論（yfinance 不要・precomputed cache） |
| `close.py` | `close_all_positions()`, `wait_and_auto_close()` |
| `backtest.py` | `run_production()` — 生産バックテスト実行管理 |
| `backtester.py` | `BacktestEngine` — 汎用的なバックテスト実行シミュレータ |

### 6. Compliance Layer (`compliance/`)
安全監査・法令遵守検証。

| モジュール | 責務 |
|---|---|
| `auditor.py` | `ComplianceAuditor.run_audit()` — バックテストや実行結果に対する時系列・数式漏洩等の包括的な安全監査の実行 |

### 7. Reporting Layer (`reporting/`)
| モジュール | 責務 |
|---|---|
| `formatter.py` | ログ出力・テキスト注文フォーマット・リスクレポート |
| `metrics.py` | 指標計算、チャート描画 |
| `results_format.py` | 結果フォルダ命名・マニフェスト出力 |

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
`production.py` や `runner/` は BrokerClient のみを参照し、kabu 固有コードに依存しない。

### Gross Exposure 調整
`domain/portfolio/optimizer.py::adjust_gross_exposure()` が正本。
`classify_actions()` による BUY/SELL/HOLD 分類もここに統合。

### リスクロジックの一本化
VaR/ES 計算・リスクチェック評価は `leadlag/core/risk.py` が正本。
`leadlag/execution/helpers.py::run_risk_checks()` はラッパーとして呼び出す。

### fast-mode (高速化モード) と本番処理最適化
`fast-mode`（高速判定ランナー `fast.py`）を本番運用の推奨とし、ボトルネック解消と高速化のために以下の設計を採用しています：

1. **データ取得の最適化**: `yfinance` による全ヒストリカルデータのダウンロードをスキップし、あらかじめ計算された `strategy_cache.npz` または `etf_data.pkl` 差分キャッシュを利用。kabuステーション API から当日分の US/JP 価格のみを直接取得。
2. **米国残差リターン計算のベクトル化**: `preprocessor.py::compute_us_residualized_returns()` の Python `for` ループを pandas のローリング窓演算（`rolling().cov()` / `rolling().var()`）に置き換えることでベクトル化。時系列計算の計算量をミリ秒単位に短縮。
3. **基準相関行列 (`c_full`) のメモリキャッシュ**: 2010〜2014年の基準期間データに対する EWMA 相関行列計算をメモリキャッシュ化。日次実行ごとに発生していた NumPy 縮約・固有値分解計算の重複を排除。
4. **残差空間事前分布 (`_prepare_residual_prior`) のメモリキャッシュ**: SRE-USRP モデルにおける Gram-Schmidt 直交化及び PCA 関連前処理結果をキャッシュ。

### 結果出力ディレクトリ方針
`results/YYYYMMDD_HHMMSS_<run_name>/` が唯一の正本ディレクトリ。
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
  tools/compute_gap_adjusted_distribution.py → (mu_gap, omega_gap) matrices
             ↓
  tools/run_daily_production_v2.py 
    ├── leadlag/models/sector_relative_ensemble_blp_enhanced.py (BLPX model)
    ├── mu_over_sigma ranking & baseline_style sizing (leadlag/core/portfolio.py)
    ├── PIT binning (RuleD ex-ante IR dynamic gross scaling: 0.75x or 1.00x)
    └── Fallback checks (v2 -> v1 -> SRE)
             ↓
  [Compliance/Risk/Order Flow]
    ├── leadlag/core/risk.py → evaluate_risk_checks()
    ├── leadlag/compliance/auditor.py (ComplianceAuditor)
    └── leadlag/broker/base.py → BrokerClient.submit_orders_batch()
             ↗ leadlag/broker/kabu/client.py  (kabuステーション)
             ↗ leadlag/broker/dry_run.py      (シミュレーション)
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
| [model_spec_sector_relative_ensemble.md](model_spec_sector_relative_ensemble.md) | SRE モデル仕様（旧本番・フォールバックモデル） |
| [deprecated_experiments.md](deprecated_experiments.md) | 廃止実験 (P2/P5/P6 等) のサマリ |
| [SCHEDULER_SETUP.md](SCHEDULER_SETUP.md) | Windows タスクスケジューラ設定（旧実行環境用） |