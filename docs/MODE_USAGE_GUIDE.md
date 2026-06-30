# CLI 実行ガイド

本ドキュメントは `leadlag` パッケージの CLI 実行方法（`leadlag.cli`）、戦略モード設定、および入出力仕様をまとめたものである。

---

## 1. 実行サブコマンド

`python3 -m leadlag.cli` のサブコマンドは次の 3 つ。

| サブコマンド | 用途 | 主な入力 | 主な出力 |
|:--|:--|:--|:--|
| `decision` | 当日の売買判定を1日分だけ作成（必要に応じて発注） | JP寄付き価格（API/Google/CSV）、資金量 | `decision_YYYYMMDD.csv`、（API時）`api_execution_log.json` |
| `backtest` | 履歴期間のバックテスト実行 | `--start-date` など | `daily_results.csv`、`metrics.csv`、`run_summary.json`、チャート |
| `close` | 保有建玉の全決済（引け反対売買） | API接続必須 | `close_execution_log.json` |

---

## 2. decision モード

### 2.1 概要

- 当日の TOPIX-17 寄付き価格を使って、即時に `BUY/SELL/HOLD` と数量を出力する。
- `--api-enable` を付けると、注文送信まで実行できる（`--api-dry-run` で疑似送信可）。
- 資金配分は信用前提のため、ロング・ショート各サイドで `capital × 1.5` まで利用可能（既定のグロス上限は 3.0）。
- API有効時は既存ポジションを取得し、ターゲット株数との差分（delta）のみを発注する（持ち越しポジションの重複発注を防止）。

### 2.2 JP 寄付き価格の入力優先順位

1. API 有効時: ブローカー API（立花証券 / kabu）
2. `--google-opens` 指定時: Google Finance
3. `--jp-opens-csv` 指定時: CSV
4. 上記なし: エラー

### 2.3 CSV 入力形式

```csv
ticker,open_price
1617.T,2500.0
1618.T,1800.0
1619.T,2100.0
... （全17銘柄）
1633.T,3100.0
```

ティッカーの順序は任意。1617.T〜1633.T の全 17 銘柄が含まれていれば OK。

### 2.4 コマンド例

```bash
# 基本: CSV入力で判定（手持ち資金 500万円）
python3 -m leadlag.cli decision \
  --trade-date 2026-04-15 \
  --jp-opens-csv jp_opens_google.csv \
  --capital 5000000

# Google Finance から寄付き価格を取得して判定（テキスト出力）
python3 -m leadlag.cli decision \
  --google-opens \
  --capital 600000 \
  --text-output

# API有効・注文は疑似実行
python3 -m leadlag.cli decision \
  --api-enable \
  --api-dry-run \
  --capital 5000000

# FAST MODE（高速判定、yfinance 不要）
python3 -m leadlag.cli decision \
  --fast-mode \
  --api-enable \
  --capital 5000000
```

### 2.5 出力形式

**コンソール出力（`--text-output` 指定時）：**
```
=== Trade Decision ===
ticker  open_price  signal  weight action etf_amount quantity
1617.T      2500.0 0.942650  0.000000   HOLD       0.0        0
1618.T      1800.0 0.961815  0.067882    BUY  302400.0      168
...

scale=0.5000, sigma_s=0.078463, long_short_mean_gap=0.182741
Available capital: 5,000,000 JPY
Total allocated: 3,436,200 JPY
Remaining capital: 1,563,800 JPY
```

**出力列の説明：**

| 列 | 説明 |
|:---|:---|
| `ticker` | TOPIX-17 銘柄コード |
| `open_price` | 当日初値 |
| `signal` | 当日の信号値（米国情報から推計） |
| `weight` | ポートフォリオウェイト（-1.0 〜 +1.0） |
| `action` | 売買判定（BUY / SELL / HOLD） |
| `etf_amount` | 配置金額（JPY）。BUY のみ金額が必要、SELL/HOLD は 0 |
| `quantity` | 発注数量（口数） |

**CSV 出力先：** `results/YYYYMMDD_HHMMSS_production_decision/decision_YYYYMMDD.csv`

### 2.6 資金配置のアルゴリズム（Weight-Proportional Allocation）

1. BUY と SELL の weight 絶対値合計の比率で、手持ち資金をロング予算・ショート予算に分割
   （信用前提のサイドレバレッジ 1.5。各サイド最大 `capital × 1.5` を使用可能）
2. 各サイドで weight の絶対値が大きい順にソート
3. 各ポジションについて、サイド予算と weight 比率で配置額を決定
4. 配置額 / 初値 を床関数で端数切り捨てして発注数量を計算

### 2.7 補助フラグ

| フラグ | 用途 | 注意点 |
|:--|:--|:--|
| `--fast-mode` | 事前計算キャッシュを使って高速に判定 | API有効が必須。`v3_mode='static'` のみ対応 |
| `--api-dry-run` | API注文を送らず疑似実行 | 接続確認とログ検証用 |
| `--auto-close` | 判定後に指定時刻で全決済 | `--fast-mode` 経路でのみ有効（API有効が必要） |
| `--auto-close-time HH:MM` | 自動クローズ時刻指定 | 既定 `14:50` |
| `--close-position-order 0-7` | 返済順序（ClosePositionOrder）指定 | close-positions / auto-close で有効。未指定は `0` |
| `--text-output` | 注文内容をテキスト表示 | 手動執行確認向け |
| `--capital-from-wallet` | ブローカー API の残高を資金量として使用（立花証券: 受入保証金を優先、kabu: 現物買付可能額） | `--api-enable` が必要 |
| `--google-opens` | Google Finance から JP 寄付き価格を取得 | API 不要で使用可能 |
| `--slippage-bps` | スリッページコスト（片道 bps）を上書き | バックテスト専用。未指定時はデフォルト 5.0 bps |

---

## 3. backtest モード

### 3.1 概要

戦略の履歴検証（AR/RISK/MDD等）をまとめて実行する。

### 3.2 コマンド例

```bash
# 基本
python3 -m leadlag.cli backtest \
  --start-date 2015-01-01 \
  --run-tag bt_20260415

# チャート生成を省略
python3 -m leadlag.cli backtest \
  --start-date 2015-01-01 \
  --skip-chart

# スリッページを変更してバックテスト
python3 -m leadlag.cli backtest \
  --slippage-bps 7.5
```

---

## 4. close サブコマンド

### 4.1 概要

現在の信用建玉を全て反対売買してクローズする。信用建玉のみ対象（ExecutionID がある建玉）。現物は自動でスキップ。

### 4.2 コマンド例

```bash
# 疑似実行
python3 -m leadlag.cli close \
  --api-enable \
  --api-dry-run

# 実発注
python3 -m leadlag.cli close \
  --api-enable

# 実発注（返済順序を指定）
python3 -m leadlag.cli close \
  --api-enable \
  --close-position-order 4
```

### 4.3 注意事項

- `--api-enable` なしでは実行不可。
- 返済順序は `ClosePositionOrder (0-7)` を使用。未指定時はデフォルト `0` を送信。

---

## 5. 本番 v2 (Residual-BLPX) および PCA-Ensemble 実行ツール

本番 v2 モデル（Residual-BLPX-RA v2）および legacy PCA-Ensemble 用の実行・検証スクリプトが `tools/` に用意されている。

### 5.1 本番 v2 日次実行スクリプト (`tools/run_daily_production_v2.py`)
本番 v2 （Residual-BLPX-RA v2: `mu_over_sigma` ランキング + `RuleD` 動的グロス）のデイリー注文生成および自動安全監査（Safety Audit）を実行する。
* 本番実行（9:10 POST_OPEN データ取得後）：
  ```bash
  python tools/run_daily_production_v2.py \
      --trade-date latest \
      --gap-input-dir results/gap_adjusted_distribution/latest
  ```
* ドライラン実行（疑似注文生成）：
  ```bash
  python tools/run_daily_production_v2.py \
      --trade-date 2026-06-16 \
      --gap-input-dir results/gap_adjusted_distribution/latest \
      --dry-run true
  ```
* セルフテスト実行：
  ```bash
  python tools/run_daily_production_v2.py --self-test true
  ```

### 5.2 ギャップ調整済み予測分布生成 (`tools/compute_gap_adjusted_distribution.py`)
当日の日本市場寄付後のギャップオープン価格および前日の米国終値に基づき、ギャップ調整済みの予測期待値 $\mu_{\text{gap}}$ と予測共分散 $\Omega_{\text{gap}}$ を計算して保存する。
  ```bash
  python tools/compute_gap_adjusted_distribution.py \
      --trade-date latest \
      --output-dir results/gap_adjusted_distribution/latest
  ```

### 5.3 日次シャドウ監視ランナー (`tools/run_daily_residual_blpx_shadow.py`)
本番 baseline (v1) と各シャドウ候補（RuleD等）を並行して計算し、ウェイト差分やシグナル特性の監視を行う。
  ```bash
  python tools/run_daily_residual_blpx_shadow.py \
      --trade-date latest \
      --gap-dir results/gap_adjusted_distribution/latest
  ```

### 5.4 Legacy PCA-Ensemble ツール
* PCA-Ensemble バックテスト（アーカイブされた PCA-Ensemble 設定ファイルを使用）：
  ```bash
  python tools/backtest_sector_relative_ensemble.py \
      --config configs/archive/production_before_residual_blpx_20260614.yaml \
      --slippage-bps 5 \
      --output-dir results/sector_relative_ensemble/
  ```
* PCA-Ensemble 日次実行（アーカイブされた PCA-Ensemble 設定ファイルを使用）：
  ```bash
  python tools/run_daily_sector_relative_ensemble.py \
      --config configs/archive/production_before_residual_blpx_20260614.yaml \
      --signal-date latest \
      --output-dir live/sector_relative_ensemble/ \
      --dry-run
  ```

---

## 6. 戦略モード（signal / weight / v3）

これらはサブコマンドとは別物で、シグナル生成とウェイト計算の内部設定を変更する。
本番の動作設定は `configs/production.yaml` の各セクション（`portfolio`, `residualization` 等）で指定し、実行時に読み込まれる。

### 6.1 signal_mode

| 値 | 用途 | ロジック概要 |
|:--|:--|:--|
| `baseline` | Gap 補正なしの基準形 | 予測シグナルをそのまま利用 |
| `gap_residual`（標準） | 寄付きギャップを控除して判定 | `signal = r_hat_jp_cc - gap_open_coef * gap` |
| `gap_tolerant` | 約定可能性を加味した保守運用 | 制限価格条件を満たした銘柄のみ残し再正規化 |

### 6.2 weight_mode

| 値 | 用途 | ロジック概要 |
|:--|:--|:--|
| `signal`（標準） | 確信度に応じて配分 | シグナルの中央値からの乖離幅で加重 |
| `equal` | 均等配分 | ロング・ショートそれぞれ同ウェイト |

> **Warning:** PCA-Ensemble 本番では `signal` モードが**必須**。`equal` は検証目的以外で使用禁止。

### 6.3 v3_mode

| 値 | 用途 | ロジック概要 | 制約 |
|:--|:--|:--|:--|
| `static`（標準） | 事前定義ベクトルを使用 | 静的な `V0` を使う | FAST MODE 対応可 |
| `dynamic` | 市場状態連動で更新 | βベースで `v3` を都度再構成 | FAST MODE 非対応 |

### 6.4 ranking.mode（Residual-BLPX-RA v2 のみ）

| 値 | 用途 | ロジック概要 |
|:--|:--|:--|
| `mu_over_sigma`（本番標準） | リスク調整リターンで順位付け | 予測期待リターン $\mu_{\text{gap}}$ を予測標準偏差 $\sigma_{\text{gap}}$ で割った値 |
| `mu_gap` | 単純期待リターンで順位付け | 予測期待リターン $\mu_{\text{gap}}$ をそのまま使用 |

### 6.5 gross_scaling.rule（Residual-BLPX-RA v2 のみ）

| 値 | 用途 | ロジック概要 |
|:--|:--|:--|
| `RuleD`（本番標準） | 予測確信度に応じた動的グロス調整 | 直近 252 営業日の ex-ante IR 実績の 33% 分位点閾値を下回る（Low ビン）場合、グロス乗数を `0.75×` (適用グロス 150%) に縮小。それ以外 (Med/High ビン) は `1.00×` (適用グロス 200%) |
| `none` | 固定グロス | 常にフルエクスポージャー（グロス 200%）を維持 |

### 6.6 検証コードでの設定例

```python
import yaml
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel

# 本番 v2 設定ファイルを読み込んでモデルを初期化
config_path = "configs/production.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
model = SectorRelativeEnsembleBLPEnhancedModel(config)
```

---

## 7. CLI オプション全一覧

`python3 -m leadlag.cli <subcommand> [options]`

### 全サブコマンド共通または decision サブコマンドのオプション

| オプション | 型 | デフォルト | 説明 |
|:---|:---|:---|:---|
| `--start-date` | str | `2015-01-01` | バックテスト開始日 |
| `--output-root` | str | `results/` | 出力ディレクトリルート |
| `--run-tag` | str | (timestamp) | 出力フォルダの識別子 |
| `--trade-date` | str | (today) | 取引日 (YYYY-MM-DD) |
| `--jp-opens-csv` | str | — | JP 寄付き価格 CSV パス |
| `--capital` | float | `1000000` | 手持ち資金 (JPY) |
| `--capital-from-wallet` | flag | — | ブローカー API 残高を資金量として使用（立花証券: 受入保証金、kabu: 現物買付可能額） |
| `--api-enable` | flag | — | ブローカー API 有効化（立花証券 / kabu） |
| `--api-url` | str | (env) | API URL |
| `--api-token` | str | (env) | API トークン |
| `--api-dry-run` | flag | — | API 疑似実行モード |
| `--fast-mode` | flag | — | 高速推論モード |
| `--auto-close` | flag | — | 自動決済 |
| `--auto-close-time` | str | `14:50` | 自動決済時刻 |
| `--close-position-order` | int | `0` | 返済順序 (0-7) |
| `--google-opens` | flag | — | Google Finance から JP 寄付き取得 |
| `--text-output` | flag | — | テキスト出力 |
| `--skip-chart` | flag | — | (backtest用) チャート生成を省略 |
| `--slippage-bps` | float | — | (backtest用) スリッページ (bps/片道) |

---

## 8. トラブルシューティング

| エラー | 原因 | 対処 |
|:---|:---|:---|
| `CSV must have at least 17 rows` | CSV が全 17 銘柄を含まない、またはフォーマット不正 | ティッカーがスペースなく正確に記載されているか確認 |
| `Previous close not found for ...` | 指定 trade-date の前営業日の終値データがない | データが存在する日付範囲を指定 |
| quantity が 0 のみ | signal が小さい、または資金が不足 | `--capital` を増やすか別の日付で試行 |
| `FAST MODE requires --api-enable` | fast-mode は API 接続が必須 | `--api-enable` を追加 |
| `--capital-from-wallet requires --api-enable` | wallet 機能は API が必要 | `--api-enable` を追加 |
| `code=11014: 現金信用区分に誤りがあります` | `TACHIBANA_MARGIN_TRADE_TYPE` と口座の信用取引区分が不一致 | `.env` の `TACHIBANA_MARGIN_TRADE_TYPE` を確認（1=制度信用, 2/3=一般信用） |
| `max_capital = 0 JPY` で発注されない | 入金がAPIに未反映、または受入保証金が0 | 入金反映後に再実行 |
