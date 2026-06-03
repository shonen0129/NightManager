# Mode一覧と使用方法

このドキュメントは、本リポジトリで使う「mode」を以下の2種類に分けて整理したものです。

1. 実行モード（`src/production.py --mode ...`）
2. 戦略モード（`signal_mode` / `weight_mode` / `v3_mode`）

---

## 1. 実行モード（`--mode`）

`src/production.py` のCLI実行モードは次の3つです。

| mode | 用途 | 主な入力 | 主な出力 |
|:--|:--|:--|:--|
| `decision`（デフォルト） | 当日の売買判定を1日分だけ作成（必要に応じて発注） | JP寄付き価格（API/Google/CSV）、資金量 | `decision_YYYYMMDD.csv`、（API時）`api_execution_log.json` |
| `backtest` | 履歴期間のバックテスト実行 | `--start-date` など | `daily_results.csv`、`metrics.csv`、`run_summary.json`、（既定で）チャート |
| `close-positions` | 保有建玉の全決済（引け反対売買） | API接続必須 | `close_execution_log.json` |

### 1.1 decision モード

#### 何に使うか
- 当日のTOPIX-17寄付き価格を使って、即時に `BUY/SELL/HOLD` と数量を出す。
- `--api-enable` を付けると、注文送信まで実行できる（`--api-dry-run` で疑似送信可）。
- 資金配分は信用前提のため、ロング・ショート各サイドで `capital × 1.5` まで利用可能（既定のグロス上限は 3.0）。

#### 入力の優先順位（JP寄付き）
- 1) API有効時: kabu API
- 2) `--google-opens` 指定時: Google Finance
- 3) `--jp-opens-csv` 指定時: CSV
- 4) 上記なし: エラー

#### 代表コマンド

```bash
# APIなし・CSV入力で判定
python src/production.py \
  --mode decision \
  --trade-date 2026-04-15 \
  --jp-opens-csv jp_opens_google.csv \
  --capital 5000000
```

```bash
# API有効・注文は疑似実行
python src/production.py \
  --mode decision \
  --api-enable \
  --api-dry-run \
  --capital 5000000
```

```bash
# FAST MODE（高速判定）
python src/production.py \
  --mode decision \
  --fast-mode \
  --api-enable \
  --capital 5000000
```

### 1.2 backtest モード

#### 何に使うか
- 戦略の履歴検証（AR/RISK/MDD等）をまとめて実行。

#### 代表コマンド

```bash
python src/production.py \
  --mode backtest \
  --start-date 2015-01-01 \
  --run-tag bt_20260415
```

#### 補足
- `--skip-chart` を付けるとチャート生成を省略可能。

### 1.3 close-positions モード

#### 何に使うか
- 現在の信用建玉を全て反対売買してクローズする。
- 信用建玉のみ対象（ExecutionIDがある建玉）。現物は自動でスキップ。

#### 代表コマンド

```bash
# 疑似実行
python src/production.py \
  --mode close-positions \
  --api-enable \
  --api-dry-run
```

```bash
# 実発注
python src/production.py \
  --mode close-positions \
  --api-enable
```

```bash
# 実発注（返済順序を指定）
python src/production.py \
  --mode close-positions \
  --api-enable \
  --close-position-order 4
```

#### 注意
- `--api-enable` なしでは実行不可。
- 返済順序は `ClosePositionOrder (0-7)` を使用。未指定時はデフォルト `0` を送信。

---

## 2. decisionに関連する補助フラグ（実運用で実質的なモード切替）

| フラグ | 用途 | 注意点 |
|:--|:--|:--|
| `--fast-mode` | 事前計算キャッシュを使って高速に判定 | API有効が必須。`v3_mode='static'` のみ対応 |
| `--api-dry-run` | API注文を送らず疑似実行 | 接続確認とログ検証用 |
| `--auto-close` | 判定後に指定時刻で全決済 | 現状は `--fast-mode` 経路でのみ有効（かつAPI有効が必要） |
| `--auto-close-time HH:MM` | 自動クローズ時刻指定 | 既定 `14:50` |
| `--close-position-order 0-7` | 返済順序（ClosePositionOrder）指定 | close-positions / auto-close で有効。未指定は `0` |
| `--text-output` | 注文内容をテキスト表示 | 手動執行確認向け |

---

## 3. 戦略モード（signal/weight/v3）

これらは `--mode` とは別物で、シグナル生成とウェイト計算の中身を変える設定です。

### 3.1 signal_mode

| 値 | 用途 | ロジック概要 | 向いている場面 |
|:--|:--|:--|:--|
| `baseline` | Gap補正なしの基準形 | 予測シグナルをそのまま利用 | 比較ベースライン |
| `gap_residual`（標準） | 寄付きギャップを控除して判定 | `signal = r_hat_jp_cc - gap_open_coef * gap` | 実運用の標準判断 |
| `gap_tolerant` | 約定可能性を加味した保守運用 | 制限価格条件を満たした銘柄のみ残し再正規化 | 実約定重視の検証 |

補足:
- `gap_tolerant` は `jp_close_sig_*` / `jp_open_trade_*` 列がある場合に約定フィルタを適用します。列がない場合は通常ウェイト計算へフォールバックします。
- FAST MODEの `PrecomputedLeadLagStrategy` では専用フィルタ分岐がなく、実質通常ウェイト計算になります。

### 3.2 weight_mode

| 値 | 用途 | ロジック概要 |
|:--|:--|:--|
| `signal`（標準） | 確信度に応じて配分 | シグナルの中央値からの乖離幅で加重 |
| `equal` | 均等配分 | ロング・ショートそれぞれ同ウェイト |

### 3.3 v3_mode

| 値 | 用途 | ロジック概要 | 制約 |
|:--|:--|:--|:--|
| `static`（標準） | 事前定義ベクトルを使用 | 静的な `V0` を使う | FAST MODE対応可 |
| `dynamic` | 市場状態連動で更新 | βベースで `v3` を都度再構成 | FAST MODE非対応 |

---

## 4. 戦略モードの設定方法

現状の `src/production.py` では、`signal_mode` / `weight_mode` / `v3_mode` をCLI引数で直接変更できません。

変更方法は次の2通りです。

1. `src/config.py` の `STRATEGY_DEFAULTS`（および `PRODUCTION_DEFAULTS` 反映）を編集する。
2. 検証スクリプト側で `LeadLagStrategy(...)` や `StrategyConfig(...)` の引数として明示指定する。

例（検証コード内）:

```python
from strategy import LeadLagStrategy

strategy = LeadLagStrategy(
    df_exec,
    signal_mode="baseline",
    weight_mode="equal",
    v3_mode="static",
)
```

---

## 5. 参考: run_forced.py

ルートの `run_forced.py` は検証用ユーティリティで、以下を強制的に上書きして実行します。

- `dispersion_filter = False`
- `max_net_exposure = 10.0`
- `max_gross_exposure = 10.0`

本番運用向けの通常モードではありません。安全性検証やデバッグ用途に限定して利用してください。

## 6. Proposal B（TOPIXベータ補正）に関する実務メモ

- Production defaults: `topix_beta_coef` をデフォルトで `0.6` に更新しました（適用日: 2026-05-27）。`beta_window` は `60` のままです。
- FAST mode 注意点: `--fast-mode` は事前計算キャッシュ（`data/decision_cache.npz`）を使用します。本番で Proposal B を有効にする場合、キャッシュに `topix_night_return` と `jp_beta_*.T`（例: `jp_beta_1617.T`）が含まれている必要があります。キャッシュに欠損があると Proposal B の補正が適用されません。
- キャッシュ再構築（推奨）:

```bash
# decision cache を再作成し、Proposal B の配列を含める
python tools/rebuild_and_test_proposal_b.py
```

- サニティチェック（再構築後）:

```bash
# 3-way 比較プロット/時系列出力
python tools/compare_three_strategies.py

# MDD 等の簡易指標出力
python tools/compute_mdd_three.py
```

- 無効化（緊急ロールバック）: 一時的に Proposal B を無効化したい場合は `src/config.py` の `topix_beta_coef` を `0` に戻し、キャッシュを再構築してください。

- 実運用運用時の注意: FAST モードで運用する場合、キャッシュのバージョン管理（どの date/run で生成したか）を `results/` 出力に含める運用手順を推奨します。

