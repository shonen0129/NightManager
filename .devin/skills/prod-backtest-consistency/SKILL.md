---
name: prod-backtest-consistency
description: 本番環境・バックテスト・docs の設定・コード間で差分が出ないかを体系チェックする。モデル/コスト/グロス/オーバーナイト/PIT/gap 分布の不整合・ドキュメントとの乖離・バックテストと本番 shadow の指標差を網羅的に確認し、再現可能なレポートを残す。本番昇格・リファクタリング・backtest 指標差調査・docs 更新時に必ず参照すること。
---

# 本番・バックテスト・docs 整合性チェック スキル

## 目的

- 本番実行・バックテスト・docs の設定値とコード経路が一致しているかを機械的に確認
- 理論値バックテスト（オプティミスティック）と悲観的バックテスト（5分足不利側・整数株丸め）の比較を標準化
- バックテスト vs 実口座 shadow の差分が「ロジック」「PIT」「執行面」のどこに由来するかを特定
- 不変条件（ルックアヘッド禁止・ベースライン分離・市場中立・ティッカー定義・当日 gap 行列のみ）を維持

## 前提

- `AGENTS.md` の不変条件とデータ整合規約を遵守
- 本番 config 正本は `configs/production/production.yaml`（`production_v2_primary_ruleD.yaml` は旧版）
- 本番 entrypoint は `tools/production/run_daily_production_v2.py`
- バックテスト入口は `scripts/experiments/run_v2_backtest_exact_production.py`（`BacktestEngine.run_v2_backtest`）
- 本番 shadow は `scripts/experiments/build_v2_production_shadow_run.py` + `tools/validation/monitor_residual_blpx_shadow_performance.py`

## チェック項目（必須・順序）

### 1. config の差分

`configs/production/production.yaml` を正本として以下を確認:

- `blpx` パラメータ: `param_set`, `rho`, `alpha_xx`, `alpha_yx`, `alpha_yy`, `lambda_pca`, `lambda_sector`, `beta_conf`, `winsor_sigma`, `blp_window`
- `portfolio`: `side_leverage`, `gross_exposure_target`, `net_exposure_limit`
- `costs`: `slippage_bps_per_side`, `cost_bps_per_gross`, `overnight_alpha_long`, `overnight_alpha_short`, `buy_interest_annual`, `borrow_fee_annual`, `reverse_fee_bps`
- `ml_order_overlay`: `enabled`, `model_dir`

確認コマンド:

```bash
diff -u configs/production/production.yaml configs/production/production_v2_primary_ruleD.yaml
grep -A20 "costs:" configs/production/production.yaml
grep -A5 "portfolio:" configs/production/production.yaml
```

基準: 本番用の値が `run_daily_production_v2.py` と `run_v2_backtest_exact_production.py` で同一であること。旧版 config を本番参照に使っていないこと。

### 2. コード経路の同一性

本番とバックテストが同一のポートフォリオ生成ロジックを使っているか:

```bash
grep -E "generate_v2_production_portfolio_with_overlay|generate_v2_production_portfolio" tools/production/run_daily_production_v2.py
grep -E "generate_v2_production_portfolio_with_overlay|generate_v2_production_portfolio" scripts/experiments/run_v2_backtest_exact_production.py
```

基準: 本番・バックテストともに `generate_v2_production_portfolio_with_overlay` を使用し、同一 `cfg` オブジェクトを読み込むこと。手動ハック（`latest` 別 config、コメントアウトのパラメータ変更）がないこと。

### 3. gap 分布・PIT 履歴

本番は当日 `latest`、バックテストはフル履歴 `20260731_024303` を使うため、PIT 履歴の影響を確認する:

```bash
# 本番の PIT 状態
cat live/production_residual_blpx/pit_binning.json

# 本番 shadow を PIT 制限付きで実行
python3 scripts/experiments/build_v2_production_shadow_run.py \
  --start 2020-01-06 --end 2026-07-29 --overlay true \
  --max-pit-history 20 \
  --shadow-root shadow_runs/v2_production_pit20_20200106_20260729 \
  --clean true

# 監視
python3 tools/validation/monitor_residual_blpx_shadow_performance.py \
  --shadow-root shadow_runs/v2_production_pit20_20200106_20260729 \
  --gap-input-dir live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --output-dir results/shadow_monitor_v2_production_pit20
```

基準: `--max-pit-history` を変更しても戦略の AR/Sharpe が大きく崩れないことを確認。本番 `pit_binning.json` の `fallback_flag` / `history_count` / `assigned_bin` を記録し、バックテストの `PIT bin=High/Low` との差を説明できること。

### 4. ウェイト再現性（本番ロジックとバックテストの差分）

本番の `decision_YYYYMMDD.csv` と同一の gap 分布を使ったバックテストが、同じウェイトを生成するかを確認する:

```bash
python3 scripts/experiments/run_live_aligned_v2_backtest.py \
  --start-date 2026-07-29 --end-date 2026-08-05
```

基準: 各日で `sign_agreement=100.0%` かつ `weight_rmse < 0.02` であること。一致しない場合、使用している gap 分布が本番と異なっている可能性を優先調査。

### 5. コスト・オーバーナイト

`production.yaml` の `costs` と `BacktestEngine.run_v2_backtest` デフォルト/呼び出しが一致しているか確認:

```bash
# バックテストのコスト使用箇所
grep -A30 "overnight_alpha" src/leadlag/execution/backtester.py
grep -A5 "side_leverage" src/leadlag/execution/backtester.py

# オーバーナイト感度分析（実行済みであれば結果参照）
python3 scripts/experiments/overnight_sensitivity_v2.py
```

基準: `overnight_alpha_long=0.75`, `overnight_alpha_short=0.5`, `slippage_bps_per_side=5`, `buy_interest_annual=0.025`, `borrow_fee_annual=0.0115`, `reverse_fee_bps=2.0`, `side_leverage=1.5` が config とコードで一致していること。これ以外の値を使っている場合、正本 config に合わせて更新・コメントで理由を残す。

### 6. 5分足データ・執行モデル

理論値バックテストは 5分足 09:10 (High+Low)/2 を仮定している。本番は実約定があるため差分が出る。執行面の差分を評価する:

```bash
# 5分足 cache 更新・範囲確認
python3 scripts/experiments/update_5m_cache.py
python3 scripts/experiments/check_5m_cache.py

# 悲観的バックテスト（5分足不利側＋整数株丸め）
python3 scripts/experiments/run_v2_backtest_pessimistic.py
python3 scripts/experiments/run_v2_backtest_theoretical_5m.py
```

基準: 理論値と悲観的の乖離が「執行・丸め・資本」由来であることが説明できること。5分足データが限定的な場合はその期間で評価を完了させ、全体期間に必要であれば別途データ取得方法を検討する。

### 7. docs との整合

`docs/ARCHITECTURE.md`, `docs/モデル技術仕様書.md`, `AGENTS.md` に書かれた数値・挙動がコードと一致するか:

```bash
grep -E "side_leverage|overnight|slippage|borrow|reverse|gross|fallback|High\+Low|RuleD|PIT" docs/ARCHITECTURE.md
grep -E "side_leverage|overnight|slippage|borrow|reverse|gross|fallback|RuleD|PIT" docs/モデル技術仕様書.md
```

基準: docs 記述と `production.yaml`・`backtester.py`・`production_v2.py`・`ml_order_overlay.py` の実装値が一致していること。差分がある場合は docs を正本 config に合わせて更新する。

### 8. 不変条件の機械的検証

`ComplianceAuditor` で以下を確認:

```bash
# ルックアヘッド・残余化・重み検証
python3 -m leadlag.compliance.auditor --config configs/production/production.yaml
```

基準: `check_pit_binning_lookahead`, `check_residualization_leakage`, `check_market_neutral`, `check_gross_exposure_limit` がすべて PASS。FAIL の場合は修正してからリリース。

### 9. 実口座とのズレ（受入保証金）

実口座の `wallet_close_*.json` と理論値バックテストの日次リターンを突合する:

```bash
python3 scripts/experiments/analyze_v2_bt_vs_actual.py
```

基準: 差分の主因が「PIT 履歴」「gap 分布」「執行/整数株」「ukeire_hosyoukin の代理性」に特定できること。同日のポートフォリオ一致度が低い場合は `run_live_aligned_v2_backtest.py` を優先実行。

### 10. 本番 shadow との比較（side leverage・overnight・コストの影響）

`monitor_residual_blpx_shadow_performance.py` は side_leverage/overnight/financing/borrow/reverse を含まない簡易評価。 exact バックテストは含む。両者の差分を理解する:

```bash
# shadow 評価
python3 tools/validation/monitor_residual_blpx_shadow_performance.py \
  --shadow-root shadow_runs/v2_production_20200106_20260729_overlay \
  --gap-input-dir live/pipeline_data/gap_adjusted_distribution/20260731_024303 \
  --output-dir results/shadow_monitor_v2_production

# exact バックテスト
python3 scripts/experiments/run_v2_backtest_exact_production.py
```

基準: shadow の AR が exact バックテストより小さいこと（side_leverage/overnight/コストの不足分）。差分が原因不明な場合は、使用している `gap_input_dir` と `cfg` を比較する。

## 実行手順（推奨）

1. `git status` で変更ファイルを確認
2. `grep` で config・docs の差分を点検
3. `run_live_aligned_v2_backtest.py` でウェイト再現性を確認
4. `run_v2_backtest_exact_production.py` で理論値を更新
5. `run_v2_backtest_pessimistic.py` で悲観的を更新
6. `analyze_v2_bt_vs_actual.py` で実口座ズレを更新
7. `build_v2_production_shadow_run.py` + `monitor_residual_blpx_shadow_performance.py` で shadow を更新
8. 不整合があれば `AGENTS.md`・`docs/ARCHITECTURE.md`・`docs/モデル技術仕様書.md`・`production.yaml` を同期修正

## 不変条件（このスキルでも絶対に守る）

1. **ルックアヘッド禁止**: すべてのローリング統計・PITビニングは strictly historical。`ComplianceAuditor` の監査項目を無効化しない
2. **ベースライン分離**: 2010–2014 は相関・事前分布の基準期間。backtest start は 2015-01-05 以降
3. **市場中立制約**: net ±0.05、gross ≤ 2.0（RuleD 適用後）
4. **ティッカー定義**: `src/leadlag/data/tickers.py` が単一正本。ユニバース変更時は `core/correlation.py` の感応度ラベルも同時更新
5. **当日 gap 行列の使用**: 前日行列をコピーして当日日付で使用してはならない（2026-07-14 の事故を踏まえる）
6. **テスト弱め禁止**: 変更後は `bash scripts/run_tests_parallel.sh` または `python3 -m pytest tests/ -v` を通す

## 実行時の注意

- `python3 -c "..."` のインライン実行は禁止。必ず `scripts/experiments/` 配下のスクリプト経由で実行する
- 長時間実行には timeout を付ける（`bash scripts/run_tests_parallel.sh` 約8分、`update_5m_cache.py` 約1分）
- ハングパターンに注意: yfinance ダウンロード、fcntl ロック、auto-close 待機、API 再試行。`docs/スタック再発防止策.md` 参照
- レポートは `reports/<sprint名>/` に markdown で残す（既存形式に倣う）

## 参照する既存スクリプト

| 目的 | スクリプト |
|------|-----------|
| 本番同一設定バックテスト | `scripts/experiments/run_v2_backtest_exact_production.py` |
| 本番ロジック再現（当日 latest gap 使用） | `scripts/experiments/run_live_aligned_v2_backtest.py` |
| 理論値バックテスト（5分足期間） | `scripts/experiments/run_v2_backtest_theoretical_5m.py` |
| 悲観的バックテスト | `scripts/experiments/run_v2_backtest_pessimistic.py` |
| 実口座ズレ分析 | `scripts/experiments/analyze_v2_bt_vs_actual.py` |
| PIT 制限 shadow | `scripts/experiments/build_v2_production_shadow_run.py` |
| shadow 評価 | `tools/validation/monitor_residual_blpx_shadow_performance.py` |
| オーバーナイト感度 | `scripts/experiments/overnight_sensitivity_v2.py` |
| 5分足 cache 更新 | `scripts/experiments/update_5m_cache.py` |
| 5分足 cache 確認 | `scripts/experiments/check_5m_cache.py` |

## 本番データパイプライン cron 運用チェック

本番の発注は、**当日の gap 行列 `mu_gap_YYYYMMDD.npy` / `omega_gap_YYYYMMDD.npy`** が新鮮かどうかに依存する。ここでは、yfinance の遅延や前日行列コピーなどによる **stale gap 分布** を防ぐための cron 運用と検証手順を定める。

### 推奨 cron（launchd）スケジュール

| 時刻 (JST) | ジョブ | 内容 |
|-----------|--------|------|
| **08:00** | `scripts/batch/update_market_data.sh` | yfinance から US/JP ETF OHLC を **force** 再取得。US クローズは夏時間 05:00 / 冬時間 06:00 JST なので、08:00 ならデータが安定している。 |
| **08:15** | `scripts/batch/run_distribution_diagnostics.sh` | `etf_data.pkl` を使い `Omega_struct` を計算。`download_data(force=True)` により、07:00 以前に cron などで更新していた TTL 切れ cache も吸収できる。 |
| **09:10** | `scripts/batch/run_decision_v2.sh` | `run_gap_distribution.sh` → `run_v2_decision.py` を実行。Tachibana 9:10 価格を gap 行列に注入して発注。 |
| **14:50** | `scripts/batch/run_close_positions.sh` | 大引け成り寄りでポジション解消。 |

### 鮮度チェック（失敗時は安全側に倒れる）

```bash
# 1. distribution_diagnostics の最新 trade_date を確認
cat live/pipeline_data/distribution_diagnostics/latest/distribution_panel_long.csv | \
  awk -F, 'NR>1 {gsub(/ .*/, "", $2); print $2}' | sort | tail -1
# 期待値: 本日 YYYY-MM-DD

# 2. gap_adjusted_distribution の最新 mu_gap の日付を確認
ls live/pipeline_data/gap_adjusted_distribution/latest/matrices | grep mu_gap | tail -1
# 期待値: mu_gap_YYYYMMDD.npy（本日）

# 3. live/production_residual_blpx/pit_binning.json の確認
cat live/production_residual_blpx/pit_binning.json
# fallback_flag=false, history_count>=1500 であること
```

基準:

- `run_distribution_diagnostics.sh` では `max trade_date == TODAY` のとき INFO、それ以外は WARNING。
- `run_gap_distribution.sh` では `max trade_date != TODAY` のとき **ERROR で exit 1**。`run_decision_v2.sh` はこれを検知し、decision がフラットポジション（w_final=0）になる。
- 何らかの理由で stale な場合、**前日行列をコピーして当日日付で使わない**。本戦略の不変条件に抵触し、誤ったポジションで発注するリスクがある。

### yfinance 遅延・データ未取得への対応

`download_data` は 12 時間 TTL を持つ。08:00 の `update_market_data.sh` は `download_data(force=True)` を呼び TTL を無視して強制更新する。更新後は必ずログを確認:

```bash
tail -5 logs/update_market_data_YYYYMMDD.log
# us_close: ... last index=YYYY-MM-DD
# jp_close: ... last index=YYYY-MM-DD
```

基準:

- `us_close` 最終日付は **前営業日の US close**（本日 JST 08:00 時点で）。
- `jp_close` / `jp_open` 最終日付は **本日または前営業日**（本日 09:00 寄り前のため）。
- もし `us_close` が 2 営業日以上遅れている場合、`update_market_data.sh` を手動再実行、または `run_distribution_diagnostics.sh` を遅らせる。
- 手動更新: `bash scripts/batch/update_market_data.sh`
- 手動 diagnostics: `bash scripts/batch/run_distribution_diagnostics.sh`

### PIT 履歴正本の管理

- 正本ファイル: `live/pipeline_data/gap_adjusted_distribution/full_history_diagnostics.csv`
- `run_gap_distribution.sh` は新しい diagnostics 行をこの正本にマージする。
- `production_v2.py` の `load_pit_ir_history` は、`gap_input_dir` 内の `full_history_diagnostics.csv` を優先して読む。
- RuleD PIT ビニングに必要な 252 日以上の IR 履歴は、この正本によって維持される。
- もし `pit_binning.json` で `fallback_flag=true` / `history_count < 1000` が出た場合、正本ファイルや `latest` シンボリックリンクを確認する。

### 月曜 / 祝日 / 夏時間の注意

- **月曜**: US close は前営業日（金曜）の 05:00/06:00 JST なので、08:00 `update_market_data.sh` まで待つ。
- **日本祝日**: `run_decision_v2.sh` は平日 cron だが、取引日でない場合 `run_gap_distribution.sh` の鮮度チェックが失敗しフラットポジションになる。これは正常。
- **アメリカ祝日**: US データが存在しない場合、`distribution_diagnostics` は最新の US 取引日を signal_date として使用する。本日の `trade_date` が存在しない場合、鮮度チェックが失敗する。日本市場が開いていても US 信号がない場合は取引を見送る。
