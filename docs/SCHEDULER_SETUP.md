# 自動スケジューラ セットアップガイド

日米ラグ戦略の自動実行を設定する手順です。現行の正本は macOS の launchd です。
Windows 用スクリプトは `archive/legacy_scripts/windows/` に保管しており、現行運用では使用しません。

## スケジュール一覧

| タスク名 | 実行時刻 | Windows | macOS | 内容 |
|---|---|---|---|---|
| `日米ラグ_AutoLogin` | 毎朝 7:00 | —（legacy archive） | — | kabuステーション自動ログイン |
| `日米ラグ_DistributionDiagnostics` | 月〜土 8:15 | — | `run_distribution_diagnostics.sh` | 分布診断の事前計算 |
| `日米ラグ_Decision` | 毎朝 9:10 | —（legacy archive） | `run_decision_v2.sh` | 売買判定 (`leadlag cli decision`) |
| `日米ラグ_ClosePositions` | 毎日 14:50 | —（legacy archive） | `run_close_positions.sh` | 引け反対売買 (`leadlag cli close`) |
| `日米ラグ_PnlReport` | 毎日 15:40 | — | `run_pnl_report.sh` | 未完了runの読取専用照合と引け損益レポート |

## 前提条件

### 共通
- `.env` に `KABU_ACCOUNT_NUMBER`, `KABU_PASSWORD` 等の環境変数が設定済みであること
- `creds/credentials.json` (Gmail API) が配置済みで、初回認証（`token.json` 生成）が完了していること

### macOS
- Python 仮想環境 (`.venv`) がプロジェクトルートに存在すること
- プロジェクトディレクトリが iCloud 外にあること（iCloud 内では launchd が `Operation not permitted` エラーで実行できません）

## セットアップ手順

### Windows

現行リポジトリにはWindows用のアクティブなscheduler入口はありません。
過去の `.bat` / `.ps1` は `archive/legacy_scripts/windows/` にあり、必要な移行は
Windows側で別途設計してください。

### macOS

#### 1. 自動セットアップ（推奨）

```bash
bash scripts/batch/setup_scheduler_macos.sh
```

これで5つのタスク（market data update / distribution diagnostics / Decision / Close / P&L report）が launchd に登録されます。既存のplistはプロジェクトルートから生成し直されるため、作業ディレクトリを移動した後も古いパスを残しません。
旧コマンド `bash scripts/batch/install_launchd.sh` はこの手順へ委譲する互換入口です。

> [!WARNING]
> プロジェクトディレクトリが iCloud 内にある場合、launchd からスクリプトにアクセスできません（`Operation not permitted`）。iCloud 外のディレクトリに移動してからセットアップしてください。

#### 2. 手動テスト実行

```bash
bash scripts/batch/run_decision_v2.sh
bash scripts/batch/run_close_positions.sh
```

#### タスク状態確認

```bash
launchctl list | grep leadlag
```

#### タスクの削除

```bash
launchctl unload ~/Library/LaunchAgents/com.leadlag.decision.plist
launchctl unload ~/Library/LaunchAgents/com.leadlag.close.plist
launchctl unload ~/Library/LaunchAgents/com.leadlag.pnl_report.plist
launchctl unload ~/Library/LaunchAgents/com.leadlag.update-market-data.plist
```

## 動作確認

### ログ確認

実行ログは `var/logs/` ディレクトリに日付別で出力され、job guardの結果は
`var/logs/job_guard/` に保存されます：

```
var/logs/
├── auto_login_20260507.log
├── decision_20260507.log
└── close_positions_20260507.log
```

### macOS タスク状態確認

```bash
launchctl list | grep leadlag
```

## タスクの削除

### macOS

```bash
launchctl unload ~/Library/LaunchAgents/com.leadlag.update-market-data.plist
launchctl unload ~/Library/LaunchAgents/com.leadlag.decision.plist
launchctl unload ~/Library/LaunchAgents/com.leadlag.close.plist
launchctl unload ~/Library/LaunchAgents/com.leadlag.pnl_report.plist
```

## スクリプトのカスタマイズ

### `run_decision_v2.sh` のオプション（macOS正本）

`leadlag cli decision` に渡すオプションを変更できます。macOSの正本入口は
`scripts/batch/run_decision_v2.sh`です：

| オプション | 説明 | デフォルト |
|---|---|---|
| `--config configs/production/production.yaml` | 継承を解決した本番設定 | 指定済み |
| `--api-enable` | 設定で選ばれたbrokerへ注文送信 | 有効 |
| `--capital-from-wallet` | broker余力から配分資本を取得 | 有効 |
| `--text-output` | コンソールにテキスト注文表を出力 | 有効 |
| `--api-dry-run` | 注文をシミュレーション（実際には送信しない） | 無効 |

gap参照は本番設定のSQLiteを使い、batchから`latest` directoryで上書きしない。
gap生成に失敗した場合も、当日cache → 許可されたon-demand → flatの順に判定する。
`--google-opens`や固定`--capital`は現行batchでは指定していない。

> [!IMPORTANT]
> 本番運用前に必ず `--api-dry-run` を追加してテスト実行してください。

### `run_close_positions.sh` のオプション（macOS正本）

`leadlag cli close` に渡すオプションを変更できます：

| オプション | 説明 | デフォルト |
|---|---|---|
| `--api-dry-run` | 注文をシミュレーション（実際には送信しない） | 無効 |
| `--close-position-order 0-7` | 返済順序（ClosePositionOrder）指定 | 0 |

## トラブルシューティング

| 症状 | 対処法 |
|---|---|
| タスクが実行されない (macOS) | プロジェクトがiCloud内にないか確認。iCloud内ではlaunchdが`Operation not permitted`で失敗します |
| `venv not found` (macOS) | `.venv` がプロジェクトルートに存在するか確認 |
| ログインが失敗する | kabuステーションがタスクバーにピン留めされているか確認 |
| OTP取得に失敗 | `creds/token.json` が有効か確認（初回は手動で認証フローを実行） |
| 土日祝に実行される | schedulerの起動日とCLIの市場休業日判定は別。発注可否はCLIの営業日判定とログを確認する |

## 期限・排他と未完了runの復旧

decision・close・gap・引け後レポートbatchは`job_guard`を通る。decision内のgap生成は親のleaseを共有する。
decision/close/gapの既定の全体期限は1800秒、TERM後の猶予は10秒で、環境変数
`LEADLAG_DECISION_TIMEOUT_SECONDS` / `LEADLAG_CLOSE_TIMEOUT_SECONDS` /
`LEADLAG_GAP_TIMEOUT_SECONDS` / `LEADLAG_JOB_GRACE_SECONDS`で調整できる。
gapがdecisionの子である場合はdecision全体の期限が適用される。
引け後の照合・レポート全体の既定期限は300秒で、`LEADLAG_RECONCILIATION_TIMEOUT_SECONDS`で調整する。

exit 124は期限超過、73はlease競合である。runは注文がFILLEDになっただけでは完了しない。
約定・建玉・余力・journalの照合を終えるまで`executing`または`reconciliation_required`に残る。
同じ口座・戦略の未解決runがあると、別日・別jobの送信も停止する。

brokerへ接続せず、保存台帳の復旧候補を表示する:

```bash
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 30 \
  .venv/bin/python -m leadlag.execution.reconcile \
  --state-db var/live/pipeline_data/execution/execution_state.sqlite
```

大引け後など、保存済み注文の終端を確認できる時点で、対象runを読取専用で照合する:

```bash
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 \
  .venv/bin/python -m leadlag.execution.reconcile \
  --state-db var/live/pipeline_data/execution/execution_state.sqlite \
  --run-id '<一覧に表示されたrun_id>' \
  --output-dir var/results/reconciliation
```

このコマンドは設定済みbrokerから注文状態・約定・建玉・余力を取得し、照合結果を保存する。
発注・取消・再送は行わない。注文ID不明、開始在庫未保存、数量/価格/費用の不足・不一致は
未解決のまま残す。runの状態だけを手動でcompletedへ書き換えて再送しない。

未解決runがない場合でも、実口座の残高・建玉を発注なしで記録するには次を使う。これは
brokerの注文ID一覧を取得するAPIではないため、約定の突合が必要な場合は上記`--run-id`を使う。

```bash
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 180 \
  .venv/bin/python -m leadlag.execution.reconcile \
  --account-snapshot \
  --output-dir var/results/reconciliation
```

15:40の既存`run_pnl_report.sh`は、最初に`reconcile --pending`を呼ぶ。
設定済みbroker・production_v2の未解決runだけを照合し、dry-runや別口座を対象にしない。
候補がなければbroker接続を省く。照合失敗時も独立した損益レポート作成を試みるが、
batchの終了コードは非ゼロのままとする。レポート成功だけで実行台帳を完了にしない。
実schedulerへの登録状態と実brokerでの照合証跡は、引き続きS5bの運用確認事項である。

## 引け損益レポート（オプション）

大引け（15:30）後に当日の実現・未実現損益をまとめた Markdown レポートを作成し、Gmail API で送信できます。
レポート処理は **15:40** に起動する `com.leadlag.pnl_report` として独立スケジュール化されており、`close` ジョブとは分離されています。

### 準備

1. スケジューラをセットアップし直します（新規ジョブを登録）：
   ```bash
   bash scripts/batch/setup_scheduler_macos.sh
   ```

2. Gmail 送信用 OAuth トークンを発行（初回のみ）:
   ```bash
   python tools/production/send_daily_close_pnl_report.py --authorize
   ```
   - 使用スコープは `https://www.googleapis.com/auth/gmail.send`
   - トークンは `creds/token_gmail_send.json` に保存されます
   - `creds/credentials.json` が必要です（OTP自動取得で使っているものを流用可能）

3. `.env` に以下を追加:
   ```bash
   LEADLAG_PNL_REPORT_SEND=1
   LEADLAG_PNL_REPORT_RECIPIENTS=you@example.com,ops@example.com
   # LEADLAG_PNL_FROM_EMAIL=optional-sender@example.com
   ```

### 動作

- `run_pnl_report.sh` は 15:40 に起動
- 同じ口座・戦略の未完了runを読取専用で照合し、約定・建玉・余力・journalの確認後に完了を記録
- 直近の `var/results/...production_close_positions` ディレクトリを自動検出
- 約定情報をブローカー API から再取得し `close_execution_log.json` を更新
- 引け後の残存ポジション・ウォレットスナップショットを取得（`positions_pnl_YYYYMMDD.json`, `wallet_pnl_YYYYMMDD.json`）
- `daily_pnl_report_YYYYMMDD.md` を作成し、設定されていればメール送信
- `LEADLAG_PNL_REPORT_SEND=1` かつ `LEADLAG_PNL_REPORT_RECIPIENTS` が設定されている場合のみメール送信されます
- デフォルトは **dry-run / ファイル保存のみ** で、勝手にメールは送信されません
- レポート失敗してもポジションクローズには影響しません
