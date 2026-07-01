# 自動スケジューラ セットアップガイド

日米ラグ戦略の自動実行を設定する手順です。Windows と macOS の両方に対応しています。

## スケジュール一覧

| タスク名 | 実行時刻 | Windows | macOS | 内容 |
|---|---|---|---|---|
| `日米ラグ_AutoLogin` | 毎朝 7:00 | `run_auto_login.bat` | — | kabuステーション自動ログイン |
| `日米ラグ_Decision` | 毎朝 9:05 | `run_decision.bat` | `run_decision.sh` | 売買判定 (`leadlag cli decision`) |
| `日米ラグ_ClosePositions` | 毎日 14:50 | `run_close_positions.bat` | `run_close_positions.sh` | 引け反対売買 (`leadlag cli close`) |

## 前提条件

### 共通
- `.env` に `KABU_ACCOUNT_NUMBER`, `KABU_PASSWORD` 等の環境変数が設定済みであること
- `creds/credentials.json` (Gmail API) が配置済みで、初回認証（`token.json` 生成）が完了していること

### Windows
- Windows 10/11
- Python 仮想環境 (`.venv`) がプロジェクトルートに存在すること

### macOS
- Python 仮想環境 (`.venv-mac`) がプロジェクトルートに存在すること
- プロジェクトディレクトリが iCloud 外にあること（iCloud 内では launchd が `Operation not permitted` エラーで実行できません）

## セットアップ手順

### Windows

#### 1. 自動セットアップ（推奨）

PowerShell を **管理者として** 開き、以下を実行：

```powershell
cd "プロジェクトディレクトリのパス\scripts\batch"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_scheduler.ps1
```

これで3つのタスクがすべて登録されます。

#### 2. 手動セットアップ

タスクスケジューラ GUI (`taskschd.msc`) から手動で登録する場合：

1. **タスクスケジューラ**を開く（`Win + R` → `taskschd.msc`）
2. **タスクの作成** を選択
3. 以下を設定：
   - **全般タブ**: タスク名を入力、「最上位の特権で実行する」にチェック
   - **トリガータブ**: 「毎日」を選択し、実行時刻を設定
   - **操作タブ**: 「プログラムの開始」で `cmd.exe` を指定し、引数に `/c "バッチファイルのフルパス"` を入力
   - **条件タブ**: 「コンピュータを AC 電源で使用している場合のみ」のチェックを外す
   - **設定タブ**: 「スケジュールされた時刻にタスクを開始できなかった場合、すぐにタスクを実行する」にチェック

### macOS

#### 1. 自動セットアップ（推奨）

```bash
bash scripts/batch/setup_scheduler_macos.sh
```

これで2つのタスク（Decision / Close）が launchd に登録されます。

> [!WARNING]
> プロジェクトディレクトリが iCloud 内にある場合、launchd からスクリプトにアクセスできません（`Operation not permitted`）。iCloud 外のディレクトリに移動してからセットアップしてください。

#### 2. 手動テスト実行

```bash
bash scripts/batch/run_decision.sh
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
```

## 動作確認

### ログ確認

実行ログは `logs/` ディレクトリに日付別で出力されます：

```
logs/
├── auto_login_20260507.log
├── decision_20260507.log
└── close_positions_20260507.log
```

### Windows タスク状態確認

```powershell
Get-ScheduledTask -TaskName "日米ラグ*" | Format-Table TaskName, State
```

### macOS タスク状態確認

```bash
launchctl list | grep leadlag
```

## タスクの削除

### Windows

```powershell
Unregister-ScheduledTask -TaskName "日米ラグ_AutoLogin" -Confirm:$false
Unregister-ScheduledTask -TaskName "日米ラグ_Decision" -Confirm:$false
Unregister-ScheduledTask -TaskName "日米ラグ_ClosePositions" -Confirm:$false
```

### macOS

```bash
launchctl unload ~/Library/LaunchAgents/com.leadlag.decision.plist
launchctl unload ~/Library/LaunchAgents/com.leadlag.close.plist
```

## スクリプトのカスタマイズ

### `run_decision` のオプション (Windows: `.bat` / macOS: `.sh`)

`leadlag cli decision` に渡すオプションを変更できます：

| オプション | 説明 | デフォルト |
|---|---|---|
| `--api-enable` | kabuステーション API 経由で注文送信 | 有効 |
| `--google-opens` | Google Finance から寄付値を取得 | 有効 |
| `--text-output` | コンソールにテキスト注文表を出力 | 有効 |
| `--api-dry-run` | 注文をシミュレーション（実際には送信しない） | 無効 |
| `--capital 1000000` | 運用資本（JPY） | 1,000,000 |

> [!IMPORTANT]
> 本番運用前に必ず `--api-dry-run` を追加してテスト実行してください。
> `run_decision.bat` 内の python コマンド行に `--api-dry-run` を追加するだけです。

### `run_close_positions` のオプション (Windows: `.bat` / macOS: `.sh`)

`leadlag cli close` に渡すオプションを変更できます：

| オプション | 説明 | デフォルト |
|---|---|---|
| `--api-dry-run` | 注文をシミュレーション（実際には送信しない） | 無効 |
| `--close-position-order 0-7` | 返済順序（ClosePositionOrder）指定 | 0 |

## トラブルシューティング

| 症状 | 対処法 |
|---|---|
| タスクが実行されない (Windows) | PCがスリープ状態。電源オプションで「スリープ解除タイマーを許可」を有効化 |
| タスクが実行されない (macOS) | プロジェクトがiCloud内にないか確認。iCloud内ではlaunchdが`Operation not permitted`で失敗します |
| `仮想環境が見つかりません` (Windows) | `.venv` がプロジェクトルートに存在するか確認 |
| `venv not found` (macOS) | `.venv-mac` がプロジェクトルートに存在するか確認 |
| ログインが失敗する | kabuステーションがタスクバーにピン留めされているか確認 |
| OTP取得に失敗 | `creds/token.json` が有効か確認（初回は手動で認証フローを実行） |
| 土日祝に実行される | 現在は毎日実行。休日判定が必要な場合はスクリプトにロジック追加が必要 |
