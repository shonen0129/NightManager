# 自動スケジューラ セットアップガイド

Windows タスクスケジューラを使用して、日米ラグ戦略の自動実行を設定する手順です。

## スケジュール一覧

| タスク名 | 実行時刻 | スクリプト | 内容 |
|---|---|---|---|
| `日米ラグ_AutoLogin` | 毎朝 7:00 | `run_auto_login.bat` | kabuステーション自動ログイン |
| `日米ラグ_Decision` | 毎朝 9:00 | `run_decision.bat` | 売買判定 (`--mode decision`) |
| `日米ラグ_ClosePositions` | 毎日 14:00 | `run_close_positions.bat` | 引け反対売買 (`--mode close-positions`) |

## 前提条件

- Windows 10/11
- Python 仮想環境 (`.venv`) がプロジェクトルートに存在すること
- `src/.env` に `KABU_ACCOUNT_NUMBER`, `KABU_PASSWORD` 等の環境変数が設定済みであること
- `creds/credentials.json` (Gmail API) が配置済みで、初回認証（`token.json` 生成）が完了していること

## セットアップ手順

### 1. 自動セットアップ（推奨）

PowerShell を **管理者として** 開き、以下を実行：

```powershell
cd "プロジェクトディレクトリのパス\scripts"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_scheduler.ps1
```

これで3つのタスクがすべて登録されます。

### 2. 手動セットアップ

タスクスケジューラ GUI (`taskschd.msc`) から手動で登録する場合：

1. **タスクスケジューラ**を開く（`Win + R` → `taskschd.msc`）
2. **タスクの作成** を選択
3. 以下を設定：
   - **全般タブ**: タスク名を入力、「最上位の特権で実行する」にチェック
   - **トリガータブ**: 「毎日」を選択し、実行時刻を設定
   - **操作タブ**: 「プログラムの開始」で `cmd.exe` を指定し、引数に `/c "バッチファイルのフルパス"` を入力
   - **条件タブ**: 「コンピュータを AC 電源で使用している場合のみ」のチェックを外す
   - **設定タブ**: 「スケジュールされた時刻にタスクを開始できなかった場合、すぐにタスクを実行する」にチェック

## 動作確認

### 手動テスト実行

```powershell
# 個別タスクを手動で実行
schtasks /run /tn "日米ラグ_AutoLogin"
schtasks /run /tn "日米ラグ_Decision"
schtasks /run /tn "日米ラグ_ClosePositions"
```

### ログ確認

実行ログは `logs/` ディレクトリに日付別で出力されます：

```
logs/
├── auto_login_20260507.log
├── decision_20260507.log
└── close_positions_20260507.log
```

### タスク状態確認

```powershell
Get-ScheduledTask -TaskName "日米ラグ*" | Format-Table TaskName, State
```

## タスクの削除

```powershell
Unregister-ScheduledTask -TaskName "日米ラグ_AutoLogin" -Confirm:$false
Unregister-ScheduledTask -TaskName "日米ラグ_Decision" -Confirm:$false
Unregister-ScheduledTask -TaskName "日米ラグ_ClosePositions" -Confirm:$false
```

## バッチファイルのカスタマイズ

### `run_decision.bat` のオプション

`production.py --mode decision` に渡すオプションを変更できます：

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

### `run_close_positions.bat` のオプション

`production.py --mode close-positions` に渡すオプションを変更できます：

| オプション | 説明 | デフォルト |
|---|---|---|
| `--api-dry-run` | 注文をシミュレーション（実際には送信しない） | 無効 |
| `--close-position-order 0-7` | 返済順序（ClosePositionOrder）指定 | 0 |

## トラブルシューティング

| 症状 | 対処法 |
|---|---|
| タスクが実行されない | PCがスリープ状態。電源オプションで「スリープ解除タイマーを許可」を有効化 |
| `仮想環境が見つかりません` | `.venv` がプロジェクトルートに存在するか確認 |
| ログインが失敗する | kabuステーションがタスクバーにピン留めされているか確認 |
| OTP取得に失敗 | `creds/token.json` が有効か確認（初回は手動で認証フローを実行） |
| 土日祝に実行される | 現在は毎日実行。休日判定が必要な場合はバッチファイルにロジック追加が必要 |
