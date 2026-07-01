# ============================================================
# Windows タスクスケジューラ自動登録スクリプト
# 管理者権限で実行してください: 右クリック → 「管理者として実行」
# ============================================================

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# プロジェクトルートの自動検出
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)

$UserId = "$env:USERDOMAIN\$env:USERNAME"
$RunLevel = "Highest"
if (-not (Test-IsAdmin)) {
    Write-Host "[WARN] 管理者権限ではありません。タスクは通常権限で登録されます。" -ForegroundColor Yellow
    $RunLevel = "Limited"
}
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel $RunLevel

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " 日米ラグ 自動スケジューラ セットアップ" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "プロジェクトディレクトリ: $ProjectDir" -ForegroundColor Yellow
Write-Host ""

# ログディレクトリ作成
$LogDir = Join-Path $ProjectDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
    Write-Host "[OK] ログディレクトリ作成: $LogDir" -ForegroundColor Green
}

# 共通設定
$TaskSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

# ------------------------------------------------------------
# タスク 1: kabuステーション自動ログイン（毎朝 7:00）
# ------------------------------------------------------------
$TaskName1 = "日米ラグ_AutoLogin"
$BatPath1 = Join-Path $ScriptDir "run_auto_login.bat"

$Trigger1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
$Action1 = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatPath1`"" `
    -WorkingDirectory $ScriptDir

# 既存タスクがあれば削除
if (Get-ScheduledTask -TaskName $TaskName1 -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName1 -Confirm:$false
    Write-Host "[INFO] 既存タスク '$TaskName1' を削除しました" -ForegroundColor Yellow
}

Register-ScheduledTask `
    -TaskName $TaskName1 `
    -Description "kabuステーション自動ログイン（毎朝7:00）" `
    -Trigger $Trigger1 `
    -Action $Action1 `
    -Principal $Principal `
    -Settings $TaskSettings | Out-Null

Write-Host "[OK] タスク登録完了: $TaskName1 (毎朝 7:00)" -ForegroundColor Green

# ------------------------------------------------------------
# タスク 2: production.py --mode decision（毎朝 9:00）
# ------------------------------------------------------------
$TaskName2 = "日米ラグ_Decision"
$BatPath2 = Join-Path $ScriptDir "run_decision.bat"

$Trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "09:00"
$Action2 = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatPath2`"" `
    -WorkingDirectory $ScriptDir

if (Get-ScheduledTask -TaskName $TaskName2 -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName2 -Confirm:$false
    Write-Host "[INFO] 既存タスク '$TaskName2' を削除しました" -ForegroundColor Yellow
}

Register-ScheduledTask `
    -TaskName $TaskName2 `
    -Description "production decision 売買判定（毎朝9:00）" `
    -Trigger $Trigger2 `
    -Action $Action2 `
    -Principal $Principal `
    -Settings $TaskSettings | Out-Null

Write-Host "[OK] タスク登録完了: $TaskName2 (毎朝 9:00)" -ForegroundColor Green

# ------------------------------------------------------------
# タスク 3: production.py --mode close-positions（毎日 14:00）
# ------------------------------------------------------------
$TaskName3 = "日米ラグ_ClosePositions"
$BatPath3 = Join-Path $ScriptDir "run_close_positions.bat"

$Trigger3 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "14:00"
$Action3 = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatPath3`"" `
    -WorkingDirectory $ScriptDir

if (Get-ScheduledTask -TaskName $TaskName3 -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName3 -Confirm:$false
    Write-Host "[INFO] 既存タスク '$TaskName3' を削除しました" -ForegroundColor Yellow
}

Register-ScheduledTask `
    -TaskName $TaskName3 `
    -Description "production close-positions 引け反対売買（毎日14:00）" `
    -Trigger $Trigger3 `
    -Action $Action3 `
    -Principal $Principal `
    -Settings $TaskSettings | Out-Null

Write-Host "[OK] タスク登録完了: $TaskName3 (毎日 14:00)" -ForegroundColor Green

# ------------------------------------------------------------
# 登録確認
# ------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " 登録済みタスク一覧" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$RegisteredTasks = @($TaskName1, $TaskName2, $TaskName3)
foreach ($tn in $RegisteredTasks) {
    $task = Get-ScheduledTask -TaskName $tn -ErrorAction SilentlyContinue
    if ($task) {
        $info = $task | Get-ScheduledTaskInfo
        $trigger = $task.Triggers[0]
        Write-Host ""
        Write-Host "  タスク名    : $tn" -ForegroundColor White
        Write-Host "  状態        : $($task.State)" -ForegroundColor White
        Write-Host "  実行時刻    : $($trigger.StartBoundary)" -ForegroundColor White
        Write-Host "  説明        : $($task.Description)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " セットアップ完了！" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "ログ出力先: $LogDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "タスクの確認・変更:" -ForegroundColor Yellow
Write-Host "  taskschd.msc（タスクスケジューラ GUI）で確認できます" -ForegroundColor Gray
Write-Host ""
Write-Host "手動テスト実行:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn `"$TaskName1`"" -ForegroundColor Gray
Write-Host "  schtasks /run /tn `"$TaskName2`"" -ForegroundColor Gray
Write-Host "  schtasks /run /tn `"$TaskName3`"" -ForegroundColor Gray
Write-Host ""
