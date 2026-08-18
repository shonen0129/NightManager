@echo off
REM ============================================================
REM leadlag cli decision（毎朝 9:00 実行）
REM ============================================================
chcp 65001 >nul

set "PROJECT_DIR=%~dp0../.."
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "LOG_DIR=%PROJECT_DIR%\logs"

REM ログディレクトリ作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ログファイル名（日付付き）
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "DATESTR=%%i"
set "LOG_FILE=%LOG_DIR%\decision_%DATESTR%.log"

echo [%date% %time%] === leadlag decision 開始 === >> "%LOG_FILE%"

REM 仮想環境のアクティベート
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo [ERROR] 仮想環境が見つかりません: %VENV_DIR% >> "%LOG_FILE%"
    exit /b 1
)

REM スクリプト実行
cd /d "%PROJECT_DIR%"
set PYTHONPATH=src
python -m leadlag.cli decision --api-enable --capital-from-wallet --text-output >> "%LOG_FILE%" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] === 終了コード: %EXIT_CODE% === >> "%LOG_FILE%"

exit /b %EXIT_CODE%
