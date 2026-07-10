---
name: hang-prevention
description: CLI実行中のスタック（ハング）を防止・診断する。既知の5パターン（auto-close無限待機・yfinanceハング・fcntlロック競合・API再試行バックオフ・注文フィル確認待ち）のタイムアウトガード実装・ハング診断を行う。docs/スタック再発防止策.mdのP1-P9対策に対応。CLI実行・長時間プロセス実行時に必ず参照すること。
---

# Hang Prevention スキル

## 目的

CLI実行中のプロセス停止（ハング）を防止し、発生時には迅速に診断・復旧する。

## 既知の5パターン

### パターンA: `wait_and_auto_close` の無限待機

- **場所**: `src/leadlag/execution/close.py:298-306`
- **原因**: `while True` + `time.sleep(300)` で14:50まで待機。launchd/cronから見ると停止しているように見える
- **対策 P1**: `--auto-close` フラグを削除し、close を別プロセス（`com.leadlag.close.plist`）に分離
- **対策 P5**: ハートビートログを追加（スリープ前後にログ出力）

### パターンB: yfinance ダウンロードのハング

- **場所**: `src/leadlag/data/fetcher.py:237-248`, `src/leadlag/broker/tachibana/client.py:206-223`
- **原因**: `yf.download()` にタイムアウトが未設定。Yahoo Finance側のレート制限・メンテナンス時に無限待機
- **対策 P2**: `signal.alarm` または `ThreadPoolExecutor` + `future.result(timeout=60)` で60秒タイムアウト
- **対策 P7**: Tachibana broker の yfinance 依存をローカルキャッシュに切り替え

### パターンC: ファイルロックの競合

- **場所**: `src/leadlag/data/cache.py:136-147`
- **原因**: `fcntl.flock(LOCK_EX)` がブロッキングモード。前回プロセスのクラッシュでロックファイルが残存すると次回がハング
- **対策 P3**: `exclusive_lock` にタイムアウト引数を追加（デフォルト30秒）。タイムアウト時は `LockTimeoutError` を送出

### パターンD: API再試行の指数バックオフ

- **場所**: `src/leadlag/broker/kabu/api.py:268-311`
- **原因**: `backoff_factor * (2**attempt)` でスリープ時間が無限増大。`max_retries` 回まで繰り返す
- **対策 P4**: `min(backoff_factor * (2**attempt), max_sleep)` を適用（`max_sleep=10`）

### パターンE: 注文後のフィル確認待ち

- **場所**: `src/leadlag/execution/helpers.py:806`
- **原因**: 注文送信後の `time.sleep(wait_seconds)` が長い場合、停止しているように見える
- **対策 P5**: ハートビートログを追加

## 対策優先度マトリクス

| 対策 | 影響度 | 難易度 | 優先度 | 対象 |
|------|--------|--------|--------|------|
| P1: auto-close分離 | 高 | 低 | **即座** | A |
| P2: yfinanceタイムアウト | 高 | 中 | **即座** | B |
| P3: ロックタイムアウト | 中 | 低 | **即座** | C |
| P4: API再試行上限 | 中 | 低 | **即座** | D |
| P5: ハートビートログ | 低 | 低 | 中期 | A, E |
| P6: シグナルハンドラ | 中 | 中 | 中期 | 全般 |
| P7: yfinance依存除去 | 中 | 高 | 中期 | B |
| P8: プロセス監視 | 低 | 中 | 長期 | 全般 |
| P9: 自動ダンプ | 低 | 中 | 長期 | 全般 |

## 実行前チェックリスト

CLI実行前に以下を確認:

- [ ] **broker API起動確認**: kabuステーション または 立花証券アプリが起動しているか
- [ ] **`--auto-close` 未使用確認**: decision実行に `--auto-close` が付いていないか
- [ ] **前回プロセス残存確認**: `ps aux | grep leadlag` で前回プロセスが残っていないか
- [ ] **ロックファイル確認**: `results/.cache/*.lock` に古いロックファイルが残っていないか
- [ ] **ネットワーク確認**: Yahoo Finance / Google Finance に到達可能か
- [ ] **`--fast-mode` 使用確認**: 標準mode（yfinance使用）を避け、fast-mode（API経由）を使用しているか

## ハング発生時の復旧手順

```bash
# 1. プロセス確認・強制終了
ps aux | grep leadlag
kill -9 <PID>

# 2. ロックファイル削除
rm -f results/.cache/*.lock

# 3. ログ確認
tail -50 logs/decision_*.log

# 4. 手動再実行（fast-mode推奨）
PYTHONPATH=src .venv-mac/bin/python -m leadlag.cli decision \
    --api-enable --fast-mode --capital-from-wallet --text-output
```

## タイムアウト実装パターン

### yfinanceタイムアウト（P2）

```python
import concurrent.futures

def download_with_timeout(tickers, start, end, timeout=60):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(yf.download, tickers, start=start, end=end, auto_adjust=False)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutExpired:
            logger.error("yfinance download timed out after %ds", timeout)
            raise
```

### ファイルロックタイムアウト（P3）

```python
import fcntl
import time

@contextlib.contextmanager
def exclusive_lock(lock_path: str, timeout: float = 30.0):
    with open(lock_path, "a+b") as lock_file:
        deadline = time.time() + timeout
        while True:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Lock acquisition timed out after {timeout}s: {lock_path}")
                time.sleep(0.5)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
```

### API再試行上限（P4）

```python
max_sleep = 10  # seconds
wait_time = min(backoff_factor * (2 ** attempt), max_sleep)
time.sleep(wait_time)
```

## 注意事項

- **長時間実行はタイムアウト付きで**: launchd/cron の `TimeoutSeconds` または外部 watchdog を設定
- **詳細は `docs/スタック再発防止策.md` を参照**: 本スキルは同ドキュメントのサマリー+実装ガイド
- **P1は運用対応**: コード変更不要。バッチスクリプトから `--auto-close` を削除するだけ
